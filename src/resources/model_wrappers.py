"""
Model wrappers.

Hackathon standard: every model call goes through **OpenRouter** with one
``OPENROUTER_API_KEY``. The paper's model names (``gpt-4.1``, ``claude-3-7-sonnet-latest``,
``gemini-2.5-flash-preview-04-17``...) are accepted as aliases and mapped to OpenRouter ids
(see ``cda_types.MODEL_ALIASES``); any other OpenRouter id (``qwen/qwen3-14b``,
``mistralai/mistral-small-3.2-24b-instruct``...) works as-is. The paper's default,
``gpt-4.1``, resolves to ``openai/gpt-4.1``.

Set ``LLM_BASE_URL`` to point the client at another OpenAI-compatible server (a local
Ollama, a mock in tests). To serve several models from local vLLM servers at once, set
``LLM_ENDPOINTS`` to a JSON map from model id to base URLs, e.g.
``{"Qwen/Qwen3-32B-FP8": ["http://localhost:8001/v1", "http://localhost:8002/v1"]}``; each client
(one per agent) sticks to one URL so the server's prefix cache keeps that agent's history warm.
The original direct-SDK clients (OpenAI, Anthropic, Vertex AI)
are kept below behind ``LLM_PROVIDER=openai|anthropic|google`` for reproduction only; their
SDKs are optional extras and imported lazily.
"""

from abc import ABC, abstractmethod
from functools import lru_cache
from itertools import count
from typing import List, Optional, TypedDict
import json
import logging
import os
import time

from openai import OpenAI, APIError, BadRequestError
from tqdm import tqdm

from poseidon_compute import make_client, resolve_model

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

# Offset by PID so parallel simulation processes spread their agents over the servers
_endpoint_counter = count(os.getpid())


@lru_cache(maxsize=1)
def local_endpoints() -> dict[str, list[str]]:
    """LLM_ENDPOINTS: model id -> base URLs of local OpenAI-compatible servers (see module docstring)."""
    raw = os.getenv("LLM_ENDPOINTS")
    return {model: [urls] if isinstance(urls, str) else urls for model, urls in json.loads(raw).items()} if raw else {}


class Message(TypedDict):
    role: str
    content: str


class ModelWrapper(ABC):
    """Abstract base class for model API wrappers."""

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.7,
        max_tokens: int = 4000,
        top_p: float = 1.0,
        max_retries: int = 3,
        initial_retry_delay: float = 1.0,
        **kwargs,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.max_retries = max_retries
        self.initial_retry_delay = initial_retry_delay
        self.additional_params = kwargs

    @abstractmethod
    def generate(self, messages: List[Message]) -> Optional[str]:
        """Generate a response for a single prompt."""

    def batch_generate(self, messages_list: List[List[Message]]) -> List[Optional[str]]:
        return [self.generate(m) for m in tqdm(messages_list, desc="Batch Generation")]

    @classmethod
    def create(cls, model_name: str, **kwargs) -> "ModelWrapper":
        """Factory: OpenRouter unless LLM_PROVIDER selects a direct SDK."""
        provider = os.getenv("LLM_PROVIDER", "openrouter").lower()
        if provider == "openai":
            return OpenAIClient(model_name, **kwargs)
        if provider == "anthropic":
            return AnthropicClient(model_name, **kwargs)
        if provider == "google":
            return GoogleClient(model_name, **kwargs)
        return OpenRouterClient(model_name, **kwargs)

    def _exponential_backoff(self, attempt: int) -> None:
        if attempt < self.max_retries:
            time.sleep(self.initial_retry_delay * (2**attempt))


class OpenRouterClient(ModelWrapper):
    """
    Default client. ``model_name`` must already be an OpenRouter id (use
    ``cda_types.resolve_model`` for the paper's aliases).

    If a model rejects ``response_format={"type": "json_object"}`` the request is retried
    without it (the prompts already demand JSON-only output).
    """

    is_remote = True

    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name, **kwargs)
        self.model_name = resolve_model(model_name)
        urls = local_endpoints().get(self.model_name)
        if urls:
            # Long timeout: a busy local server queues requests instead of rate-limiting them
            url = urls[next(_endpoint_counter) % len(urls)]
            self.client = OpenAI(base_url=url, api_key=os.getenv("LLM_API_KEY") or "EMPTY", timeout=900)
        else:
            self.client = make_client(timeout=120)
        self.is_remote = "localhost" not in str(self.client.base_url) and "127.0.0.1" not in str(self.client.base_url)

    def generate(self, messages: List[Message], **overrides) -> Optional[str]:
        """``overrides`` replace request parameters for this call only; a value of None drops the parameter."""
        params = {"temperature": self.temperature, "max_tokens": self.max_tokens, "top_p": self.top_p,
                  **self.additional_params, **overrides}
        params = {k: v for k, v in params.items() if v is not None}
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    **params,
                )
                content = response.choices[0].message.content or ""
                self.last_thinking = ""
                if "</think>" in content:  # reasoning models: keep only the answer, and the thinking for the logs
                    thinking, content = content.rsplit("</think>", 1)
                    self.last_thinking = thinking.replace("<think>", "").strip()
                return content
            except BadRequestError as e:
                if "response_format" in params and "response_format" in str(e).lower():
                    logger.warning(f"{self.model_name} rejected JSON mode; retrying without response_format")
                    params.pop("response_format")
                    self.additional_params.pop("response_format", None)
                    continue
                logger.warning(f"OpenRouter API error (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt == self.max_retries - 1:
                    return None
                self._exponential_backoff(attempt)
            except APIError as e:
                logger.warning(f"OpenRouter API error (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt == self.max_retries - 1:
                    logger.error(f"Failed after {self.max_retries} attempts: {e}")
                    return None
                self._exponential_backoff(attempt)
        return None


# --------------------------------------------------------------------------- #
# Original direct-SDK clients (opt-in via LLM_PROVIDER; not used in the hackathon)
# --------------------------------------------------------------------------- #
class OpenAIClient(ModelWrapper):
    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name, **kwargs)
        self.client = OpenAI(timeout=60)

    def generate(self, messages: List[Message]) -> Optional[str]:
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    max_completion_tokens=self.max_tokens,
                    top_p=self.top_p,
                    **self.additional_params,
                )
                return response.choices[0].message.content
            except APIError as e:
                logger.warning(f"OpenAI API error (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt == self.max_retries - 1:
                    return None
                self._exponential_backoff(attempt)
        return None


class AnthropicClient(ModelWrapper):
    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name, **kwargs)
        from anthropic import Anthropic  # optional extra

        self.client = Anthropic(timeout=60)

    def generate(self, messages: List[Message]) -> Optional[str]:
        from anthropic import APIConnectionError, APIStatusError, RateLimitError

        if messages[0]["role"] == "system":
            system, messages = messages[0]["content"], messages[1:]
        else:
            system = "You are a helpful assistant."
        for attempt in range(self.max_retries):
            try:
                response = self.client.messages.create(
                    model=self.model_name,
                    system=system,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    **self.additional_params,
                )
                return response.content[0].text
            except (APIConnectionError, RateLimitError, APIStatusError) as e:
                logger.warning(f"Anthropic API error (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt == self.max_retries - 1:
                    return None
                self._exponential_backoff(attempt)
        return None


class GoogleClient(ModelWrapper):
    def __init__(self, model_name: str, project: Optional[str] = None, location: str = "us-central1", **kwargs):
        super().__init__(model_name, **kwargs)
        import vertexai  # optional extra
        from vertexai.generative_models import GenerativeModel

        project = project or os.getenv("GOOGLE_PROJECT_ID")
        if project is None:
            raise ValueError("GOOGLE_PROJECT_ID must be set for LLM_PROVIDER=google")
        vertexai.init(project=project, location=location)
        self.model = GenerativeModel(model_name=self.model_name)

    def generate(self, messages: List[Message]) -> Optional[str]:
        from vertexai.generative_models import Content, GenerationConfig, Part

        if messages and messages[0]["role"] == "system":
            messages = messages[1:]
        contents = [
            Content(role="model" if m["role"] == "assistant" else "user", parts=[Part.from_text(m["content"])])
            for m in messages
            if m["role"] in ("user", "assistant")
        ]
        config = GenerationConfig(temperature=self.temperature, max_output_tokens=65535, top_p=self.top_p)
        for attempt in range(self.max_retries):
            try:
                response = self.model.generate_content(contents=contents, generation_config=config)
                if response.candidates and response.candidates[0].content.parts:
                    return response.candidates[0].content.parts[0].text
                return None
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Google API error (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt == self.max_retries - 1:
                    return None
                self._exponential_backoff(attempt)
        return None
