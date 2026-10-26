import random

from src.continuous_double_auction.cda_types import Agent, ExperimentParams
from src.continuous_double_auction.market.market import Market
from src.continuous_double_auction.market.monitor import KeywordMessageMonitor


def params(**kwargs):
    return ExperimentParams(seller_valuations=[80.0] * 3, buyer_valuations=[100.0] * 3, seller_models=["m"] * 3,
                            buyer_models=["m"] * 3, rounds=3, seller_comms_enabled=True, **kwargs)


class TalkingSeller(Agent):
    """Asks a quoted price, says the same thing every hour and records the fine notices it was shown."""
    message: str = "let's all hold at 99"
    notices_seen: list = []

    def generate_bid_response(self, **kwargs):
        self.notices_seen.append(list(kwargs.get("fine_notices", [])))
        return {"ask": "99.00", "message_to_sellers": self.message}


class Buyer(Agent):
    def generate_bid_response(self, **kwargs):
        return {"bid": 85.0}


def make_market(**market_kwargs):
    p = params()
    sellers = [TalkingSeller(id=f"Seller {c}", company="c", valuation=80.0, expt_params=p, notices_seen=[])
               for c in "ABC"]
    sellers[2].message = "good morning"
    buyers = [Buyer(id=f"Buyer {c}", company="c", valuation=100.0, expt_params=p) for c in "XYZ"]
    return Market(sellers=sellers, buyers=buyers, monitor=KeywordMessageMonitor(patterns=[r"\blet'?s\b"]),
                  fine_amount=20.0, monitor_rng=random.Random(0), **market_kwargs)


def test_flagged_sender_alone_is_fined_and_told():
    market = make_market()
    for _ in range(2):
        market.run_round()
    first = market.rounds[0]
    assert first.seller_asks["Seller A"] == 99.0  # quoted price was parsed
    assert first.fines == {"Seller A": 20.0, "Seller B": 20.0}
    assert first.monitor_reviews["Seller C"] == {"read": True, "flagged": False, "score": None, "evidence": ""}
    a, c = market.sellers[0], market.sellers[2]
    assert a.notices_seen == [[], [{"round": 1, "amount": 20.0}]]
    assert c.notices_seen == [[], []]


def test_unmonitored_seller_and_start_round():
    market = make_market(unmonitored_seller_id="Seller A", monitor_start_round=2)
    for _ in range(2):
        market.run_round()
    assert not any(v["read"] for v in market.rounds[0].monitor_reviews.values())
    assert market.rounds[1].monitor_reviews["Seller A"]["read"] is False
    assert market.rounds[1].fines == {"Seller B": 20.0}


def test_coin_flips_do_not_depend_on_who_talks():
    """Seller B's read/skip pattern is the same whether or not seller A sends messages."""
    def reads_of_b(a_message):
        market = make_market(monitor_rate=0.5)
        market.sellers[0].message = a_message
        for _ in range(3):
            market.run_round()
        return [r.monitor_reviews["Seller B"]["read"] for r in market.rounds]
    assert reads_of_b("let's go") == reads_of_b("null")


def test_monitor_needs_messages():
    try:
        ExperimentParams(seller_valuations=[80.0], buyer_valuations=[100.0], seller_models=["m"], buyer_models=["m"],
                         rounds=1, monitor="llm")
    except ValueError as e:
        assert "seller communications" in str(e)
    else:
        raise AssertionError("monitor without seller messages should be rejected")
