import random

from src.continuous_double_auction.cda_types import Agent, ExperimentParams
from src.continuous_double_auction.market.agents import ZIPBuyer
from src.continuous_double_auction.market.market import Market


class FixedSeller(Agent):
    ask: float

    def generate_bid_response(self, **kwargs):
        return {"ask": self.ask}


def run(asks, hours, start_bid=88.0):
    """Three ZIP buyers (value $100) starting at start_bid against sellers who never change their asks."""
    random.seed(0)
    p = ExperimentParams(seller_valuations=[80.0] * 3, buyer_valuations=[100.0] * 3, seller_models=["m"] * 3,
                         buyer_models=["zip"] * 3, rounds=hours)
    sellers = [FixedSeller(id=f"Seller {c}", company="c", valuation=80.0, expt_params=p, ask=a) for c, a in zip("ABC", asks)]
    buyers = [ZIPBuyer(id=f"Buyer {c}", company="c", valuation=100.0, expt_params=p, profit_margin=1 - start_bid / 100)
              for c in "XYZ"]
    market = Market(sellers=sellers, buyers=buyers)
    for _ in range(hours):
        market.run_round()
    return market


def test_sellers_holding_out_together_pull_bids_up():
    market = run([99.0, 99.0, 99.0], hours=15)
    first_sale = next(r.round_number for r in market.rounds if r.trades)
    assert 3 < first_sale <= 12  # bids climb about $1.40 an hour while nothing trades
    assert all(t.price >= 99.0 for t in market.past_trades)


def test_seller_holding_out_alone_loses_the_sale():
    market = run([88.0, 88.0, 99.0], hours=15)
    assert market.past_trades and all(t.seller_id != "Seller C" for t in market.past_trades)
    assert all(t.price < 89.0 for t in market.past_trades)


def test_bids_never_exceed_value():
    market = run([120.0, 120.0, 120.0], hours=30)
    assert not market.past_trades
    assert max(max(r.buyer_bids.values()) for r in market.rounds) <= 100.0
