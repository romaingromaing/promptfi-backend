from pydantic import BaseModel, Field, field_validator, ValidationError
from typing import List, Literal, Optional, Dict

GOOD_THRESH = 0.30
BAD_THRESH  = -0.30

class GateExpr(BaseModel):
    expr: str

class Gates(BaseModel):
    all_of: List[GateExpr] = []
    any_of: List[GateExpr] = []

class Weighting(BaseModel):
    mode: Literal["composite"] = "composite"

class Rebalance(BaseModel):
    cadence: Literal["weekly"] = "weekly"
    band_pp: float = 5.0
    turnover_max: float = 0.15
    hard_cap: float = 0.50

class Risk(BaseModel):
    max_weight: float = 0.40
    slippage_max_bps: int = 80
    order_max_usd: float = 2000
    cooldown_hours: int = 6

class Plan(BaseModel):
    name: str = "UserPlan"
    universe: List[Literal["BTC","ETH","SOL"]] = ["BTC","ETH","SOL"]
    gates: Gates = Gates()
    weighting: Weighting = Weighting()
    rebalance: Rebalance = Rebalance()
    risk: Risk = Risk()

    @field_validator("universe")
    @classmethod
    def non_empty(cls, v):
        if not v: raise ValueError("Universe cannot be empty.")
        return v
