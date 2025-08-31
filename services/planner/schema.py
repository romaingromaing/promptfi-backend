# class GateExpr(BaseModel):
#     expr: str

# class Gates(BaseModel):
#     all_of: List[GateExpr] = []
#     any_of: List[GateExpr] = []

# class Weighting(BaseModel):
#     mode: Literal["composite"] = "composite"

# class Rebalance(BaseModel):
#     cadence: Literal["weekly"] = "weekly"
#     band_pp: float = 5.0
#     turnover_max: float = 0.15
#     hard_cap: float = 0.50

# class Risk(BaseModel):
#     max_weight: float = 0.40
#     slippage_max_bps: int = 80
#     order_max_usd: float = 2000
#     cooldown_hours: int = 6

# class Plan(BaseModel):
#     name: str = "Auto_Expert"
#     universe: List[Literal["BTC","ETH","SOL"]] = ["BTC","ETH","SOL"]
#     gates: Gates = Gates()
#     weighting: Weighting = Weighting()
#     rebalance: Rebalance = Rebalance()
#     risk: Risk = Risk()

#     @field_validator("universe")
#     @classmethod
#     def non_empty(cls, v):
#         if not v: raise ValueError("Universe cannot be empty.")
#         return v

# {
#   "name": "Auto_Expert",
#   "regime": "auto",
#   "direction_bias": "neutral",
#   "universe_list": ["BTC","ETH","SOL"],
#   "custom_rules": [],
#   "gates": {
#     "trend":    { "ema_short": 50, "ema_long": 200, "adx_min": 20 },
#     "range":    { "adx_max": 20,  "bb_bw_pct_max": 0.30 },
#     "breakout": { "donchian_n": 20, "min_vol_mult": 1.5, "adx_rising": true },
#     "support":  { "atr_mult": 0.8, "rsi_min": 40 },
#     "sentiment": "AUTO"
#   },
#   "weighting": {
#     "mode": "composite",
#     "coeffs": { "trend": 0.35, "momentum": 0.35, "volume": 0.15, "sentiment": 0.15 },
#     "tilt_sentiment_pct": 0.10
#   },
#   "rebalance": { "cadence": "weekly", "band_pp": 5.0, "turnover_max": 0.15 },
#   "risk": { "max_weight": 0.40, "hard_cap": 0.50, "slippage_max_bps": 80, "order_max_usd": 2000, "cooldown_hours": 6 },
#   "execution": { "chunk_usd": 2000, "use_yield": false },
#   "sentiment_cfg": { "good_threshold": 0.30, "bad_threshold": -0.30, "shock_delta_24h": 0.50 }
# }

from pydantic import BaseModel, Field, field_validator, ValidationError
from typing import List, Literal, Optional, Dict, Union

GOOD_THRESH = 0.30
BAD_THRESH = -0.30

class TrendGates(BaseModel):
    ema_short: Optional[int] = None
    ema_long: Optional[int] = None
    adx_min: Optional[float] = None

class RangeGates(BaseModel):
    adx_max: Optional[float] = None
    bb_bw_pct_max: Optional[float] = None

class BreakoutGates(BaseModel):
    donchian_n: Optional[int] = None
    min_vol_mult: Optional[float] = None
    adx_rising: Optional[bool] = None

class SupportGates(BaseModel):
    atr_mult: Optional[float] = None
    rsi_min: Optional[int] = None

class Gates(BaseModel):
    trend: Optional[TrendGates] = Field(default_factory=TrendGates)
    range: Optional[RangeGates] = Field(default_factory=RangeGates)
    breakout: Optional[BreakoutGates] = Field(default_factory=BreakoutGates)
    support: Optional[SupportGates] = Field(default_factory=SupportGates)
    sentiment: Union[Literal["AUTO"], Optional[str]] = None

class WeightingCoeffs(BaseModel):
    trend: Optional[float] = None
    momentum: Optional[float] = None
    volume: Optional[float] = None
    sentiment: Optional[float] = None

class Weighting(BaseModel):
    mode: Literal["composite"] = "composite"
    coeffs: Optional[WeightingCoeffs] = Field(default_factory=WeightingCoeffs)
    tilt_sentiment_pct: Optional[float] = None

class Rebalance(BaseModel):
    cadence: Literal["weekly"] = "weekly"
    band_pp: float = 5.0
    turnover_max: float = 0.15

class Risk(BaseModel):
    max_weight: float = 0.40
    hard_cap: float = 0.50
    slippage_max_bps: int = 80
    order_max_usd: float = 2000
    cooldown_hours: int = 6

class Execution(BaseModel):
    chunk_usd: Optional[float] = None
    use_yield: Optional[bool] = None

class SentimentCfg(BaseModel):
    good_threshold: Optional[float] = None
    bad_threshold: Optional[float] = None
    shock_delta_24h: Optional[float] = None

class Plan(BaseModel):
    name: str = "Auto_Expert"
    regime: Union[Literal["auto"], Optional[str]] = None
    direction_bias: Union[Literal["neutral", "bullish", "bearish"], Optional[str]] = None
    universe_list: List[str] = Field(default_factory=list)
    custom_rules: List[str] = Field(default_factory=list)
    gates: Gates = Field(default_factory=Gates)
    weighting: Weighting = Field(default_factory=Weighting)
    rebalance: Rebalance = Field(default_factory=Rebalance)
    risk: Risk = Field(default_factory=Risk)
    execution: Optional[Execution] = Field(default_factory=Execution)
    sentiment_cfg: Optional[SentimentCfg] = Field(default_factory=SentimentCfg)

    @field_validator("universe_list")
    @classmethod
    def non_empty_universe(cls, v):
        if not v:
            raise ValueError("Universe list cannot be empty.")
        return v

# {
#   "name": "Auto_Expert",
#   "regime": "auto",
#   "direction_bias": "neutral",
#   "universe_list": ["BTC","ETH","SOL"],
#   "custom_rules": [],
#   "gates": {
#     "trend":    { "ema_short": 50, "ema_long": 200, "adx_min": 20 },
#     "range":    { "adx_max": 20,  "bb_bw_pct_max": 0.30 },
#     "breakout": { "donchian_n": 20, "min_vol_mult": 1.5, "adx_rising": true },
#     "support":  { "atr_mult": 0.8, "rsi_min": 40 },
#     "sentiment": "AUTO"
#   },
#   "weighting": {
#     "mode": "composite",
#     "coeffs": { "trend": 0.35, "momentum": 0.35, "volume": 0.15, "sentiment": 0.15 },
#     "tilt_sentiment_pct": 0.10
#   },
#   "rebalance": { "cadence": "weekly", "band_pp": 5.0, "turnover_max": 0.15 },
#   "risk": { "max_weight": 0.40, "hard_cap": 0.50, "slippage_max_bps": 80, "order_max_usd": 2000, "cooldown_hours": 6 },
#   "execution": { "chunk_usd": 2000, "use_yield": false },
#   "sentiment_cfg": { "good_threshold": 0.30, "bad_threshold": -0.30, "shock_delta_24h": 0.50 }
# }