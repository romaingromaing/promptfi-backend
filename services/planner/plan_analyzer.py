# import re
# from typing import Dict, List

# MAX_LOOKBACK = 540

# def analyze_plan(plan: Dict) -> Dict:
#     assets = plan["universe"]
#     feats: List[str] = []

#     def add_feat(tok: str):
#         if tok not in feats: feats.append(tok)

#     patt_sma = re.compile(r"SMA\((\d+)(?:,VOLUME)?\)")
#     patt_ema = re.compile(r"EMA\((\d+)\)")
#     gates = plan.get("gates", {}).get("all_of", []) + plan.get("gates", {}).get("any_of", [])
#     for g in gates:
#         expr = g["expr"].upper()
#         if "SMA(" in expr:
#             for n in patt_sma.findall(expr):
#                 if ",VOLUME" in expr: add_feat(f"VOL_SMA_{n}")
#                 else: add_feat(f"SMA_{n}")
#         if "EMA(" in expr:
#             for n in patt_ema.findall(expr):
#                 add_feat(f"EMA_{n}")
#         if "RSI(14)" in expr: add_feat("RSI_14")
#         if "RET_60D" in expr: add_feat("RET_60D")
#         if "SENTIMENT" in expr: add_feat("SENTIMENT")

#     # derive lookback: max window + 20% buffer (min 90)
#     windows = []
#     for f in feats:
#         if f.startswith("SMA_") or f.startswith("EMA_") or f.startswith("VOL_SMA_"):
#             windows.append(int(f.split("_")[-1]))
#         if f == "RET_60D": windows.append(60)
#         if f == "RSI_14": windows.append(14)
#     need = max(windows) if windows else 60
#     need = int(min(MAX_LOOKBACK, max(90, need * 1.2)))

#     return {"assets": assets, "features": feats, "lookback_days": need}

# #aren't we replacing sentiment with thresholds in plan_analyzer?

import re
from engine.engine import Plan
from typing import List
from dotenv import load_dotenv
import os
import google.generativeai as genai
import json
load_dotenv()

api_key= os.getenv("GOOGLE_API_KEY")

DEFAULT_JSON = {
  "name": "Auto_Expert",
  "regime": "auto",
  "direction_bias": "neutral",
  "universe_list": ["BTC","ETH","SOL"],
  "custom_rules": [],
  "gates": {
    "trend": {"ema_short":50,"ema_long":200,"adx_min":20},
    "range": {"adx_max":20,"bb_bw_pct_max":0.30},
    "breakout": {"donchian_n":20,"min_vol_mult":1.5,"adx_rising":True},
    "support": {"atr_mult":0.8,"rsi_min":40},
    "sentiment":"AUTO"
  },
  "weighting": {
    "mode":"composite",
    "coeffs":{"trend":0.35,"momentum":0.35,"volume":0.15,"sentiment":0.15},
    "tilt_sentiment_pct":0.10
  },
  "rebalance":{"cadence":"weekly","band_pp":5.0,"turnover_max":0.15},
  "risk":{"max_weight":0.40,"hard_cap":0.50,"slippage_max_bps":80,"order_max_usd":2000,"cooldown_hours":6},
  "execution":{"chunk_usd":2000,"use_yield":False},
  "sentiment_cfg":{"good_threshold":0.30,"bad_threshold":-0.30,"shock_delta_24h":0.50}
}


def build_plan_json_from_text(user_text: str) -> dict:
    t = user_text.lower()
    plan = {**DEFAULT_JSON}
    # universe
    uni = []
    for k in ["btc","eth","sol"]:
        if re.search(rf"\b{k}\b", t): uni.append(k.upper())
    if uni: plan["universe_list"] = uni
    # direction bias
    if "bullish" in t: plan["direction_bias"] = "bullish"
    elif "bearish" in t: plan["direction_bias"] = "bearish"
    # regime hints
    for r in ["trend","range","breakout","support"]:
        if r in t: plan["regime"] = r
    # explicit rules
    rules = []
    if "price > 30d ma" in t or "above 30d" in t:
        rules.append("CLOSE>SMA(30)")
    m = re.search(r"rsi\s*>\s*([4-9]\d)", t)  # FIXED: correct whitespace and capture
    if m:
        rules.append(f"RSI(14)>{m.group(1)}")
    if "volume strong" in t:
        rules.append("VOLUME>1.5*VOL_SMA(20)")
    if "sentiment good" in t:
        rules.append("SENTIMENT>=GOOD")
    plan["custom_rules"] = rules
    return plan

def build_plan_with_gemini(user_text: str) -> dict:
    """
    Use Gemini to generate a structured Plan JSON.
    Falls back to regex parser if Gemini fails.
    Ensures return format matches DEFAULT_JSON.
    """
    try:
        genai.configure(api_key=api_key)
    except Exception as e:
        print(f"Gemini config error: {e}")
        return build_plan_json_from_text(user_text)

    # Instead of Plan.model_json_schema(), just show Gemini the expected keys
    prompt_parts = [
        "You are an expert crypto trading strategist. "
        "The user will provide a natural language description of a strategy. "
        "Your job is to fill out a Plan JSON object that strictly conforms to the schema below. "
        "If the user does not mention some fields, auto-populate them with robust defaults "
        "from technical analysis and risk management. "
        "Never leave required fields as null or None. "
        "\n\nJSON Schema Example:\n"
        f"{json.dumps(DEFAULT_JSON, indent=2)}\n\n"
        f"User's Strategy Description: \"{user_text}\"\n\n"
        "Output only the JSON object, no explanations."
    ]

    model = genai.GenerativeModel('gemini-2.0-flash')

    try:
        response = model.generate_content(
            prompt_parts,
            generation_config={"response_mime_type": "application/json"}
        )
        raw_json_output = response.text
        plan_data = json.loads(raw_json_output)

        # Merge Gemini output with DEFAULT_JSON (to ensure consistent keys)
        merged_plan = {**DEFAULT_JSON, **plan_data}
        return merged_plan

    except Exception as e:
        print(f"Gemini generation error: {e}")
        # Fallback: regex parser
        return build_plan_json_from_text(user_text)


    model = genai.GenerativeModel('gemini-2.0-flash')

    try:
        response = model.generate_content(
            prompt_parts,
            generation_config={"response_mime_type": "application/json"}
        )
        raw_json_output = response.text
        plan_data = json.loads(raw_json_output)

        # Validate against Pydantic Plan schema
        validated_plan = Plan.model_validate(plan_data)
        return validated_plan.model_dump()

    except Exception as e:
        # Fallback: regex parser
        return build_plan_json_from_text(user_text)


def analyze_features(plan_json: dict) -> dict:
    feats: List[str]= []
    def add_feats(tok:str):
        if tok not in feats: feats.append(tok)
    return {
        "assets": plan_json["universe_list"],
        "features": ["OHLCV","SMA/EMA","RSI/StochRSI","MACD","ADX/DI","ATR","Bollinger/Keltner","Donchian","OBV/CMF/MFI","VWAP","Sentiment"],
        "lookback_days": 540
    }
