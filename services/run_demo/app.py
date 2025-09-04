# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# import pandas as pd
# from services.planner.plan_analyzer import analyze_plan
# from services.data.ohlcv import load_ohlcv
# from services.data.sentiment import fetch_cp_headlines, score_headlines
# from engine.backtest import run_backtest
# from dotenv import load_dotenv
# from services.planner.app import plan_debug
# import os  
# load_dotenv()
# auth_token=os.getenv("CP_AUTH_TOKEN")
# app = FastAPI()

# class BacktestReq(BaseModel):
#     plan: dict
#     start: str | None = None
#     end: str | None = None
#     cp_key: str | None = None

# # #for debugging 
# # plan_res= plan_debug("Trade BTC and ETH when CLOSE is greater than SMA(30) use best possible techniques")
# # BackTest= {
# #     "plan": plan_res,
# #     "start": None,
# #     "end": None,
# #     "cp_key": auth_token
# # }

# @app.post("/backtest")
# def backtest(req: BacktestReq):
#     try:
#         meta = analyze_plan(req.plan)
#         assets = meta["assets"]; days = meta["lookback_days"]
#         ohlcv = {a: load_ohlcv(a, days) for a in assets}
#         cp = fetch_cp_headlines(auth_token)
#         sent = score_headlines(cp)
#         ec, stats = run_backtest(req.plan, ohlcv, sent, start=req.start, end=req.end)
#         return {"stats": stats, "equity_curve": [{"t": str(t), "equity": float(v)} for t,v in ec["equity"].items()]}
#     except Exception as e:
#         raise HTTPException(400, str(e))

# # def main(req: BacktestReq=BackTest):
# #     try:
# #         meta = analyze_plan(req.plan)
# #         assets = meta["assets"]; days = meta["lookback_days"]
# #         ohlcv = {a: load_ohlcv(a, days) for a in assets}
# #         cp = fetch_cp_headlines(auth_token)
# #         sent = score_headlines(cp)
# #         ec, stats = run_backtest(req.plan, ohlcv, sent, start=req.start, end=req.end)
# #         print({"stats": stats, "equity_curve": [{"t": str(t), "equity": float(v)} for t,v in ec["equity"].items()]})
# #     except Exception as e:
# #         raise HTTPException(400, str(e))
    
# # if __name__ == "__main__":
# #     print(main())


import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pprint import pprint
from services.planner.plan_analyzer import build_plan_json_from_text, analyze_features, build_plan_with_gemini
from services.data.data_layer import load_universe
from services.data.sentiment import fetch_headlines, rolling_sentiment
from engine.engine import Plan
from engine.backtest import run_backtest
from dotenv import load_dotenv
import pandas as pd

load_dotenv()
auth_token = os.getenv("CP_AUTH_TOKEN")

app = FastAPI(title="Run Demo API")


class PlanRequest(BaseModel):
    text: str


class BacktestReq(BaseModel):
    plan: dict
    start: str | None = None
    end: str | None = None
    cp_key: str | None = None
    debug: bool = False


def to_plan_obj(pj: dict) -> Plan:
    return Plan(
        regime=pj["regime"],
        direction_bias=pj.get("direction_bias", "neutral"),
        universe=pj["universe_list"],
        gates=pj["gates"],
        custom_rules=pj.get("custom_rules", []),
        weighting=pj["weighting"],
        rebalance=pj["rebalance"],
        risk=pj["risk"],
        execution=pj["execution"],
        sentiment_cfg=pj.get("sentiment_cfg", {}),
    )


@app.post("/plan")
def get_plan(req: PlanRequest):
    """Convert user text into Plan JSON using existing helper (no Gemini)."""
    try:
        plan_json = build_plan_with_gemini(req.text)
        analysis = analyze_features(plan_json)
        return {"plan": plan_json, "analysis": analysis}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/get_headlines")
def get_headlines():
    try:
        cp = fetch_headlines(auth_token)
        if isinstance(cp, pd.DataFrame):
            return {"headlines": cp.reset_index(drop=True).to_dict(orient="records")}
        return {"headlines": []}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/backtest")
def backtest(req: BacktestReq):
    try:
        meta = analyze_features(req.plan)
        assets = meta["assets"]
        days = meta["lookback_days"]
        ohlcv = load_universe(assets, since_days=days)
        cp = fetch_headlines(auth_token if req.cp_key is None else req.cp_key)
        sent = rolling_sentiment(cp) if isinstance(cp, pd.DataFrame) and not cp.empty else {}

        plan_obj = to_plan_obj(req.plan)
        ec, stats = run_backtest(plan_obj, ohlcv, sent, start=req.start, end=req.end)

        # Handle both DataFrame and dict output for ec
        equity_curve = []
        if isinstance(ec, dict) and "equity" in ec and isinstance(ec["equity"], dict):
            for t, v in ec["equity"].items():
                equity_curve.append({"t": str(t), "equity": float(v)})
        elif hasattr(ec, "index") and hasattr(ec, "__getitem__"):
            # Assume ec is a DataFrame with an 'equity' column
            for t, v in zip(ec.index, ec["equity"]):
                equity_curve.append({"t": str(t), "equity": float(v)})

        return {"stats": stats, "equity_curve": equity_curve}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    # keep previous script behavior when run directly for convenience
    user_text = "I'm bullish on BTC and ETH. You choose the best technicals with sentiment good."
    plan_json = build_plan_with_gemini(user_text)
    print("\n--- Plan JSON ---")
    pprint(plan_json)

    feats = analyze_features(plan_json)
    ohlcv = load_universe(feats["assets"], since_days=feats["lookback_days"])
    news = fetch_headlines(os.getenv("CP_AUTH_TOKEN"))
    sent = rolling_sentiment(news) if not news.empty else {}

    plan = to_plan_obj(plan_json)
    recent = {a: df.tail(250) for a, df in ohlcv.items()}
    # best-effort: call target_weights if available, else skip explain
    try:
        from engine.engine import target_weights, build_trade_plan
        tw, explains = target_weights(plan, recent, sent)
        print("\n--- Target Weights ---")
        pprint(tw)
        print("\n--- Explain ---")
        pprint(explains)
    except Exception:
        print("target_weights not available or raised an error; skipping explain")

    ec, stats = run_backtest(plan, ohlcv, sent, start="2024-01-01")
    print("\n--- Backtest Stats ---")
    pprint(stats)
    try:
        print("\n--- Equity tail ---")
        print(ec.tail())
    except Exception:
        pass

