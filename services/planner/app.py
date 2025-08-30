from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from .schema import Plan, GOOD_THRESH, BAD_THRESH, Gates, GateExpr, Weighting # Import Weighting
import google.generativeai as genai
import os
import json 
from dotenv import load_dotenv
from .plan_analyzer import analyze_plan
# from data.ohlcv import load_ohlcv
from services.data.ohlcv import ccxt_ohlcv, cg_market_chart_range
import pandas as pd
from services.data.sentiment import fetch_cp_headlines, score_headlines


load_dotenv()  
app = FastAPI()
auth_token=os.getenv("CP_AUTH_TOKEN")
class PlanRequest(BaseModel):
    """Pydantic model for the incoming API request body."""
    text: str

def map_sentiment_words(s: str) -> str:
    """Replace descriptive sentiment terms with their numerical equivalents."""
    s = s.replace("SENTIMENT >= GOOD", f"SENTIMENT >= {GOOD_THRESH}")
    s = s.replace("SENTIMENT <= BAD",  f"SENTIMENT <= {BAD_THRESH}")
    return s

def clean_llm_output(llm_output_string: str) -> str:
    """Removes markdown code block delimiters from the LLM output."""
    cleaned_output = llm_output_string.strip()
    if cleaned_output.startswith("```json"):
        cleaned_output = cleaned_output[len("```json"):].lstrip('\n')
    elif cleaned_output.startswith("```"):
        cleaned_output = cleaned_output[len("```"):].lstrip('\n')
    if cleaned_output.endswith("```"):
        cleaned_output = cleaned_output[:-3].rstrip('\n')
    return cleaned_output.strip()

def naive_parse(text: str) -> Plan:
    """
    Converts user free text into a Plan JSON using Gemini.
    This function replaces the naive rule-based parsing.
    """
    try: 
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY environment variable is not set.")
            else:
                 print("Warning: Using deprecated GEMINI_API_KEY. Please use GOOGLE_API_KEY instead.")

        genai.configure(api_key=api_key)

        model = genai.GenerativeModel(model_name="gemini-2.0-flash")

    except Exception as e:
        raise ValueError(f"Failed to initialize Gemini model: {e}")

    prompt = f"""
    You are an AI assistant that converts natural-language investment strategy descriptions into structured JSON plans.
    Output exactly one JSON object (no explanation, no markdown). The JSON must follow this structure:

    {{
    "name": "string",
    "universe": ["BTC","ETH"] (at least one),
    "gates": {{"all_of":[{{"expr":"string"}}],"any_of":[{{"expr":"string"}}]}},
    "weighting": {{"mode":"composite"}},
    "rebalance": {{"cadence":"weekly", "band_pp": number, "turnover_max": number, "hard_cap": number}},
    "risk": {{"max_weight": number, "slippage_max_bps": integer, "order_max_usd": number, "cooldown_hours": integer}}
    }}

    CRITICAL rules (must follow):
    1. Do NOT output JSON null values for any numeric field. If a numeric value is unknown, EITHER:
    - omit that key entirely from the object, OR
    - set it to a reasonable numeric default (see rule 2).
    2. Reasonable defaults if needed:
    - rebalance.band_pp = 0.0
    - rebalance.turnover_max = 0.0
    - rebalance.hard_cap = 1.0
    - risk.max_weight = 1.0
    - risk.slippage_max_bps = 0
    - risk.order_max_usd = 0.0
    - risk.cooldown_hours = 0
    3. For sentiment terms:
    - "SENTIMENT >= GOOD" -> "SENTIMENT >= {GOOD_THRESH}"
    - "SENTIMENT <= BAD"  -> "SENTIMENT <= {BAD_THRESH}"
    4. Only include the `weighting` object with a single field `"mode": "composite"` — do not include an `assets` or other weighting fields.
    5. Output must be valid JSON that exactly matches the types above (numbers as numbers, integers as integers). No extra text.

    Example output for the user strategy "Buy BTC when sentiment is positive and price above 50-day SMA":
    {{
    "name": "BTC Sentiment & Price Momentum",
    "universe": ["BTC"],
    "gates": {{"all_of":[{{"expr":"SENTIMENT >= 0.3"}},{{"expr":"PRICE > SMA(50)"}}],"any_of":[]}},
    "weighting": {{"mode":"composite"}},
    "rebalance": {{"cadence":"weekly", "band_pp": 0.0, "turnover_max": 0.0, "hard_cap": 1.0}},
    "risk": {{"max_weight": 1.0, "slippage_max_bps": 0, "order_max_usd": 0.0, "cooldown_hours": 0}}
    }}

    User strategy: "{text}"
    Please output the single JSON object now.
    """

    try:
        response = model.generate_content(prompt)
        llm_output_string = response.text
        print(f"LLM Raw Output: {llm_output_string}") 
    except Exception as e:
        raise ValueError(f"Gemini API call failed: {e}")

    try:
        cleaned_output = clean_llm_output(llm_output_string)
        print(f"LLM Cleaned Output: {cleaned_output}") 
        plan = Plan.model_validate_json(cleaned_output)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM generated invalid JSON after cleaning: {e}. Raw output: {llm_output_string}. Cleaned output: {cleaned_output}")
    except Exception as e:
        raise ValueError(f"LLM generated JSON that does not match Plan schema after cleaning: {e}. Raw output: {llm_output_string}. Cleaned output: {cleaned_output}")

    for gate_list in [plan.gates.all_of, plan.gates.any_of]:
        for gate_expr in gate_list:
            gate_expr.expr = map_sentiment_words(gate_expr.expr)

    return plan

@app.post("/plan")
def plan(req: PlanRequest):
    """API endpoint to convert user text strategy into Plan JSON."""
    try:
        generated_plan = naive_parse(req.text)
        analysis = analyze_plan(generated_plan.model_dump())
        print(f"Plan Analysis: {analysis}") 
        return generated_plan.model_dump() 
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An internal error occurred.")

@app.get("/get_ohclv", response_model=None)
def get_ohlcv(symbol: str, days: int):
    """Return OHLCV as JSON-serializable records.

    FastAPI cannot create a response model from pandas.DataFrame types, so
    convert the DataFrame to a list of dicts with ISO timestamps.
    """
    try:
        pair = f"{symbol}/USDT"
        df = ccxt_ohlcv(pair, since_days=days)
    except Exception:
        df = cg_market_chart_range(symbol, days=days)

    # Reset index (time), convert timestamps to ISO strings and return records
    df = df.reset_index()
    # rename index column to time if it's named 't'
    if 't' in df.columns:
        df = df.rename(columns={'t': 'time'})
    # ensure timestamps are JSON-serializable strings
    if 'time' in df.columns:
        df['time'] = df['time'].astype(str)

    return {"ohlcv": df.to_dict(orient='records')}

@app.get("/get_headlines")
def get_headlines():
    try:
        cp = fetch_cp_headlines(auth_token=auth_token)
        return {"headlines": cp.to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error fetching headlines: {e}")

@app.get("/get_sentiment")
def get_sentiment(symbols: str = None):
    """
    Get sentiment scores for all headlines, or filter by comma-separated symbols (e.g., ETH,BTC).
    """
    try:
        cp = fetch_cp_headlines(auth_token=auth_token)
        sent = score_headlines(cp)
        if symbols:
            symbol_list = [s.strip().upper() for s in symbols.split(",")]
            filtered = {k: v.to_dict() for k, v in sent.items() if k in symbol_list}
            return {"sentiment": filtered}
        else:
            return {"sentiment": {k: v.to_dict() for k, v in sent.items()}}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error fetching sentiment: {e}")