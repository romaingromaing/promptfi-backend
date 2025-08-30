# import requests, pandas as pd
# from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# CP_API = "https://cryptopanic.com/api/v1/posts/"
# CP_DEVELOPER_API="https://cryptopanic.com/api/developer/v2/posts/"

# sid = SentimentIntensityAnalyzer()

# def fetch_cp_headlines(auth_token: str, page_size=50):
#     r = requests.get(CP_DEVELOPER_API, params={
#         "auth_token": auth_token,
#         "kind": "news",
#         "filter": "rising|hot|bullish|bearish",
#         "public": "true"
#     }, timeout=20)
#     r.raise_for_status()
#     items = r.json().get("results", [])
#     rows = []
#     for it in items:
#         ts = pd.to_datetime(it["published_at"], utc=True)
#         title = it.get("title","")
#         assets = [t.get("code","").upper() for t in it.get("currencies", []) if t.get("code")]
#         rows.append({"time":ts, "title":title, "assets":assets})
#     return pd.DataFrame(rows)

# def score_headlines(df: pd.DataFrame):
#     # Expand per asset and roll 6h mean
#     if df.empty: return {}
#     rows = []
#     for _,r in df.iterrows():
#         s = sid.polarity_scores(r["title"])["compound"]
#         for a in r["assets"] or []:
#             rows.append({"time": r["time"], "asset": a, "score": s})
#     sdf = pd.DataFrame(rows)
#     if sdf.empty: return {}
#     sdf = sdf.set_index("time").groupby("asset")["score"].rolling("6H").mean().reset_index()
#     out = {}
#     for a, g in sdf.groupby("asset"):
#         s = g.set_index("time")["score"].clip(-1,1)
#         out[a] = s
#     return out

import requests, pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

CP_API = "https://cryptopanic.com/api/developer/v2/posts/"

sid = SentimentIntensityAnalyzer()

def fetch_cp_headlines(auth_token: str, page_size=50):
    r = requests.get(CP_API, params={
        "auth_token": auth_token,
        "kind": "news",
        "filter": "rising|hot|bullish|bearish",
        "public": "true"
    }, timeout=20)
    r.raise_for_status()
    items = r.json().get("results", [])
    rows = []
    for it in items:
        # print(it)
        ts = pd.to_datetime(it["published_at"], utc=True)
        title = it.get("title","")
        assets = []
        if "bitcoin" in title.lower(): assets.append("BTC")
        if "ethereum" in title.lower(): assets.append("ETH")
        if "solana" in title.lower(): assets.append("SOL")
        # assets = [t.get("code","").upper() for t in it.get("currencies", []) if t.get("code")]
        # assets = [inst.get("code","").upper() for inst in it.get("instruments", []) if inst.get("code")]
        panic_score = it.get("panic_score", 0) 
        rows.append({"time":ts, "title":title, "assets":assets, "panic_score": panic_score})
        # rows.append({ "title":title, "assets":assets, "panic_score": panic_score})
    return pd.DataFrame(rows)

def score_headlines(df: pd.DataFrame):
    # Expand per asset and roll 6h mean
    if df.empty: return {}
    rows = []
    for _,r in df.iterrows():
        s_vader = sid.polarity_scores(r["title"])["compound"]
        # s_panic = r.get("panic_score", 0) / 100.0 
        # Combine VADER sentiment and normalized panic_score
        # s_combined = (0.7 * s_vader) + (0.3 * s_panic)
        s_combined = s_vader
        
        for a in r["assets"] or []:
            rows.append({"time": r["time"], "asset": a, "score": s_combined})
            
    sdf = pd.DataFrame(rows)
    if sdf.empty: return {}
    sdf = sdf.set_index("time").groupby("asset")["score"].rolling("6H").mean().reset_index()
    out = {}
    for a, g in sdf.groupby("asset"):
        s = g.set_index("time")["score"].clip(-1,1)
        out[a] = s
    return out