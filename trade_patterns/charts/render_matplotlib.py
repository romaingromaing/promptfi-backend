import io, base64
import matplotlib             
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from typing import Dict, Any, List
from .chart_schema import ChartJSON

#helper function to transform the raw OHLCV data into a Pandas DataFrame.
def _ohlcv_to_df(ohlcv: Dict[str, List[float]]) -> pd.DataFrame:
    # ohlcv = {"t":[...ms],"o":[...],"h":[...],"l":[...],"c":[...],"v":[...]}
    df = pd.DataFrame({
        "time": pd.to_datetime(ohlcv["t"], unit="ms", utc=True),
        "open": ohlcv["o"], "high": ohlcv["h"], "low": ohlcv["l"], "close": ohlcv["c"], "volume": ohlcv["v"]
    }).set_index("time")
    return df

# def render_png(ohlcv: Dict[str, Any], overlays: ChartJSON, width=900, height=500) -> str:
#     df = _ohlcv_to_df(ohlcv)
#     fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)

#     # Simple OHLC (candles)
#     x = mdates.date2num(df.index.to_pydatetime())
#     for i, (t, row) in enumerate(df.iterrows()):
#         color = "green" if row["close"] >= row["open"] else "red"
#         ax.plot([x[i], x[i]], [row["low"], row["high"]], linewidth=1, color=color)
#         ax.add_line(plt.Line2D([x[i]-0.2, x[i]+0.2], [row["open"], row["open"]], color=color, linewidth=3))
#         ax.add_line(plt.Line2D([x[i]-0.2, x[i]+0.2], [row["close"], row["close"]], color=color, linewidth=3))

#     # Overlays
#     for shp in overlays.get("series", []):
#         t = shp.get("type")
#         if t == "line":
#             pts = shp["points"]
#             ax.plot([mdates.epoch2num(p[0]/1000) for p in pts], [p[1] for p in pts],
#                     linestyle="--" if shp.get("style", {}).get("dashed") else "-",
#                     linewidth=shp.get("style", {}).get("width", 1))
#         elif t == "ray":
#             start = shp["from_"]
#             x0 = mdates.epoch2num(start[0]/1000)
#             y0 = start[1]
#             x1 = x[-1]
#             ax.plot([x0, x1], [y0, y0], linestyle="--")
#         elif t == "box":
#             p1, p2 = shp["p1"], shp["p2"]
#             xs = [mdates.epoch2num(p1[0]/1000), mdates.epoch2num(p2[0]/1000)]
#             ys = [p1[1], p2[1]]
#             ax.fill_between(xs, ys[0], ys[1], alpha=shp.get("style", {}).get("alpha", 0.1))
#         elif t == "poly":
#             pts = shp["points"]
#             ax.plot([mdates.epoch2num(p[0]/1000) for p in pts], [p[1] for p in pts])
#         elif t == "label":
#             at = shp["at"]; ax.text(mdates.epoch2num(at[0]/1000), at[1], shp["text"])
#         elif t == "level":
#             y = shp["y"]; ax.axhline(y, linestyle="--")

#     ax.xaxis_date(); ax.set_title("Auto Chart")
#     fig.autofmt_xdate()
#     buf = io.BytesIO()
#     plt.tight_layout()
#     plt.savefig(buf, format="png", bbox_inches="tight")
#     plt.close(fig)
#     return base64.b64encode(buf.getvalue()).decode("ascii")


def render_png(ohlcv: Dict[str, Any], overlays: ChartJSON, width=900, height=500) -> str:
    df = _ohlcv_to_df(ohlcv)
    fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)

    # Simple OHLC (candles) - This part is already correct
    x = mdates.date2num(df.index.to_pydatetime())
    for i, (t, row) in enumerate(df.iterrows()):
        color = "green" if row["close"] >= row["open"] else "red"
        ax.plot([x[i], x[i]], [row["low"], row["high"]], linewidth=1, color=color)
        ax.add_line(plt.Line2D([x[i]-0.2, x[i]+0.2], [row["open"], row["open"]], color=color, linewidth=3))
        ax.add_line(plt.Line2D([x[i]-0.2, x[i]+0.2], [row["close"], row["close"]], color=color, linewidth=3))

    # Overlays - THIS IS WHERE THE FIXES ARE
    for shp in overlays.get("series", []):
        t = shp.get("type")
        if t == "line":
            pts = shp["points"]
            # FIX: Replace mdates.epoch2num(p[0]/1000)
            ax.plot([p[0] / 86400000 for p in pts], [p[1] for p in pts],
                    linestyle="--" if shp.get("style", {}).get("dashed") else "-",
                    linewidth=shp.get("style", {}).get("width", 1))
        elif t == "ray":
            start = shp["from_"]
            # FIX: Replace mdates.epoch2num(start[0]/1000)
            x0 = start[0] / 86400000
            y0 = start[1]
            x1 = x[-1]
            ax.plot([x0, x1], [y0, y0], linestyle="--")
        elif t == "box":
            p1, p2 = shp["p1"], shp["p2"]
            # FIX: Replace mdates.epoch2num for both points
            xs = [p1[0] / 86400000, p2[0] / 86400000]
            ys = [p1[1], p2[1]]
            ax.fill_between(xs, ys[0], ys[1], alpha=shp.get("style", {}).get("alpha", 0.1))
        elif t == "poly":
            pts = shp["points"]
            # FIX: Replace mdates.epoch2num(p[0]/1000)
            ax.plot([p[0] / 86400000 for p in pts], [p[1] for p in pts])
        elif t == "label":
            at = shp["at"]
            # FIX: Replace mdates.epoch2num(at[0]/1000)
            ax.text(at[0] / 86400000, at[1], shp["text"])
        elif t == "level":
            y = shp["y"]; ax.axhline(y, linestyle="--")

    ax.xaxis_date(); ax.set_title("Auto Chart")
    fig.autofmt_xdate()
    buf = io.BytesIO()
    plt.tight_layout()
    # Corrected the method call here from plt.savefig to fig.savefig
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    # return base64.b64encode(buf.getvalue()).decode("ascii")
    return buf.getvalue() 