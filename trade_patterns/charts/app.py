from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
from .render_matplotlib import render_png

app = FastAPI()

class ChartReq(BaseModel):
    ohlcv: Dict[str, Any]
    overlays: Dict[str, Any]
    width: int = 900
    height: int = 500

@app.post("/charts/render")
def render(req: ChartReq):
    try:
        pngb64 = render_png(req.ohlcv, req.overlays, req.width, req.height)
        return {"png_base64": pngb64}
    except Exception as e:
        raise HTTPException(400, str(e))
