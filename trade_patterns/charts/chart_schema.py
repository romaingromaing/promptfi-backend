from __future__ import annotations
from typing import List, Tuple, TypedDict, Literal, Dict, Any

XY = Tuple[int, float]  # epoch ms, price

class Line(TypedDict, total=False):
    type: Literal["line"]
    points: List[XY]
    style: Dict[str, Any]  # {"dashed":bool,"width":int}

class Ray(TypedDict, total=False):
    type: Literal["ray"]
    from_: XY
    angleDeg: float
    style: Dict[str, Any]

class Box(TypedDict, total=False):
    type: Literal["box"]
    p1: XY
    p2: XY
    style: Dict[str, Any]  # {"alpha":float}

class Poly(TypedDict, total=False):
    type: Literal["poly"]
    points: List[XY]
    style: Dict[str, Any]

class Label(TypedDict, total=False):
    type: Literal["label"]
    at: XY
    text: str

class Level(TypedDict, total=False):
    type: Literal["level"]
    y: float
    style: Dict[str, Any]

Shape = Line | Ray | Box | Poly | Label | Level

class ChartJSON(TypedDict):
    version: int  # =1
    series: List[Shape]

