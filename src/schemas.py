from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field, field_validator

BBox = list[float]


class ProductGroup(BaseModel):
    id: str = ""
    sku_id: str | None = None
    sku_name: str = ""
    bbox: BBox = Field(default_factory=lambda: [0, 0, 0, 0])
    confidence: float = 0.0
    facing_count: int | None = None
    observed_text: str = ""

    @field_validator("bbox")
    @classmethod
    def validate_bbox(cls, value: BBox) -> BBox:
        if len(value) != 4:
            raise ValueError("bbox 必须为 [x1,y1,x2,y2]")
        return [float(x) for x in value]


class PriceTag(BaseModel):
    id: str = ""
    bbox: BBox = Field(default_factory=lambda: [0, 0, 0, 0])
    raw_text: str = ""
    name_hint: str = ""
    sku_id: str | None = None
    price: float | None = None
    price_type: Literal["normal", "member", "promotion", "unknown"] = "unknown"
    confidence: float = 0.0

    @field_validator("bbox")
    @classmethod
    def validate_bbox(cls, value: BBox) -> BBox:
        if len(value) != 4:
            raise ValueError("bbox 必须为 [x1,y1,x2,y2]")
        return [float(x) for x in value]


class MatchItem(BaseModel):
    product_group_id: str
    price_tag_id: str | None = None
    sku_id: str | None = None
    sku_name: str = ""
    price: float | None = None
    price_type: str = "unknown"
    score: float = 0.0
    status: Literal["accepted", "review", "unmatched"] = "unmatched"
    reasons: list[str] = Field(default_factory=list)
    raw_text: str = ""
    # 一个商品陈列组可能对应零个、一个或多个价签。保留候选价签明细，
    # 供前端判断“价格一致 / 价格不一致 / 无法判断”。
    candidate_price_tag_ids: list[str] = Field(default_factory=list)
    candidate_prices: list[float | None] = Field(default_factory=list)


class AnalysisResult(BaseModel):
    filename: str = ""
    scene_type: Literal[
        "standard_shelf",
        "hook_display",
        "basket_display",
        "promotion_stack",
        "replenishment",
        "unknown",
    ] = "unknown"
    scene_confidence: float = 0.0
    image_width: int = 0
    image_height: int = 0
    products: list[ProductGroup] = Field(default_factory=list)
    price_tags: list[PriceTag] = Field(default_factory=list)
    matches: list[MatchItem] = Field(default_factory=list)
    unmatched_price_tags: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    model_raw_response: str = ""
    elapsed_seconds: float = 0.0
    extra: dict[str, Any] = Field(default_factory=dict)
