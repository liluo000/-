from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PIL import Image

from .catalog import catalog_by_id, resolve_sku
from .image_utils import crop_normalized, draw_analysis, normalize_bbox
from .matcher import match_products_and_tags

if TYPE_CHECKING:
    from .gemini_client import QwenVisionClient
from .schemas import AnalysisResult, PriceTag, ProductGroup


class RecognitionPipeline:
    def __init__(self, client: QwenVisionClient, catalog: list[dict], reference_contact_path: str | Path):
        self.client = client
        self.catalog = catalog
        self.by_id = catalog_by_id(catalog)
        path = Path(reference_contact_path)
        self.reference_contact = Image.open(path).convert("RGB") if path.exists() else None

    def run(
        self,
        image: Image.Image,
        filename: str,
        use_reference_contact: bool = True,
        refine_price_tags: bool = True,
        max_refine_tags: int = 16,
        accept_threshold: float = 0.72,
        review_threshold: float = 0.50,
    ) -> tuple[AnalysisResult, Image.Image]:
        started = time.perf_counter()
        width, height = image.size
        payload, raw_response = self.client.analyze(
            image,
            self.catalog,
            self.reference_contact if use_reference_contact else None,
        )
        scene_type = payload.get("scene_type", "unknown")
        if scene_type not in {
            "standard_shelf", "hook_display", "basket_display", "promotion_stack", "replenishment", "unknown"
        }:
            scene_type = "unknown"

        products = self._parse_products(payload.get("products", []), width, height)
        tags = self._parse_tags(payload.get("price_tags", []), width, height)

        if refine_price_tags:
            for tag in tags[:max_refine_tags]:
                try:
                    crop = crop_normalized(image, tag.bbox, padding=0.10)
                    refined, _ = self.client.read_price_tag(crop, self.catalog)
                    self._merge_refined_tag(tag, refined)
                except Exception as exc:  # noqa: BLE001
                    # 单个价签二次识别失败不应中断整张图。
                    tag.raw_text = tag.raw_text or f"二次OCR失败：{exc}"

        matches, unmatched_tags, score_matrix = match_products_and_tags(
            products,
            tags,
            scene_type,
            self.catalog,
            accept_threshold=accept_threshold,
            review_threshold=review_threshold,
        )
        result = AnalysisResult(
            filename=filename,
            scene_type=scene_type,
            scene_confidence=_safe_confidence(payload.get("scene_confidence")),
            image_width=width,
            image_height=height,
            products=products,
            price_tags=tags,
            matches=matches,
            unmatched_price_tags=unmatched_tags,
            notes=[str(note) for note in payload.get("notes", [])],
            model_raw_response=raw_response,
            elapsed_seconds=round(time.perf_counter() - started, 3),
            extra={"score_matrix": score_matrix.tolist()},
        )
        annotated = draw_analysis(
            image,
            [p.model_dump() for p in products],
            [t.model_dump() for t in tags],
            [m.model_dump() for m in matches],
        )
        return result, annotated

    def _parse_products(self, values: list[dict[str, Any]], width: int, height: int) -> list[ProductGroup]:
        products: list[ProductGroup] = []
        for index, item in enumerate(values):
            sku_id = item.get("sku_id")
            if sku_id not in self.by_id:
                sku_id, _, _ = resolve_sku(
                    f'{item.get("sku_name", "")} {item.get("observed_text", "")}', self.catalog
                )
            sku_name = self.by_id.get(sku_id or "", {}).get("name") or str(item.get("sku_name") or "未知SKU")
            products.append(ProductGroup(
                id=f"P{index + 1:02d}",
                sku_id=sku_id,
                sku_name=sku_name,
                bbox=normalize_bbox(item.get("bbox", [0, 0, 0, 0]), width, height),
                confidence=_safe_confidence(item.get("confidence")),
                facing_count=_safe_int(item.get("facing_count")),
                observed_text=str(item.get("observed_text") or ""),
            ))
        return products

    def _parse_tags(self, values: list[dict[str, Any]], width: int, height: int) -> list[PriceTag]:
        tags: list[PriceTag] = []
        for index, item in enumerate(values):
            raw_text = str(item.get("raw_text") or "")
            name_hint = str(item.get("name_hint") or "")
            sku_id = item.get("sku_id")
            if sku_id not in self.by_id:
                sku_id, _, _ = resolve_sku(f"{name_hint} {raw_text}", self.catalog)
            tags.append(PriceTag(
                id=f"T{index + 1:02d}",
                bbox=normalize_bbox(item.get("bbox", [0, 0, 0, 0]), width, height),
                raw_text=raw_text,
                name_hint=name_hint,
                sku_id=sku_id,
                price=_safe_float(item.get("price")),
                price_type=_normalize_price_type(item.get("price_type")),
                confidence=_safe_confidence(item.get("confidence")),
            ))
        return tags

    def _merge_refined_tag(self, tag: PriceTag, refined: dict[str, Any]) -> None:
        if refined.get("raw_text"):
            tag.raw_text = str(refined["raw_text"])
        if refined.get("name_hint"):
            tag.name_hint = str(refined["name_hint"])
        price = _safe_float(refined.get("price"))
        if price is not None:
            tag.price = price
        tag.price_type = _normalize_price_type(refined.get("price_type") or tag.price_type)
        tag.confidence = max(tag.confidence, _safe_confidence(refined.get("confidence")))
        sku_id = refined.get("sku_id")
        if sku_id not in self.by_id:
            sku_id, _, _ = resolve_sku(f"{tag.name_hint} {tag.raw_text}", self.catalog)
        if sku_id in self.by_id:
            tag.sku_id = sku_id


def _safe_float(value: Any) -> float | None:
    value = _first_scalar(value)
    if value is None or value == "":
        return None
    try:
        number = float(str(value).replace("￥", "").replace("¥", "").replace("元", "").strip())
    except (ValueError, TypeError):
        return None
    if number <= 0 or number > 9999:
        return None
    return round(number, 2)


def _safe_int(value: Any) -> int | None:
    value = _first_scalar(value)
    try:
        number = int(value)
        return number if number >= 0 else None
    except (ValueError, TypeError):
        return None


def _safe_confidence(value: Any) -> float:
    number = _safe_float(value)
    if number is None:
        return 0.0
    if 1.0 < number <= 100.0:
        number /= 100.0
    return max(0.0, min(1.0, number))


def _first_scalar(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        for item in value:
            scalar = _first_scalar(item)
            if scalar not in (None, ""):
                return scalar
        return None
    if isinstance(value, dict):
        for key in ("value", "score", "confidence", "price", "count"):
            if key in value:
                return _first_scalar(value[key])
        return None
    return value


def _normalize_price_type(value: Any) -> str:
    value = str(value or "unknown").lower()
    return value if value in {"normal", "member", "promotion", "unknown"} else "unknown"
