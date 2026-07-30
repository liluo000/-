from __future__ import annotations

import base64
import io
import json
import os
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw, ImageFont


def image_to_data_url(image: Image.Image, fmt: str = "JPEG", quality: int = 90) -> str:
    buffer = io.BytesIO()
    converted = image.convert("RGB")
    converted.save(buffer, format=fmt, quality=quality, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    mime = "image/jpeg" if fmt.upper() in {"JPG", "JPEG"} else "image/png"
    return f"data:{mime};base64,{encoded}"


def normalize_bbox(bbox: Any, width: int, height: int) -> list[float]:
    """兼容扁平坐标、嵌套点坐标，以及模型偶发返回的字符串坐标。"""
    if isinstance(bbox, dict):
        if all(key in bbox for key in ("x1", "y1", "x2", "y2")):
            bbox = [bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]]
        else:
            bbox = bbox.get("bbox") or bbox.get("box") or bbox.get("coordinates") or []
    if isinstance(bbox, str):
        try:
            bbox = json.loads(bbox)
        except json.JSONDecodeError:
            bbox = bbox.replace("，", ",").strip("[]() ").split(",")

    values: list[float] = []

    def collect(value: Any) -> None:
        if len(values) >= 4:
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                collect(item)
            return
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            return

    collect(bbox)
    if len(values) != 4:
        return [0.0, 0.0, 0.0, 0.0]
    max_value = max(abs(v) for v in values)
    if max_value <= 1.5:
        values = [v * 1000.0 for v in values]
    elif max_value > 1000.0 and width > 0 and height > 0:
        x1, y1, x2, y2 = values
        values = [x1 / width * 1000, y1 / height * 1000, x2 / width * 1000, y2 / height * 1000]
    x1, y1, x2, y2 = values
    x1, x2 = sorted((max(0.0, min(1000.0, x1)), max(0.0, min(1000.0, x2))))
    y1, y2 = sorted((max(0.0, min(1000.0, y1)), max(0.0, min(1000.0, y2))))
    return [x1, y1, x2, y2]


def bbox_to_pixels(bbox: Iterable[float], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return (
        int(round(x1 / 1000 * width)),
        int(round(y1 / 1000 * height)),
        int(round(x2 / 1000 * width)),
        int(round(y2 / 1000 * height)),
    )


def crop_normalized(image: Image.Image, bbox: Iterable[float], padding: float = 0.04) -> Image.Image:
    width, height = image.size
    x1, y1, x2, y2 = bbox_to_pixels(bbox, width, height)
    pad_x = int(max(4, (x2 - x1) * padding))
    pad_y = int(max(4, (y2 - y1) * padding))
    return image.crop((max(0, x1 - pad_x), max(0, y1 - pad_y), min(width, x2 + pad_x), min(height, y2 + pad_y)))


def get_font(size: int = 24) -> ImageFont.ImageFont:
    candidates = [
        os.environ.get("CHINESE_FONT_PATH", ""),
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "/System/Library/Fonts/PingFang.ttc",
    ]
    for path in candidates:
        if path and Path(path).exists():
            try:
                return ImageFont.truetype(path, size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def draw_analysis(image: Image.Image, products: list[dict], tags: list[dict], matches: list[dict]) -> Image.Image:
    result = image.convert("RGB").copy()
    draw = ImageDraw.Draw(result)
    width, height = result.size
    font = get_font(max(16, int(width / 70)))
    small_font = get_font(max(13, int(width / 90)))

    accepted = {m.get("product_group_id"): m for m in matches}
    for product in products:
        box = bbox_to_pixels(product.get("bbox", [0, 0, 0, 0]), width, height)
        match = accepted.get(product.get("id"), {})
        status = match.get("status", "unmatched")
        outline = {"accepted": "#16A34A", "review": "#F59E0B", "unmatched": "#DC2626"}.get(status, "#2563EB")
        draw.rectangle(box, outline=outline, width=max(3, width // 500))
        label = f'{product.get("id", "")} {product.get("sku_id") or "?"} {product.get("sku_name", "")}'
        _draw_label(draw, box[0], max(0, box[1] - 28), label, outline, font)

    for tag in tags:
        box = bbox_to_pixels(tag.get("bbox", [0, 0, 0, 0]), width, height)
        draw.rectangle(box, outline="#7C3AED", width=max(3, width // 500))
        price = tag.get("price")
        label = f'{tag.get("id", "")} ￥{price:.2f}' if isinstance(price, (int, float)) else f'{tag.get("id", "")} 未识别价格'
        _draw_label(draw, box[0], box[1], label, "#7C3AED", small_font)

    product_map = {p.get("id"): p for p in products}
    tag_map = {t.get("id"): t for t in tags}
    for match in matches:
        if match.get("status") == "unmatched" or not match.get("price_tag_id"):
            continue
        product = product_map.get(match.get("product_group_id"))
        tag = tag_map.get(match.get("price_tag_id"))
        if not product or not tag:
            continue
        pb = bbox_to_pixels(product["bbox"], width, height)
        tb = bbox_to_pixels(tag["bbox"], width, height)
        p_center = ((pb[0] + pb[2]) // 2, pb[3])
        t_center = ((tb[0] + tb[2]) // 2, (tb[1] + tb[3]) // 2)
        line_color = "#16A34A" if match.get("status") == "accepted" else "#F59E0B"
        draw.line([p_center, t_center], fill=line_color, width=max(2, width // 700))
    return result


def _draw_label(draw: ImageDraw.ImageDraw, x: int, y: int, text: str, color: str, font: ImageFont.ImageFont) -> None:
    try:
        bbox = draw.textbbox((x, y), text, font=font)
        pad = 3
        draw.rectangle((bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad), fill=color)
        draw.text((x, y), text, fill="white", font=font)
    except Exception:
        draw.text((x, y), text, fill=color, font=font)
