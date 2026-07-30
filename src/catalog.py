from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from rapidfuzz import fuzz

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG_PATH = ROOT / "data" / "catalog.json"


def load_catalog(path: str | Path = DEFAULT_CATALOG_PATH) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as file:
        catalog = json.load(file)
    if len(catalog) != 12:
        raise ValueError(f"目录应包含12个SKU，当前为{len(catalog)}个")
    return catalog


def normalize_text(text: str | None) -> str:
    if not text:
        return ""
    replacements = {
        "（": "(", "）": ")", "：": ":", " ": "", "\n": "",
        "雪天盐": "雪天", "食盐": "盐", "克": "g", "Ｇ": "g",
        "未加典": "未加碘", "加典": "加碘", "海澡": "海藻",
        "井矿": "井矿", "低纳": "低钠", "松态": "松態",
    }
    result = str(text).lower()
    for old, new in replacements.items():
        result = result.replace(old.lower(), new.lower())
    for token in ["雪天", "高碘版", "省内", "版", "(高碘)", "高碘"]:
        result = result.replace(token.lower(), "")
    return result


def catalog_by_id(catalog: Iterable[dict]) -> dict[str, dict]:
    return {item["sku_id"]: item for item in catalog}


def resolve_sku(text: str | None, catalog: list[dict]) -> tuple[str | None, float, list[tuple[str, float]]]:
    """把OCR或模型输出的品名映射为12个SKU之一。"""
    normalized = normalize_text(text)
    if not normalized:
        return None, 0.0, []

    ranked: list[tuple[str, float]] = []
    for item in catalog:
        score = 0.0
        primary_candidates = [item["name"], item.get("short_name", "")]
        keyword_candidates = item.get("keywords", [])

        for candidate in primary_candidates:
            c = normalize_text(candidate)
            if not c:
                continue
            if c in normalized:
                local = 0.99
            elif normalized in c and len(normalized) >= 4:
                local = 0.93
            else:
                local = max(
                    fuzz.ratio(normalized, c),
                    fuzz.partial_ratio(normalized, c),
                    fuzz.token_set_ratio(normalized, c),
                ) / 100.0 * 0.90
            score = max(score, local)

        for candidate in keyword_candidates:
            c = normalize_text(candidate)
            if not c:
                continue
            if c in normalized:
                # 短关键词只能作为弱证据，长关键词才接近完整品名。
                coverage = len(c) / max(1, len(normalized))
                local = 0.68 + min(0.27, coverage * 0.32)
            else:
                local = max(fuzz.ratio(normalized, c), fuzz.partial_ratio(normalized, c)) / 100.0 * 0.82
            score = max(score, local)

        for negative in item.get("negative_keywords", []):
            n = normalize_text(negative)
            if not n:
                continue
            # “未加碘”包含“加碘”字面子串，不能因此误判为负关键词。
            if n.startswith("加碘") and "未加碘" in normalized:
                continue
            if n in normalized:
                score -= 0.28

        # “未加碘”等是强区分词，防止相似包装混淆。
        item_text = normalize_text(item["name"])
        if "未加碘" in normalized and "未加碘" not in item_text:
            score -= 0.42
        if "未加碘" not in normalized and "未加碘" in item_text and "加碘" in normalized:
            score -= 0.42
        if "低钠" in normalized and "低钠" not in item_text:
            score -= 0.32
        if "997" in normalized and "997" not in item_text:
            score -= 0.45

        ranked.append((item["sku_id"], max(0.0, min(1.0, score))))

    ranked.sort(key=lambda x: x[1], reverse=True)
    best_id, best_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    # 分数过低或前两名过于接近时，不强制映射。
    if best_score < 0.58 or (best_score - second_score < 0.06 and best_score < 0.88):
        return None, best_score, ranked[:3]
    return best_id, best_score, ranked[:3]


def catalog_prompt(catalog: list[dict]) -> str:
    lines = []
    for item in catalog:
        lines.append(
            f'{item["sku_id"]}: {item["name"]}，规格{item["spec"]}，'
            f'参考价￥{item["reference_price"]:.2f}，关键词：{"、".join(item.get("keywords", []))}'
        )
    return "\n".join(lines)
