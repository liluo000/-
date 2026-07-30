from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from .catalog import catalog_by_id, normalize_text, resolve_sku
from .schemas import MatchItem, PriceTag, ProductGroup


def _center(bbox: list[float]) -> tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)


def _horizontal_overlap(a: list[float], b: list[float]) -> float:
    overlap = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    base = max(1.0, min(a[2] - a[0], b[2] - b[0]))
    return min(1.0, overlap / base)


def _interval_gap(value: float, low: float, high: float) -> float:
    if low <= value <= high:
        return 0.0
    return low - value if value < low else value - high


def _distance_score(a: list[float], b: list[float]) -> float:
    """陈列组到价签的距离。

    商品组可能横跨十几袋商品，因此不能比较两个框的中心点；价签只要位于
    商品组的水平范围内，就应视为水平距离为零。
    """
    bx, _ = _center(b)
    horizontal_gap = _interval_gap(bx, a[0], a[2])
    vertical_gap = abs(b[1] - a[3])
    product_width = max(1.0, a[2] - a[0])
    horizontal_scale = max(70.0, product_width * 0.22)
    horizontal_score = max(0.0, 1.0 - horizontal_gap / horizontal_scale)
    vertical_score = max(0.0, 1.0 - vertical_gap / 230.0)
    return 0.62 * horizontal_score + 0.38 * vertical_score


def _shelf_anchor_score(product: list[float], tag: list[float]) -> float:
    """判断价签是否可靠地锚定在标准货架商品组下方。"""
    tx, _ = _center(tag)
    product_width = max(1.0, product[2] - product[0])
    margin = max(10.0, product_width * 0.035)
    horizontal_gap = _interval_gap(tx, product[0] - margin, product[2] + margin)
    horizontal_score = max(0.0, 1.0 - horizontal_gap / max(55.0, product_width * 0.16))

    vertical_gap = tag[1] - product[3]
    if -35.0 <= vertical_gap <= 125.0:
        vertical_score = 1.0
    elif 125.0 < vertical_gap <= 240.0:
        vertical_score = max(0.0, 1.0 - (vertical_gap - 125.0) / 115.0)
    elif -80.0 <= vertical_gap < -35.0:
        vertical_score = max(0.0, 1.0 - (-35.0 - vertical_gap) / 45.0)
    else:
        vertical_score = 0.0
    return 0.58 * horizontal_score + 0.42 * vertical_score


def _same_price(a: PriceTag, b: PriceTag, tolerance: float = 0.011) -> bool:
    return a.price is not None and b.price is not None and abs(float(a.price) - float(b.price)) <= tolerance


def _direction_score(product: list[float], tag: list[float], scene: str) -> float:
    px, py = _center(product)
    tx, ty = _center(tag)
    if scene == "standard_shelf":
        vertical_gap = tag[1] - product[3]
        if -30 <= vertical_gap <= 260:
            return 1.0
        if ty > py and vertical_gap <= 400:
            return 0.55
        return 0.05
    if scene == "hook_display":
        return 1.0 if abs(tx - px) < 170 and abs(ty - py) < 350 else 0.35
    if scene == "basket_display":
        return 1.0 if abs(tx - px) < 260 and abs(ty - py) < 300 else 0.35
    if scene == "promotion_stack":
        return 1.0 if abs(tx - px) < 350 and abs(ty - py) < 400 else 0.35
    return 0.45


def _price_plausibility(price: float | None, reference: float | None) -> float:
    if price is None or not reference or reference <= 0:
        return 0.5
    ratio = abs(price - reference) / reference
    if ratio <= 0.15:
        return 1.0
    if ratio <= 0.40:
        return 0.72
    if ratio <= 0.80:
        return 0.35
    return 0.05


def score_pair(product: ProductGroup, tag: PriceTag, scene: str, catalog: list[dict]) -> tuple[float, list[str]]:
    by_id = catalog_by_id(catalog)
    reasons: list[str] = []

    product_sku = product.sku_id
    tag_sku = tag.sku_id
    if not tag_sku:
        tag_sku, tag_name_score, _ = resolve_sku(f"{tag.name_hint} {tag.raw_text}", catalog)
    else:
        tag_name_score = 1.0

    if product_sku and tag_sku:
        if product_sku == tag_sku:
            name_score = 1.0
            reasons.append("商品SKU与价签品名一致")
        else:
            name_score = 0.0
            reasons.append("商品SKU与价签品名冲突")
    else:
        product_text = product.sku_name or product.observed_text
        tag_text = tag.name_hint or tag.raw_text
        resolved_product, product_score, _ = resolve_sku(product_text, catalog)
        resolved_tag, resolved_tag_score, _ = resolve_sku(tag_text, catalog)
        if resolved_product and resolved_tag and resolved_product == resolved_tag:
            name_score = min(product_score, resolved_tag_score)
            reasons.append("商品文字与价签文字指向同一SKU")
        else:
            name_score = max(0.0, tag_name_score * 0.65)
            reasons.append("品名证据不足")

    overlap = _horizontal_overlap(product.bbox, tag.bbox)
    distance = _distance_score(product.bbox, tag.bbox)
    direction = _direction_score(product.bbox, tag.bbox, scene)
    shelf_anchor = _shelf_anchor_score(product.bbox, tag.bbox) if scene == "standard_shelf" else 0.0
    reference_price = by_id.get(product_sku or tag_sku or "", {}).get("reference_price")
    price_score = _price_plausibility(tag.price, reference_price)

    if overlap >= 0.45:
        reasons.append("水平投影明显重叠")
    if direction >= 0.8:
        reasons.append("价签方向符合当前陈列场景")
    if distance >= 0.75:
        reasons.append("价签靠近商品组的水平范围")
    if shelf_anchor >= 0.82 and tag.price is not None:
        reasons.append("价签位于该商品陈列组正下方")
    if price_score < 0.2:
        reasons.append("现场价与参考价差异很大，仅作异常提示")

    if scene == "standard_shelf" and name_score < 0.58 and shelf_anchor >= 0.72 and tag.price is not None:
        # 小价签常只能读出价格。对标准货架，价签位于商品组水平范围内且紧贴下方时，
        # 允许以几何锚定为主要证据，避免宽陈列组两端价签被中心距离误伤。
        score = (
            0.08 * name_score
            + 0.27 * overlap
            + 0.22 * direction
            + 0.28 * shelf_anchor
            + 0.10 * distance
            + 0.05 * price_score
        )
        reasons.append("价签品名不清，采用标准货架几何锚定")
    else:
        if scene == "standard_shelf":
            weights = (0.55, 0.16, 0.12, 0.12, 0.05)
        elif scene in {"hook_display", "basket_display", "promotion_stack"}:
            weights = (0.62, 0.10, 0.10, 0.13, 0.05)
        else:
            weights = (0.76, 0.05, 0.05, 0.09, 0.05)

        score = (
            weights[0] * name_score
            + weights[1] * overlap
            + weights[2] * direction
            + weights[3] * distance
            + weights[4] * price_score
        )

    # 补货或未知场景只有品名高度一致时才允许进入人工复核以上等级。
    if scene in {"replenishment", "unknown"} and name_score < 0.93:
        score = min(score, 0.43)
    if product_sku and tag_sku and product_sku != tag_sku:
        score = min(score, 0.18)
    return max(0.0, min(1.0, score)), reasons


def _candidate_tag_indices(
    product: ProductGroup,
    tags: list[PriceTag],
    row_scores: np.ndarray,
    scene: str,
    review_threshold: float,
) -> list[int]:
    """返回可能属于该商品陈列组的价签索引。

    这不是最终一对一分配，而是为价格核验保留全部合理候选。标准货架中，
    位于商品组水平范围内并紧贴下方的价签会被保留，即使小字品名没有读清。
    明确识别为其他SKU的价签会被排除。
    """
    candidates: list[int] = []
    for index, tag in enumerate(tags):
        if product.sku_id and tag.sku_id and product.sku_id != tag.sku_id:
            continue

        same_sku = bool(product.sku_id and tag.sku_id and product.sku_id == tag.sku_id)
        score_ok = float(row_scores[index]) >= max(0.42, review_threshold - 0.08)
        if scene == "standard_shelf":
            anchored = _shelf_anchor_score(product.bbox, tag.bbox) >= 0.80
            if same_sku or score_ok or anchored:
                candidates.append(index)
        elif same_sku or float(row_scores[index]) >= review_threshold:
            candidates.append(index)

    return sorted(candidates, key=lambda idx: float(row_scores[idx]), reverse=True)


def match_products_and_tags(
    products: list[ProductGroup],
    tags: list[PriceTag],
    scene: str,
    catalog: list[dict],
    accept_threshold: float = 0.72,
    review_threshold: float = 0.50,
    unmatched_score: float = 0.46,
) -> tuple[list[MatchItem], list[str], np.ndarray]:
    by_id = catalog_by_id(catalog)
    m, n = len(products), len(tags)
    if m == 0:
        return [], [tag.id for tag in tags], np.empty((0, n))
    if n == 0:
        return [
            MatchItem(
                product_group_id=p.id,
                sku_id=p.sku_id,
                sku_name=p.sku_name,
                status="unmatched",
                reasons=["未检测到价签"],
            )
            for p in products
        ], [], np.empty((m, 0))

    scores = np.zeros((m, n), dtype=float)
    reason_map: dict[tuple[int, int], list[str]] = {}
    for i, product in enumerate(products):
        for j, tag in enumerate(tags):
            scores[i, j], reason_map[(i, j)] = score_pair(product, tag, scene, catalog)

    size = m + n
    high_cost = 10.0
    cost = np.full((size, size), high_cost, dtype=float)
    cost[:m, :n] = 1.0 - scores

    # 每个商品只能连接自己的“无价签”虚拟列。
    for i in range(m):
        cost[i, n + i] = 1.0 - unmatched_score
    # 每个价签只能连接自己的“无商品”虚拟行。
    for j in range(n):
        cost[m + j, j] = 1.0 - unmatched_score
    cost[m:, n:] = 0.0

    row_ind, col_ind = linear_sum_assignment(cost)
    selected: dict[int, int] = {}
    matched_tag_indices: set[int] = set()
    for row, col in zip(row_ind, col_ind):
        if row < m and col < n:
            selected[row] = col
            matched_tag_indices.add(col)

    results: list[MatchItem] = []
    for i, product in enumerate(products):
        if i not in selected:
            results.append(MatchItem(
                product_group_id=product.id,
                sku_id=product.sku_id,
                sku_name=product.sku_name,
                status="unmatched",
                reasons=["全局匹配选择了无价签节点"],
            ))
            continue
        j = selected[i]
        tag = tags[j]
        score = float(scores[i, j])
        candidate_indices = _candidate_tag_indices(
            product, tags, scores[i], scene, review_threshold
        )
        # 最终被匈牙利算法选中的价签至少要进入候选明细。
        if j not in candidate_indices:
            candidate_indices.insert(0, j)
        candidate_ids = [tags[idx].id for idx in candidate_indices]
        candidate_prices = [tags[idx].price for idx in candidate_indices]
        row_order = sorted(range(n), key=lambda idx: scores[i, idx], reverse=True)
        second_j = row_order[1] if len(row_order) > 1 else None
        ambiguity_gap = scores[i, row_order[0]] - scores[i, second_j] if second_j is not None else 1.0
        duplicate_consistent = False
        if second_j is not None and ambiguity_gap < 0.05 and scene == "standard_shelf":
            first_tag = tags[row_order[0]]
            second_tag = tags[second_j]
            duplicate_consistent = (
                _same_price(first_tag, second_tag)
                and _shelf_anchor_score(product.bbox, first_tag.bbox) >= 0.80
                and _shelf_anchor_score(product.bbox, second_tag.bbox) >= 0.80
                and (not first_tag.sku_id or first_tag.sku_id == product.sku_id)
                and (not second_tag.sku_id or second_tag.sku_id == product.sku_id)
            )

        if score >= accept_threshold and (ambiguity_gap >= 0.05 or duplicate_consistent):
            status = "accepted"
        elif score >= review_threshold:
            status = "review"
        else:
            status = "unmatched"

        reasons = reason_map[(i, j)].copy()
        if duplicate_consistent:
            reasons.append("同一陈列组下存在多个相同价格价签，按重复陈列标签合并")
        elif ambiguity_gap < 0.05:
            reasons.append("两个候选价签得分接近，需要人工确认")
        if status == "unmatched":
            reasons.append("最高得分低于自动配对阈值")

        results.append(MatchItem(
            product_group_id=product.id,
            price_tag_id=tag.id if status != "unmatched" else None,
            sku_id=product.sku_id or tag.sku_id,
            sku_name=(by_id.get(product.sku_id or tag.sku_id or "", {}).get("name") or product.sku_name),
            price=tag.price if status != "unmatched" else None,
            price_type=tag.price_type,
            score=score,
            status=status,
            reasons=reasons,
            raw_text=tag.raw_text,
            candidate_price_tag_ids=candidate_ids,
            candidate_prices=candidate_prices,
        ))

    actual_matched_tags = {r.price_tag_id for r in results if r.price_tag_id}
    # 同一个宽商品陈列组的左右两端可能各放一张相同价签。业务只需一个价格，
    # 但第二张不应被报告为“多余价签”。
    product_lookup = {p.id: p for p in products}
    tag_lookup = {t.id: t for t in tags}
    for result in results:
        if not result.price_tag_id or result.status == "unmatched":
            continue
        product = product_lookup.get(result.product_group_id)
        chosen = tag_lookup.get(result.price_tag_id)
        if not product or not chosen or chosen.price is None or scene != "standard_shelf":
            continue
        for candidate in tags:
            if candidate.id in actual_matched_tags or not _same_price(chosen, candidate):
                continue
            if candidate.sku_id and product.sku_id and candidate.sku_id != product.sku_id:
                continue
            if _shelf_anchor_score(product.bbox, candidate.bbox) >= 0.80:
                actual_matched_tags.add(candidate.id)

    unmatched_tags = [tag.id for tag in tags if tag.id not in actual_matched_tags]
    return results, unmatched_tags, scores


def _format_candidate_prices(tag_ids: list[str], prices: list[float | None]) -> str:
    if not tag_ids:
        return ""
    parts: list[str] = []
    for index, tag_id in enumerate(tag_ids):
        price = prices[index] if index < len(prices) else None
        price_text = f"￥{float(price):.2f}" if price is not None else "未识别"
        parts.append(f"{tag_id} {price_text}")
    return "；".join(parts)


def _deduplicate_candidate_prices(
    candidate_map: dict[str, float | None],
) -> tuple[list[str], list[float | None], int]:
    """相同的可读价格只保留第一张价签；不可读价格不擅自合并。"""
    tag_ids: list[str] = []
    prices: list[float | None] = []
    seen_prices: set[float] = set()
    merged_count = 0
    for tag_id, price in candidate_map.items():
        if price is not None:
            normalized = round(float(price), 2)
            if normalized in seen_prices:
                merged_count += 1
                continue
            seen_prices.add(normalized)
            price = normalized
        tag_ids.append(tag_id)
        prices.append(price)
    return tag_ids, prices, merged_count


def aggregate_matches(matches: list[MatchItem], catalog: list[dict] | None = None) -> list[dict[str, Any]]:
    """按SKU汇总，并把目录参考价与现场候选价签进行核验。

    业务规则：
    - “价格”来自12个SKU目录中的参考/标准价格；
    - “候选价格”来自照片中识别到的现场价签；
    - 多张价签价格相同时按一个候选价格处理；
    - 无价、无法读价或存在多个不同候选价格时显示“无法判断”。
    """
    grouped: dict[str, list[MatchItem]] = defaultdict(list)
    for match in matches:
        grouped[match.sku_id or match.product_group_id].append(match)

    by_id = catalog_by_id(catalog or [])
    rows: list[dict[str, Any]] = []
    for group_key, items in grouped.items():
        best = max(items, key=lambda item: item.score)
        sku_id = best.sku_id or (group_key if group_key.startswith("SKU") else None)
        catalog_item = by_id.get(sku_id or "", {})
        reference_price = catalog_item.get("reference_price")
        try:
            reference_price = round(float(reference_price), 2) if reference_price is not None else None
        except (TypeError, ValueError):
            reference_price = None

        # 先按价签ID收集，再把多个相同的可读价格合并为一个候选价格。
        candidate_map: dict[str, float | None] = {}
        for item in items:
            if item.candidate_price_tag_ids:
                for index, tag_id in enumerate(item.candidate_price_tag_ids):
                    price = item.candidate_prices[index] if index < len(item.candidate_prices) else None
                    candidate_map.setdefault(tag_id, price)
            elif item.price_tag_id:
                candidate_map.setdefault(item.price_tag_id, item.price)

        candidate_tag_ids, candidate_prices, merged_count = _deduplicate_candidate_prices(candidate_map)
        candidate_display = _format_candidate_prices(candidate_tag_ids, candidate_prices)

        if reference_price is None or len(candidate_tag_ids) != 1 or not candidate_prices or candidate_prices[0] is None:
            verification = "无法判断"
        elif abs(reference_price - float(candidate_prices[0])) <= 0.011:
            verification = "价格一致"
        else:
            verification = "价格不一致"

        status = best.status
        note_prefix = ""
        if len(candidate_tag_ids) > 1:
            status = "review"
            note_prefix = f"检测到{len(candidate_tag_ids)}个不同候选价格，按规则无法判断"
        elif not candidate_tag_ids or not candidate_prices or candidate_prices[0] is None:
            note_prefix = "未识别到唯一且可读的现场价格"
        else:
            if merged_count and status != "unmatched":
                status = "accepted"
                note_prefix = f"已合并{merged_count + 1}个相同价格价签"
            if verification == "价格不一致":
                mismatch_note = f"目录价格￥{reference_price:.2f}，现场价签{candidate_display}"
                note_prefix = "；".join(part for part in [note_prefix, mismatch_note] if part)

        explanation = "；".join(part for part in [note_prefix, "；".join(best.reasons)] if part)
        rows.append({
            "sku_id": sku_id,
            "品名": catalog_item.get("name") or best.sku_name,
            "价格": reference_price,
            "候选价格": candidate_display,
            "识别结果": verification,
            "配对置信度": round(best.score, 3),
            "状态": status,
            "说明": explanation,
        })
    return rows
