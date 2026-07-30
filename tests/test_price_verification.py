from src.catalog import load_catalog
from src.matcher import aggregate_matches
from src.schemas import MatchItem


def _match(*, sku: str, tag_ids: list[str], prices: list[float | None], status: str = "accepted") -> MatchItem:
    return MatchItem(
        product_group_id="P01",
        price_tag_id=tag_ids[0] if tag_ids else None,
        sku_id=sku,
        sku_name=sku,
        price=prices[0] if prices else None,
        score=0.9,
        status=status,
        candidate_price_tag_ids=tag_ids,
        candidate_prices=prices,
    )


def test_single_price_equal_reference_is_consistent():
    rows = aggregate_matches([_match(sku="SKU12", tag_ids=["T04"], prices=[3.5])], load_catalog())
    assert rows[0]["价格"] == 3.5
    assert rows[0]["候选价格"] == "T04 ￥3.50"
    assert rows[0]["识别结果"] == "价格一致"


def test_single_price_different_from_reference_is_inconsistent():
    rows = aggregate_matches([_match(sku="SKU06", tag_ids=["T07"], prices=[6.0])], load_catalog())
    assert rows[0]["价格"] == 6.8
    assert rows[0]["候选价格"] == "T07 ￥6.00"
    assert rows[0]["识别结果"] == "价格不一致"


def test_missing_price_is_unknown():
    rows = aggregate_matches([_match(sku="SKU01", tag_ids=[], prices=[], status="unmatched")], load_catalog())
    assert rows[0]["识别结果"] == "无法判断"


def test_multiple_tags_with_same_price_are_merged_and_consistent():
    rows = aggregate_matches(
        [_match(sku="SKU12", tag_ids=["T04", "T05"], prices=[3.5, 3.5])],
        load_catalog(),
    )
    assert rows[0]["候选价格"] == "T04 ￥3.50"
    assert rows[0]["识别结果"] == "价格一致"
    assert rows[0]["状态"] == "accepted"


def test_multiple_tags_with_same_non_reference_price_are_merged_and_inconsistent():
    rows = aggregate_matches(
        [_match(sku="SKU08", tag_ids=["T01", "T02"], prices=[7.0, 7.0], status="review")],
        load_catalog(),
    )
    assert rows[0]["候选价格"] == "T01 ￥7.00"
    assert rows[0]["识别结果"] == "价格不一致"
    assert rows[0]["状态"] == "accepted"


def test_multiple_different_prices_remain_unknown():
    rows = aggregate_matches(
        [_match(sku="SKU06", tag_ids=["T04", "T06"], prices=[6.0, 9.51])],
        load_catalog(),
    )
    assert rows[0]["候选价格"] == "T04 ￥6.00；T06 ￥9.51"
    assert rows[0]["识别结果"] == "无法判断"
    assert rows[0]["状态"] == "review"
