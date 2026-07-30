from src.catalog import load_catalog
from src.matcher import match_products_and_tags
from src.schemas import PriceTag, ProductGroup


def test_hungarian_supports_missing_and_extra_tags():
    catalog = load_catalog()
    products = [
        ProductGroup(id="P01", sku_id="SKU04", sku_name="加碘生态井矿盐", bbox=[0, 0, 300, 400], confidence=0.9),
        ProductGroup(id="P02", sku_id="SKU05", sku_name="未加碘生态井矿盐", bbox=[400, 0, 700, 400], confidence=0.9),
    ]
    tags = [
        PriceTag(id="T01", sku_id="SKU04", name_hint="加碘生态井矿盐", price=4.0, bbox=[50, 410, 250, 480], confidence=0.9),
        PriceTag(id="T02", sku_id="SKU11", name_hint="未加碘海晶盐", price=4.5, bbox=[800, 410, 950, 480], confidence=0.9),
    ]
    matches, unmatched_tags, _ = match_products_and_tags(products, tags, "standard_shelf", catalog)
    assert matches[0].price_tag_id == "T01"
    assert matches[0].status == "accepted"
    assert matches[1].status == "unmatched"
    assert "T02" in unmatched_tags


def test_wide_group_accepts_price_only_edge_tags_and_merges_duplicates():
    from src.matcher import match_products_and_tags
    from src.schemas import ProductGroup, PriceTag
    from src.catalog import load_catalog

    catalog = load_catalog()
    products = [ProductGroup(
        id="P01", sku_id="SKU12", sku_name="南岳寿山未加碘食用盐",
        bbox=[180, 300, 930, 520], confidence=0.95,
    )]
    tags = [
        PriceTag(id="T01", bbox=[185, 520, 245, 570], price=3.50, confidence=0.9),
        PriceTag(id="T02", bbox=[865, 520, 925, 570], price=3.50, confidence=0.9),
    ]
    matches, unmatched, scores = match_products_and_tags(products, tags, "standard_shelf", catalog)
    assert len(matches) == 1
    assert matches[0].status == "accepted"
    assert matches[0].price == 3.5
    assert matches[0].score >= 0.72
    assert matches[0].candidate_price_tag_ids == ["T01", "T02"]
    assert matches[0].candidate_prices == [3.5, 3.5]
    assert unmatched == []
    assert scores.shape == (1, 2)


def test_price_only_tag_does_not_cross_to_distant_product_group():
    from src.matcher import match_products_and_tags
    from src.schemas import ProductGroup, PriceTag
    from src.catalog import load_catalog

    catalog = load_catalog()
    products = [
        ProductGroup(id="P01", sku_id="SKU12", sku_name="南岳寿山未加碘食用盐", bbox=[100, 300, 450, 500]),
        ProductGroup(id="P02", sku_id="SKU10", sku_name="加碘活水盐", bbox=[550, 300, 900, 500]),
    ]
    tags = [
        PriceTag(id="T01", bbox=[150, 505, 210, 555], price=3.5),
        PriceTag(id="T02", bbox=[760, 505, 820, 555], price=3.5),
    ]
    matches, unmatched, _ = match_products_and_tags(products, tags, "standard_shelf", catalog)
    assert [m.price_tag_id for m in matches] == ["T01", "T02"]
    assert all(m.status == "accepted" for m in matches)
    assert unmatched == []
