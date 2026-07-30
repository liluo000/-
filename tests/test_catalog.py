from src.catalog import load_catalog, resolve_sku


def test_catalog_has_12_skus():
    catalog = load_catalog()
    assert len(catalog) == 12


def test_resolve_distinguishes_iodized_and_non_iodized():
    catalog = load_catalog()
    sku, score, _ = resolve_sku("雪天未加碘生态井矿盐320克", catalog)
    assert sku == "SKU05"
    assert score > 0.7


def test_resolve_low_sodium():
    catalog = load_catalog()
    sku, _, _ = resolve_sku("海藻碘低钠盐", catalog)
    assert sku == "SKU03"
