from src.image_utils import normalize_bbox
from src.pipeline import _safe_confidence, _safe_float, _safe_int


def test_nested_bbox_is_flattened():
    assert normalize_bbox([[100, 200], [300, 400]], 1000, 1000) == [100.0, 200.0, 300.0, 400.0]


def test_bbox_dict_and_string_are_supported():
    assert normalize_bbox({"x1": 10, "y1": 20, "x2": 30, "y2": 40}, 1000, 1000) == [10.0, 20.0, 30.0, 40.0]
    assert normalize_bbox("[10,20,30,40]", 1000, 1000) == [10.0, 20.0, 30.0, 40.0]


def test_list_wrapped_scalar_fields_are_unwrapped():
    assert _safe_float([["￥3.50"]]) == 3.5
    assert _safe_int([3]) == 3
    assert _safe_confidence([0.92]) == 0.92
    assert _safe_confidence([92]) == 0.92
