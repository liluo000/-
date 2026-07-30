from __future__ import annotations

from PIL import Image

from src.catalog import load_catalog
from src.gemini_client import QwenVisionClient


class FakeResponse:
    ok = True
    status_code = 200
    text = ""

    @staticmethod
    def json():
        return {
            "choices": [{
                "message": {
                    "content": (
                            '{"scene_type":"standard_shelf","scene_confidence":0.9,'
                            '"products":[],"price_tags":[],"notes":[]}'
                        )
                }
            }]
        }


class FakeSession:
    def __init__(self):
        self.headers = {}
        self.last_url = None
        self.last_headers = None
        self.last_payload = None

    def post(self, url, headers, json, timeout):
        self.last_url = url
        self.last_headers = headers
        self.last_payload = json
        return FakeResponse()


def test_qwen_client_uses_official_endpoint_and_bearer_header():
    session = FakeSession()
    client = QwenVisionClient(api_key="test-key-not-real", session=session)
    payload, _ = client.analyze(Image.new("RGB", (640, 480), "white"), load_catalog(), None)

    assert payload["scene_type"] == "standard_shelf"
    assert session.last_url.endswith("/compatible-mode/v1/chat/completions")
    assert session.last_headers["Authorization"] == "Bearer test-key-not-real"
    assert session.last_payload["model"] == "qwen3.7-flash"
    assert session.last_payload["response_format"] == {"type": "json_object"}
    assert session.last_payload["enable_thinking"] is False
    content = session.last_payload["messages"][0]["content"]
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_placeholder_key_is_rejected():
    client = QwenVisionClient(api_key="在这里填写你的DASHSCOPE_API_KEY")
    assert not client.is_configured
