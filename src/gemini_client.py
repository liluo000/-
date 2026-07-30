from __future__ import annotations

import base64
import io
import json
import time
from typing import Any

import requests
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from .catalog import catalog_prompt

DEFAULT_MODEL = "qwen3.7-flash"
DEFAULT_API_ROOT = "https://dashscope.aliyuncs.com/compatible-mode/v1"


ANALYSIS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["scene_type", "scene_confidence", "products", "price_tags", "notes"],
    "properties": {
        "scene_type": {
            "type": "string",
            "enum": [
                "standard_shelf",
                "hook_display",
                "basket_display",
                "promotion_stack",
                "replenishment",
                "unknown",
            ],
        },
        "scene_confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "products": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "sku_id",
                    "sku_name",
                    "bbox",
                    "confidence",
                    "facing_count",
                    "observed_text",
                ],
                "properties": {
                    "sku_id": {"type": ["string", "null"]},
                    "sku_name": {"type": "string"},
                    "bbox": {
                        "type": "array",
                        "minItems": 4,
                        "maxItems": 4,
                        "items": {"type": "number"},
                    },
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "facing_count": {"type": ["integer", "null"], "minimum": 0},
                    "observed_text": {"type": "string"},
                },
            },
        },
        "price_tags": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "bbox",
                    "raw_text",
                    "name_hint",
                    "sku_id",
                    "price",
                    "price_type",
                    "confidence",
                ],
                "properties": {
                    "bbox": {
                        "type": "array",
                        "minItems": 4,
                        "maxItems": 4,
                        "items": {"type": "number"},
                    },
                    "raw_text": {"type": "string"},
                    "name_hint": {"type": "string"},
                    "sku_id": {"type": ["string", "null"]},
                    "price": {"type": ["number", "null"], "minimum": 0},
                    "price_type": {
                        "type": "string",
                        "enum": ["normal", "member", "promotion", "unknown"],
                    },
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
            },
        },
        "notes": {"type": "array", "items": {"type": "string"}},
    },
}


PRICE_TAG_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["raw_text", "name_hint", "sku_id", "price", "price_type", "confidence"],
    "properties": {
        "raw_text": {"type": "string"},
        "name_hint": {"type": "string"},
        "sku_id": {"type": ["string", "null"]},
        "price": {"type": ["number", "null"], "minimum": 0},
        "price_type": {
            "type": "string",
            "enum": ["normal", "member", "promotion", "unknown"],
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
}


class QwenAPIError(RuntimeError):
    """Qwen API 调用失败，错误信息已转换为适合前端显示的中文。"""


class QwenVisionClient:
    """通过阿里云百炼 OpenAI 兼容接口调用 Qwen 多模态模型。"""

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        api_root: str = DEFAULT_API_ROOT,
        timeout: int = 180,
        max_retries: int = 3,
        session: requests.Session | None = None,
    ) -> None:
        self.api_key = (api_key or "").strip()
        self.model = model
        self.api_root = api_root.rstrip("/")
        self.timeout = timeout
        self.max_retries = max(1, int(max_retries))
        self.session = session or requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "User-Agent": "salt-price-matcher-qwen/1.0",
        })

    @property
    def is_configured(self) -> bool:
        return self._key_is_configured()

    @property
    def endpoint(self) -> str:
        return f"{self.api_root}/chat/completions"

    def health_check(self) -> tuple[bool, str]:
        if not self._key_is_configured():
            return False, "未配置 Qwen API Key。请编辑项目根目录的 .env 文件。"
        try:
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": "只回复 OK"}],
                "enable_thinking": False,
                "max_completion_tokens": 16,
            }
            response = self._post(payload)
            text = self._extract_text(response).strip()
            return True, f"Qwen API 连接成功，模型：{self.model}，响应：{text[:40] or 'OK'}"
        except Exception as exc:  # noqa: BLE001
            return False, str(exc)

    def analyze(
        self,
        image: Image.Image,
        catalog: list[dict],
        reference_contact: Image.Image | None = None,
        max_tokens: int = 9000,
    ) -> tuple[dict[str, Any], str]:
        self._require_key()
        prompt = self._analysis_prompt(catalog, image.size)
        prompt += (
            "\n\n请严格只输出一个 JSON 对象，不要添加解释或 Markdown。"
            "输出必须符合以下 JSON Schema：\n"
            + json.dumps(ANALYSIS_SCHEMA, ensure_ascii=False)
        )
        content: list[dict[str, Any]] = [
            {"type": "text", "text": prompt},
            self._image_part(image, quality=92, max_side=2800),
        ]
        if reference_contact is not None:
            content.extend([
                {
                    "type": "text",
                    "text": (
                        "下一张图片是固定12个SKU的参考总图，SKU编号已经标注。"
                        "参考图只用于判断包装品名，不得把参考价当作现场价格。"
                    )
                },
                self._image_part(reference_contact, quality=90, max_side=2600),
            ])

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "response_format": {"type": "json_object"},
            "enable_thinking": False,
        }
        response = self._post(payload)
        raw = self._extract_text(response)
        return self._parse_json(raw), raw

    def read_price_tag(self, crop: Image.Image, catalog: list[dict]) -> tuple[dict[str, Any], str]:
        self._require_key()
        prompt = f"""
你是零售价格标签OCR助手。只读取图片中真实可见的价签内容。

任务：
- 尽可能完整抄录价签文字；
- 从固定12个SKU中判断品名，无法确定时 sku_id 必须为 null；
- 读取现场实际价格，数字或小数点看不清时 price 必须为 null；
- 如果同时出现普通价、会员价、促销价，返回当前最醒目的有效售价，并在 raw_text 中保留其他文字；
- 绝不使用目录参考价补全现场价格。
- price、confidence 必须是单个数字或 null，禁止返回数组；
- bbox 如需返回，必须是 [x1,y1,x2,y2] 四个数字组成的扁平数组，禁止嵌套数组。

固定12个SKU目录：
{catalog_prompt(catalog)}

请严格只输出一个 JSON 对象，不要添加解释或 Markdown。输出必须符合以下 JSON Schema：
{json.dumps(PRICE_TAG_SCHEMA, ensure_ascii=False)}
""".strip()
        prepared = self._prepare_tag_crop(crop)
        payload = {
            "model": self.model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    self._image_part(prepared, quality=96, max_side=2400),
                ],
            }],
            "response_format": {"type": "json_object"},
            "enable_thinking": False,
        }
        response = self._post(payload)
        raw = self._extract_text(response)
        return self._parse_json(raw), raw

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        last_error: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                response = self.session.post(
                    self.endpoint,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
            except requests.Timeout:
                last_error = QwenAPIError("Qwen API 请求超时，请稍后重试或减少一次上传的图片数量。")
            except requests.RequestException as exc:
                last_error = QwenAPIError(f"无法连接 Qwen API：{exc}")
            else:
                if response.ok:
                    try:
                        return response.json()
                    except ValueError as exc:
                        raise QwenAPIError("Qwen API 返回了无法解析的响应。") from exc

                message = self._extract_error_message(response)
                status = response.status_code
                if status in {401, 403}:
                    raise QwenAPIError(
                        "Qwen API Key 无效、已禁用或当前业务空间没有权限。请检查百炼密钥和模型授权。"
                        f"\n服务端信息：{message}"
                    )
                if status == 400:
                    if "DataInspectionFailed" in message or "inappropriate content" in message.lower():
                        raise QwenAPIError(
                            "图片被阿里云百炼内容安全审核拦截，模型尚未开始识别。"
                            "请确认照片不含人脸、隐私或敏感内容，并裁剪到商品货架区域后重试；"
                            "如果是正常门店照片被误判，请携带完整错误信息向阿里云提交工单申请内容加白。"
                            f"\n服务端信息：{message}"
                        )
                    raise QwenAPIError(f"Qwen API 请求格式或图片内容不被接受：{message}")
                if status == 404:
                    raise QwenAPIError(
                        f"模型 {self.model} 不可用。请确认该 API Key 所属项目可以访问此模型。服务端信息：{message}"
                    )
                if status == 429:
                    last_error = QwenAPIError(
                        "Qwen 额度或速率限制已达到，请等待一段时间再试。"
                        f"\n服务端信息：{message}"
                    )
                elif status >= 500:
                    last_error = QwenAPIError(f"Qwen 服务暂时不可用：{message}")
                else:
                    raise QwenAPIError(f"Qwen API 调用失败（HTTP {status}）：{message}")

            if attempt < self.max_retries - 1:
                time.sleep(1.5 * (2 ** attempt))

        raise last_error or QwenAPIError("Qwen API 调用失败。")

    @staticmethod
    def _extract_error_message(response: requests.Response) -> str:
        try:
            data = response.json()
            error = data.get("error") or {}
            message = str(error.get("message") or data)
            code = error.get("code")
            request_id = data.get("request_id") or data.get("id")
            response_headers = getattr(response, "headers", {}) or {}
            request_id = request_id or response_headers.get("x-request-id")
            details = []
            if code:
                details.append(f"code={code}")
            if request_id:
                details.append(f"request_id={request_id}")
            suffix = f"（{', '.join(details)}）" if details else ""
            return f"{message}{suffix}"[:1200]
        except ValueError:
            return response.text[:1000]

    @staticmethod
    def _extract_text(response: dict[str, Any]) -> str:
        choices = response.get("choices") or []
        if not choices:
            raise QwenAPIError(f"Qwen 没有返回候选结果：{str(response)[:800]}")
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, list):
            text = "".join(str(item.get("text") or "") for item in content if isinstance(item, dict)).strip()
        else:
            text = str(content or "").strip()
        if not text:
            finish_reason = choices[0].get("finish_reason")
            raise QwenAPIError(f"Qwen 返回内容为空，结束原因：{finish_reason}")
        return text

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.removeprefix("```json").removeprefix("```").strip()
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3].strip()
        try:
            value = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise QwenAPIError(f"Qwen 返回的JSON无法解析：{exc}\n原始内容：{cleaned[:800]}") from exc
        if not isinstance(value, dict):
            raise QwenAPIError("Qwen 返回结果不是JSON对象。")
        return value

    def _require_key(self) -> None:
        if not self._key_is_configured():
            raise QwenAPIError(
                "尚未配置 Qwen API Key。请打开项目根目录的 .env，"
                "把 DASHSCOPE_API_KEY= 后面的占位文字替换为你的百炼密钥。"
            )

    def _key_is_configured(self) -> bool:
        key = self.api_key.strip()
        placeholders = ("请填写", "在这里填写", "your", "replace", "xxxxxxxx")
        return bool(key) and not any(token in key.lower() for token in placeholders)

    @staticmethod
    def _image_part(image: Image.Image, quality: int = 92, max_side: int = 2800) -> dict[str, Any]:
        converted = image.convert("RGB")
        longest = max(converted.size)
        if longest > max_side:
            scale = max_side / longest
            converted = converted.resize(
                (max(1, round(converted.width * scale)), max(1, round(converted.height * scale))),
                Image.Resampling.LANCZOS,
            )
        buffer = io.BytesIO()
        converted.save(buffer, format="JPEG", quality=quality, optimize=True)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
        }

    @staticmethod
    def _prepare_tag_crop(crop: Image.Image) -> Image.Image:
        image = crop.convert("RGB")
        if image.width < 1200:
            scale = min(5.0, max(1.8, 1200 / max(1, image.width)))
            image = image.resize(
                (int(image.width * scale), int(image.height * scale)),
                Image.Resampling.LANCZOS,
            )
        image = ImageOps.autocontrast(image, cutoff=1)
        image = ImageEnhance.Contrast(image).enhance(1.18)
        image = ImageEnhance.Sharpness(image).enhance(1.40)
        return image.filter(ImageFilter.UnsharpMask(radius=1.2, percent=115, threshold=3))

    @staticmethod
    def _analysis_prompt(catalog: list[dict], image_size: tuple[int, int]) -> str:
        width, height = image_size
        return f"""
你是固定12个食盐SKU的门店商品与价签识别系统。当前第一张图片是待识别门店照片。

核心目标：
1. 识别连续摆放的同款商品“陈列组”，而不是逐袋检测；
2. 检测真实可见的价签并读取品名与价格；
3. 商品和价签数量不要求相等；允许商品无价签、缺货留签、漏检和多余价签；
4. 补货、纸箱、地面堆放或库存场景没有明确价签时，不得虚构价格。

坐标规则：bbox 使用0到1000的归一化坐标 [x1,y1,x2,y2]，左上角为(0,0)，右下角为(1000,1000)。原图尺寸为 {width}×{height}。

场景含义：
- standard_shelf：标准层板货架；
- hook_display：挂钩陈列；
- basket_display：有明确零售陈列关系的筐装商品；
- promotion_stack：独立堆头或促销陈列；
- replenishment：补货、仓储、纸箱、地面或工作台堆放；
- unknown：无法判断。

固定12个SKU目录：
{catalog_prompt(catalog)}

识别要求：
- sku_id 只允许为 SKU01 至 SKU12；不确定必须为 null；
- 同层连续且外观相同的商品合并成一条 products 记录；
- 包装上的数字、广告条和宣传文案不能当作价签；
- price 只能来自现场价签；看不清必须为 null；
- 价签上只读到部分品名时保留 name_hint，不要为了配对硬猜SKU；
- 对横向很宽的同款商品陈列组，左右两端可能各有一张相同价格价签；两张都要检测，不能因为重复而遗漏；
- 小价签若只能看清价格，仍需返回 price，sku_id 可为 null，后续程序会依据货架位置配对；
- confidence 必须反映真实不确定性。
- scene_confidence、confidence、price、facing_count 必须是单个数字或 null，禁止返回数组；
- 每个 bbox 必须严格为 [x1,y1,x2,y2] 四个数字组成的扁平数组，禁止返回 [[x1,y1],[x2,y2]] 等嵌套格式。
""".strip()


# 兼容旧模块命名，便于 pipeline 与第三方调用。
GeminiAPIError = QwenAPIError
GeminiVisionClient = QwenVisionClient
VisionModelClient = QwenVisionClient
