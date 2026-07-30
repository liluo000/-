from __future__ import annotations

import io
import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from PIL import Image, ImageOps

from src.catalog import load_catalog
from src.gemini_client import DEFAULT_API_ROOT, DEFAULT_MODEL, QwenVisionClient
from src.matcher import aggregate_matches
from src.pipeline import RecognitionPipeline
from src.storage import HistoryStore

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env", override=False)

st.set_page_config(page_title="商品与标价识别配对", page_icon="🧂", layout="wide")
st.markdown(
    """
<style>
.block-container {padding-top: 1.0rem; padding-bottom: 2rem;}
[data-testid="stMetricValue"] {font-size: 1.45rem;}
[data-testid="stSidebar"] {min-width: 315px;}
.small-note {color:#64748b;font-size:0.88rem;}
.status-ok {padding:.7rem .85rem;border-radius:.55rem;background:#ecfdf5;color:#166534;border:1px solid #bbf7d0;}
.status-bad {padding:.7rem .85rem;border-radius:.55rem;background:#fef2f2;color:#991b1b;border:1px solid #fecaca;}
</style>
""",
    unsafe_allow_html=True,
)

catalog = load_catalog(ROOT / "data" / "catalog.json")
history = HistoryStore(ROOT / "data" / "history.db")
api_key = os.getenv("DASHSCOPE_API_KEY", "")
client = QwenVisionClient(
    api_key=api_key,
    model=os.getenv("QWEN_MODEL", DEFAULT_MODEL),
    api_root=os.getenv("QWEN_API_ROOT", DEFAULT_API_ROOT),
)
pipeline = RecognitionPipeline(client, catalog, ROOT / "data" / "reference_contact.jpg")


@st.cache_data(show_spinner=False)
def load_image_from_path(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def sidebar_settings() -> dict:
    with st.sidebar:
        st.header("阿里云百炼 Qwen")
        st.write(f"固定模型：`{client.model}`")
        if client.is_configured:
            st.markdown('<div class="status-ok">✅ 已从 .env 读取 API Key</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-bad">❌ 尚未配置 DASHSCOPE_API_KEY</div>', unsafe_allow_html=True)
            st.caption("请编辑项目根目录的 `.env`，填写 `DASHSCOPE_API_KEY=你的百炼密钥`，然后重新启动程序。")

        if st.button("测试 Qwen 连接", use_container_width=True):
            with st.spinner("正在连接阿里云百炼 Qwen……"):
                ok, message = client.health_check()
            st.success(message) if ok else st.error(message)

        st.divider()
        st.header("识别设置")
        use_reference = st.checkbox(
            "发送12个SKU参考总图",
            value=True,
            help="同时发送门店照片和12个SKU参考图，通常有助于区分相似包装。",
        )
        refine_tags = st.checkbox(
            "对价签进行二次放大识别",
            value=True,
            help="主分析后逐个裁剪价签，再调用Qwen读取小字。更准确，但会增加请求次数。",
        )
        max_tags = st.number_input("最多二次识别价签数", min_value=1, max_value=40, value=12)
        accept = st.slider("自动通过阈值", 0.50, 0.95, 0.72, 0.01)
        review = st.slider("人工复核阈值", 0.25, accept, min(0.50, accept), 0.01)

        st.divider()
        st.caption("程序使用阿里云百炼 Qwen3.7-Flash 多模态 API，不依赖本地模型。")
        return {
            "use_reference": use_reference,
            "refine_tags": refine_tags,
            "max_tags": int(max_tags),
            "accept": float(accept),
            "review": float(review),
        }


def aggregate_matches_from_dicts(matches: list[dict]) -> list[dict]:
    from src.schemas import MatchItem

    return aggregate_matches([MatchItem.model_validate(match) for match in matches], catalog)


def open_uploaded_image(upload) -> Image.Image:
    image = Image.open(upload)
    image = ImageOps.exif_transpose(image)
    return image.convert("RGB")


settings = sidebar_settings()

tab_recognition, tab_catalog, tab_history, tab_system = st.tabs(
    ["📷 识别与配对", "📚 12个SKU", "🗂️ 历史记录", "⚙️ 系统检查"]
)

with tab_recognition:
    st.title("12个SKU商品与标价识别配对")
    st.caption(
        "上传门店照片后，Qwen识别固定12个SKU和价签；程序将SKU目录价格与现场候选价签比较，输出价格一致、价格不一致或无法判断。"
    )

    if not client.is_configured:
        st.error("尚未配置API Key。请先编辑 `.env` 中的 `DASHSCOPE_API_KEY`，保存后重新启动。")

    uploads = st.file_uploader(
        "上传一张或多张门店照片",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        help="建议正面拍摄，尽量让商品包装和价签文字清晰可见。",
    )

    sample_col1, sample_col2, clear_col, _ = st.columns([1.3, 1.3, 1, 4])
    with sample_col1:
        if st.button("载入标准货架示例", use_container_width=True):
            st.session_state["sample_paths"] = [str(ROOT / "data" / "samples" / "standard_shelf.jpg")]
    with sample_col2:
        if st.button("载入补货场景示例", use_container_width=True):
            st.session_state["sample_paths"] = [str(ROOT / "data" / "samples" / "replenishment.jpg")]
    with clear_col:
        if st.button("清空结果", use_container_width=True):
            st.session_state.pop("analysis_results", None)
            st.session_state.pop("sample_paths", None)
            st.rerun()

    has_inputs = bool(uploads) or bool(st.session_state.get("sample_paths"))
    run = st.button(
        "开始识别配对",
        type="primary",
        disabled=(not has_inputs or not client.is_configured),
        use_container_width=False,
    )

    if run:
        inputs: list[tuple[str, Image.Image]] = []
        if uploads:
            for upload in uploads:
                try:
                    inputs.append((upload.name, open_uploaded_image(upload)))
                except Exception as exc:  # noqa: BLE001
                    st.error(f"{upload.name} 读取失败：{exc}")
        else:
            for path in st.session_state.pop("sample_paths", []):
                inputs.append((Path(path).name, load_image_from_path(path).copy()))

        st.session_state["analysis_results"] = []
        progress = st.progress(0, text="准备识别")
        for index, (filename, image) in enumerate(inputs):
            try:
                progress.progress(index / max(1, len(inputs)), text=f"正在识别：{filename}")
                result, annotated = pipeline.run(
                    image,
                    filename,
                    use_reference_contact=settings["use_reference"],
                    refine_price_tags=settings["refine_tags"],
                    max_refine_tags=settings["max_tags"],
                    accept_threshold=settings["accept"],
                    review_threshold=settings["review"],
                )
                st.session_state["analysis_results"].append(
                    {"result": result.model_dump(), "original": image, "annotated": annotated}
                )
            except Exception as exc:  # noqa: BLE001
                st.session_state["analysis_results"].append(
                    {"error": str(exc), "filename": filename, "original": image}
                )
        progress.progress(1.0, text="识别完成")

    for result_index, item in enumerate(st.session_state.get("analysis_results", [])):
        st.divider()
        if "error" in item:
            st.error(f'{item.get("filename", "图片")} 识别失败：{item["error"]}')
            st.image(item.get("original"), use_container_width=True)
            continue

        result = item["result"]
        st.subheader(result["filename"])
        accepted_count = sum(match["status"] == "accepted" for match in result["matches"])
        review_count = sum(match["status"] == "review" for match in result["matches"])
        unmatched_count = sum(match["status"] == "unmatched" for match in result["matches"])
        metrics = st.columns(7)
        metrics[0].metric("场景", result["scene_type"])
        metrics[1].metric("商品组", len(result["products"]))
        metrics[2].metric("价签", len(result["price_tags"]))
        metrics[3].metric("自动通过", accepted_count)
        metrics[4].metric("需复核", review_count)
        metrics[5].metric("未匹配", unmatched_count)
        metrics[6].metric("耗时", f'{result["elapsed_seconds"]:.1f}s')

        left, right = st.columns(2)
        with left:
            st.image(item["original"], caption="原图", use_container_width=True)
        with right:
            st.image(item["annotated"], caption="识别框与配对连线", use_container_width=True)

        if result.get("notes"):
            st.info("；".join(str(note) for note in result["notes"]))

        aggregated = aggregate_matches_from_dicts(result["matches"])
        frame = pd.DataFrame(aggregated)
        if frame.empty:
            st.warning("未形成商品—价格结果。补货场景、无价签或画面信息不足时，这是合理结果。")
        else:
            sku_options = [sku["sku_id"] for sku in catalog]
            status_options = ["accepted", "review", "unmatched"]
            edited = st.data_editor(
                frame,
                key=f"editor_{result_index}",
                hide_index=True,
                num_rows="dynamic",
                use_container_width=True,
                column_config={
                    "sku_id": st.column_config.SelectboxColumn("SKU", options=sku_options, required=False),
                    "价格": st.column_config.NumberColumn(
                        "价格", min_value=0.0, max_value=1000.0, step=0.1, format="￥%.2f", disabled=True,
                        help="12个SKU目录中的参考/标准价格。"
                    ),
                    "候选价格": st.column_config.TextColumn(
                        "候选价格", disabled=True,
                        help="照片中识别到的现场价签；多个价签会分别列出。"
                    ),
                    "识别结果": st.column_config.TextColumn(
                        "识别结果", disabled=True,
                        help="唯一现场价与目录价格相同：价格一致；不同：价格不一致；无价或多个价签：无法判断。"
                    ),
                    "配对置信度": st.column_config.ProgressColumn(
                        "置信度", min_value=0.0, max_value=1.0, format="%.3f"
                    ),
                    "状态": st.column_config.SelectboxColumn("状态", options=status_options),
                },
            )
            result_rows = edited.to_dict(orient="records")
            export_col1, export_col2, export_col3, save_col = st.columns(4)
            with export_col1:
                st.download_button(
                    "下载CSV",
                    data=edited.to_csv(index=False).encode("utf-8-sig"),
                    file_name=f'{Path(result["filename"]).stem}_识别结果.csv',
                    mime="text/csv",
                    use_container_width=True,
                )
            with export_col2:
                export_payload = {"rows": result_rows, "analysis": result}
                st.download_button(
                    "下载JSON",
                    data=json.dumps(export_payload, ensure_ascii=False, indent=2).encode("utf-8"),
                    file_name=f'{Path(result["filename"]).stem}_识别结果.json',
                    mime="application/json",
                    use_container_width=True,
                )
            with export_col3:
                buffer = io.BytesIO()
                item["annotated"].save(buffer, format="JPEG", quality=92)
                st.download_button(
                    "下载标注图",
                    data=buffer.getvalue(),
                    file_name=f'{Path(result["filename"]).stem}_标注.jpg',
                    mime="image/jpeg",
                    use_container_width=True,
                )
            with save_col:
                if st.button("保存人工确认结果", key=f"save_{result_index}", use_container_width=True):
                    record_id = history.save(result["filename"], result["scene_type"], result_rows, result)
                    st.success(f"已保存，记录编号 {record_id}")

        with st.expander("查看商品组、价签、得分和Qwen原始响应"):
            st.markdown("**商品陈列组**")
            st.dataframe(pd.DataFrame(result["products"]), hide_index=True, use_container_width=True)
            st.markdown("**价签**")
            st.dataframe(pd.DataFrame(result["price_tags"]), hide_index=True, use_container_width=True)
            st.markdown("**未匹配价签**")
            st.write(result["unmatched_price_tags"] or "无")
            st.markdown("**匹配得分矩阵**")
            st.write(result.get("extra", {}).get("score_matrix", []))
            st.markdown("**Qwen原始JSON响应**")
            st.code(result["model_raw_response"], language="json")

with tab_catalog:
    st.title("12个SKU目录")
    st.caption("程序只使用品名、包装参考图、规格和关键词；不使用69码。参考价只用于异常提示，不代替现场价签。")
    columns = st.columns(4)
    for index, sku in enumerate(catalog):
        with columns[index % 4]:
            st.image(ROOT / sku["reference_image"], use_container_width=True)
            st.markdown(f'**{sku["sku_id"]}｜{sku["name"]}**')
            st.write(f'规格：{sku["spec"]}　参考价：￥{sku["reference_price"]:.2f}')
            st.caption("关键词：" + "、".join(sku.get("keywords", [])))
            if sku.get("source_note"):
                st.warning(sku["source_note"])

with tab_history:
    st.title("历史记录")
    records = history.list()
    if not records:
        st.info("还没有保存记录。请在识别结果页面完成人工确认后保存。")
    else:
        history_frame = pd.DataFrame(records)
        st.dataframe(history_frame, hide_index=True, use_container_width=True)
        selected_id = st.selectbox("查看详细记录", options=[record["id"] for record in records])
        selected = history.get(int(selected_id))
        if selected:
            st.json(selected["result"], expanded=False)
            st.download_button(
                "下载该记录JSON",
                data=json.dumps(selected, ensure_ascii=False, indent=2).encode("utf-8"),
                file_name=f"history_{selected_id}.json",
                mime="application/json",
            )

with tab_system:
    st.title("系统检查")
    st.write(f"Qwen模型：`{client.model}`")
    st.write(f"Qwen接口：`{client.endpoint}`")
    st.write(f"API Key状态：**{'已配置' if client.is_configured else '未配置'}**")
    st.write(f"SKU数量：**{len(catalog)}**")
    st.write(f"参考图数量：**{sum((ROOT / item['reference_image']).exists() for item in catalog)}**")
    st.write(f"历史数据库：`{ROOT / 'data' / 'history.db'}`")
    if st.button("运行Qwen连接测试", type="primary"):
        with st.spinner("正在连接阿里云百炼 Qwen……"):
            ok, message = client.health_check()
        st.success(message) if ok else st.error(message)
    st.info(
        "免费额度可能存在每分钟和每日限制。程序使用结构化JSON输出、高分辨率图片分析、二次价签OCR及人工复核；"
        "补货或仓储照片没有明确价签时会拒绝虚构价格。"
    )
