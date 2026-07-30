# 12个SKU商品与标价识别配对系统（Qwen3.7-Flash 版）

本项目固定使用阿里云百炼 Qwen3.7-Flash 多模态 API，不需要本地模型，也不包含 OpenRouter 或 OpenAI 接口。

## 主要功能

- 上传一张或多张门店照片；
- 在固定12个SKU中识别商品陈列组；
- 检测价签并进行二次放大识别；
- 使用品名、货架位置和全局分配进行商品—价签配对；
- 允许商品无价签、价签无商品以及多个候选价签；
- 输出价格核验结果、标注图、CSV、JSON，并保存人工确认记录。

## 价格核验规则

结果表中的字段含义：

- **价格**：`data/catalog.json` 中该SKU的参考/标准价格；
- **候选价格**：Qwen从门店照片中识别出的现场价签价格，并显示价签编号；
- **识别结果**：
  - 恰好一个对应价签，且现场价等于标准价：`价格一致`；
  - 恰好一个对应价签，但现场价不等于标准价：`价格不一致`；
  - 没有识别到价格、价签价格不可读，或同一商品对应多个价签：`无法判断`。

例如 SKU12 下方同时检测到 `T04 ￥3.50` 和 `T05 ￥3.50`，即使两个价格相同，仍显示 `无法判断`，避免程序擅自选择一个标签。

## Windows 使用方法

1. 安装 Python 3.10—3.13，并在安装时勾选 **Add Python to PATH**；
2. 解压到简单路径，例如 `D:\salt_price_matcher_v4`；
3. 双击 `OPEN_CONFIG.bat`；
4. 在 `.env` 中只填写：

```env
DASHSCOPE_API_KEY=你的阿里云百炼_API_Key
QWEN_MODEL=qwen3.7-flash
QWEN_API_ROOT=https://dashscope.aliyuncs.com/compatible-mode/v1
```

5. 保存后双击 `START_HERE.bat`；
6. 第一次运行会创建虚拟环境并安装依赖；
7. 浏览器访问 `http://127.0.0.1:8501`。

启动失败时运行 `RUN_DEBUG.bat`；如虚拟环境损坏，先运行 `RESET_VENV.bat`，再运行 `START_HERE.bat`。

## 目录说明

- `app.py`：Streamlit前端；
- `src/gemini_client.py`：Qwen图片识别接口（保留旧文件名以兼容现有模块）；
- `src/pipeline.py`：识别流程；
- `src/matcher.py`：候选价签、匈牙利匹配及价格核验；
- `data/catalog.json`：12个SKU及标准价格；
- `data/reference_images/`：12个SKU参考图片；
- `tests/`：自动测试。

## 测试

```bash
pip install -r requirements-dev.txt
pytest -q
```

自动测试覆盖SKU解析、Qwen请求格式、漏检和多余价签、宽商品组双价签及价格一致性判断。
