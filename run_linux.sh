#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "已创建 .env。请先编辑 DASHSCOPE_API_KEY，然后重新运行本脚本。"
  exit 1
fi

if grep -q "在这里填写" .env; then
  echo "请先编辑 .env，把 DASHSCOPE_API_KEY= 后面的占位文字替换为阿里云百炼 API Key。"
  exit 1
fi

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
