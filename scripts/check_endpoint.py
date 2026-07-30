from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.gemini_client import DEFAULT_API_ROOT, DEFAULT_MODEL, QwenVisionClient

load_dotenv(ROOT / ".env")
client = QwenVisionClient(
    api_key=os.getenv("DASHSCOPE_API_KEY", ""),
    model=os.getenv("QWEN_MODEL", DEFAULT_MODEL),
    api_root=os.getenv("QWEN_API_ROOT", DEFAULT_API_ROOT),
)
ok, message = client.health_check()
print(message)
raise SystemExit(0 if ok else 1)
