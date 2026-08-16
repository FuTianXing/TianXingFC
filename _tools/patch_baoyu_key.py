# -*- coding: utf-8 -*-
"""改 baoyu-image-gen openai.ts：魔搭场景下 key 从 MODELSCOPE_API_KEY 读取（fallback）。"""
import io
import sys

PATH = r"C:\Users\Administrator\.agents\skills\baoyu-image-gen\scripts\providers\openai.ts"

OLD = '''  const baseURL = process.env.OPENAI_BASE_URL || "https://api.openai.com/v1";
  const apiKey = process.env.OPENAI_API_KEY;

  if (!apiKey) {
    throw new Error(
      "OPENAI_API_KEY is required. Codex/ChatGPT desktop login does not automatically grant OpenAI Images API access to this script."
    );
  }'''

NEW = '''  const baseURL = process.env.OPENAI_BASE_URL || "https://api.openai.com/v1";
  // ModelScope(魔搭) 场景：优先用 MODELSCOPE_API_KEY，普通 OpenAI 用 OPENAI_API_KEY
  const apiKey = process.env.OPENAI_API_KEY ||
    (baseURL.includes("modelscope") ? process.env.MODELSCOPE_API_KEY : undefined);

  if (!apiKey) {
    throw new Error(
      "OPENAI_API_KEY is required (or MODELSCOPE_API_KEY when OPENAI_BASE_URL points to ModelScope). Codex/ChatGPT desktop login does not automatically grant OpenAI Images API access to this script."
    );
  }'''

with io.open(PATH, "r", encoding="utf-8") as f:
    src = f.read()

if OLD not in src:
    print("PATTERN NOT FOUND, aborting", file=sys.stderr)
    sys.exit(1)

with io.open(PATH, "w", encoding="utf-8", newline="") as f:
    f.write(src.replace(OLD, NEW, 1))

print("PATCHED OK")
print("has MODELSCOPE_API_KEY:", "MODELSCOPE_API_KEY" in src.replace(OLD, NEW, 1))
