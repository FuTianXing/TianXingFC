# -*- coding: utf-8 -*-
"""改 baoyu-image-gen openai.ts：支持独立中转站端点 OPENAI_IMAGE_BASE_URL / OPENAI_IMAGE_API_KEY。

优先级设计：
- baseURL: OPENAI_IMAGE_BASE_URL（中转站）> OPENAI_BASE_URL（魔搭）> 官方默认
- apiKey : OPENAI_IMAGE_API_KEY（中转站）> OPENAI_API_KEY > MODELSCOPE_API_KEY（仅魔搭）
这样魔搭（OPENAI_BASE_URL）与 tu-zi 中转站（OPENAI_IMAGE_BASE_URL）互不冲突。
"""
import io
import sys

PATH = r"C:\Users\Administrator\.agents\skills\baoyu-image-gen\scripts\providers\openai.ts"

OLD = '''  const baseURL = process.env.OPENAI_BASE_URL || "https://api.openai.com/v1";
  // ModelScope(魔搭) 场景：优先用 MODELSCOPE_API_KEY，普通 OpenAI 用 OPENAI_API_KEY
  const apiKey = process.env.OPENAI_API_KEY ||
    (baseURL.includes("modelscope") ? process.env.MODELSCOPE_API_KEY : undefined);

  if (!apiKey) {
    throw new Error(
      "OPENAI_API_KEY is required (or MODELSCOPE_API_KEY when OPENAI_BASE_URL points to ModelScope). Codex/ChatGPT desktop login does not automatically grant OpenAI Images API access to this script."
    );
  }'''

NEW = '''  // 端点优先级：OPENAI_IMAGE_BASE_URL（中转站）> OPENAI_BASE_URL（魔搭/自定义）> 官方默认
  const baseURL = process.env.OPENAI_IMAGE_BASE_URL ||
    process.env.OPENAI_BASE_URL ||
    "https://api.openai.com/v1";
  // key 优先级：OPENAI_IMAGE_API_KEY（中转站）> OPENAI_API_KEY > MODELSCOPE_API_KEY（仅魔搭）
  const apiKey = process.env.OPENAI_IMAGE_API_KEY ||
    process.env.OPENAI_API_KEY ||
    (baseURL.includes("modelscope") ? process.env.MODELSCOPE_API_KEY : undefined);

  if (!apiKey) {
    throw new Error(
      "OPENAI_API_KEY is required (or OPENAI_IMAGE_API_KEY for a relay, MODELSCOPE_API_KEY for ModelScope). Codex/ChatGPT desktop login does not automatically grant OpenAI Images API access to this script."
    );
  }'''

with io.open(PATH, "r", encoding="utf-8") as f:
    src = f.read()

if OLD not in src:
    print("PATTERN NOT FOUND, aborting", file=sys.stderr)
    sys.exit(1)

new_src = src.replace(OLD, NEW, 1)

with io.open(PATH, "w", encoding="utf-8", newline="") as f:
    f.write(new_src)

print("PATCHED OK")
print("has OPENAI_IMAGE_BASE_URL:", "OPENAI_IMAGE_BASE_URL" in new_src)
print("has OPENAI_IMAGE_API_KEY:", "OPENAI_IMAGE_API_KEY" in new_src)
