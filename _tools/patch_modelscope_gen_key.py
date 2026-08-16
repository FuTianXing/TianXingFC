# -*- coding: utf-8 -*-
"""改 baoyu-modelscope-gen main.ts：API key 优先读环境变量 MODELSCOPE_API_KEY。"""
import io
import sys

PATH = r"C:\Users\Administrator\.agents\skills\baoyu-modelscope-gen\scripts\main.ts"

OLD = '''// --- 内部配置 (Internal Configuration) ---
// 将配置直接写在技能内部，不再依赖外部 .env 文件
const CONFIG = {
  API_KEY: "ms-ced4ea53-c2f3-4637-85d5-f388acb4c48d",
  DEFAULT_MODEL: "Tongyi-MAI/Z-Image-Turbo",
  BASE_URL: "https://api-inference.modelscope.cn/v1"
};'''

NEW = '''// --- 内部配置 (Internal Configuration) ---
// API Key 优先从环境变量 MODELSCOPE_API_KEY 读取，未设置时回退到内置默认值
const CONFIG = {
  API_KEY: process.env.MODELSCOPE_API_KEY || "ms-ced4ea53-c2f3-4637-85d5-f388acb4c48d",
  DEFAULT_MODEL: "Tongyi-MAI/Z-Image-Turbo",
  BASE_URL: "https://api-inference.modelscope.cn/v1"
};'''

with io.open(PATH, "r", encoding="utf-8") as f:
    src = f.read()

if OLD not in src:
    print("PATTERN NOT FOUND, aborting", file=sys.stderr)
    sys.exit(1)

new_src = src.replace(OLD, NEW, 1)

with io.open(PATH, "w", encoding="utf-8", newline="") as f:
    f.write(new_src)

print("PATCHED OK")
print("has MODELSCOPE_API_KEY:", "process.env.MODELSCOPE_API_KEY" in new_src)
