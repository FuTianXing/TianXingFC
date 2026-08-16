# -*- coding: utf-8 -*-
"""探测魔搭(ModelScope)可用的图像生成模型：逐个提交小任务，判断模型是否存在（返回 task_id=存在，404/400=不存在或不可用）。"""
import json
import sys
import time
import urllib.request
import urllib.error

KEY = "ms-ced4ea53-c2f3-4637-85d5-f388acb4c48d"
BASE = "https://api-inference.modelscope.cn/v1"

# 候选文生图模型（覆盖魔搭常见命名 + 已验证的 Z-Image 系列）
CANDIDATES = [
    "Tongyi-MAI/Z-Image-Turbo",       # 已确认可用
    "Tongyi-MAI/Z-Image",             # 完整版
    "Tongyi-MAI/Z-Image-Lite",
    "Qwen/Qwen-Image",
    "Qwen/Qwen-Image-Edit",
    "MusePublic/Qwen-Image-Edit",
    "black-forest-labs/FLUX.1-schnell",
    "black-forest-labs/FLUX.1-dev",
    "Kwai-Kolors/Kolors",
    "ByteDance-Seed/Seedream-3.0",
    "ByteDance/Seedream-3.0",
    "AIGCDesign/TIMM-4x",
    "dataautogpt3/ProteusV0.4",
    "dataroaming/sana",
    "eastl/playground-v2.5",
    "stabilityai/stable-diffusion-3-5-large",
    "Tencent-Hunyuan/Hy3",
]

def probe(model):
    payload = {"model": model, "prompt": "一只橘猫", "size": "1024x1024"}
    req = urllib.request.Request(
        BASE + "/images/generations",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": "Bearer " + KEY,
            "Content-Type": "application/json",
            "X-ModelScope-Async-Mode": "true",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            return resp.status, body
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        return e.code, body
    except Exception as e:
        return -1, f"{type(e).__name__}: {e}"

def main():
    ok, fail = [], []
    for m in CANDIDATES:
        code, body = probe(m)
        try:
            j = json.loads(body)
            has_task = bool(j.get("task_id"))
        except Exception:
            has_task = False
        if code == 200 and has_task:
            ok.append(m)
            print(f"✅ {m}  (task_id 已创建)")
        else:
            fail.append((m, code, body[:150]))
            print(f"❌ {m}  status={code}  {body[:120]}")
        time.sleep(0.8)
    print("\n===== 汇总 =====")
    print("可用:", ok)
    print("不可用:", [m for m, _, _ in fail])

if __name__ == "__main__":
    main()
