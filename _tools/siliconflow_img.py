# -*- coding: utf-8 -*-
"""通过硅基流动 SiliconFlow API 生成图片。支持模型列表查询 + 文生图。"""
import json
import os
import sys
import urllib.request

KEY = os.environ.get("SILICONFLOW_API_KEY", "")
BASE = "https://api.siliconflow.cn/v1"

def api(path, payload, method="POST"):
    kwargs = {
        "headers": {
            "Authorization": "Bearer " + KEY,
            "Content-Type": "application/json",
        },
        "method": method,
    }
    if payload is not None:
        kwargs["data"] = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(BASE + path, **kwargs)
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8", errors="replace"))

def main():
    if not KEY:
        print("NO_SILICONFLOW_KEY")
        sys.exit(2)

    if len(sys.argv) > 1 and sys.argv[1] == "models":
        status, data = api("/models", None, method="GET")
        print("STATUS", status)
        for m in data.get("data", []):
            mid = m.get("id", "")
            if any(k in mid.lower() for k in ("flux", "qwen-image", "image", "kolors", "sana", "seedream")):
                print(" -", mid)
        return

    # 文生图
    prompt = sys.argv[1]
    model = sys.argv[2] if len(sys.argv) > 2 else "black-forest-labs/FLUX.1-schnell"
    size = sys.argv[3] if len(sys.argv) > 3 else "1024x576"
    out = sys.argv[4] if len(sys.argv) > 4 else "siliconflow_out.png"
    payload = {"model": model, "prompt": prompt, "image_size": size, "num_inference_steps": 30, "batch_size": 1}
    print("MODEL:", model, "SIZE:", size)
    status, data = api("/images/generations", payload)
    print("STATUS", status)
    if status != 200:
        print(json.dumps(data, ensure_ascii=False)[:2000])
        sys.exit(1)
    imgs = data.get("images", data.get("data", []))
    if not imgs:
        print(json.dumps(data, ensure_ascii=False)[:2000])
        sys.exit(1)
    url = imgs[0].get("url", "")
    if not url:
        print(json.dumps(data, ensure_ascii=False)[:2000])
        sys.exit(1)
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=120) as resp:
        blob = resp.read()
    with open(out, "wb") as f:
        f.write(blob)
    print("SAVED:", out, len(blob), "bytes")

if __name__ == "__main__":
    main()
