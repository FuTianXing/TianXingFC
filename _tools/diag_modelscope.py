# -*- coding: utf-8 -*-
"""验证魔搭(ModelScope) key 有效性 + 列出可用图像模型 + 测试生图。"""
import json
import sys
import urllib.request
import urllib.error

KEY = "ms-ced4ea53-c2f3-4637-85d5-f388acb4c48d"
BASE = "https://api-inference.modelscope.cn/v1"

def req(path, payload=None, method="POST"):
    kwargs = {
        "headers": {
            "Authorization": "Bearer " + KEY,
            "Content-Type": "application/json",
        },
        "method": method,
    }
    if payload is not None:
        kwargs["data"] = json.dumps(payload).encode("utf-8")
    r = urllib.request.Request(BASE + path, **kwargs)
    try:
        with urllib.request.urlopen(r, timeout=90) as resp:
            return resp.status, resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", errors="replace")
    except Exception as e:
        return -1, f"{type(e).__name__}: {e}"

def main():
    # 1. 模型列表（GET）
    status, body = req("/models", None, "GET")
    print("=== GET /models ===")
    print("STATUS", status)
    if status == 200:
        try:
            data = json.loads(body)
            models = data.get("data", [])
            imgs = [m for m in models if any(k in str(m.get("id", "")).lower() for k in ("image", "z-image", "flux", "sana", "kolors", "seedream", "wanx", "cogview"))]
            print(f"共 {len(models)} 个模型, 图像相关 {len(imgs)} 个:")
            for m in imgs:
                print(" -", m.get("id"))
        except Exception as e:
            print("parse error:", e, body[:800])
    else:
        print(body[:800])

    # 2. 直接测一张图
    status, body = req("/images/generations",
                       {"model": "Tongyi-MAI/Z-Image-Turbo",
                        "prompt": "一只橘猫在窗台上晒太阳，插画风格",
                        "size": "1024x1024"})
    print("\n=== POST /images/generations ===")
    print("STATUS", status)
    print(body[:1200])

if __name__ == "__main__":
    main()
