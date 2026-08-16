# -*- coding: utf-8 -*-
"""对照测试不同合法 size 与参数组合，定位 500。"""
import json
import os
import time
import urllib.request
import urllib.error

KEY = os.environ.get("SENSENOVA_API_KEY", "")

def probe(payload, label):
    print(f"\n===== {label} =====")
    print(f"PAYLOAD: {json.dumps(payload, ensure_ascii=False)}")
    req = urllib.request.Request(
        "https://token.sensenova.cn/v1/images/generations",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            print(f"STATUS: {resp.status}")
            print(f"BODY: {body[:800]}")
            return ("OK", body)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        print(f"HTTP ERROR: {e.code} {e.reason}")
        print(f"BODY: {body[:800]}")
        return (e.code, body)
    except Exception as e:
        print(f"OTHER ERROR: {type(e).__name__}: {e}")
        return ("EXC", str(e))

def main():
    prompt = "一只猫在窗台上晒太阳，卡通风格"

    # 合法 size 逐个测试（避开已测的 2752x1536）
    for size in ["2048x2048", "2560x720", "3072x864", "2496x1664", "1536x2752", "1664x2496"]:
        code, _ = probe({"model": "sensenova-u1-fast", "prompt": prompt, "size": size, "n": 1},
                        f"合法 size={size}")
        if code == 200:
            print(">>> 找到可用的 size:", size)
            break
        time.sleep(1)

    # 其他参数变体（在合法 size 上）
    probe({"model": "sensenova-u1-fast", "prompt": prompt, "size": "2048x2048"},
          "无 n 字段, size=2048x2048")
    probe({"model": "sensenova-u1-fast", "prompt": prompt, "size": "2048x2048", "quality": "high"},
          "带 quality=high")
    probe({"model": "sensenova-u1-fast", "prompt": prompt, "size": "2048x2048", "response_format": "url"},
          "带 response_format=url")
    probe({"model": "sensenova-u1-fast", "prompt": prompt[:50], "size": "2048x2048"},
          "短 prompt")

if __name__ == "__main__":
    main()
