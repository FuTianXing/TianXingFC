# -*- coding: utf-8 -*-
"""最终验证：1) 同一 key 的文本模型是否正常（排除 key 禁用）；2) 图像接口重试。"""
import json
import os
import time
import urllib.request
import urllib.error

KEY = os.environ.get("SENSENOVA_API_KEY", "")

def post(endpoint, payload, label, timeout=60):
    print(f"\n===== {label} =====")
    req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            print(f"STATUS: {resp.status}")
            print(f"BODY: {body[:600]}")
            return resp.status
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        print(f"HTTP ERROR: {e.code} {e.reason} | BODY: {body[:600]}")
        return e.code
    except Exception as e:
        print(f"OTHER ERROR: {type(e).__name__}: {e}")
        return -1

def main():
    # 1. 文本模型 chat/completions（同一 key）
    post("https://token.sensenova.cn/v1/chat/completions",
         {"model": "deepseek-v4-flash",
          "messages": [{"role": "user", "content": "回复：OK"}],
          "max_tokens": 10},
         "文本模型 deepseek-v4-flash (同 key)")

    time.sleep(1)

    # 2. 图像接口重试（最小合法 payload）
    post("https://token.sensenova.cn/v1/images/generations",
         {"model": "sensenova-u1-fast", "prompt": "一只猫", "size": "2752x1536", "n": 1},
         "图像接口重试 2752x1536")

    time.sleep(1)

    # 3. 图像接口重试（另一个 size）
    post("https://token.sensenova.cn/v1/images/generations",
         {"model": "sensenova-u1-fast", "prompt": "一只猫", "size": "2048x2048", "n": 1},
         "图像接口重试 2048x2048")

if __name__ == "__main__":
    main()
