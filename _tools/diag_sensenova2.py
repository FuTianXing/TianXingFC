# -*- coding: utf-8 -*-
"""查看 sensenova-u1-fast 模型完整信息 + 尝试多种调用格式定位 500。"""
import json
import os
import urllib.request
import urllib.error

KEY = os.environ.get("SENSENOVA_API_KEY", "")

def probe(endpoint, payload, label, method="POST"):
    print(f"\n===== {label} =====")
    print(f"ENDPOINT: {endpoint}")
    req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8") if payload is not None else None,
        headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json"},
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            print(f"STATUS: {resp.status}")
            print(f"BODY: {body[:1500]}")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        print(f"HTTP ERROR: {e.code} {e.reason}")
        print(f"BODY: {body[:1500]}")
    except Exception as e:
        print(f"OTHER ERROR: {type(e).__name__}: {e}")

def main():
    # 1. 模型完整描述
    req = urllib.request.Request(
        "https://token.sensenova.cn/v1/models",
        headers={"Authorization": "Bearer " + KEY},
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    for m in data.get("data", []):
        if m.get("id") == "sensenova-u1-fast":
            print("===== sensenova-u1-fast 完整信息 =====")
            print(json.dumps(m, ensure_ascii=False, indent=2))

    prompt = "一只猫在窗台上晒太阳"

    # 2. 不同 size 格式：1024x1024 / 1:1
    probe("https://token.sensenova.cn/v1/images/generations",
          {"model": "sensenova-u1-fast", "prompt": prompt, "size": "1024x1024", "n": 1},
          "size=1024x1024")

    # 3. OpenAI 标准 images 参数（response_format=b64_json）
    probe("https://token.sensenova.cn/v1/images/generations",
          {"model": "sensenova-u1-fast", "prompt": prompt, "size": "1024x1024", "n": 1, "response_format": "b64_json"},
          "带 response_format")

    # 4. 尝试 /v1/image/generations（单数）
    probe("https://token.sensenova.cn/v1/image/generations",
          {"model": "sensenova-u1-fast", "prompt": prompt, "size": "1024x1024"},
          "端点 /v1/image/generations")

    # 5. 尝试 chat/completions + tools image_generation（Responses 风格）
    probe("https://token.sensenova.cn/v1/chat/completions",
          {"model": "sensenova-u1-fast",
           "messages": [{"role": "user", "content": prompt}],
           "tools": [{"type": "image_generation", "image_generation": {"size": "1024x1024", "quality": "auto"}}]},
          "chat/completions + image_generation tool")

    # 6. Responses API 风格
    probe("https://token.sensenova.cn/v1/responses",
          {"model": "sensenova-u1-fast",
           "input": prompt,
           "tools": [{"type": "image_generation", "image_generation": {"size": "1024x1024"}}]},
          "/v1/responses + image_generation tool")

if __name__ == "__main__":
    main()
