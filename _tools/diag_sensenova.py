# -*- coding: utf-8 -*-
"""诊断商汤 sensenova 文生图 API：打印原始请求/响应，定位 500 原因。"""
import json
import os
import sys
import urllib.request
import urllib.error

KEY = os.environ.get("SENSENOVA_API_KEY", "")

def probe(endpoint, payload, label):
    print(f"\n===== {label} =====")
    print(f"ENDPOINT: {endpoint}")
    print(f"PAYLOAD: {json.dumps(payload, ensure_ascii=False)}")
    req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            print(f"STATUS: {resp.status}")
            print(f"HEADERS: {dict(resp.headers)}")
            print(f"BODY: {body[:3000]}")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        print(f"HTTP ERROR: {e.code} {e.reason}")
        print(f"HEADERS: {dict(e.headers)}")
        print(f"BODY: {body[:3000]}")
    except Exception as e:
        print(f"OTHER ERROR: {type(e).__name__}: {e}")

def main():
    if not KEY:
        print("NO_SENSENOVA_API_KEY")
        sys.exit(2)
    print(f"KEY: set, len={len(KEY)}, prefix={KEY[:8]}...")
    prompt = "一只猫在窗台上晒太阳"

    # 1. 技能里默认的端点 + 模型
    probe("https://token.sensenova.cn/v1/images/generations",
          {"model": "sensenova-u1-fast", "prompt": prompt, "size": "2752x1536", "n": 1},
          "技能默认: token.sensenova.cn + sensenova-u1-fast")

    # 2. 平台主端点
    probe("https://api.sensenova.cn/v1/images/generations",
          {"model": "sensenova-u1-fast", "prompt": prompt, "size": "2752x1536", "n": 1},
          "候选: api.sensenova.cn + sensenova-u1-fast")

    # 3. 无 n 字段（部分实现不接受 n）
    probe("https://token.sensenova.cn/v1/images/generations",
          {"model": "sensenova-u1-fast", "prompt": prompt, "size": "2752x1536"},
          "候选: token 端点 + 去掉 n")

    # 4. 模型列表查询（探测模型是否还存在）
    req = urllib.request.Request(
        "https://token.sensenova.cn/v1/models",
        headers={"Authorization": "Bearer " + KEY},
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            print(f"\n===== GET /v1/models =====")
            print(f"STATUS: {resp.status}")
            print(f"BODY: {resp.read().decode('utf-8', errors='replace')[:3000]}")
    except urllib.error.HTTPError as e:
        print(f"\n===== GET /v1/models =====")
        print(f"HTTP ERROR: {e.code} {e.reason}")
        print(f"BODY: {e.read().decode('utf-8', errors='replace')[:3000]}")
    except Exception as e:
        print(f"\n===== GET /v1/models =====")
        print(f"OTHER ERROR: {type(e).__name__}: {e}")

if __name__ == "__main__":
    main()
