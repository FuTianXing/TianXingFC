# -*- coding: utf-8 -*-
"""魔搭完整生图链路：提交任务 + 轮询 + 下载。"""
import json
import sys
import time
import urllib.request
import urllib.error

KEY = "ms-ced4ea53-c2f3-4637-85d5-f388acb4c48d"
BASE = "https://api-inference.modelscope.cn/v1"

def req(path, payload=None, method="POST", extra_headers=None):
    headers = {
        "Authorization": "Bearer " + KEY,
        "Content-Type": "application/json",
    }
    if extra_headers:
        headers.update(extra_headers)
    kwargs = {"headers": headers, "method": method}
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
    prompt = "清爽夏日海报，主题夏日微风，浅薄荷绿到天蓝色渐变背景，白色细长手写体大字，笔画飘逸带风尾，中央留白，穿插小号英文 summer breeze 和细装饰线，品牌风格，16:9"
    out = sys.argv[1] if len(sys.argv) > 1 else r"D:\software\TianXingFC\images\modelscope_test.png"

    # 异步模式提交（和 baoyu-modelscope-gen 一致）
    status, body = req("/images/generations",
                       {"model": "Tongyi-MAI/Z-Image-Turbo", "prompt": prompt, "size": "2048x1152"},
                       extra_headers={"X-ModelScope-Async-Mode": "true"})
    print("=== SUBMIT ===")
    print("STATUS", status, body[:600])
    if status != 200:
        sys.exit(1)
    task_id = json.loads(body).get("task_id")
    if not task_id:
        print("no task_id")
        sys.exit(1)

    # 轮询
    for i in range(30):
        time.sleep(4)
        st, b = req(f"/tasks/{task_id}", None, "GET",
                    extra_headers={"X-ModelScope-Task-Type": "image_generation"})
        print(f"[poll {i}] status={st} {b[:400]}")
        if st == 200:
            data = json.loads(b)
            ts = data.get("task_status")
            if ts == "SUCCEED":
                imgs = data.get("output_images") or data.get("output") or []
                print("IMAGES:", json.dumps(imgs, ensure_ascii=False)[:800])
                if imgs:
                    url = imgs[0] if isinstance(imgs[0], str) else imgs[0].get("url")
                    if url:
                        with urllib.request.urlopen(url, timeout=120) as resp:
                            blob = resp.read()
                        with open(out, "wb") as f:
                            f.write(blob)
                        print("SAVED:", out, len(blob), "bytes")
                        sys.exit(0)
            elif ts == "FAILED":
                print("TASK FAILED:", b[:600])
                sys.exit(1)
    print("TIMEOUT")

if __name__ == "__main__":
    main()
