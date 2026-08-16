# -*- coding: utf-8 -*-
"""修补 sensenova_u1_gen.py：HTTP 错误时输出原始响应体，便于排查 500/400 等。"""
import io
import sys

PATH = r"C:\Users\Administrator\.agents\skills\sensenova-u1-fast\scripts\sensenova_u1_gen.py"

OLD = '''        msgs = {400: "参数不合法 — 请检查 prompt、size 等参数",
                403: "权限不足 — 请检查 API Key 是否正确",
                429: "额度超限 — 每5小时1500次限制，请稍后重试",
                500: "服务端错误 — 请稍后重试"}
        print(f"错误 {status}: {msgs.get(status, error_body)}", file=sys.stderr)
        sys.exit(1)'''

NEW = '''        msgs = {400: "参数不合法 — 请检查 prompt、size 等参数",
                403: "权限不足 — 请检查 API Key 是否正确",
                429: "额度超限 — 每5小时1500次限制，请稍后重试",
                500: "服务端错误 — 请稍后重试"}
        print(f"错误 {status}: {msgs.get(status, error_body)}", file=sys.stderr)
        if error_body:
            print(f"原始响应: {error_body}", file=sys.stderr)
        sys.exit(1)'''

with io.open(PATH, "r", encoding="utf-8") as f:
    src = f.read()

if OLD not in src:
    print("PATTERN NOT FOUND, aborting", file=sys.stderr)
    sys.exit(1)

new_src = src.replace(OLD, NEW, 1)

with io.open(PATH, "w", encoding="utf-8", newline="") as f:
    f.write(new_src)

print("PATCHED OK")
print("has '原始响应' line:", "原始响应" in new_src)
