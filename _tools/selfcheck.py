# -*- coding: utf-8 -*-
"""落盘复扫器（永久化）：新增行复扫。**不要再用 shell 内联跑复扫**（反引号与 $ 会被 shell 处理）。
用法：python _tools/selfcheck.py <文件路径> [更多路径...]
"""
import io
import os
import re
import subprocess
import sys

CTRL = "\t\x0b\x0c\r\x07"
MISS = ["imes", "heta", "lpha", "pprox", "rac{", "egin{", "ext{", "amma", "abla"]
BARE = ["langle", "sigma", "sqrt", "frac", "vert", "pi", "max"]
LAW = ("值得注意的是", "众所周知", "首先", "其次", "深刻", "极其")
CONTRAST = ("不是……而是", "并非……而是", "不在于……而在于")

if len(sys.argv) < 2:
    print("用法：python _tools/selfcheck.py <文件路径> [...]")
    sys.exit(2)

hits = []
for path in sys.argv[1:]:
    if not os.path.exists(path):
        print("MISSING:", path)
        continue
    out = subprocess.run(["git", "diff", "--ignore-cr-at-eol", "-U0", "HEAD", "--", path],
                         capture_output=True, text=True, encoding="utf-8", errors="replace")
    for line in out.stdout.split("\n"):
        if not line.startswith("+") or line.startswith("+++"):
            continue
        t = line[1:]
        for c in CTRL:
            j = t.find(c)
            while j != -1:
                if j + 1 < len(t) and t[j + 1].isalpha():
                    hits.append((path, "CTRL+letter", repr(t[max(0, j - 16):j + 16])))
                j = t.find(c, j + 1)
        for m in re.finditer(r"\\vert(?=[A-Za-z])", t):
            hits.append((path, "vert-adjacent", ""))
        for m in re.finditer(r"\$[^$]*\$", t):
            inner = m.group(0)[1:-1]
            if "`" in t:
                continue
            for w in BARE:
                if re.search(r"(?<!\\)\b" + w + r"\b", inner):
                    hits.append((path, "bare-cmd:" + w, m.group(0)[:70]))
        for w in MISS:
            if re.search(r"(?<![\w\\])" + re.escape(w), t):
                hits.append((path, "missing:" + w, t[:90]))
        if "——" in t:
            hits.append((path, "em-dash", t[:100]))
        for w in LAW:
            if w in t:
                hits.append((path, "law:" + w, t[:90]))
        for pat in CONTRAST:
            if pat in t:
                hits.append((path, "contrast:" + pat, t[:90]))
        st = t.strip()
        is_item = st.startswith(("|", "-", "*", ">", "#", "```", "$", "![", "!")) or re.match(r"^\d+\.", st)
        if not st or is_item:
            continue
        if len(st) > 300:
            hits.append((path, "para>300", "len=%d" % len(st)))

print("=" * 66)
print("新增行复扫（%d 个文件）" % (len(sys.argv) - 1))
if hits:
    print("命中 %d 处：" % len(hits))
    for h in hits[:14]:
        print("  [%s] %s | %s" % (h[1], h[0], h[2][:110]))
    sys.exit(1)
print("零命中")
print("=" * 66)
