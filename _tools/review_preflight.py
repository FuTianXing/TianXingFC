# -*- coding: utf-8 -*-
"""评审前客观核查：转义残缺 / 引用对应 / 术语"""
import io, re

t = io.open("温度的波粒互补定义：复相干无序度作为热力学标尺.md", encoding="utf-8").read()

# 1. 字面 \uXXXX 转义残缺
esc = re.findall(chr(92) * 2 + "u[0-9a-fA-F]{4}", t)
print("literal uXXXX escapes:", len(esc))
lines = t.splitlines()
for i, l in enumerate(lines, 1):
    if (chr(92) + "u") in l:
        print("line", i, ":", l[:100])

# 2. 文内引用核对
for n in range(1, 17):
    c = t.count("[" + str(n) + "]")
    if c == 0:
        print("UNCITED: [" + str(n) + "]")

# 3. 术语与数字核对
for pat in ["相位-only", "单元相子", "Jensen", "150 MeV", "2.725"]:
    print(pat, ":", t.count(pat))

# 4. em-dash 与控制字节
print("emdash:", t.count(chr(0x2014)))
bad = [hex(ord(c)) for c in t if (7 <= ord(c) <= 12 and ord(c) != 10) or ord(c) == 27]
print("ctrl:", bad)
