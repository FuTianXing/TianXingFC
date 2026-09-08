# -*- coding: utf-8 -*-
"""复审修复终检：C1-C5 落位 + 无副作用"""
import io

p = "温度的波粒互补定义：复相干无序度作为热力学标尺.md"
t = io.open(p, encoding="utf-8").read()
BS = chr(92)

checks = [
    ("C1 引用数学模式", all(("$[%d]$") % n in t for n in [8, 9, 10, 11, 13, 14, 15, 16, 17])),
    ("C2 直引号清零", t.count('"') == 0 and t.count("\u201c") == t.count("\u201d")),
    ("C3/C4 双符号统一", ("p_" + BS + "mathrm r") not in t
     and ("1-" + BS + "xi " + BS + "approx") in t
     and ("本节记号 $p$ 即第 4 节的粒侧无序度") in t),
    ("C5 拆括注", ("数值自检采用精确模长 $P = |z|$（保留全部非线性项），确认") in t
     and ("（数值自检采用") not in t),
    ("无 em-dash", t.count("\u2014") == 0),
    ("无坏字节", not [hex(ord(c)) for c in t
                      if (7 <= ord(c) <= 12 and ord(c) != 10) or ord(c) == 27]),
    ("无旧名与混排", t.count("复相干度亏损") == 0 and ("相位-only") not in t),
]
fails = [k for k, ok in checks if not ok]
for k, ok in checks:
    print(("PASS " if ok else "FAIL ") + k)
print("TOTAL:", "ALL PASS" if not fails else "FAIL: " + str(fails))
print("lines:", t.count(chr(10)) + 1)
