# -*- coding: utf-8 -*-
"""复审 C 级五项修复：双符号统一、引用样式、引号样式、嵌套括注"""
import io

p = "温度的波粒互补定义：复相干无序度作为热力学标尺.md"
t = io.open(p, encoding="utf-8").read()

# C3/C4: §2.3 已在 Edit 工具修复，此处校验
assert "p_{\\mathrm r}" not in t, "p_r symbol remains"
assert "1-\\eta \\approx \\frac{\\langle b^2\\rangle}" in t, "C3 not applied"

# C1: 纯文本 [n] -> 数学模式 $[n]$（仅六处既有纯文本）
plain = [("推广 [10]）", "推广 $[10]$）"),
         ("能量-熵关系上 [8]", "能量-熵关系上 $[8]$"),
         ("于奇点 [11]）", "于奇点 $[11]$）"),
         ("替代奇点 [13]", "替代奇点 $[13]$"),
         ("温度关系 [9] 的", "温度关系 $[9]$ 的"),
         ("暴胀子势能 [15]）", "暴胀子势能 $[15]$）")]
for old, new in plain:
    assert old in t, old
    t = t.replace(old, new, 1)

# C5: 双层嵌套括注拆句
old = ("弥散加大后映射弯折（数值自检采用精确模长 $P = |z|$（保留全部非线性项），"
       "$\\sigma = \\bar P$ 处 $p/w \\approx 0.45$；线性近似在此处失效），但单参数结构保持："
       "知道弥散，两个读数同时确定。")
new = ("弥散加大后映射弯折。数值自检采用精确模长 $P = |z|$（保留全部非线性项），"
       "确认 $\\sigma = \\bar P$ 处 $p/w \\approx 0.45$，线性近似在此处失效；"
       "但单参数结构保持：知道弥散，两个读数同时确定。")
assert old in t, "C5 anchor missing"
t = t.replace(old, new, 1)

# C2: 直引号转弯引号（行内逐对替换，全局交替易受既有弯引号干扰）
lines = t.split(chr(10))
n_pairs = 0
fixed_lines = []
for l in lines:
    if '"' in l:
        parts = l.split('"')
        # 行内必须偶数个直引号才能逐对配对
        assert (len(parts) - 1) % 2 == 0, "odd quotes in line: " + l[:60]
        rebuilt = parts[0]
        for k in range(1, len(parts), 2):
            rebuilt += "\u201c" + parts[k] + "\u201d" + parts[k + 1]
            n_pairs += 1
        l = rebuilt
    fixed_lines.append(l)
t = chr(10).join(fixed_lines)
assert t.count('"') == 0
assert t.count("\u201c") == t.count("\u201d")

# 落盘前自检
bad = [hex(ord(c)) for c in t if (7 <= ord(c) <= 12 and ord(c) != 10) or ord(c) == 27]
assert not bad and t.count("\u2014") == 0
io.open(p, "w", encoding="utf-8", newline="").write(t)
print("C1-C5 done; straight quotes converted:", n_pairs, "pairs")
