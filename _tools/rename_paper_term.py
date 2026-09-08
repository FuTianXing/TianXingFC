# -*- coding: utf-8 -*-
"""论文温度载体术语定名：复相干度亏损/非相干份额 -> 复相干无序度（作者提案 2026-09-08）"""
import io, os

root = "."
paper_old = "温度的波粒互补定义：复相干度亏损作为热力学标尺.md"
paper_new = "温度的波粒互补定义：复相干无序度作为热力学标尺.md"
BS = chr(92)

# --- 论文 ---
t = io.open(paper_old, encoding="utf-8").read()
assert "复相干无序度" not in t
n_term = t.count("非相干份额")
t = t.replace("非相干份额", "复相干无序度")
# 摘要与结论的去重复（"复振幅场的复相干无序度" -> "场的复相干无序度"）
t = t.replace("本文将温度定义为该复振幅场的复相干无序度", "本文将温度定义为场的复相干无序度")
t = t.replace("温度是复振幅场的复相干无序度乘以基态振荡能量子", "温度是场的复相干无序度乘以基态振荡能量子")
# 标题内的旧名
t = t.replace("复相干度亏损作为热力学标尺", "复相干无序度作为热力学标尺")
# 定义处补一次白话解释（首次出现后）
anchor = "**温度就是场的复相干无序度乘以普朗克温度**。"
assert anchor in t
t = t.replace(anchor, anchor + "复相干无序度即场强的非相干份额：合唱之外的部分，波侧的方向散乱与粒侧的响度失谐都算在内。", 1)
# 落盘前控制字节自检
bad = [c for c in t if (7 <= ord(c) <= 12 and ord(c) != 10) or ord(c) == 27]
assert not bad, bad
assert t.count("\u2014") == 0
io.open(paper_old, "w", encoding="utf-8", newline="").write(t)
os.rename(paper_old, paper_new)
print("paper renamed; 非相干份额 replaced:", n_term)

# --- 子理论索引 ---
p = "理论总纲/子理论索引.md"
t = io.open(p, encoding="utf-8").read()
assert "复相干无序度" not in t
t = t.replace("复相干度亏损作为热力学标尺", "复相干无序度作为热力学标尺")
t = t.replace("（$D$ = 复相干度亏损 ", "（$D$ = 复相干无序度，即 $1 - |\\langle z\\rangle|/\\sqrt{\\langle|z|^2\\rangle}$；")
bad = [c for c in t if (7 <= ord(c) <= 12 and ord(c) != 10) or ord(c) == 27]
assert not bad, bad
io.open(p, "w", encoding="utf-8", newline="").write(t)
print("index updated")

# --- 扩展注记状态行补记定名 ---
p = "内部草稿/注记-温度定义的波粒互补扩展：双侧激发度与等相干平衡.md"
t = io.open(p, encoding="utf-8").read()
old = "温度主式取 v3 复相干度亏损载体 $k_BT = E_P D$"
if old in t:
    t = t.replace(old, "温度主式取 v3 复相干无序度载体 $k_BT = E_P D$（主式定名复相干无序度，2026-09-08）", 1)
    io.open(p, "w", encoding="utf-8", newline="").write(t)
    print("extension note status updated")
else:
    print("extension note: anchor not found, skip")
