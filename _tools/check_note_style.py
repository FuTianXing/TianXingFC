# -*- coding: utf-8 -*-
"""临时工具：注记修改后铁律自检（em-dash、禁句式、"账"字、标题编号一致性）。"""
import sys, re
sys.stdout.reconfigure(encoding="utf-8")

PATH = r"D:\software\TianXingFC\内部草稿\注记-螺旋位错与元相子螺旋阶梯：同构、角色对调与三维位错分类的探索方向.md"
with open(PATH, encoding="utf-8") as f:
    text = f.read()
lines = text.splitlines()

problems = []
for i, line in enumerate(lines, 1):
    if "—" in line:
        problems.append(f"L{i} em-dash: {line[:60]}")
    for pat in ["不是……而是", "不是...而是", "而是", "并非……而是", "不在于……而在于"]:
        if pat in line:
            problems.append(f"L{i} 禁句式[{pat}]: {line[:60]}")
    if "账" in line:
        problems.append(f"L{i} '账'字: {line[:60]}")

print("=== 铁律检查 ===")
print("\n".join(problems) if problems else "通过：无 em-dash、无禁句式、无'账'字")

print("\n=== 章节结构 ===")
for i, line in enumerate(lines, 1):
    if line.startswith("#"):
        print(f"L{i}: {line}")

print("\n=== 残留旧编号引用 ===")
stale = [f"L{i}: {l[:70]}" for i, l in enumerate(lines, 1) if re.search(r"见 5\.3|见 4\.3\)|（见 四）", l)]
print("\n".join(stale) if stale else "无")
