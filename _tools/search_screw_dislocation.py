# -*- coding: utf-8 -*-
"""临时工具：检索"螺旋位错/螺型位错/刃位错"等材料物理位错术语在工作区中的出现位置。"""
import os, sys

io_stdout_enc = 'utf-8'
sys.stdout.reconfigure(encoding=io_stdout_enc)

ROOT = r"D:\software\TianXingFC"
TERMS = ["螺旋位错", "螺型位错", "刃位错", "刃型位错", "screw dislocation", "edge dislocation",
         "Burgers", "伯格斯", "沛尔斯", "Peierls", "滑移", "攀移"]

hits = {}
for dirpath, dirnames, filenames in os.walk(ROOT):
    if ".git" in dirpath:
        continue
    for fn in filenames:
        if not fn.endswith(".md"):
            continue
        p = os.path.join(dirpath, fn)
        try:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        except OSError:
            continue
        for t in TERMS:
            if t in text:
                hits.setdefault(t, []).append(os.path.relpath(p, ROOT))

for t in TERMS:
    files = hits.get(t, [])
    if files:
        print(f"[{t}] {len(files)} 个文件:")
        for rel in files[:6]:
            print("   ", rel)

# 重点：螺旋阶梯几何基准文件中"螺旋"与"位错"是否同段出现
target = os.path.join(ROOT, "理论总纲", "螺旋阶梯几何.md")
if os.path.exists(target):
    with open(target, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    print("\n=== 理论总纲/螺旋阶梯几何.md 中同时含'螺旋'与'位错'的行 ===")
    n = 0
    for i, line in enumerate(lines, 1):
        if "位错" in line:
            n += 1
            print(f"L{i}: {line[:120].strip()}")
    if n == 0:
        print("(该文件无'位错'字样)")
    # 螺旋阶梯几何的来源说明
    print("\n=== 文件头 30 行 ===")
    for line in lines[:30]:
        print(line.rstrip()[:100])
else:
    print("\n未找到 理论总纲/螺旋阶梯几何.md")
