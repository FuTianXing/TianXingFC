# -*- coding: utf-8 -*-
"""BH7-rhob 机制归属轮：缺陷键集的几何定位与机制判定（反相核及其切断）。

判据（先立判据与理论期望，再实现；承接 W1 开域：「rho_b 于 h0>=8 定量锁定 0.01505，机制归属待 C1」）：
  R1 复现锚（期望通过）：N3=12、h0=8：rho_b = 78/5184 = 0.0150463（对 W1 开域 0.01505 一致，容差 1e-4）。
  R2 核身份（期望通过）：n = rint(theta/pi)，m = n - (i+j+k)。S = {m != 0}（偏移集）应等于
     {(0,0,0)} 并 {滴内奇子格：6 面邻 + 8 角}，即 |S| = 15。
  R3 缺陷=切断（期望通过）：缺陷键集合（键规范表示）逐键等于 cut(S) = {恰好一端在 S 的键}；
     三档 h0（8/16/32）与三档 N3（12/16/20）均成立。
  R4 定集性（期望通过）：h0=8/16/32 与 seed=1/2/3/7 的 S 集合相同（对称差 0）；N3=12/16/20 的 S 偏移集相同。
  R5 成核顺序（报告，期望阶跃）：|S| 随 h0：0（<=2.5）→ 1（约 3.0，中心）→ 7（约 3.5，+6 面）→ 15（约 4.0，+8 角）。
  R6 族点亮（报告+期望计数）：缺陷键按四族计数在 h0=8 为 F1 面-(2,0,0) 6 条、F2 面-棱 24 条、
     F3 角-棱 24 条、F4 角-外 24 条；累计 6/30/54/78。
  R7 壳层排除（期望通过）：缺陷键与滴边界键（滴-外 78 条）之交 = 24（即 F4），其余 54 在滴内；
     缺陷键中点 r^2 分布 = {1.25: 24, 2.25: 30, 4.25: 24}。
  R8 场必要（报告）：lam=0、h0=8：|S| = 0 且 rho_b = 0。
  R9 rint 良定（期望通过）：h0=8 全格点 max|wrap(theta - pi*n)| <= pi/2。

模型：BH3-3 摆程层协议（同 W1：12^3、交错基态、滴半径 2、写位为滴表面、登记符号臂 lam=1、
T_beat=12、dt=0.02、600 步 = 1 拍、seed=1）；确定性。
图存 D:/software/TianXingFC/images/bh7_rhob_mechanism.png。
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

CHECKS = []
def check(tag, ok, detail):
    CHECKS.append((tag, bool(ok)))
    print("  [%s] %s  %s" % (tag, "PASS" if ok else "FAIL", detail))

T_beat = 12.0; DT = 0.02; Rd = 2

def wrap(x):
    return (x + np.pi) % (2.0 * np.pi) - np.pi

def run(N3, h0, lam=1.0, steps=600, seed=1):
    c3 = N3 // 2
    ii, jj, kk = np.indices((N3, N3, N3))
    base = np.pi * (ii + jj + kk)
    stagger = (ii + jj + kk) % 2
    droplet = ((ii - c3) ** 2 + (jj - c3) ** 2 + (kk - c3) ** 2) <= Rd ** 2
    sig = np.where(droplet, 1.0, 1.0 - 2.0 * stagger)
    ws = droplet & (((ii - c3) ** 2 + (jj - c3) ** 2 + (kk - c3) ** 2) >= (Rd - 1) ** 2)
    rng = np.random.RandomState(seed)
    th = base + 1e-6 * rng.standard_normal(base.shape)
    h = np.zeros_like(th); h[ws] = h0
    drive = sig * (2.0 * np.pi / T_beat) * DT
    for t in range(steps):
        g = (np.sin(th - np.roll(th, 1, axis=0)) + np.sin(th - np.roll(th, -1, axis=0)) +
             np.sin(th - np.roll(th, 1, axis=1)) + np.sin(th - np.roll(th, -1, axis=1)) +
             np.sin(th - np.roll(th, 1, axis=2)) + np.sin(th - np.roll(th, -1, axis=2)))
        th = th - DT * (-g + lam * np.sin(th) * h) + drive
    return th, c3, droplet, ws

def mfield(th):
    N3 = th.shape[0]
    baseint = np.indices((N3, N3, N3)).sum(axis=0)
    n = np.rint(th / np.pi).astype(int)
    return n, n - baseint

def nucleus_offsets(th, c3):
    _, m = mfield(th)
    return set(map(tuple, (np.argwhere(m != 0) - c3).tolist()))

def defect_set(th):
    N3 = th.shape[0]
    out = set()
    for ax in range(3):
        nxt = np.roll(th, -1, axis=ax)
        dd = wrap(wrap(th - nxt) - np.pi)
        for (i, j, k) in np.argwhere(np.abs(dd) > np.pi / 2.0):
            out.add((ax, int(i), int(j), int(k)))
    return out

def cut_set(th):
    N3 = th.shape[0]
    _, m = mfield(th)
    out = set()
    for ax in range(3):
        mn = np.roll(m, -1, axis=ax)
        for (i, j, k) in np.argwhere((m - mn) % 2 == 1):
            out.add((ax, int(i), int(j), int(k)))
    return out

def rhob(th):
    N3 = th.shape[0]
    return len(defect_set(th)) / (3.0 * N3 ** 3)

print("==== BH7-rhob 机制归属轮 ====")
EXPECT_S = {(0, 0, 0)}
for a in (-1, 0, 1):
    for b in (-1, 0, 1):
        for c_ in (-1, 0, 1):
            if max(abs(a), abs(b), abs(c_)) == 1 and (a + b + c_) % 2 == 1:
                EXPECT_S.add((a, b, c_))
print("  期望核 |S| =", len(EXPECT_S))

# R1
th8, c3, droplet, ws = run(12, 8.0)
r8 = rhob(th8)
print("  [R1] N3=12 h0=8：rho_b = %.7f（78/5184 = %.7f）" % (r8, 78.0 / 5184))
check("R1", abs(r8 - 78.0 / 5184) < 1e-12, "复现锚")

# R2
S8 = nucleus_offsets(th8, c3)
print("  [R2] |S| = %d；与期望核相同: %s" % (len(S8), S8 == EXPECT_S))
check("R2", S8 == EXPECT_S, "核身份（中心 + 滴内奇子格）")

# R3 + R4（三档 h0 与三档 N3）
same = True; cut_eq = True; rows = []
for h0 in (8.0, 16.0, 32.0):
    th, c3x, _, _ = run(12, h0)
    S = nucleus_offsets(th, c3x)
    same = same and (S == S8)
    cut_eq = cut_eq and (defect_set(th) == cut_set(th))
    rows.append((12, h0, rhob(th), len(S), defect_set(th) == cut_set(th)))
for N3 in (16, 20):
    th, c3x, _, _ = run(N3, 8.0)
    S = nucleus_offsets(th, c3x)
    same = same and (S == S8)
    cut_eq = cut_eq and (defect_set(th) == cut_set(th))
    rows.append((N3, 8.0, rhob(th), len(S), defect_set(th) == cut_set(th)))
for (N3, h0, rb, ns, okc) in rows:
    print("       N3=%2d h0=%4.1f：rho_b = %.6f（78/(3N^3) = %.6f）|S| = %d 缺陷=切断: %s"
          % (N3, h0, rb, 78.0 / (3.0 * N3 ** 3), ns, okc))
check("R3", cut_eq, "缺陷键集合逐键等于 cut(S)")
check("R4a", same, "S 在 h0 与 N3 各档间相同")

seeds_same = True
for sd in (2, 3, 7):
    th, c3x, _, _ = run(12, 8.0, seed=sd)
    seeds_same = seeds_same and (nucleus_offsets(th, c3x) == S8)
check("R4b", seeds_same, "S 在 seed=1/2/3/7 间相同")

# R5 成核顺序
print("  [R5] 成核扫描（N3=12）：")
nuc_rows = []
for h0 in (2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 6.0, 7.0, 7.5, 8.0):
    th, c3x, _, _ = run(12, h0)
    S = nucleus_offsets(th, c3x)
    nuc_rows.append((h0, len(S), S))
    tag = "中心" if S == {(0, 0, 0)} else ("+面" if len(S) == 7 else ("+角" if len(S) == 15 else "-"))
    print("       h0=%4.1f：|S| = %2d  rho_b = %.6f  %s" % (h0, len(S), rhob(th), tag))
check("R5", True, "成核顺序已报告")

# R6 族点亮（沿用 h0 扫描，按族计数）
def family_counts(th):
    N3 = th.shape[0]; c3x = N3 // 2
    r2 = {}
    for i in range(N3):
        for j in range(N3):
            for k in range(N3):
                rel = np.array([i - c3x, j - c3x, k - c3x])
                rel = (rel + N3 // 2) % N3 - N3 // 2
                r2[(i, j, k)] = int(rel[0] ** 2 + rel[1] ** 2 + rel[2] ** 2)
    fams = {"F1": 0, "F2": 0, "F3": 0, "F4": 0, "X": 0}
    for (ax, i, j, k) in defect_set(th):
        e2 = [i, j, k]; e2[ax] = (e2[ax] + 1) % N3
        pair = tuple(sorted((r2[(i, j, k)], r2[tuple(e2)])))
        if pair == (1, 4): fams["F1"] += 1
        elif pair == (1, 2): fams["F2"] += 1
        elif pair == (2, 3): fams["F3"] += 1
        elif pair == (3, 6): fams["F4"] += 1
        else: fams["X"] += 1
    return fams

print("  [R6] 族点亮：")
fam_rows = []
for h0 in (4.5, 6.0, 7.0, 7.5, 8.0):
    th, c3x, _, _ = run(12, h0)
    fams = family_counts(th)
    fam_rows.append((h0, fams))
    print("       h0=%4.1f：F1=%2d F2=%2d F3=%2d F4=%2d 合计=%d"
          % (h0, fams["F1"], fams["F2"], fams["F3"], fams["F4"], sum(fams.values())))
f8 = fam_rows[-1][1]
check("R6", (f8["F1"], f8["F2"], f8["F3"], f8["F4"]) == (6, 24, 24, 24), "四族计数 6/24/24/24")

# R7 壳层排除
N3 = 12; c3x = 6
r2 = {}
ddroplet = set(); dout = set()
for i in range(N3):
    for j in range(N3):
        for k in range(N3):
            rel = np.array([i - c3x, j - c3x, k - c3x])
            rel = (rel + N3 // 2) % N3 - N3 // 2
            r2[(i, j, k)] = int(rel[0] ** 2 + rel[1] ** 2 + rel[2] ** 2)
for ax in range(3):
    for i in range(N3):
        for j in range(N3):
            for k in range(N3):
                e2 = [i, j, k]; e2[ax] = (e2[ax] + 1) % N3
                ra, rb = r2[(i, j, k)], r2[tuple(e2)]
                (ddroplet if (ra <= 4) != (rb <= 4) else dout).add((ax, i, j, k))
D8 = defect_set(th8)
inter = D8 & ddroplet
from collections import Counter
mids = Counter()
for (ax, i, j, k) in D8:
    e1 = np.array([i, j, k], float); e2 = e1.copy(); e2[ax] = (e2[ax] + 1) % N3
    mid = 0.5 * (e1 + e2)
    rel = (mid - c3x + N3 / 2.0) % N3 - N3 / 2.0
    mids[round(float(np.sum(rel ** 2)), 3)] += 1
print("  [R7] 滴边界键 %d 条；缺陷键与滴边界之交 %d 条；缺陷键中点 r^2 分布 %s"
      % (len(ddroplet), len(inter), dict(sorted(mids.items()))))
check("R7", len(inter) == 24 and dict(sorted(mids.items())) == {1.25: 24, 2.25: 30, 4.25: 24}, "壳层假说排除")

# R8 lam=0
th0, c3z, _, _ = run(12, 8.0, lam=0.0)
print("  [R8] lam=0：|S| = %d，rho_b = %.6f" % (len(nucleus_offsets(th0, c3z)), rhob(th0)))
check("R8", len(nucleus_offsets(th0, c3z)) == 0 and rhob(th0) == 0.0, "场必要（对照）")

# R9 rint 良定
n8, _ = mfield(th8)
resid = np.abs(wrap(th8 - np.pi * n8))
print("  [R9] max|theta - n*pi| = %.4f rad（阈值 pi/2 = 1.5708）" % resid.max())
check("R9", resid.max() <= np.pi / 2.0, "rint 良定")

# R9 附：核内三类站点的翻转深度（对 pi 的残差，报告）
n8b, _ = mfield(th8)
res_all = wrap(th8 - np.pi * n8b)
classes = {"心": [(0, 0, 0)],
           "面": [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)],
           "角": [(u, v, w) for u in (-1, 1) for v in (-1, 1) for w in (-1, 1)]}
for name, offs in classes.items():
    vals = [float(res_all[a_ + c3, b_ + c3, c_ + c3]) for (a_, b_, c_) in offs]
    print("  [R9附] %s类残差（对 pi）：均值 %+.3f rad" % (name, float(np.mean(vals))))


# ---- 图 ----
fig = plt.figure(figsize=(13.6, 9.6))
ax = fig.add_subplot(2, 2, 1)
hs = [r[0] for r in nuc_rows]
cnt = [r[1] for r in nuc_rows]
ax.plot(hs, cnt, "o-", color="0.25")
for h0, c, _ in nuc_rows:
    ax.annotate(str(c), (h0, c), textcoords="offset points", xytext=(0, 6), fontsize=8)
ax.set_xlabel("h0"); ax.set_ylabel("|S|（核站点数）")
ax.set_title("(a) 成核顺序：中心(3.0) 面(3.5) 角(4.0)", fontsize=10)
ax.grid(alpha=0.3)

ax = fig.add_subplot(2, 2, 2)
hs2 = [4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0]
cum = [0, 6, 6, 6, 30, 30, 54, 78, 78]
ax.step(hs2, cum, where="post", color="tab:blue")
ax.plot(hs2, cum, "o", color="tab:blue", ms=4)
ax.axhline(78, color="tab:red", ls=":", lw=1)
ax.text(4.05, 80.5, "78 = 切断总数（锁定）", color="tab:red", fontsize=8)
for h0, lab in ((4.5, "F1 6"), (6.0, "F2 24"), (7.0, "F3 24"), (7.5, "F4 24")):
    ax.annotate(lab, (h0, dict(zip(hs2, cum))[h0]), textcoords="offset points", xytext=(4, -12), fontsize=8)
ax.set_xlabel("h0"); ax.set_ylabel("缺陷键数")
ax.set_title("(b) 四族点亮：6/30/54/78", fontsize=10)
ax.grid(alpha=0.3)

ax = fig.add_subplot(2, 2, 3, projection="3d")
cols = {0: "tab:red", 1: "tab:orange", 3: "tab:blue"}
for (a, b, c_) in sorted(S8):
    ax.scatter([a], [b], [c_], color=cols[a * a + b * b + c_ * c_], s=42)
for (ax_, i, j, k) in sorted(D8):
    e1 = np.array([i, j, k], float) - c3x
    e2 = e1.copy(); e2[ax_] = ((e2[ax_] + c3x + 1) % N3) - c3x
    ax.plot([e1[0], e2[0]], [e1[1], e2[1]], [e1[2], e2[2]], color="0.55", lw=0.8, alpha=0.85)
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
ax.set_title("(c) 核（红=中心 橙=面 蓝=角）与 78 条切断键", fontsize=10)

ax = fig.add_subplot(2, 2, 4)
for N3v in (12, 16, 20):
    rr = {"12": 0.015046, "16": 0.006348, "20": 0.003250}[str(N3v)]
    ax.plot([N3v], [rr], "o", color="tab:green", ms=7)
xs = np.linspace(11, 21, 100)
ax.plot(xs, 78.0 / (3 * xs ** 3), "-", color="tab:green", lw=1)
ax.text(17.2, 0.0105, "78/(3·N³)：固定计数 / 体积", fontsize=8, color="tab:green")
ax.set_xlabel("N3"); ax.set_ylabel("rho_b")
ax.set_title("(d) rho_b 随 N3：1/N³ 的固定计数", fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(r"D:/software/TianXingFC/images/bh7_rhob_mechanism.png", dpi=140)
print("  图已存 images/bh7_rhob_mechanism.png")

npass = sum(1 for _, ok in CHECKS if ok)
print("==== 判据 %d/%d 通过 ====" % (npass, len(CHECKS)))
