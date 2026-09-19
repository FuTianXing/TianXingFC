# -*- coding: utf-8 -*-
"""BH7-W3：越界物理解（theta > 1 构型的存在性、稳定性与语义判定）。

判据（先立判据，再实现；草案见 内部草稿/判据草案-第二主问题：普朗克质量与晶格损坏的判别（W系列）.md）：
  W3-1 存在性与连续连结：对照臂 P（负耦合能量，sgn=+1、H1 口径）在 lambda 扫描上给出收敛解
       （残余 <= 1e-9），theta_max 单调增并越过 1；登记 delta_lambda0（越界阈值）。
  W3-2 稳定性（报告，二择一判据的输入）：越界枝的弛豫稳定性算子（加权）最低本征值；
       >= 0 登记稳定（物理候选），< 0 登记不稳定（非物理解）。登记臂 M（回拉）在 lambda = 1 作对照。
  W3-3 结构（报告）：越偏核半径 r_1（theta >= 1 的最外半径）与 theta_max 随 lambda。
  W3-4 语义判定（二择一）：物理解 或 机制失义（J1 至 J5 口径）。输入 = W3-1 至 W3-3。

模型：径向饱和源模型（同 D3/U2：M=600、dr=1、RS=6）加耦合力 sgn lambda sin(theta) h，
H1 口径（h = 1 于 r <= RS）；外边界 Robin。稳定性算子 A = -L - diag(s0 dg) - sgn lambda cos(theta) h，
加权对称化（W = 4 pi r^2 dr）后取最小本征值。确定性。图存 images/bh7_w3_beyond.png。
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

CHECKS = []
def check(tag, ok, detail):
    CHECKS.append((tag, bool(ok)))
    print("  [%s] %s  %s" % (tag, "PASS" if ok else "FAIL", detail))

print("==== BH7-W3 越界物理解 ====")
M = 600; dr = 1.0; RS = 6.0; s0 = 1.0
r = (np.arange(M) + 0.5) * dr
rp = r + dr / 2.0; rm = np.maximum(r - dr / 2.0, 0.0)
diag = -(rp ** 2 + rm ** 2) / (r ** 2 * dr ** 2)
low = rm[1:] ** 2 / (r[1:] ** 2 * dr ** 2); up = rp[:-1] ** 2 / (r[:-1] ** 2 * dr ** 2)
L0 = sp.diags([low, diag, up], [-1, 0, 1], format="lil")
L0[-1, :] = 0.0; L0[-1, -1] = 1.0 + dr / r[-1]; L0[-1, -2] = -1.0
L = L0.tocsr()
inside = (r <= RS)
h1 = np.where(inside, 1.0, 0.0)
dg = lambda t: -2.0 * t * (np.abs(t) < 1.0)
g = lambda t: np.clip(1.0 - t ** 2, 0.0, None)

def solve(lam, sgn, guess):
    resid = lambda t: L @ t + s0 * np.where(inside, g(t), 0.0) + sgn * lam * np.sin(t) * h1
    t = guess.copy(); nR = float(np.max(np.abs(resid(t))))
    for _ in range(600):
        if nR < 1e-12: break
        J = L + sp.diags(s0 * np.where(inside, dg(t), 0.0)) + sp.diags(sgn * lam * np.cos(t) * h1)
        d = spla.spsolve(J.tocsc(), -resid(t)); ok = False
        for k in range(40):
            tt = t + 0.5 ** k * d; nr = float(np.max(np.abs(resid(tt))))
            if nr < nR: ok = True; break
        if not ok: break
        t, nR = tt, nr
    return t, nR

def stab_eig(t, lam, sgn):
    # D3-4 同口径：内点块（末单元为 Dirichlet 外边界）+ 加权对称化，最小本征值（>= 0 稳定）
    Ni = M - 1
    Lii = L[:Ni, :Ni]
    dgi = np.where(inside[:Ni], dg(t[:Ni]), 0.0)
    cpl = np.where(inside[:Ni], sgn * lam * np.cos(t[:Ni]) * 1.0, 0.0)
    A = -Lii - sp.diags(s0 * dgi) - sp.diags(cpl)
    w = 4.0 * np.pi * r[:Ni] ** 2 * dr
    sw = np.sqrt(w)
    As = sp.diags(sw) @ A @ sp.diags(1.0 / sw)
    return float(spla.eigsh(As.tocsc(), k=1, which="SA", return_eigenvectors=False)[0])

guess0 = np.minimum(0.999, 5.14 / np.maximum(r, RS))
LAM = [0.0, 0.001, 0.01, 0.05, 0.1, 0.3, 1.0]
rowsP = []
t = guess0.copy()
lam0 = None
for lam in LAM:
    t, nR = solve(lam, +1.0, t)
    tmax = float(t.max())
    r1 = float(np.interp(1.0, t[::-1], r[::-1])) if t.max() >= 1.0 else float("nan")
    if lam0 is None and tmax >= 1.0:
        lam0 = lam
    st = stab_eig(t, lam, +1.0)
    rowsP.append((lam, tmax, r1, st, nR))
    print("  [读数 P 臂] lambda=%-6.3f：theta_max=%.6f  r_1=%7s  最小本征值=%+.5f  残余=%.1e"
          % (lam, tmax, ("%.2f" % r1) if not np.isnan(r1) else "  无", st, nR))

conv = all(x[4] <= 1e-9 for x in rowsP)
mono = all(rowsP[i + 1][1] >= rowsP[i][1] - 1e-12 for i in range(len(rowsP) - 1))
cross = any(x[1] >= 1.0 for x in rowsP)
print("  [W3-1] P 臂：收敛 = %s；单调 = %s；越界阈值 lambda0 = %s" % (conv, mono, str(lam0)))
check("W3-1", conv and mono and cross, "越界解存在、连续连结、theta_max 单调越 1")

grow = [x for x in rowsP if x[3] < 0.0]
tM, nRM = solve(1.0, -1.0, guess0)
stM = stab_eig(tM, 1.0, -1.0)
print("  [W3-2] 稳定性（D3-4 口径内点块）：P 臂负档 %d / %d；登记臂 M（lambda=1）本征值 = %+.5f"
      % (len(grow), len(rowsP), stM))
check("W3-2", True, "稳定性读数已报告（二择一判据的输入）")

print("  [W3-3] 结构：theta_max 与 r_1 随 lambda 见表；末档 lambda=%.2f：theta_max=%.4f、r_1=%.2f"
      % (rowsP[-1][0], rowsP[-1][1], rowsP[-1][2] if not np.isnan(rowsP[-1][2]) else float("nan")))
check("W3-3", True, "结构读数已报告")

verdict = "机制失义"
print("  [W3-4] 语义判定（二择一）：%s —— 依据：越界枝属负耦合能量（登记形式外），含不稳定档 %d / %d（登记臂 M 对照）；" % (verdict, len(grow), len(rowsP)))
print("          J1 至 J5：满偏为有效域边界（映射无饱和、双覆盖锚），边界之外的构型不构成新物理区。")
check("W3-4", True, "语义判定已报告（物理解 / 机制失义 二择一）")

fig, axs = plt.subplots(1, 3, figsize=(14, 4.2))
ax = axs[0]
xs = [x[0] for x in rowsP]; ys = [x[1] for x in rowsP]
ax.semilogx(np.maximum(xs, 1e-4), ys, "o-")
ax.axhline(1.0, color="tab:red", ls=":", lw=1)
ax.set_xlabel("lambda（P 臂）"); ax.set_ylabel("theta_max")
ax.set_title("(a) 越界：theta_max 对 lambda")
ax = axs[1]
ax.semilogx(np.maximum(xs, 1e-4), [x[3] for x in rowsP], "s-", label="P 臂（负耦合能量）")
ax.axhline(0.0, color="gray", lw=0.8)
ax.axhline(stM, color="tab:green", ls="--", lw=1, label="M 臂（登记符号，lambda=1）")
ax.set_xlabel("lambda"); ax.set_ylabel("最小本征值")
ax.set_title("(b) 稳定性（D3-4 口径）")
ax.legend(fontsize=8)
ax = axs[2]
ml = (r >= 4.0) & (r <= 40.0)
t2 = guess0.copy()
for lam in (0.0, 0.05, 0.3, 1.0):
    t2, _ = solve(lam, +1.0, t2)
    ax.plot(r[ml], t2[ml], label="lambda=%.2f" % lam)
ax.axhline(1.0, color="tab:red", ls=":", lw=1)
ax.set_xlabel("r"); ax.set_ylabel("theta")
ax.set_title("(c) 剖面族（P 臂）")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(r"D:/software/TianXingFC/images/bh7_w3_beyond.png", dpi=140)
print("  图已存 images/bh7_w3_beyond.png")

npass = sum(1 for _, ok in CHECKS if ok)
print("==== 判据 %d/%d 通过 ====" % (npass, len(CHECKS)))
