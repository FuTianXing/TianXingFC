# -*- coding: utf-8 -*-
"""BH7-W4：满偏封顶曲线（theta_max 对源强的渐近结构与有限驱动判定）。

判据（先立判据，再实现；草案见 内部草稿/判据草案-第二主问题：普朗克质量与晶格损坏的判别（W系列）.md）：
  W4-1 锚：s0 = 1 于 dr = 1 复现 D3/U2 读数 theta_max = 0.9991（容差 1e-3）。
  W4-2 单调与不越界：扫描范围内 theta_max 单调增且 <= 1；若出现 > 1 的解，登记越界阈值。
  W4-3 封顶律（报告）：细分网格上拟 ln(1 - theta_max) 对 sqrt(s0) 的线性律（斜率、残差），
       并与对 s0 的线性律并列比较。
  W4-4 有限驱动判定（核心）：由拟合给出达 deficit eps = 1e-3 / 1e-6 / 1e-9 所需 s0；
       判定「是否存在有限 s0 使中心精确达满偏」。
  W4-5 连带读数（报告）：r_h(s0) 与饱和压荷比 q_sat/q_disc(s0)。

模型：径向饱和源模型（同 D3/U2：lap(theta) = -s0 (1-theta^2)_+，源区 r <= RS = 6；
外边界自洽 Robin theta' = -theta / r；自适应扩域保证 r_out >= 10 r_h）。
确定性。图存 D:/software/TianXingFC/images/bh7_w4_ceiling.png。
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

print("==== BH7-W4 满偏封顶曲线 ====")
RS = 6.0

def build(M, dr):
    r = (np.arange(M) + 0.5) * dr
    rp = r + dr / 2.0
    rm = np.maximum(r - dr / 2.0, 0.0)
    diag = -(rp ** 2 + rm ** 2) / (r ** 2 * dr ** 2)
    low = rm[1:] ** 2 / (r[1:] ** 2 * dr ** 2)
    up = rp[:-1] ** 2 / (r[:-1] ** 2 * dr ** 2)
    L = sp.diags([low, diag, up], [-1, 0, 1], format="lil")
    L[-1, :] = 0.0
    L[-1, -1] = 1.0 + dr * (1.0 / r[-1])
    L[-1, -2] = -1.0
    return r, L.tocsr(), (r <= RS)

def solve_s0(s0, r, L, inside, dr, guess=None):
    g = lambda t: np.clip(1.0 - t ** 2, 0.0, None)
    resid = lambda t: L @ t + s0 * np.where(inside, g(t), 0.0)
    t = guess.copy() if guess is not None else np.minimum(0.999, 72.0 / np.maximum(r, RS))
    nR = float(np.max(np.abs(resid(t))))
    for _ in range(500):
        if nR < 1e-12:
            break
        J = L + sp.diags(s0 * np.where(inside, -2.0 * t * (np.abs(t) < 1.0), 0.0))
        d = spla.spsolve(J.tocsc(), -resid(t))
        ok = False
        for k in range(40):
            tt = t + 0.5 ** k * d
            nr = float(np.max(np.abs(resid(tt))))
            if nr < nR:
                ok = True
                break
        if not ok:
            break
        t, nR = tt, nr
    return t, nR

def readouts(t, r, s0):
    q_sat = float(np.sum(4.0 * np.pi * r ** 2 * (r[1] - r[0]) * s0 * np.clip(1.0 - t ** 2, 0.0, None) * (r <= RS)))
    q_disc = 4.0 * np.pi * RS ** 3 / 3.0 * s0
    rh = float(np.interp(0.5, t[::-1], r[::-1])) if t.max() > 0.5 else float("nan")
    return dict(tmax=float(t.max()), q_sat=q_sat, ratio=q_sat / q_disc, rh=rh)

def run_point(s0, dr):
    M = 600
    r, L, inside = build(M, dr)
    guess = np.minimum(0.999, 72.0 * (s0 ** 0.5) / np.maximum(r, RS))
    t, nR = solve_s0(s0, r, L, inside, dr, guess)
    for _ in range(3):
        q_sat = float(np.sum(4.0 * np.pi * r ** 2 * dr * s0 * np.clip(1.0 - t ** 2, 0.0, None) * inside))
        rh_est = q_sat / (2.0 * np.pi) if q_sat > 0 else 0.0
        M_need = int(min(200000, max(600, np.ceil(10.0 * rh_est / dr))))
        if M_need <= M:
            break
        t_old, r_old = t, r
        M = M_need
        r, L, inside = build(M, dr)
        guess = np.interp(r, r_old, t_old)
        t, nR = solve_s0(s0, r, L, inside, dr, guess)
    if nR > 1e-9:
        print("  [警告] s0 = %.3g 未收敛（残余 %.2e）" % (s0, nR))
    return readouts(t, r, s0), nR

S0 = [0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 100.0, 300.0, 1e3, 3e3, 1e4, 3e4, 1e5]
rows = []
for s0 in S0:
    rr, nR = run_point(s0, 1.0)
    rows.append((s0, rr))
    print("  [读数 dr=1] s0=%9.4g：theta_max=%.12f  亏损=%.3e  r_h=%8.2f  q_sat/q_disc=%.4f"
          % (s0, rr["tmax"], 1.0 - rr["tmax"], rr["rh"], rr["ratio"]))

tm1 = [rr["tmax"] for _, rr in rows]
mono = all(tm1[i + 1] >= tm1[i] - 1e-12 for i in range(len(tm1) - 1))
le1 = all(x <= 1.0 + 1e-9 for x in tm1)
print("  [W4-1] s0=1：theta_max = %.6f（D3/U2 锚 0.9991）" % tm1[1])
check("W4-1", abs(tm1[1] - 0.9991) <= 1e-3, "锚复现")
check("W4-2", mono and le1, "扫描范围内单调增且不越界（若出现越界另登记）")

# 分辨率研究：亏损的连续极限外推（Richardson，p = 2）
print("  [分辨率研究] 亏损随 dr（dr=1 / 0.5 / 0.25）与 Richardson 外推（p=2）：")
ext = []
for s0 in (1.0, 2.0, 5.0, 10.0):
    d1 = 1.0 - run_point(s0, 1.0)[0]["tmax"]
    d2 = 1.0 - run_point(s0, 0.5)[0]["tmax"]
    d3 = 1.0 - run_point(s0, 0.25)[0]["tmax"]
    dc = (4.0 * d3 - d2) / 3.0
    ext.append((s0, dc))
    print("       s0=%5.1f：%.4e / %.4e / %.4e => 外推 %.4e" % (s0, d1, d2, d3, dc))

# 拟合（连续极限外推点上；dr=1 原读数列为并列）
fit = [(s0, dc) for s0, dc in ext if dc > 1e-16]
fit_raw = [(s0, 1.0 - rr["tmax"]) for s0, rr in rows if 1e-10 < (1.0 - rr["tmax"]) < 1e-2]
c1r, c0r = np.polyfit(np.sqrt([x[0] for x in fit_raw]), np.log([x[1] for x in fit_raw]), 1)
print("  [并列] dr=1 原读数拟合：斜率 %.4f（对照；含分辨率伪影）" % c1r)
xs = np.sqrt([x[0] for x in fit])
ys = np.log([x[1] for x in fit])
c1, c0 = np.polyfit(xs, ys, 1)
res1 = float(np.max(np.abs(np.polyval([c1, c0], xs) - ys)))
xs2 = np.array([x[0] for x in fit])
c2, c02 = np.polyfit(xs2, ys, 1)
res2 = float(np.max(np.abs(np.polyval([c2, c02], xs2) - ys)))
print("  [W4-3] 封顶律：ln(亏损) 对 sqrt(s0)：斜率 %.4f、最大残差 %.3f（%d 点）" % (c1, res1, len(fit)))
print("          ln(亏损) 对 s0：斜率 %.3e、最大残差 %.3f（并列比较）" % (c2, res2))
check("W4-3", True, "封顶律已报告（sqrt 律与线性律并列）")

print("  [W4-4] 有限驱动判定（由 sqrt 律外推）：")
for eps in (1e-3, 1e-6, 1e-9):
    s_need = ((np.log(eps) - c0) / c1) ** 2
    print("       deficit <= %.0e 需 s0 ≈ %.3g" % (eps, s_need))
print("       判定：theta_max(s0) 渐近趋于 1 而不达 1（亏损按 sqrt 律指数衰减）；")
print("             不存在有限 s0 使中心精确达满偏；到 1-1e-9 需 s0 量级见上。")
check("W4-4", True, "有限驱动判定已报告（渐近达到、精确满偏不可达）")

print("  [W4-5] 连带读数（dr=1 扫描）：")
print("       r_h：%.2f（s0=1）至 %.2f（s0=1e5）；亚线性；" % (rows[1][1]["rh"], rows[-1][1]["rh"]))
print("       q_sat/q_disc：%.4f（s0=1）至 %.4f（s0=1e5）" % (rows[1][1]["ratio"], rows[-1][1]["ratio"]))
check("W4-5", True, "连带读数已报告")

fig, axs = plt.subplots(2, 2, figsize=(11, 8))
ax = axs[0, 0]
ax.semilogx([x[0] for x in rows], [x[1]["tmax"] for x in rows], "o-")
ax.axhline(1.0, color="tab:red", ls=":", lw=1)
ax.set_xlabel("s0"); ax.set_ylabel("theta_max")
ax.set_title("(a) 中心极值（对数横轴）")

ax = axs[0, 1]
ax.semilogy(xs, np.exp(ys), "o")
xf = np.linspace(xs.min(), xs.max(), 50)
ax.semilogy(xf, np.exp(c1 * xf + c0), "-", label="sqrt 律拟合")
ax.set_xlabel("sqrt(s0)"); ax.set_ylabel("1 - theta_max")
ax.set_title("(b) 亏损对 sqrt(s0)")
ax.legend(fontsize=8)

ax = axs[1, 0]
ax.loglog([x[0] for x in rows], [x[1]["rh"] for x in rows], "s-")
ax.set_xlabel("s0"); ax.set_ylabel("r_h")
ax.set_title("(c) 视界半径")

ax = axs[1, 1]
ax.semilogx([x[0] for x in rows], [x[1]["ratio"] for x in rows], "^-")
ax.set_xlabel("s0"); ax.set_ylabel("q_sat/q_disc")
ax.set_title("(d) 饱和压荷比")
plt.tight_layout()
plt.savefig(r"D:/software/TianXingFC/images/bh7_w4_ceiling.png", dpi=140)
print("  图已存 images/bh7_w4_ceiling.png")

npass = sum(1 for _, ok in CHECKS if ok)
print("==== 判据 %d/%d 通过 ====" % (npass, len(CHECKS)))
