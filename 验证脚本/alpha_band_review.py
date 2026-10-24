# -*- coding: utf-8 -*-
"""靶的重审：alpha 带的输入灵敏度与 1/2 通道的判定（第 433 条，出口其三）。

判据：
  R1 基准带复核（与第 413 条一致）：p = 3/2（辐射期稀释）、f = 2.12e-105、w = 3.80e21
  R2 输入灵敏度：f、w_def、t_eq、z_eq 各乘 1/2 与 2，看带端移动
  R3 稀释指数变体：p in {0.5, 1, 3/2, 2, 3}（kination、辐射、物质），看带与「0.5 是否可行」
  R4 结构性核：alpha = 1/2 的相变时刻 t*(1/2) 与宇宙年龄的比较；g(beta) 的最小值

模型（全部闭式）：
  曲线：xi_c(t) = C (t/t_eq)^beta，C = (z_eq^-3/f)^(1/3)，beta = p/3
  生长律：xi = t^alpha
  交点：t* = exp(B/(alpha-beta))，B = ln C - beta ln t_eq
  带端：alpha_min = beta + B/ln t_age（t*=t_age）
        alpha_max = beta ln w / (ln w - B)（xi*=w）
  0.5 可行性（beta < 1/2）：t*(1/2) <= t_age  <==>  g(beta) := B/(1/2-beta) <= ln t_age

运行：python 验证脚本/alpha_band_review.py
"""
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG = os.path.join(ROOT, "images", "alpha_band_review.png")

F0 = 2.12e-105
T_EQ0 = 2.745e55
Z_EQ0 = 3400.0
W0 = 3.80e21
T_AGE0 = 4.35e17 / 5.391247e-44


def band(p=1.5, f=F0, w=W0, t_eq=T_EQ0, z=Z_EQ0, t_age=T_AGE0):
    C = (z ** -3.0 / f) ** (1.0 / 3.0)
    beta = p / 3.0
    lnC = math.log(C); lneq = math.log(t_eq); lna = math.log(t_age); lnw = math.log(w)
    B = lnC - beta * lneq
    out = {"C": C, "beta": beta, "B": B, "ln_age": lna}
    ok = (B > 0.0)
    if ok:
        out["a_min"] = beta + B / lna
        if lnw > B:
            out["a_max"] = beta * lnw / (lnw - B)
            ok = out["a_min"] <= out["a_max"]
        else:
            out["a_max"] = float("inf")
    out["band_ok"] = ok
    if beta < 0.5:
        g = B / (0.5 - beta)
        out["half_ok"] = bool(ok) and (g <= lna)
        out["t_half_log10"] = g / math.log(10.0)
    else:
        out["half_ok"] = False
        out["t_half_log10"] = float("inf")
    return out


print("== R1 基准复核（p=3/2，标准口径） ==")
b0 = band()
print("  C = %.4e, beta = %.4f, B = %.4f, ln(t_age) = %.3f" % (b0["C"], b0["beta"], b0["B"], b0["ln_age"]))
print("  band = [%.4f, %.4f]（第 413 条：[0.5598, 0.6015]）" % (b0["a_min"], b0["a_max"]))
print("  alpha = 1/2：beta = 1/2 为临界支（同标度，无交点）；对 beta < 1/2 的结构核见 R4")
print()

print("== R2 输入灵敏度（各乘 1/2 与 2，记录带端） ==")
print("  input        factor   a_min     a_max")
for name, kw, fac in (("f", {"f": F0 / 2}, 0.5), ("f", {"f": F0 * 2}, 2.0),
                       ("w_def", {"w": W0 / 2}, 0.5), ("w_def", {"w": W0 * 2}, 2.0),
                       ("t_eq", {"t_eq": T_EQ0 / 2}, 0.5), ("t_eq", {"t_eq": T_EQ0 * 2}, 2.0),
                       ("z_eq", {"z": Z_EQ0 / 2}, 0.5), ("z_eq", {"z": Z_EQ0 * 2}, 2.0),
                       ("t_age", {"t_age": T_AGE0 / 2}, 0.5), ("t_age", {"t_age": T_AGE0 * 2}, 2.0)):
    bb = band(**kw)
    print("  %-10s   x%-5s   %.4f    %.4f" % (name, fac, bb["a_min"], bb["a_max"]))
print()

print("== R3 稀释指数变体（p：kination 1、辐射 3/2、物质 2/3? 见下） ==")
print("  p      beta    band               0.5 可行")
for p in (0.5, 1.0, 1.5, 2.0, 3.0):
    bb = band(p=p)
    if bb["band_ok"]:
        hi = bb["a_max"]
        hitxt = ("%.4f" % hi) if hi != float("inf") else "inf"
        print("  %-6.2f %.4f  [%.4f, %s]     %s" % (p, bb["beta"], bb["a_min"], hitxt, bb["half_ok"]))
    else:
        print("  %-6.2f %.4f  empty (B<=0 or a_min>a_max)   %s" % (p, bb["beta"], bb["half_ok"]))
print()

print("== R4 结构性核：g(beta) 的最小值（beta -> 0） ==")
print("  g(beta) = (ln C - beta ln t_eq)/(1/2 - beta)，beta in (0, 1/2)，单调性判据：ln C - (1/2) ln t_eq = %.3f" % (math.log(b0["C"]) - 0.5 * math.log(T_EQ0)))
gmin = math.log(b0["C"]) / 0.5
print("  g_min = ln C / 0.5 = %.3f（对应 beta -> 0，即 p -> 0）" % gmin)
print("  与 ln(t_age) = %.3f 比较：g_min - ln(t_age) = %+.3f（>0 即 1/2 一律不可行）" % (b0["ln_age"], gmin - b0["ln_age"]))
print()

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

ax = axes[0]
labels = []
vals_lo = []
vals_hi = []
for name, kw, fac in (("f x1/2", {"f": F0 / 2}, 0.5), ("f x2", {"f": F0 * 2}, 2.0),
                      ("w x1/2", {"w": W0 / 2}, 0.5), ("w x2", {"w": W0 * 2}, 2.0),
                      ("t_eq x1/2", {"t_eq": T_EQ0 / 2}, 0.5), ("t_eq x2", {"t_eq": T_EQ0 * 2}, 2.0),
                      ("z x1/2", {"z": Z_EQ0 / 2}, 0.5), ("z x2", {"z": Z_EQ0 * 2}, 2.0),
                      ("t_age x1/2", {"t_age": T_AGE0 / 2}, 0.5), ("t_age x2", {"t_age": T_AGE0 * 2}, 2.0)):
    bb = band(**kw)
    labels.append(name)
    vals_lo.append(bb["a_min"])
    vals_hi.append(bb["a_max"])
y = range(len(labels))
ax.hlines(y, vals_lo, vals_hi, color="#2b6cb0", lw=3)
ax.axvline(0.5598, color="0.5", ls="--", lw=1)
ax.axvline(0.6015, color="0.5", ls="--", lw=1)
ax.set_yticks(list(y)); ax.set_yticklabels(labels, fontsize=8)
ax.set_xlabel("alpha band"); ax.set_title("R2: band ends under input x1/2 / x2")
ax.grid(alpha=0.3, axis="x")

ax = axes[1]
betas = [i * 0.005 for i in range(1, 100)]
gs = []
for bt in betas:
    B = math.log(b0["C"]) - bt * math.log(T_EQ0)
    gs.append(B / (0.5 - bt))
ax.plot(betas, gs, color="#c45911", lw=2, label="g(beta) = t*(1/2) in ln units")
ax.axhline(b0["ln_age"], color="#2f855a", ls="--", lw=1.5, label="ln(t_age) = %.1f" % b0["ln_age"])
ax.axvspan(0.5, 1.0, color="0.8", alpha=0.5)
ax.text(0.72, min(gs) + 2, "no crossing (beta >= 1/2)", fontsize=8, ha="center")
ax.set_xlabel("beta = p/3"); ax.set_ylabel("g (ln units)")
ax.set_title("R4: 1/2 needs g <= ln(age); minimum is above")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(IMG, dpi=160)
print("figure saved:", IMG)
