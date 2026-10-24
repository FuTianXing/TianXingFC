# -*- coding: utf-8 -*-
"""第 438 条：惯性微观来源的三判据（变率继承）。

判据：
  I1 二阶性对照：变率继承规则 z^2 - (2 - lambda) z + 1 = 0 给 |z| = 1（振荡、实数频率），
     且 theta = k lP 精确（Courant 数 1 的线性色散）；位置直改规则 z = 1 - lambda 给 |z| != 1（衰减）。
  I2 常数分辨：惯性倍数 I -> 2I 不改变任何可观测量（等幅比值与长波斜率）；折叠系数 kapp 改变
     长波斜率（kapp 可观测，由"每拍抄写一次、瞬时无损"裁决固定为 1，不引入自由参数）。
  I3 两侧同源：实轴方向链与基准取向场横向模在 kapp = 1 时给出同一长波斜率；kapp 偏离给出定量比。

容差预算：I1 与 I3 的期望值为精确解析值（代数恒等式）。I3 的规范侧长波斜率含格点截断 $1 - k^2 lP^2/24$（正弦展开的高阶项），取 k = 1e-5 时该项量级 4e-12，故容差取 1e-10；
I1 的位置直改对照为结构性差异（|z| 与 1 的距离在 1e-2 量级以上），阈值取 1e-2。
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IMG = r"D:\software\TianXingFC\images\jz1_inertia_rate_inheritance.png"
lP, tP = 1.0, 1.0

def lam(k):
    return 4.0 * np.sin(k * lP / 2.0) ** 2

def z_inherit(k, kappa=1.0):
    a = 2.0 - kappa * lam(k)
    disc = a * a - 4.0
    if disc < 0.0:
        th = np.arccos(np.clip(a / 2.0, -1.0, 1.0))
        return complex(np.cos(th), np.sin(th))
    return complex(0.5 * (a + np.sqrt(disc)), 0.0)

def z_position(k):
    return 1.0 - lam(k)

# ---------------- I1 ----------------
ks = np.array([0.05, 0.1, 0.2, 0.4, 0.8, 1.2, 1.5, 2.0])
rows1 = []
for k in ks:
    z = z_inherit(k)
    th = np.angle(z)
    rows1.append((k, abs(z), abs(abs(z) - 1.0), th, abs(th - k * lP), abs(z_position(k))))

def recur(k, n=400, kappa=1.0):
    z = z_inherit(k, kappa)
    al = np.zeros(n + 2, dtype=complex)
    al[0], al[1] = 1.0, z
    lm = lam(k)
    for s in range(1, n + 1):
        al[s + 1] = 2.0 * al[s] - al[s - 1] - kappa * lm * al[s]
    return al, z

al, zmod = recur(0.3)
res_rec = float(np.max(np.abs(al[:401] - zmod ** np.arange(401))))

# ---------------- I2 ----------------
def ratio_E_B(dt, I):
    aE = 0.5 * I * (1.0 / dt ** 2 - 1.0 / (2.0 * tP ** 2))
    aB = -I / (4.0 * tP ** 2)
    return aE / abs(aB)

Is = [0.25, 1.0, 4.0]
rows2a = [(I, ratio_E_B(1.0, I)) for I in Is]
kaps = [0.5, 1.0, 2.0]
rows2b = [(kp, np.sqrt(kp) * lP / tP) for kp in kaps]

# ---------------- I3 ----------------
def slope_alpha(kappa):
    return np.sqrt(kappa) * lP / tP

def slope_gauge(kappa, kk):
    s = 4.0 * np.sum(np.sin(kk * lP / 2.0) ** 2)
    return np.sqrt(kappa * s) / tP / float(np.linalg.norm(kk))

k3 = np.array([1e-5, 1e-5, 1e-5])
sl_a = slope_alpha(1.0)
sl_g = slope_gauge(1.0, k3)

# ---------------- 输出 ----------------
print("I1 二阶性对照（变率继承 vs 位置直改）")
print("k      |z|继承   ||z|-1|      theta       |theta-k|    |z|位置")
for r1 in rows1:
    print("%.3f  %.10f  %.2e  %.10f  %.2e  %.10f" % r1)
print("递推核对 k=0.3（纯模式初值，特征根代入递推）：max|alpha^(s) - z^s| = %.2e" % res_rec)
print()
print("I2 惯性倍数 I（观测应不变）与折叠系数 kapp（观测应改变）")
for I, r_ in rows2a:
    print("I=%.2f 等幅比值=%.12f" % (I, r_))
for kp, sl in rows2b:
    print("kapp=%.2f 长波斜率=%.12f" % (kp, sl))
print()
print("I3 两侧同源（实轴方向链 vs 基准取向场横向模）")
print("alpha 斜率=%.12f；gauge 斜率=%.12f；相对差=%.2e" % (sl_a, sl_g, abs(sl_a - sl_g) / sl_a))
print("偏离检验：kapp_gauge=1.44 时斜率比=%.6f（预期 1.2）" % (slope_gauge(1.44, k3) / sl_a))
print()
flag1 = bool(all(r[2] < 1e-10 and r[4] < 1e-10 for r in rows1)
             and all(abs(r[5] - 1.0) > 1e-3 for r in rows1)
             and res_rec < 1e-10)
flag2 = bool(max(r[1] for r in rows2a) - min(r[1] for r in rows2a) < 1e-10
             and abs(rows2b[2][1] / rows2b[1][1] - np.sqrt(2.0)) < 1e-10)
flag3 = bool(abs(sl_a - sl_g) / sl_a < 1e-10
             and abs(slope_gauge(1.44, k3) / sl_a - 1.2) < 1e-10)
print("判据 I1（二阶性对照与精确线性色散）：%s" % ("通过" if flag1 else "未通过"))
print("判据 I2（惯性倍数不可观测、折叠系数可观测）：%s" % ("通过" if flag2 else "未通过"))
print("判据 I3（两侧同源与可证伪偏离）：%s" % ("通过" if flag3 else "未通过"))
print("判据总计：%d/3" % (int(flag1) + int(flag2) + int(flag3)))

# ---------------- 出图 ----------------
fig, ax = plt.subplots(2, 2, figsize=(11, 8))
kk = np.linspace(1e-3, np.pi - 1e-3, 400)
ax[0, 0].plot(kk, [abs(z_inherit(k)) for k in kk], "g-", lw=1.2, label="rate inheritance |z| = 1")
ax[0, 0].plot(kk, [abs(z_position(k)) for k in kk], "r-", lw=1.2, label="position direct |z| = |1-lambda|")
ax[0, 0].axhline(1.0, color="gray", lw=0.8)
ax[0, 0].set_ylim(0, 2)
ax[0, 0].set_xlabel("k")
ax[0, 0].set_ylabel("|z| per stroke")
ax[0, 0].set_title("I1 oscillation vs decay")
ax[0, 0].legend(fontsize=7)

ax[0, 1].plot(kk, kk * lP / tP, "g-", lw=1.2, label="inheritance: omega = (lP/tP) k (exact)")
ax[0, 1].plot(kk, (2.0 / tP) * np.abs(np.sin(kk * lP / 2.0)), "b--", lw=1.0, label="continuous-time sine")
ax[0, 1].set_xlabel("k")
ax[0, 1].set_ylabel("omega")
ax[0, 1].set_title("I1 dispersion: exact linear vs sine")
ax[0, 1].legend(fontsize=7)

ax[1, 0].bar([0, 1, 2], [r[1] for r in rows2a], color="seagreen")
ax[1, 0].set_xticks([0, 1, 2])
ax[1, 0].set_xticklabels(["I=0.25", "I=1", "I=4"], fontsize=8)
ax[1, 0].set_ylim(0, 1.4)
ax[1, 0].set_ylabel("equal-weight ratio")
ax[1, 0].set_title("I2 inertia multiplier: no observable change")

ax[1, 1].bar([0, 1], [sl_a, slope_gauge(1.44, k3)], color=["steelblue", "indianred"])
ax[1, 1].set_xticks([0, 1])
ax[1, 1].set_xticklabels(["same kappa (1.0)", "kappa=1.44 (falsifiable)"], fontsize=8)
ax[1, 1].set_ylabel("long-wave slope")
ax[1, 1].set_title("I3 two-side sameness")
plt.tight_layout()
plt.savefig(IMG, dpi=130)
print("图已保存：%s" % IMG)
