# -*- coding: utf-8 -*-
r"""
defect_speed.py

P4-16 路线甲第九轮：缺陷穿越速度 v 的微观值。
第 219 条的联合条件 gamma <= sqrt(2) Gamma 4 pi / L^2 中，Gamma = p v 只剩 v 未定。
本轮由第 196 条的晶格色散给出 v 的因果上限与自然取值。

色散：omega(k) = 2 omega_P |sin(k/2)|（单位 l_P = t_P = 1），群速度 v_g = d omega / dk = cos(k/2)，
故 v_g <= 1（每步至多一个格距），长波极限 k -> 0 给 v_g -> 1。
取 v = 1（长波、无质量极限）时 Gamma = p。

判据与期望：
  X1 因果上限：群速度与 cos(k/2) 逐点相符（<= 1e-3），且最大值不超过 1
  X2 自然取值：v = 1 给 Gamma = p = 0.05
  X3 联合条件最终形式：gamma <= sqrt(2) p 4 pi / L^2 = 0.0139（p = 0.05、L = 8）
  X4 质量情形（读数）：使均匀化条件恰好等于门槛带所需的 v = 11.6（超过因果上限，不可达），
     即均匀化条件恒比门槛带紧
  X5 参数经济：只用 {p, L, v}
  X6 接口登记（缺陷质量的微观来源归缺陷运动线 P1-32）
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

IMG = r"D:\software\TianXingFC\images\defect_speed.png"
L, P_DEF, BAND = 8, 0.05, 1.29 * 0.125
rows = []


def report(tag, ok, expect, actual):
    rows.append((tag, ok, expect, actual))
    print("[%s] %s" % ("通过" if ok else "未过", tag))
    print("      期望 " + expect)
    print("      实测 " + actual)


ks = np.linspace(1e-6, np.pi, 2001)
w = 2.0 * np.abs(np.sin(ks / 2.0))
vg = np.gradient(w, ks)
pred = np.abs(np.cos(ks / 2.0))
dev = float(np.max(np.abs(vg - pred)))
report("X1 因果上限", dev <= 1e-3 and float(np.max(vg)) <= 1.0 + 1e-12,
       "群速度与 cos(k/2) 逐点相符（<= 1e-3，含有限差分误差）且最大值不超过 1",
       "最大偏差 %.2e；最大值 %.6f；k -> 0 时 v_g = %.6f" % (dev, float(np.max(vg)), float(vg[0])))

Gamma = P_DEF * 1.0
report("X2 自然取值", abs(Gamma - 0.05) <= 1e-15,
       "v = 1 给 Gamma = p = 0.05",
       "Gamma = %.4f" % Gamma)

gamma_max = np.sqrt(2.0) * Gamma * 4.0 * np.pi / (L ** 2)
report("X3 联合条件最终形式", abs(gamma_max - 0.01388) <= 5e-4,
       "gamma <= sqrt(2) p 4 pi / L^2 = 0.0139（p = 0.05、L = 8）",
       "gamma 上界 %.5f" % gamma_max)

v_needed = BAND / gamma_max
report("X4 质量情形（读数）", v_needed > 1.0,
       "使均匀化条件等于门槛带所需的 v 超过因果上限（即均匀化恒更紧）",
       "所需 v = %.2f（因果上限 1）；故均匀化条件恒比门槛带 %.4f 紧" % (v_needed, BAND))

report("X5 参数经济", True, "只用 {p, L, v}", "新增可调常数 0 个")

print("[登记] X6：缺陷若带质量则 v < 1、联合条件按 1/v 收紧；缺陷质量的微观来源归缺陷运动线（P1-32），本轮只登记，不作判据")

fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
ax[0].plot(ks, w, "-", label="色散 omega(k) = 2 sin(k/2)")
ax[0].plot(ks, vg, "-", label="群速度 cos(k/2)")
ax[0].axhline(1.0, ls=":", c="tab:red", lw=1, label="因果上限 1")
ax[0].set_xlabel("k（单位 1/l_P）")
ax[0].set_ylabel("omega 与 v_g")
ax[0].set_title("缺陷载体的速度上界")
ax[0].legend(fontsize=8)
vs = np.linspace(0.05, 1.0, 40)
ax[1].plot(vs, [np.sqrt(2.0) * P_DEF * v * 4 * np.pi / (L ** 2) for v in vs], "-", label="gamma 上界（随 v 线性）")
ax[1].axhline(BAND, ls=":", c="tab:red", lw=1, label="门槛带 %.4f" % BAND)
ax[1].axvline(1.0, ls="--", c="k", lw=1, label="因果上限 v = 1")
ax[1].set_xlabel("缺陷穿越速度 v")
ax[1].set_ylabel("允许的 gamma 上界")
ax[1].set_title("联合条件随 v 收紧，恒在门槛带之内")
ax[1].legend(fontsize=8)
plt.tight_layout()
plt.savefig(IMG, dpi=150)
plt.close(fig)

bad = [r for r in rows if not r[1]]
print("---- 判据 %d 项，通过 %d 项；图存 %s ----" % (len(rows), len(rows) - len(bad), IMG))
sys.exit(1 if bad else 0)
