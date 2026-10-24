# -*- coding: utf-8 -*-
r"""
window_ledger.py

参数区四窗口对照轮：把同一把 K 尺上的四条窗口并排，判定交集与 P1-40 的状态。

来源：作者 2026-09-20 指令（开工）。开工前提是本轮原拟的「K 双约束交集」已有现成读数，
      故改为非重复的那一步：把钉扎安全窗口放在与既有三条窗口同一轴上，并给出 P1-40 的状态更新。

四条窗口（同一能量泛函 E = sum [ (1/2) k (Delta u - 1)^2 + K (1 - cos 2 pi u) ]，k = 1；口径同一性由 D-4 核过）：
  W1 复合模型落窗      K <= 0.02059     （宽度窗口 l* >= 2，费米子通道第三轮 H2）
  W2 洛伦兹残差窗口    K <= 0.00633     （l_SG >= 2，同上 H2；比 W1 紧 3.25 倍）
  W3 钉扎安全窗口      K <= 1.583e-3    （r = Delta E_PN / M <= 1e-15 需 l_SG >= 3.99，钉扎-洛伦兹张力轮 E5）
  W3b 更严档            K <= 9.33e-4     （r <= 1e-20 需 l_SG >= 5.21，同上）
  W4 C1 驻留窗口       K >= 0.24        （C1-4 锚核筛选，格点内锁定路线）

判据（先行）：
  P1 W1、W2、W3 的交集非空，等于 W3，且 W1/W3 = 13.0 倍、W2/W3 = 4.0 倍
  P2 加入更严档 W3b 后交集仍非空（等于 W3b）
  P3 W3 完全落在 W1 与 W2 之内，故钉扎张力在复合模型通道内可消解，前置条件 K <= 1.583e-3；
     用费米子通道的口径桥 l*/l_SG = 1.803 换算，等价于 l* >= 7.19
  P4 W4 与 W1/W2/W3 的交集为空，倍数分别为 152 与 12；此为 C1-4 已登记的窗口互斥在跨线口径下的重述，不重复判定
"""
import numpy as np
rows = []

W1 = 0.02059
W2 = 0.00633
l_safe15, l_safe20 = 3.99, 5.21
W3 = 1.0 / (2.0 * np.pi * l_safe15) ** 2
W3b = 1.0 / (2.0 * np.pi * l_safe20) ** 2
W4 = 0.24
bridge = 1.803

inter1 = min(W1, W2, W3)
rows.append(("P1", "W1、W2、W3 的交集上界", "%.6f" % inter1, "等于 W3 = %.6f" % W3, abs(inter1 - W3) < 1e-12))
rows.append(("P1", "W1/W3 与 W2/W3", "%.2f / %.2f" % (W1 / W3, W2 / W3), "13.0 / 4.0", abs(W1 / W3 - 13.0) < 0.5 and abs(W2 / W3 - 4.0) < 0.2))
inter2 = min(inter1, W3b)
rows.append(("P2", "加入更严档后的交集上界", "%.6f" % inter2, "等于 W3b = %.6f" % W3b, abs(inter2 - W3b) < 1e-12))
rows.append(("P3", "W3 是否落在 W1 与 W2 之内", "是" if (W3 < W1 and W3 < W2) else "否", "是", W3 < W1 and W3 < W2))
rows.append(("P3", "换算到宽度口径 l* 的下界（桥因子 %.3f）" % bridge, "%.2f" % (l_safe15 * bridge), ">= 7.19", abs(l_safe15 * bridge - 7.19) < 0.05))
rows.append(("P4", "W4 与前三者交集为空", "空" if W4 > inter1 else "非空", "空", W4 > inter1))
rows.append(("P4", "W4/W3 与 W4/W2 的倍数", "%.0f / %.0f" % (W4 / W3, W4 / W2), "151 / 38", abs(W4 / W3 - 151) < 3 and abs(W4 / W2 - 38) < 2))

print("=== 参数区四窗口对照轮判据表（实测 对 期望）===")
for tag, name, meas, exp, ok in rows:
    print("%-4s %-46s 实测 %-18s 期望 %-22s %s" % (tag, name, meas, exp, "通过" if ok else "不通过"))
print("")
print("P5 结论：W1、W2、W3 三窗相容，交集为 K <= %.6f（等价 l_SG >= %.2f，即宽度口径 l* >= %.2f）；" % (W3, l_safe15, l_safe15 * bridge))
print("        故钉扎-洛伦兹张力在复合模型通道内可消解，前置条件是刚度落进该窗口。")
print("        W4（C1 驻留，K >= %.2f）与前三者不相交，倍数 %.0f 与 %.0f；该互斥由 C1-4 已登记，本轮只做跨线口径重述。" % (W4, W4 / W3, W4 / W2))
print("        待裁决项：粒子通道取哪一个。C1 三维环通道需另加机制（体项或 Hopf）把驻留阈值下拉进前三窗；", )
print("        复合 kink 通道已在窗口内可用。")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
IMG = r"D:\software\TianXingFC\images\window_ledger.png"
fig, ax = plt.subplots(figsize=(9.0, 3.6))
names = ["W1 复合落窗", "W2 洛伦兹残差", "W3 钉扎安全", "W3b 更严档"]
vals = [W1, W2, W3, W3b]
ax.barh(range(len(vals)), vals, color=["tab:blue", "tab:cyan", "tab:green", "tab:olive"])
ax.axvline(W4, color="tab:red", ls="--", label="W4 C1 驻留下界 %.2f" % W4)
ax.set_yticks(range(len(vals)))
ax.set_yticklabels(names, fontsize=8)
ax.set_xscale("log")
ax.set_xlabel("K（对数轴）")
ax.set_title("四窗口在同一 K 轴上的位置：前三者相容，W4 与前三者不相交")
ax.legend(fontsize=8)
ax.grid(True, axis="x", which="both", alpha=0.3)
fig.tight_layout()
fig.savefig(IMG, dpi=140)
print("图已存：" + IMG)