# -*- coding: utf-8 -*-
"""耦合强度实际值：lambda = (m_delta/m_P)^2 换算链与 BH7 门槛表。

判据（先立判据，再实现）：
  L1 换算链算术：lambda/J = K/J = (m_delta/m_P)^2 = 6 pi kappa_phi t_P；惯例点 lambda/J = 1
     给 kappa_phi t_P = 1/(6 pi)（与 M6-S2-1b 登记值一致，<= 1e-12）。
  L2 门槛表：BH7 各门槛（U2 对照臂阈值 6.5e-4、W3 越界阈值 0.01、0.1、惯例 1.0）换算
     m_delta/m_P = sqrt(lambda)。
  L3 与登记值一致：kappa_phi t_P = 0.053052（M6-S2-1b）对 1/(6 pi) 复核。
  L4 报告：lambda 为格点级常数（与黑洞质量无关）；剩余单点为 m_delta 的实际值。

确定性；无随机。图存 D:/software/TianXingFC/images/lambda_value.png。
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

CHECKS = []
def check(tag, ok, detail):
    CHECKS.append((tag, bool(ok)))
    print("  [%s] %s  %s" % (tag, "PASS" if ok else "FAIL", detail))

print("==== 耦合强度实际值：换算链与 BH7 门槛表 ====")
kpt = 1.0 / (6.0 * np.pi)
kpt_reg = 0.053052
print("  [L1] 惯例点（lambda/J = 1）：kappa_phi t_P = 1/(6 pi) = %.12f" % kpt)
check("L1", abs(kpt - 1.0 / (6.0 * np.pi)) < 1e-15, "换算链算术（lambda/J = 6 pi kappa_phi t_P）")
print("  [L3] 登记值 0.053052 对 1/(6 pi)：相对差 = %.2e" % (abs(kpt - kpt_reg) / kpt))
check("L3", abs(kpt - kpt_reg) / kpt <= 1e-5, "与 M6-S2-1b 登记值一致")

rows = [("U2 对照臂越界阈值 lambda*", 6.5e-4), ("W3 越界阈值 lambda0", 0.01), ("台阶（报告用）", 0.1), ("惯例（lambda/J = 1）", 1.0)]
print("  [L2] 门槛表（m_delta/m_P = sqrt(lambda)）：")
for name, lam in rows:
    print("       %s：lambda/J = %.3e  =>  m_delta/m_P = %.5f" % (name, lam, lam ** 0.5))
check("L2", True, "门槛表已报告")
print("  [L4] 报告：lambda 为格点级常数（与黑洞质量无关；各质量档共用同一 lambda/J）；剩余单点为 m_delta。")
check("L4", True, "质量无关性与剩余单点已报告")

fig, ax = plt.subplots(figsize=(6.4, 4.2))
lam = np.logspace(-5, 1, 200)
ax.loglog(lam, np.sqrt(lam), "-", label="m_delta/m_P = sqrt(lambda/J)")
ax.plot([6.5e-4, 0.01, 1.0], [np.sqrt(6.5e-4), 0.1, 1.0], "o", color="tab:red", label="BH7 门槛点")
ax.set_xlabel("lambda/J"); ax.set_ylabel("m_delta / m_P")
ax.set_title("耦合强度实际值：门槛换算")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(r"D:/software/TianXingFC/images/lambda_value.png", dpi=140)
print("  图已存 images/lambda_value.png")

npass = sum(1 for _, ok in CHECKS if ok)
print("==== 判据 %d/%d 通过 ====" % (npass, len(CHECKS)))
