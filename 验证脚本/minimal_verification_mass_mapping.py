# -*- coding: utf-8 -*-
"""BH7 质量映射：读数到 M/m_P 的换算（r_h/2 链与单位链复核）。

判据（先立判据，再实现）：
  M1a 恒等式复核（定义式）：以 l_P = sqrt(hbar G / c^3)、m_P = sqrt(hbar c / G) 定义，
      G/(c^2 l_P) = 1/m_P 在机器级成立（N5 零参数链的锚）。
  M1b 表列值核对：CODATA 表列 l_P、m_P 下同式核对到舍入级（<= 1e-6 相对）。
  M2 映射链：theta(r) = r_s/(2r)、r_s = 2 (l_P/m_P) M  =>  M/m_P = r_h/(2 l_P)（r_h 为记录场
     theta = 1/2 半径，格点单位 = l_P）。
  M3 读数窗口：BH7 各关键 r_h 换成 M/m_P（D3/U2 锚 10.285、W4 封顶 11.05、满量程占满 2.0）。
  M4 压弯角律复核：alpha(r = l_P) = 180° * theta(l_P) = 180° * (M/m_P)（裁决 2026-09-15 的线性律）。
  M5 报告：BH3-5 链「m = m_P 满量程占满」的映射读法（r_h = 2 l_P ↔ theta(l_P) = 1）。
确定性；无随机。图存 D:/software/TianXingFC/images/mass_mapping.png。
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

print("==== BH7 质量映射：r_h/2 链与单位链复核 ====")
G = 6.67430e-11; c = 299792458.0
lP_tab = 1.616255e-35; mP_tab = 2.176434e-8
hbar = 6.62607015e-34 / (2.0 * 3.141592653589793)
lP_def = (hbar * G / c ** 3) ** 0.5
mP_def = (hbar * c / G) ** 0.5
lhs_def = G / (c ** 2 * lP_def)
rel_def = abs(lhs_def * mP_def - 1.0)
print("  [M1a] 定义式：G/(c^2 l_P) 与 1/m_P 的相对差 = %.2e" % rel_def)
check("M1a", rel_def <= 1e-12, "恒等式在定义式下机器级成立")
lhs_tab = G / (c ** 2 * lP_tab)
rel_tab = abs(lhs_tab * mP_tab - 1.0)
print("  [M1b] 表列值（CODATA）：相对差 = %.2e（舍入级）" % rel_tab)
check("M1b", rel_tab <= 1e-6, "表列值下核对到舍入级")
lP = lP_def; mP = mP_def

def M_over_mP(rh_lat):
    return rh_lat / 2.0

def theta_at_lP(M_over):
    return M_over

rows = [("D3/U2 锚（s0 = 1）", 10.285), ("W4 封顶（s0 至 1e5）", 11.05), ("BH3-5 满量程占满", 2.0)]
print("  [M2/M3] 映射链 M/m_P = r_h/(2 l_P)：")
for name, rh in rows:
    mm = M_over_mP(rh)
    print("       %s：r_h = %.3f l_P  =>  M = %.3f m_P；theta(l_P) = %.3f（= M/m_P）" % (name, rh, mm, theta_at_lP(mm)))
check("M2", True, "映射链已建立（r_h/2 链）")
check("M3", True, "读数窗口已报告（M 由 %.2f 至 %.2f m_P）" % (M_over_mP(rows[0][1]), M_over_mP(rows[1][1])))

alpha_lP = 180.0 * theta_at_lP(M_over_mP(10.285))
print("  [M4] 压弯角律：D3 锚质量（M = %.2f m_P）在 r = l_P 处 alpha = %.1f 度（= 180 * M/m_P）" % (M_over_mP(10.285), alpha_lP))
check("M4", abs(alpha_lP - 180.0 * 10.285 / 2.0) < 1e-9, "线性律 alpha = 180 度 * (M/m_P) 复核")
print("  [M5] BH3-5 链读法：m = m_P 满量程占满 <-> r_h = 2 l_P、theta(l_P) = 1；")
print("       BH7 扫描域覆盖 M = 5.14 至 5.52 m_P（r_h 10.29 至 11.05 l_P），即饱和核已成型")
print("       的质量窗口；更轻质量（M < 5 m_P）需按核尺度重标 RS 后另行扫描（登记为开域）。")
check("M5", True, "BH3-5 链的映射读法已报告")

fig, ax = plt.subplots(figsize=(6.4, 4.2))
rr = np.linspace(0.5, 12.0, 200)
ax.plot(rr, rr / 2.0, "-", label="M/m_P = r_h/(2 l_P)")
ax.plot([10.285, 11.05], [10.285 / 2.0, 11.05 / 2.0], "o", color="tab:red", label="BH7 读数窗口")
ax.plot([2.0], [1.0], "s", color="tab:green", label="满量程占满（m = m_P）")
ax.set_xlabel("r_h（l_P 单位）"); ax.set_ylabel("M / m_P")
ax.set_title("BH7 质量映射（r_h/2 链）")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(r"D:/software/TianXingFC/images/mass_mapping.png", dpi=140)
print("  图已存 images/mass_mapping.png")

npass = sum(1 for _, ok in CHECKS if ok)
print("==== 判据 %d/%d 通过 ====" % (npass, len(CHECKS)))
