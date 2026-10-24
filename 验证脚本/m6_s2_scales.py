# -*- coding: utf-8 -*-
"""M6-S2（可做部分）：参数来源、单位链与时间标定。

判据（先写期望，再算；设计见《黑洞线攻坚计划：坍缩机制重定位与强场终态》§7.6 的 S2 步）：
  S2a 单位链锚点：记录场的换算链（h = l_P theta、满偏 = 1 = 180 度、拍 = t_P）下，
       N5 对齐恒等式 theta(r_s) = 1/2 对四个量级跨度的质量成立（偏差 <= 1e-3）；
       且注入几何恒等式 6 N_c = 4 pi（N_c = 2 pi / 3）逐位成立。
  S2b 注入量级（报告）：单次碰撞写入 delta theta = m/m_P；源强 s_eff = 4 pi m/m_P。
  S2c 斜坡速率（报告）：坍缩驱动下视界处记录的增长速率 Gamma = (M/m_P)(l_P/r_s)^2（每拍）。
  S2d 交叉判定：以 D2 的交叉速率（一维均匀、三维均匀、1/r 漏斗）为门槛，比较 Gamma 与 Gamma_c；
       耦合先触发的条件为 Gamma < Gamma_c；报告比值与触发时刻（拍与秒）。
  S2e 噪声底口径（报告）：取单次碰撞写入量为噪声底（eps0 = m/m_P 量级），给出 Gamma_c 的相应移动。

输入（全部取既有定案材料）：每拍注入 delta theta = m/m_P、固定点条件 s_eff = 6 w_eff / alpha、
惯性换算 h = l_P theta、拍 = t_P（c = l_P/t_P）、N5 对齐 theta(r_s) = 1/2。
确定性：全部为解析式与常数计算，无随机数。
"""

import sys
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CHECKS = []


def check(tag, ok, detail):
    CHECKS.append((tag, bool(ok)))
    print("  [%s] %s  %s" % (tag, "PASS" if ok else "FAIL", detail))


# 常数（CODATA / 体系定案）
G = 6.67430e-11
c = 2.99792458e8
M_P = 2.176434e-8          # 普朗克质量 [kg]
L_P = 1.616255e-35         # 普朗克长度 [m]
T_P = 5.391247e-44         # 普朗克时间 [s]
M_NUC = 1.67262192369e-27  # 核子质量 [kg]

print("==== M6-S2（可做部分）：参数来源、单位链与时间标定 ====")

# S2a：theta(r_s) = 1/2 对四个量级跨度的质量
bodies = [("核子", M_NUC), ("1 克", 1e-3), ("地球", 5.9722e24), ("太阳", 1.98847e30)]
devs = []
print("\n  [S2a] N5 对齐恒等式 theta(r_s) = 1/2（四个量级跨度）")
for name, m in bodies:
    r_s = 2 * G * m / c ** 2
    theta_rs = (m / M_P) * (L_P / r_s)
    devs.append(abs(theta_rs - 0.5))
    print("       %-4s m = %.3e kg  r_s = %.3e m  theta(r_s) = %.12f" % (name, m, r_s, theta_rs))
nc = 2 * np.pi / 3
geom = abs(6 * nc - 4 * np.pi)
check("S2a", max(devs) <= 1e-3 and geom <= 1e-12,
      "theta(r_s) 最大偏差 %.2e；6 N_c - 4 pi = %.2e（N_c = %.6f）" % (max(devs), geom, nc))

# S2b：注入量级
d_theta = M_NUC / M_P
s_eff = 4 * np.pi * M_NUC / M_P
print("\n  [S2b] 注入量级：delta theta = m/m_P = %.3e（单次碰撞）；s_eff = %.3e（每拍）" % (d_theta, s_eff))

# S2c：坍缩驱动的斜坡速率（每拍）
def gamma_geo(m):
    r_s = 2 * G * m / c ** 2
    return (m / M_P) * (L_P / r_s) ** 2

print("\n  [S2c] 坍缩驱动的斜坡速率 Gamma = (M/m_P)(l_P/r_s)^2（每拍）")
masses = [("核子", M_NUC), ("1 克", 1e-3), ("地球", 5.9722e24), ("太阳", 1.98847e30), ("10 太阳质量", 1.98847e31)]
for name, m in masses:
    print("       %-8s Gamma = %.3e 每拍" % (name, gamma_geo(m)))

# S2d：交叉判定
GC = {"一维均匀（D2）": 3.566293e-2, "三维均匀（D2）": 2.085900e-2,
      "一维漏斗（D2）": 1.304e-3, "三维漏斗 M4-D2 探路": 1.529887e-3}
LN = np.log(1.0 / 1e-3)
print("\n  [S2d] 交叉判定（Gamma < Gamma_c 即耦合先触发）")
g_sun = gamma_geo(1.98847e30)
g_10sun = gamma_geo(1.98847e31)
for name, gc in GC.items():
    print("       门槛 %-16s Gamma_c = %.4e  比值 Gamma/Gamma_c = %.2e" % (name, gc, g_sun / gc))
ratios_small = all(g_sun / gc < 1.0 for gc in GC.values())
check("S2d", ratios_small,
      "太阳质量档 Gamma = %.3e 每拍，对四个门槛的比值均小于 1（最小 %.2e）" % (g_sun, min(g_sun / gc for gc in GC.values())))

# 触发时刻：h = Gamma t，取漏斗型 gamma(h) ~ k h（k 由 D2 的 I 标定：I = int gamma dh = k/2）
k_funnel = 2 * 0.009008  # 一维漏斗 I = k/2
t_star = np.sqrt(2 * LN / (k_funnel * g_sun))
print("       漏斗型（一维，k = %.4f）：触发时刻 t* = %.3e 拍 = %.3e 秒" % (k_funnel, t_star, t_star * T_P))
print("       坍缩时标（毫秒级）约 %.3e 拍（对照）" % (1e-3 / T_P))

# S2e：噪声底口径
eps0_alt = M_NUC / M_P
ln_alt = np.log(1.0 / eps0_alt)
I_funnel_1d = 0.009008
gc_funnel_exact = I_funnel_1d / LN
gc_funnel_alt = I_funnel_1d / ln_alt
print("\n  [S2e] 噪声底口径：eps0 = 单次碰撞写入 %.3e -> ln(1/eps0) = %.3f（对照 1e-3 的 %.3f）" % (eps0_alt, ln_alt, LN))
print("       一维漏斗门槛相应由 %.4e 移到 %.4e（缩小 %.3f 倍）" % (gc_funnel_exact, gc_funnel_alt, gc_funnel_alt / gc_funnel_exact))
check("S2e", abs((gc_funnel_alt / gc_funnel_exact) - LN / ln_alt) < 1e-12,
      "门槛移动比 = ln(1/eps0)_旧 / ln(1/eps0)_新 = %.4f" % (LN / ln_alt))

# 交叉质量：Gamma(M) = Gamma_c 的解（Gamma 正比于 1/M）
gc_ref = gc_funnel_exact
m_star = (L_P ** 2 * c ** 4 * M_P) / (4 * G ** 2 * gc_ref)
print("\n  [S2f] 交叉质量：由 Gamma(M) = Gamma_c（漏斗门槛）解出 M* = %.3e kg（约 %.1e GeV）"
      % (m_star, m_star * c ** 2 / (1.602176634e-10)))
print("       宏观天体（>= 1 克）均在 Gamma < Gamma_c 一侧（耦合先触发）")

print("\n==== 判据汇总：%d / %d 通过 ====" % (sum(1 for _, ok in CHECKS if ok), len(CHECKS)))
sys.exit(0 if all(ok for _, ok in CHECKS) else 1)