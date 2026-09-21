# -*- coding: utf-8 -*-
"""入口份额 f 与指数深度 234.4 的审查（第 363 条）。
运行：python 验证脚本/minimal_verification_f_depth_audit.py
"""
import math

m_P = 2.176434e-8
l_P = 1.616255e-35
m_e = 9.1093837015e-31
rho_b = 4.2075e-28
f_doc = 1.531335e-102
M = 1.273240 * m_e

print("== 1. 被审数字 ==")
print("  f = %.6e ; ln(1/f) = %.4f（登记 234.4）" % (f_doc, math.log(1.0 / f_doc)))
print("  -log10(f) = %.4f（登记 101.8）" % (-math.log10(f_doc)))
print()

print("== 2. f 的反解，两种 rho_P 读法 ==")
rho_site = 1.0 / l_P ** 3
rho_mass = m_P / l_P ** 3
f_site = rho_b / (M * rho_site)
f_mass = rho_b / (M * rho_mass)
f_ratio = rho_b / ((M / m_P) * rho_mass)
print("  rho_P = 1/l_P^3（站点数密度）= %.6e m^-3 -> f = %.6e，相对差 %.2e" % (rho_site, f_site, abs(f_site / f_doc - 1)))
print("  rho_P = m_P/l_P^3（质量密度）  = %.6e kg/m^3 -> f = %.6e，与登记值差 %.1f 个数量级" % (rho_mass, f_mass, math.log10(f_doc / f_mass)))
print("  质量密度配无量纲 M/m_P          -> f = %.6e，相对差 %.2e" % (f_ratio, abs(f_ratio / f_doc - 1)))
print()

print("== 3. 量纲检查 ==")
print("  论文原式：rho_def = f M rho_P，rho_P = m_P/l_P^3")
print("  M 取质量量纲时，右边量纲 = kg x kg/m^3 = kg^2/m^3，与左边 kg/m^3 不符（差一个质量量纲）。")
print("  数值上成立的读法：M 取无量纲的 M/m_P，或 rho_P 取站点数密度 1/l_P^3。")
print()

print("== 4. 每格一缺陷的密度与观测跨度 ==")
rho_one_dimless = (M / m_P) * rho_mass
rc = math.log10(rho_one_dimless / rho_b)
print("  读法一（M/m_P 乘质量密度）= %.4e kg/m^3，与观测相差 %.1f 个数量级（对上 101.8）" % (rho_one_dimless, rc))
rho_one_naive = M * rho_mass
print("  读法二（M 直接乘质量密度）= %.4e kg/m^3，与观测相差 %.1f 个数量级" % (rho_one_naive, math.log10(rho_one_naive / rho_b)))
print()

print("== 5. 结论 ==")
print("  234.4 与被审的 f 自洽；f 与 101.8 亦自洽。")
print("  需要一句口径修正：把 rho_P 定义为站点数密度 1/l_P^3，或把 M 写成 M/m_P。")
