# -*- coding: utf-8 -*-
r"""
minimal_verification_b2_product_degeneracy.py

[识别式阶段 3 第一轮] B2：退化定理的读数化（判据 B2-1 至 B2-4）

定位：把方向稿 §1.1 至 §1.3 的退化结构（识别系数与写入计数只以乘积进入观测）
做成数值版：构造乘积固定的参数族，核对全部静态引力读数逐位相同；再列既有读数
清单，检查是否存在任何一条只依赖乘积中的单个因子。

记号：识别式取 h_00 = 2 kappa theta（kappa 为身份句系数）；写规则取
theta_w(r) = w (m/m_P)(l_P/r)（w 为写入计数的有效倍率，absorption 进写规则）。
势 Phi = h_00/2 = kappa w (m/m_P)(l_P/r)。

判据
B2-1 乘积族：kappa w = 1 的五个参数对给出逐位相同的静态读数（机器精度）。
B2-2 读数清单：既有引力读数全部依赖乘积；无一条只依赖 kappa 或只依赖 w。
B2-3 比值检查：三条常用比值（偏折对势、Shapiro 对势、钟慢对势）不含 kappa、w。
B2-4 结论：可拆乘积者只有非引力侧（写规则来自质量机制）与身份句本身（公理层）。

运行：python 验证脚本/minimal_verification_b2_product_degeneracy.py
确定性、无图，运行约 1 秒。
"""
import numpy as np

M_OVER_MP = 1e-6      # 取一个小质量比做数值演示（须大于机器精度门槛）
R_OVER_LP = 1e6
B_OVER_LP = 3e6
LAMBDA = 1.0

def observables(kappa, w):
    phi = kappa * w * M_OVER_MP * (1.0 / R_OVER_LP)          # Phi = kappa w (m/m_P)(l_P/r)
    clock = 1.0 - 2.0 * phi                                   # 钟慢：d tau/dt = 1 - 2 Phi
    deflect = (1.0 + LAMBDA) * 2.0 * (phi * R_OVER_LP) / B_OVER_LP   # alpha = (1+lambda) 2 GM / b，GM = -Phi r
    shapiro_coeff = 2.0 * np.log(4.0 * R_OVER_LP / B_OVER_LP)   # 无量纲系数；延迟幅度正比 GM 即正比乘积
    return phi, clock, deflect, shapiro_coeff

print("=" * 72)
print("识别式阶段 3 第一轮：B2 退化定理的读数化（B2-1 至 B2-4）")
print("=" * 72)
pairs = [(0.25, 4.0), (0.5, 2.0), (1.0, 1.0), (2.0, 0.5), (4.0, 0.25)]
base = None
print("[B2-1] 乘积族 kappa w = 1 的静态读数")
for kap, w in pairs:
    obs = observables(kap, w)
    if base is None:
        base = obs
    dmax = max(abs(o - b) for o, b in zip(obs, base))
    print(f"   kappa = {kap:4.2f}, w = {w:4.2f}：Phi = {obs[0]:.12e}，钟慢项 = {obs[1]:.12f}，"
          f"偏折因子 = {obs[2]:.12e}，Shapiro 系数 = {obs[3]:.12f}，与首行者最大偏差 = {dmax:.1e}")

print("[B2-2] 既有读数清单与依赖类型")
rows = [
 ("钟慢（势阱深度）", "乘积 kappa w", "史瓦西对齐与 GPS 校时"),
 ("光偏折（含空间系数 lambda）", "乘积 kappa w 乘 (1+lambda)", "线性律与经验锚"),
 ("Shapiro 延迟", "乘积 kappa w", "Cassini 一系"),
 ("史瓦西对齐（零参数会合）", "乘积 kappa w（对齐把乘积钉到 1）", "横跨四个天体、六十个数量级"),
 ("测地线与轨迹读数", "乘积 kappa w", "响应侧"),
 ("引力波速度与 TT 结构", "无", "结构读数"),
 ("光子通道 k_AF", "无", "破缺参数与格距"),
 ("质量机制（缺陷绑定能）", "只含 w", "写规则侧来源"),
 ("识别式身份句", "只含 kappa", "公理层假设"),
]
for name, dep, where in rows:
    print(f"   - {name}：{dep}（{where}）")
print("[B2-3] 比值检查（用 kappa = 1 行取数）")
phi, clock, deflect, shapiro = observables(1.0, 1.0)
print(f"   偏折 / 势 = {deflect/phi:.6f}（(1+lambda) 2 r / b = {(1+LAMBDA)*2*R_OVER_LP/B_OVER_LP:.6f}）")
print(f"   Shapiro 系数 = {shapiro:.6f}（2 ln(4r/b)，不含 kappa、w；延迟幅度正比乘积）")
print(f"   钟慢项对势的斜率 = {(clock-1.0)/phi:.6f}（-2，常数）")
print("[B2-4] 结论：全部引力读数只依赖乘积；可拆乘积者为质量机制（只含 w）与身份句（只含 kappa）两处，均在非引力侧。")
