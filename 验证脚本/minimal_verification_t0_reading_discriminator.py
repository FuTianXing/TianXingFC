# -*- coding: utf-8 -*-
r"""
minimal_verification_t0_reading_discriminator.py

[识别式阶段 1 第一轮] 判定项读数化：两种读法的差异表（判据 T0-1 与 T0-2）

问题（方向稿 §2）：识别式与记录的关系是"度规结构"（记录直接改写度规，钟向与空间向
系数可独立）还是"线性换算因子"（记录只改一个局域标度因子，钟与尺同乘一个因子）。

判据（先立判据再实现）
T0-1 差异可算：两种读法在同一个势上给出不同的折射率，光偏折系数之比恰为 2。
T0-2 参数空间：换算因子读法的刻度函数只有一个（ℓ），度规结构读法有两个（钟向 κ 与空间向 λ）；
      两类读法在"能否让尺独立变化"上是结构差异，不是一个数值差异。
T0-3 既有读数归属：把三条既有读数（光偏折三读法、阶段 3 批二的交错模、阶段 4 的 λ 导出）
      逐条归到两侧之一。
T0-4 假设清单：列出度规结构侧读数所依赖的假设，并标出承重者。

运行：python 验证脚本/minimal_verification_t0_reading_discriminator.py
确定性、无图，运行约 1 秒。
"""
import numpy as np

A, B, EPS = 0.02, 20.0, 2.0

def phi(x, y):
    return -A / np.sqrt(x ** 2 + y ** 2 + EPS ** 2)

def n_clock(x, y):          # 换算因子读法：只有钟率（局部光速 sqrt(1 + 2 phi)）
    return 1.0 / np.sqrt(np.clip(1.0 + 2.0 * phi(x, y), 1e-9, None))

def n_metric(x, y):         # 度规结构读法：钟率加空间（lambda = 1）
    p = phi(x, y)
    return np.sqrt(np.clip(1.0 - 2.0 * p, 1e-9, None) / np.clip(1.0 + 2.0 * p, 1e-9, None))

def deflection(nfun, b=B, delta=0.5, span=60.0, nq=20001):
    xs = np.linspace(-span * b, span * b, nq)
    dndy = (nfun(xs, b + delta) - nfun(xs, b - delta)) / (2 * delta)
    return float(np.trapezoid(dndy, xs))

print("=" * 72)
print("识别式阶段 1 第一轮：判定项读数化（T0-1 至 T0-4）")
print("=" * 72)
al_c = deflection(n_clock)
al_m = deflection(n_metric)
print(f"[T0-1] 偏折积分：换算因子读法 {al_c:+.6e}；度规结构读法 {al_m:+.6e}；比值 {al_m/al_c:.6f}（期望 2）")
for a in (0.01, 0.04):
    A = a
    r = deflection(n_metric) / deflection(n_clock)
    print(f"       势幅 A = {a:.2f} 复算比值 = {r:.6f}")
A = 0.02
print("[T0-2] 参数空间：换算因子读法 1 个刻度函数（钟与尺同乘）；度规结构读法 2 个（κ 与 λ 独立）")
print("[T0-3] 既有读数归属（引既有材料）")
rows = [
 ("光偏折三读法（minimal_verification_light_deflection_local_frame.py）", "全值出现（钟率加空间）", "度规结构侧"),
 ("阶段 3 批二：交错模（判据 H2b）", "站点记录无法唯一决定段内容，尺有独立通道", "度规结构侧"),
 ("阶段 4：lambda 导出（判据 L1 至 L5）", "lambda = 1（结构恒等加三锚）", "度规结构侧"),
]
for a, b2, c in rows:
    print(f"   - {a}：{b2} → {c}")
print("[T0-4] 假设清单（度规结构侧读数所依赖者）")
for a in ("写规则 delta theta = m/m_P", "G 单一入口（s_eff = 4 pi m/m_P）", "稳态格林函数（离散泊松）", "度规识别身份句 h_00/2 = theta"):
    mark = "承重（公理级缺口）" if "身份句" in a else "已有读数或来源"
    print(f"   - {a}：{mark}")
