# -*- coding: utf-8 -*-
r"""
minimal_verification_negative_curvature_conversion.py

[构造轮前置轮] 负曲率的转化路径：三元组第二极小与失稳窗（判据 V1 至 V4，第 594 条）

阶段二裁决（2026-10-11）取构造层档位路线后，构造轮的判据方向需要一次前置修正：
来源乙（异类耦合）的负曲率在第 591 条判为"结构层命中"，但负曲率的物理身份是
失稳通道（恢复力翻负），不是直接的负能量项。本判据把转化路径的解析结构钉下来，
防止构造轮第 1 轮判错对象（找"单项负能量"而非"新相化总账"）。

模型（第 591 条 JB 同构，三元组精确能）
    V(δ) = -J cos(√2 δ) - 2J cos(δ/√2)
  三对异类耦合（一对配对角 √2δ、两对 δ/√2），J = 1 为模型单位，真空 δ = 0
  给 V = -3J（取相对真空读数时减去 -3J）。曲率 V''(δ) = J g(δ)，
  g(δ) = 2cos(√2δ) + cos(δ/√2)，δ_c = 1.323604 变号（第 591 条读数）。

判据（先立解析预期再实现）

V1（第二极小的解析确认）
  解析：V'(δ) = J√2[sin(√2δ) + sin(δ/√2)] = 0 的非零解两支：
    支一 δ = (2√2/3)kπ（极大），支二 δ = √2π(1+2k)（k = 0 给 δ* = √2π）。
  V''(δ*) = J g(√2π) = J[2cos(2π) + cos(π)] = J > 0：极小。
  V(δ*) = -J·cos(2π) - 2J·cos(π) = -J + 2J = +J（相对真空 -3J 的代价为 4J）。
  数值：扫描加抛物线细化定位极小，判位置对 √2π = 4.442883、能量对 +J
  （机器精度档）。

V2（失稳窗）
  解析：V'' < 0 即 g(δ) < 0，窗 = (δ_c, δ_2)，δ_2 为 g = 0 的上端点。
  数值二分 δ_2，对照解析（g = 0 无闭式，数值为准，报八位）。

V3（单对耦合无第二极小，区分力对照）
  单对 V_1(δ) = -J cos δ：极小仅 δ = 0（周期 2π 内），无失稳后承接点。
  三元组的第二极小是三对合成（两种配对角不可通约）的结构特征，非平凡。
  数值扫描确认单对在 (0, 2π) 无第二极小。

V4（异类项相对真空恒正，负线项不来自单项的解析证明）
  解析：cos ≤ 1 逐点成立，故 V(δ) = -J cos(√2δ) - 2J cos(δ/√2) ≥ -3J，
  等号仅 δ = 0。任何 δ ≠ 0 的芯区构型，异类项相对真空都是正贡献。
  数值：随机构型抽查（固定种子）确认 V > -3J 严格成立，反例即为违反。
  结论面：构造轮的能量账必须按"新相化总账"立判据（异类代价 +4J/格点
  量级 对 键拉伸节省），负线项若成立只能来自联合构型的再分布，不来自
  异类项单项符号。

可逆性：判据稳健（解析恒等式与扫描确认，不依赖参数档）。
模型单位：J = 1。确定性，无随机数（V4 抽查用固定种子伪随机）。
运行：python 验证脚本/minimal_verification_negative_curvature_conversion.py
"""
import math
import time

import numpy as np

T_START = time.time()
J = 1.0
SQRT2 = math.sqrt(2.0)
DELTA_STAR = SQRT2 * math.pi   # 4.442882938...

print('=' * 76)
print('构造轮前置轮：负曲率的转化路径（第 594 条，V1 至 V4）')
print('=' * 76)
print()
print('三元组精确能 V(δ) = -J cos(√2δ) - 2J cos(δ/√2)（第 591 条 JB 同构）')
print('裁决前置修正：负曲率是失稳通道非负能量项，构造轮判新相化总账。')
print()


def V(delta):
    return -J * np.cos(SQRT2 * delta) - 2.0 * J * np.cos(delta / SQRT2)


def g(delta):
    return 2.0 * J * np.cos(SQRT2 * delta) + J * np.cos(delta / SQRT2)


# ---------- V1：第二极小 ----------
print('--- V1（第二极小的解析确认）---')
n = 2000001
d_grid = np.linspace(0.0, 2.0 * math.pi / SQRT2 * 2.2, n)
v_grid = V(d_grid)
# 在 (π, 2π) 内找局部极小（避开 δ=0 主极小）
mask = (d_grid > math.pi) & (d_grid < 2.0 * math.pi)
i_loc = np.argmin(np.where(mask, v_grid, np.inf))
# 抛物线细化
lo, hi = d_grid[max(i_loc - 2, 0)], d_grid[min(i_loc + 2, n - 1)]
for _ in range(80):
    mid1 = lo + (hi - lo) / 3.0
    mid2 = hi - (hi - lo) / 3.0
    if V(mid1) < V(mid2):
        hi = mid2
    else:
        lo = mid1
d_num = (lo + hi) / 2.0
v_num = float(V(d_num))
err_pos = abs(d_num - DELTA_STAR)
err_v = abs(v_num - J)
print('   解析极小：δ* = √2π = %.9f，V(δ*) = %+.6f（相对真空代价 4J）' % (DELTA_STAR, J))
print('   数值极小：δ = %.9f，V = %+.9f' % (d_num, v_num))
print('   偏差：位置 %.2e，能量 %.2e' % (err_pos, err_v))
# 容差预算：三分法在曲率 J 的平坦极小处定位受浮点极限 sqrt(eps)≈1.5e-8 限制
#（首版设 1e-8 未做预算，实测 1.97e-08 恰为该极限，属容差设错非物理信号）
v1_ok = err_pos < 1e-6 and err_v < 1e-8
print('   [%s] V1' % ('PASS' if v1_ok else 'FAIL'))
print()

# ---------- V2：失稳窗 ----------
print('--- V2（失稳窗 V\'\' < 0 = g < 0）---')
def bisect_zero(f, a, b, it=200):
    fa, fb = f(a), f(b)
    assert fa * fb < 0
    for _ in range(it):
        m = 0.5 * (a + b)
        fm = f(m)
        if fa * fm < 0:
            b, fb = m, fm
        else:
            a, fa = m, fm
    return 0.5 * (a + b)

d_lo, d_hi = 1.0, 3.0
d_c = bisect_zero(g, d_lo, d_hi)          # g 降过零（下端）
d_2 = bisect_zero(g, 4.0, 5.5)            # g 升过零（上端）
print('   失稳窗：δ ∈ (%.8f, %.8f)，宽度 %.8f' % (d_c, d_2, d_2 - d_c))
print('   对照：δ_c = 1.323604（第 591 条读数）')
v2_ok = abs(d_c - 1.323604) < 1e-6
print('   [%s] V2（δ_c 与既有读数逐位相符；δ_2 = %.8f 为新读数）'
      % ('PASS' if v2_ok else 'FAIL', d_2))
print()

# ---------- V3：失稳后的承接极小（区分力对照） ----------
print('--- V3（失稳后的承接极小：单对无、三元组有）---')
# 判据：在 (0.1, 周期-0.1) 内找 V' = 0 的驻点并判 V'' 符号。
# 单对 -J cos δ：驻点仅 δ = π（极大），失稳（δ > π/2）后滑回 δ = 2π 真空周期像，
# 芯区不驻留新相；三元组：δ* = √2π 为 V'' > 0 的真极小，失稳后有承接点。
def dV(f, d, h=1e-6):
    return (f(d + h) - f(d - h)) / (2 * h)

def stationary_curvature(f, a, b, npts=400000):
    d1_ = np.linspace(a, b, npts)
    dv = dV(f, d1_)
    sign_change = np.where(dv[:-1] * dv[1:] < 0)[0]
    out = []
    for i in sign_change:
        lo, hi = d1_[i], d1_[i + 1]
        for _ in range(60):
            m = 0.5 * (lo + hi)
            if dV(f, lo) * dV(f, m) <= 0:
                hi = m
            else:
                lo = m
        d0 = 0.5 * (lo + hi)
        curv = (f(d0 + 1e-5) - 2 * f(d0) + f(d0 - 1e-5)) / 1e-10
        out.append((d0, curv))
    return out

st1 = stationary_curvature(lambda d: -J * np.cos(d), 0.1, 2 * math.pi - 0.1)
st3 = stationary_curvature(V, 0.1, 2 * SQRT2 * math.pi - 0.1)
st1_txt = [('δ=%.4f 曲率%+.3f' % (d0, c)) for d0, c in st1] if st1 else ['无']
print('   单对 (0, 2π) 内驻点：%s' % st1_txt)
st3_txt = [('δ=%.4f 曲率%+.3f' % (d0, c)) for d0, c in st3]
print('   三元组 (0, 2√2π) 内驻点：%s' % st3_txt)
v3_ok = (all(c < 0 for _, c in st1)
         and any(c > 0 and abs(d0 - DELTA_STAR) < 1e-3 for d0, c in st3))
print('   [%s] V3（单对失稳后无承接极小、三元组有：第二极小依赖两种配对角的合成）'
      % ('PASS' if v3_ok else 'FAIL'))
print()

# ---------- V4：半周期内异类项相对真空恒正 ----------
print('--- V4（半周期内异类项相对真空恒正，解析 cos ≤ 1 逐点）---')
# 周期结构：V(δ + 2√2π) = V(δ)（两种配对角同返，与第 591 条 JB-1 周期 2π√2 一致），
# 等能点 δ = 2√2π·m 为真空周期回归。判据范围取半周期 (0, √2π)（芯区 δ 的物理量级）：
# 其内 V > V(0) = -3J 严格成立（cos ≤ 1 逐点，等号仅 δ = 0）。
rng = np.random.default_rng(20261011)
worst = 1e9
for trial in range(200000):
    delta = float(rng.uniform(1e-6, SQRT2 * math.pi - 1e-6))
    excess = float(V(delta)) + 3.0 * J
    worst = min(worst, excess)
half = np.linspace(1e-6, SQRT2 * math.pi - 1e-6, n)
grid_min = float(np.min(V(half)) + 3.0 * J)
# 周期回归核对：V(2√2π) 对 V(0)
period_check = abs(float(V(2 * SQRT2 * math.pi)) + 3.0 * J)
print('   周期 2√2π = %.6f 核对：V(2√2π) - V(0) = %.2e（应 0）' % (2 * SQRT2 * math.pi, period_check))
print('   随机构型 200000 例（半周期内）：最小超出量 = %.3e（应 > 0）' % worst)
print('   半周期细扫描（%d 点）：最小超出量 = %.3e（应 > 0）' % (n, grid_min))
v4_ok = worst > 0 and grid_min > 0 and period_check < 1e-10
print('   [%s] V4（半周期内任何 δ ≠ 0 的异类项相对真空恒正；负线项若成立只能来自'
      '新相化与键拉伸节省的联合再分布，构造轮按总账立判据）' % ('PASS' if v4_ok else 'FAIL'))
print()

# ---------- 汇总 ----------
print('=' * 76)
print('汇总（第 594 条，构造轮前置轮）')
print('=' * 76)
print('V1 第二极小：δ* = √2π ≈ 4.442883，V(δ*) = +J（相对真空代价 4J）— 解析加数值逐位')
print('V2 失稳窗：(%.6f, %.6f)，上端点为新读数' % (d_c, d_2))
print('V3 单对无第二极小：第二极小是三元组合成结构特征（区分力成立）')
print('V4 异类项相对真空恒正：负线项不来自单项符号，构造轮判新相化总账')
print()
print('运行时间 %.2f 秒' % (time.time() - T_START))
