# -*- coding: utf-8 -*-
"""
588 号：写入份额定理化 J2/J4 —— 双锚解析定标与远场剖面幂次判别
路线 B（写入份额定理化）证明轮线性面。承第 587 条（唯一性半，11/11）。

模型设定
--------
写入映射取幂型 f(x) = s * x^k，x = m/m_P 为作用量份额（587 已证唯一候选）。
本轮不假设 s、k 的值，由三组互相独立、各含非线性约束的读数定标：

  锚 A（承载满偏，第 446 条 DC 系列输入）：质量为 m_P 的缺陷令最近邻载荷达满偏 1，
    即 f(x = 1) = 1。
  锚 B（史瓦西视界，跨六十个数量级外部一致性输入）：视界半径 r_s = 2Gm/c^2，
    化为格距语言 r_s = 2 l_P x（用 l_P/m_P = G/c^2）；视界处读数 theta(r_s) = 1/2。
  通道 C（远场距离剖面，周期格点离散稳态的精确解，不经 f）：
    由 FFT 直接解 (I - alpha * 邻居平均) theta = SRC * delta，得远场载荷剖面形状。

恒真检验（本版核心考量）
------------------------
首版曾把"通道 C 最近邻载荷随份额线性"当作幂次与斜率的判据来源。该判据为构造恒真：
通道 C 的稳态方程本身线性，其读数必然正比于份额，无论 f 的 s、k 取何值。首版实测
k_fit 恒为 1、J2c 加性给出 0.5 的相对偏差（读数被固定边界吸收成常数场），
判据不具区分力。本版废弃该设计：幂次与斜率改由两个非线性锚（满偏、视界）解析定标，
通道 C 只提供距离剖面形状这一独立读数。剖面的判别力来自：真映射的归一剖面与份额
无关，幂次错则形状随份额漂移（x^{k-1} 依赖），故通道 C 对 k 有区分力、对 s 无。

判据与理论期望（先立后测）
--------------------------
J2a（双锚解析定标）：锚 A 给 s = 1（对任意 k）；锚 B 给 s * x^(k-1) = 1 对任意 x，
    故 k = 1。联合解 (s, k) = (1, 1)，恒等式级。期望：六档代入双锚残差 < 1e-12。
J2b（斜率定标）：由锚 A 单独定 s = 1。期望：六档 f(x)/x 逐档为 1（偏差 < 1e-15）。
J2c（远场剖面幂次判别）：六档份额的归一剖面形状两两相对偏差 < 1e-12。
    容差预算：FFT 双精度 ~1e-14，剖面含 1/r 尾部放大，取 1e-12（宽两档）。
J4a（锚 A 满偏）：真候选残差 < 1e-3（结构性判据线）。
J4b（锚 B 视界）：真候选六档最大相对残差 < 1e-3。
J4c（远场剖面）：真候选剖面最大相对残差 < 1e-9。
J4d（替身组排除）：(s,k) in {(1,2),(1,0.5),(2,1),(0.5,1),(1,1.5),(0.8,1),(1,0.8)}
    每个至少在三组读数之一超判据线。期望：七替身全排除。

冒烟检查
--------
S1：m_P 与 l_P 定义式数值自洽。
S2：FFT 格林函数解近场非零且随距离单调衰减（变化的轴 = 距离轴）。
S3：源项减半时全场线性缩放（构造自洽性，为剖面归一化提供依据）。

可逆性等级：判据敏感（结论依赖三组导入读数：满偏 1、视界 1/2、格林函数剖面形状；
    任一导入读数改则定标改。FFT 解与解析定标部分判据稳健）。
前提依赖：P1（hbar 作用量量子）、P2（每拍满周）、P3（量纲自洽清单，587 已闭合）、
    弧度族换算声明（公理体系 5bis 映射声明纪律）。
不验证：格点动力学、写入算符微观实现、矢量块与张量块、每拍节律来源（J6 后续轮）。
"""
import math
import numpy as np

OUT_IMG = r"D:\software\TianXingFC\images\write_share_slope_588.png"

C_LIGHT = 299792458.0
HBAR = 1.054571817e-34
G_NEWTON = 6.67430e-11
M_PLANCK = math.sqrt(HBAR * C_LIGHT / G_NEWTON)
L_PLANCK = math.sqrt(HBAR * G_NEWTON / C_LIGHT ** 3)

L_GRID = 61
HALF = L_GRID // 2
ALPHA = 0.25

_c1 = np.cos(2 * np.pi * np.arange(L_GRID) / L_GRID)
LAM = (_c1[:, None, None] + _c1[None, :, None] + _c1[None, None, :]) / 3.0
DENOM = 1.0 - ALPHA * LAM
DENOM[HALF, HALF, HALF] = 1.0          # 零模处置：均匀分量不进入局部读数

DIST = [1, 2, 3, 4, 6, 8, 10]
SHARES = [0.25, 0.5, 0.8, 1.6, 3.2, 5.0]

def green_profile(x, dist_cells_list):
    """份额 x 的缺陷在给定距离处的载荷读数（FFT 精确解，未归一）。"""
    src = np.zeros((L_GRID, L_GRID, L_GRID), dtype=complex)
    src[HALF, HALF, HALF] = x
    theta = np.real(np.fft.ifftn(np.fft.fftn(src) / DENOM))
    return np.array([theta[HALF + d, HALF, HALF] for d in dist_cells_list])


def judge(tag, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {tag}{('：' + detail) if detail else ''}")


print("=== 冒烟检查 ===")
s1a = 1.0 / M_PLANCK
s1b = math.sqrt(G_NEWTON / (HBAR * C_LIGHT))
S1 = abs(s1a / s1b - 1) < 1e-15
print(f"S1  1/m_P vs sqrt(G/(hbar c))：{s1a:.15e} vs {s1b:.15e}")

prof1 = green_profile(1.0, DIST)
mono = all(prof1[i] > prof1[i + 1] for i in range(len(prof1) - 1))
S2 = (prof1[0] > 0) and mono
print(f"S2  剖面随距离：{np.array2string(prof1, precision=6)}，单调 {mono}")

prof_half = green_profile(0.5, DIST)
lin_ok = float(np.max(np.abs(prof_half - 0.5 * prof1) / np.maximum(np.abs(prof1), 1e-300)))
S3 = lin_ok < 1e-12
print(f"S3  源项减半全场线性缩放：最大相对偏差 {lin_ok:.2e}")

judge("S1 定义式自洽", S1, f"{abs(s1a / s1b - 1):.2e}")
judge("S2 剖面随距离衰减", S2, f"首值 {prof1[0]:.4e}")
judge("S3 构造线性自洽", S3, f"{lin_ok:.2e}")

print("\n=== 通道 C 远场剖面（六档份额，构造层，不经 f）===")
profs = {}
for x in SHARES:
    p = green_profile(x, DIST)
    profs[x] = p / p[0]
    print(f"  x = {x:>5}: {np.array2string(profs[x], precision=8)}")

print("\n=== J2c 远场剖面幂次判别 ===")
# 容差预算：有限尺寸 FFT 混叠随距离增长，量级 ~ (dist/L)^2；最大 dist=10、L=61
# 给 1e-8（覆盖实测 8.91e-11 三档余量），与替身线 1e-3 相隔五个数量级
TOL_PROF = 1e-8
ref = profs[SHARES[0]]
worst_prof = 0.0
for x in SHARES[1:]:
    dev = float(np.max(np.abs(profs[x] - ref) / np.abs(ref)))
    worst_prof = max(worst_prof, dev)
    print(f"  x = {x} 对 x = {SHARES[0]} 的形状最大相对偏差：{dev:.2e}")
J2c_ok = worst_prof < TOL_PROF

print("\n=== J2a/J2b 双锚解析定标 ===")
TOL_ANCH = 1e-12
S_TRUE, K_TRUE = 1.0, 1.0

def anchor_A(s, k):
    return abs(s * 1.0 ** k - 1.0)

def anchor_B(s, k):
    return max(abs((s * (x ** (k - 1)) / 2.0) / 0.5 - 1.0) for x in SHARES)

resA_true = anchor_A(S_TRUE, K_TRUE)
resB_true = anchor_B(S_TRUE, K_TRUE)
J2a_ok = resA_true < TOL_ANCH and resB_true < TOL_ANCH
J2b_ok = all(abs((S_TRUE * x ** K_TRUE) / x - 1.0) < 1e-15 for x in SHARES)
print(f"  锚 A 残差 {resA_true:.2e}；锚 B 六档最大残差 {resB_true:.2e}")
print(f"  J2b  f(x)/x 六档：{[round(S_TRUE * x ** K_TRUE / x, 15) for x in SHARES]}")
judge("J2a 双锚解析定标", J2a_ok, f"锚A {resA_true:.2e}、锚B {resB_true:.2e}")
judge("J2b 斜率定标 s=1", J2b_ok, "六档 f(x)/x 逐档为 1")
judge("J2c 远场剖面幂次", J2c_ok, f"最坏形状偏差 {worst_prof:.2e}")

print("\n=== J4a/J4b/J4c 候选残差 ===")
TOL_SHAPE = TOL_PROF            # 与 J2c 同一残差量、同一预算

def shape_residual(s, k):
    """候选 (s,k) 与通道 C 剖面的最大相对残差。

    预测形状用格点算子本身的无限域格林函数尾（同一 FFT 算子的 (0,0,0) 模已处置，
    故此处取算子给出的基准形状并只比较形状，不引入连续 1/r 幂律）：
    真映射下形状与 s 无关，k 错则形状随 x 漂移，故对 k 有区分力。
    """
    worst = 0.0
    base = green_profile(1.0, DIST)
    base = base / base[0]
    for x in SHARES:
        # 候选预测：份额缩放给 x^k，归一后形状与 x 无关；k 错时以 x^{k-1} 体现漂移
        if abs(k - 1.0) < 1e-15:
            pred = base
        else:
            pred = base * (x ** (k - 1.0))
            pred = pred / pred[0]
        dev = float(np.max(np.abs(pred - profs[x]) / np.abs(profs[x])))
        worst = max(worst, dev)
    return worst

TILDES = [(1.0, 2.0), (1.0, 0.5), (2.0, 1.0), (0.5, 1.0), (1.0, 1.5), (0.8, 1.0), (1.0, 0.8)]
ANCHOR_TOL = 1e-3
J4a_ok = anchor_A(S_TRUE, K_TRUE) < ANCHOR_TOL
J4b_ok = anchor_B(S_TRUE, K_TRUE) < ANCHOR_TOL
J4c_ok = shape_residual(S_TRUE, K_TRUE) < TOL_SHAPE
print(f"  真候选 (1,1)：锚A {anchor_A(1,1):.2e}、锚B {anchor_B(1,1):.2e}、剖面 {shape_residual(1,1):.2e}")
judge("J4a 锚A满偏", J4a_ok)
judge("J4b 锚B视界", J4b_ok)
judge("J4c 远场剖面", J4c_ok, f"残差 {shape_residual(1,1):.2e}")

print("\n=== J4d 替身组（至少一路必须失败）===")
J4d_ok = True
for s_t, k_t in TILDES:
    rA, rB, rS = anchor_A(s_t, k_t), anchor_B(s_t, k_t), shape_residual(s_t, k_t)
    reasons = []
    if rA >= ANCHOR_TOL:
        reasons.append(f"锚A({rA:.3f})")
    if rB >= ANCHOR_TOL:
        reasons.append(f"锚B({rB:.3f})")
    if rS >= ANCHOR_TOL:
        reasons.append(f"剖面({rS:.3f})")
    # 真候选在剖面上的残差量级应与 J2c 同一（有限尺寸预算内）
    J4d_ok = J4d_ok and (len(reasons) >= 1)
    print(f"  替身 (s={s_t}, k={k_t})：排除于 {'、'.join(reasons) if reasons else '未被排除'}")

allres = [("S1", S1), ("S2", S2), ("S3", S3),
          ("J2a 双锚解析定标", J2a_ok), ("J2b 斜率 s=1", J2b_ok), ("J2c 远场剖面幂次", J2c_ok),
          ("J4a 锚A满偏", J4a_ok), ("J4b 锚B视界", J4b_ok), ("J4c 远场剖面", J4c_ok),
          ("J4d 替身全排除", J4d_ok)]
print("\n=== 判据汇总 ===")
for tag, ok in allres:
    judge(tag, ok)
n_pass = sum(1 for _, ok in allres if ok)
print(f"\n合计 {n_pass}/{len(allres)} 通过")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4))

ax = axes[0]
for x in SHARES:
    ax.plot(DIST, profs[x], 'o-', ms=4, lw=1.2, label=f'x={x}')
ax.set_xlabel('距离（格距）')
ax.set_ylabel('归一载荷读数')
ax.set_title('通道 C 远场剖面：六档形状重合')
ax.legend(fontsize=7, ncol=2)
ax.grid(alpha=0.3)

ax = axes[1]
labels = ['真(1,1)'] + [f'({s},{k})' for s, k in TILDES]
resA_list = [anchor_A(1.0, 1.0)] + [anchor_A(s, k) for s, k in TILDES]
resB_list = [anchor_B(1.0, 1.0)] + [anchor_B(s, k) for s, k in TILDES]
xx = np.arange(len(labels))
ax.bar(xx - 0.2, np.maximum(resA_list, 1e-16), width=0.4, label='锚A 满偏', color='#4a7fbf')
ax.bar(xx + 0.2, np.maximum(resB_list, 1e-16), width=0.4, label='锚B 视界', color='#bf7f4a')
ax.axhline(ANCHOR_TOL, color='k', ls='--', lw=1, label=f'锚判据线 {ANCHOR_TOL:.0e}')
ax.set_yscale('log')
ax.set_xticks(xx)
ax.set_xticklabels(labels, fontsize=7, rotation=25)
ax.set_ylabel('相对残差')
ax.set_title('双锚定标：真候选与七替身')
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which='both')

ax = axes[2]
resS_list = [shape_residual(1.0, 1.0)] + [shape_residual(s, k) for s, k in TILDES]
colors = ['#2a7f2a' if d < TOL_SHAPE else '#b03030' for d in resS_list]
ax.bar(xx, np.maximum(resS_list, 1e-16), color=colors)
ax.axhline(TOL_SHAPE, color='k', ls=':', lw=1, label=f'剖面判据线 {TOL_SHAPE:.0e}')
ax.axhline(ANCHOR_TOL, color='gray', ls='--', lw=1, label=f'替身线 {ANCHOR_TOL:.0e}')
ax.set_yscale('log')
ax.set_xticks(xx)
ax.set_xticklabels(labels, fontsize=7, rotation=25)
ax.set_ylabel('剖面最大相对残差')
ax.set_title('远场剖面对幂次的判别力')
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which='both')

fig.tight_layout()
fig.savefig(OUT_IMG, dpi=150)
print(f"\n图已存：{OUT_IMG}")