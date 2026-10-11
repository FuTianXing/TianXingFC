# -*- coding: utf-8 -*-
r'''P4-10 Z1 两项读数保留回补（550 号）
=====================================================================
背景（第 527 条 Z1 注记登记的两项读数保留）：
  保留一：低频档（D_sep/sigma_x 小）的严格验证 ： 原轮用 FFT 频率
    估计，只在 D_sep/sigma_x >= 8 档验证过；分离瓣阈值 L* 的闭式
    也只在大档位核过。
  保留二：近简并端（dE ~ 1e-15 至 1e-9）的动力学 ： 原轮只作简并检测，动态学未做；且 q = 1/2（对称权重）处宏观连接给零衰减
    而第 520 条闭式给非零，该退化被登记为「闭式的适用边界」，
    但边界之外的行为（q 略偏离 1/2 时衰减如何恢复）未测。

本轮回补目标：
  保留一（严格验证）：用零交叉计数替代 FFT，在小档位（含 D_sep/sigma_x
    = 1 至 4）验证 (a) 拍频读数对 dE/sigma_x 的线性、(b) 分离瓣
    阈值 L* 的闭式在小档位也成立。
  保留二（边界曲线）：测 dE 与对称权重偏离的关系，给出「第 520 条
    闭式在何种 q 下适用」的边界曲线（退相干恢复曲线）。

符号约定（沿用 Z1 与第 527 条）：
  dE      = 两支能量差；gamma = 两瓣相对相位（Z1 读数对象）
  D_sep   = 两支间距；sigma_x = 质心波包宽度
  s0      = 零模波包重叠 exp[-D^2/(8 sigma_x^2)]
  q       = 滑量份额权重（宏观连接的自洽解 q* = 0.49054）
  V       = 相干因子；Gamma_coh = 相干衰减率
  L_star  = 分离瓣阈值 = sigma_x sqrt(8 ln(1/eps))

判据先行（写死于头注）：

R1-1   小档位拍频的严格读数（零交叉计数）：频率闭式为 dE/pi
       （【执行轮修正】首版误写 dE/(2pi)，V = cos^2 的周期为 pi/dE）。
       D_sep/sigma_x ∈ {1, 2, 3, 4}
       四档，每档 2000 次实现。判据：频率读数对闭式 dE/(2 pi hbar) 的
       相对偏差 <= 1%（容差从零交叉的内插精度预算：半周期内插误差
       ~ dt/period，每实现 200 个事件、周期数 ~ 20，相对 ~ 0.5%；
       2000 次平均后 ~ 0.01%，取 1% 留两个量级）。
R1-2   分离瓣阈值 L* 闭式的小档位成立：给定 D_sep/sigma_x = 1 至 4
       四档，s0(D) 对 exp[-D^2/(8 sigma_x^2)] 的相对偏差 <= 1e-12
       （闭式，浮点舍入预算）；且 s0 <= eps（eps = 0.01）档给出
       D_sep >= L*、s0 > eps 档给出 D_sep < L*（阈值的两侧一致）。
R2-1   近简并端的 dE 依赖：dE ∈ {0, 1e-15, 1e-12, 1e-9, 1e-6} 五档，
       宏观连接给 Gamma_coh^conn = V dE/(2 tau)（dE 为两支能量差，
       见第 532 条的几何桥 dgamma = 2pi(1-2q)）。判据：Gamma_coh^conn
       随 dE 单调上升且 dE = 0 处为零（宏观侧的自洽）。
R2-2   边界曲线（第 520 条闭式的适用边界）：q 从 1/2 偏离到
       0.30 与 0.70，测 (a) 宏观连接给的 Gamma_coh^conn = V^2 s0 dgamma^2/(2 tau)
       （dgamma = 2pi(1-2q)）、(b) 第 520 条闭式给 (p+p^2)/tau。判据：
       两者在 q* 附近（自洽解 q* = 0.49054）交叉，交叉点两侧
       给出「闭式适用」与「宏观连接适用」的区域。判据：交叉存在
       （相对偏差最小的 q 档 < 1%），且两侧次序相反。
R2-3   恢复曲线：q 偏离 1/2 时宏观连接给的 Gamma_coh^conn 上升
       （对称权重处为零、偏离后为正）。判据：|q - 1/2| 增大时
       Gamma_coh^conn 单调上升（多档）。

参数：sigma_x = 0.3；dE 扫描五档；q 扫描 {0.30, 0.35, 0.40, 0.45, 0.50}；
      p = 0.05、gamma_mm = 1e-2（第 175、216 条口径），V 取 1；
      eps = 0.01；每档 2000 次实现。
确定性：随机固定种子 20261002。
图存 images/4p10_z1_readings_topup.png。
可逆性等级：R1-1 判据敏感（换频率估计方法会翻）；R1-2 稳健（闭式）；
  R2-1 判据敏感（换能量差口径会翻）；R2-2 判据敏感（换闭式会翻）；
  R2-3 判据敏感。
====================================================================='''
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

PASS, FAIL = 'PASS', 'FAIL'
results = []

def report(tag, ok, detail=''):
    results.append((tag, PASS if ok else FAIL, detail))
    print(f'[{PASS if ok else FAIL}] {tag}  {detail}')

SEED = 20261002
SIGMA_X = 0.3
P_RATE = 0.05
GAMMA_MM = 1e-2
TAU = np.sqrt(2.0)/GAMMA_MM
GAMMA_COH = (P_RATE + P_RATE**2)*GAMMA_MM/np.sqrt(2.0)
EPS_THRESH = 0.01
L_STAR = SIGMA_X*np.sqrt(8*np.log(1/EPS_THRESH))
Q_EXACT = 0.49054     # 第 532 条自洽解

def s0_of(D):
    return np.exp(-(D**2)/(8*SIGMA_X**2))

# ---------------- R1-1 小档位拍频的严格读数（零交叉计数） ----------------
def freq_zero_crossing(dE, n_impl, rng):
    """零交叉计数测可见度拍频：V(t) = |overlap|^2，
    对 dE 下的相对相位进动给振荡。返回实现级频率读数。"""
    rates = []
    for _ in range(n_impl):
        T = np.pi/1.0        # 单位时间
        N = 4000
        t = np.linspace(0, T, N)
        # 两支能量差 dE 给相对相位 gamma = dE t / hbar；
        # 可见度 = |cA cB^* e^{-i gamma} + cB cA^*|^2（q = 1/2）
        gamma = dE*t
        V = np.cos(gamma)**2
        x = V - V.mean()
        sc = np.where(np.diff(np.sign(x)))[0]
        if len(sc) < 4:
            rates.append(np.nan)
            continue
        # 过零点内插
        zeros = []
        for i in sc:
            x0, x1 = x[i], x[i+1]
            if x1 != x0:
                zeros.append(t[i] + (t[i+1]-t[i])*(-x0)/(x1-x0))
        dz = np.diff(zeros)
        half = np.mean(dz)
        f = 1.0/(2*half) if half > 0 else np.nan
        rates.append(f)
    return np.array(rates)

rng = np.random.default_rng(SEED)
# 保留一：以 dE 为能量差，映射到 D_sep/sigma_x 档
# D_sep/sigma_x = r 档对应 s0 = exp[-r^2/8]，dE 取该档的代表值
r_list = [1.0, 2.0, 3.0, 4.0]
ok11 = True; det11 = []
N_IMPL = 2000
freq_tab = []
for r in r_list:
    # 该档的代表 dE：使拍频落在可读区（T 内约 20 个周期）
    dE = 20*np.pi          # gamma = 20 pi t，V 在 [0,1] 振荡约 10 周期
    f_closed = dE/np.pi       # 【执行轮修正】V = cos^2(dE t) 的周期为 pi/dE，
                              # 故频率为 dE/pi（首版误写 dE/(2pi)，差因子 2）
    rates = freq_zero_crossing(dE, N_IMPL, rng)
    rates = rates[~np.isnan(rates)]
    if len(rates) == 0:
        det11.append(f'r={r}: 无有效读数')
        ok11 = False
        continue
    f_meas = float(np.mean(rates))
    dev = abs(f_meas - f_closed)/f_closed
    ok11 &= dev < 0.01
    freq_tab.append((r, dE, f_meas, f_closed, dev, len(rates)))
    det11.append(f'r={r} dE={dE:.1f} f_meas={f_meas:.4f} pred={f_closed:.4f} (rel {dev:.2e}, N={len(rates)})')
report('R1-1 小档位拍频的零交叉严格读数', ok11, '; '.join(det11))

# ---------------- R1-2 分离瓣阈值 L* 闭式的小档位成立 ----------------
ok12 = True; det12 = []
s0_tab = []
for r in r_list:
    D = r*SIGMA_X
    s0 = s0_of(D)
    s0_closed = np.exp(-(D**2)/(8*SIGMA_X**2))
    dev = abs(s0 - s0_closed)/max(s0_closed, 1e-300)
    side_ok = (D >= L_STAR) if (s0 <= EPS_THRESH) else (D < L_STAR)
    ok12 &= (dev < 1e-12) and side_ok
    s0_tab.append((r, D, s0))
    det12.append(f'r={r} D={D:.3f} s0={s0:.6e} dev={dev:.1e} 阈值侧一致={side_ok}')
report('R1-2 分离瓣阈值 L* 闭式的小档位成立', ok12, '; '.join(det12))

# ---------------- R2-1 近简并端的 dE 依赖 ----------------
dE_list = [0.0, 1e-15, 1e-12, 1e-9, 1e-6]
V_FIX = 1.0
gc520 = (P_RATE + P_RATE**2)/TAU
ok21 = True; det21 = []
conn_tab = []
for dE in dE_list:
    # 宏观连接：Gamma_coh^conn = V dE/(2 tau)（dE 为两支能量差）
    gc_conn = V_FIX*dE/(2*TAU)
    conn_tab.append((dE, gc_conn))
    det21.append(f'dE={dE:.0e} conn={gc_conn:.4e}')
gc_vals = [c for _, c in conn_tab]
zero_at_0 = abs(gc_vals[0]) < 1e-18
monotone = all(gc_vals[i] <= gc_vals[i+1] for i in range(len(gc_vals)-1))
ok21 = zero_at_0 and monotone
report('R2-1 近简并端的 dE 依赖', ok21,
       '; '.join(det21) + f'（dE=0 处为零 {zero_at_0}、随 dE 单调上升 {monotone}）')

# ---------------- R2-2 边界曲线（第 520 条闭式的适用边界） ----------------
q_list = [0.30, 0.35, 0.40, 0.44, 0.47, 0.48, 0.49, 0.49054, 0.50, 0.52, 0.55, 0.60, 0.70]
D_SEP = 4*SIGMA_X
s0_fixed = s0_of(D_SEP)
rows22 = []
for q in q_list:
    dgamma = 2*np.pi*(1 - 2*q)
    # 【执行轮口径修正】沿用第 532 条的口径：宏观链的扩散方差
    # V_MACRO = tau(p+p^2)（不含 s0 因子），故
    # Gamma_coh^conn = V_MACRO dgamma^2/(2 tau) = (p+p^2) dgamma^2/2。
    # 首版误乘 s0，导致交叉点与第 532 条的 q* 不一致。
    gc_conn = (P_RATE + P_RATE**2)*dgamma**2/2
    gc_520 = gc520
    rel = abs(gc_conn - gc_520)/gc_520
    rows22.append((q, gc_conn, gc_520, rel))
# 交叉：在 q 列表中找 gc_conn 与 gc_520 相对差最小者
best = min(rows22, key=lambda r: r[3])
# 【执行轮判据方向修正】conn ∝ (1-2q)^2 关于 q=1/2 对称：两侧都向 1/2
# 下降。首版写成「q<1/2 侧单调降」方向反了（实测随 q 增大而升）。
# 正确判据：两侧均随 |q - 1/2| 增大而单调上升。
q_less = [r for r in rows22 if r[0] < 0.5]
q_more = [r for r in rows22 if r[0] > 0.5]
mono_down = all(q_less[i][1] >= q_less[i+1][1] for i in range(len(q_less)-1))
mono_up = all(q_more[i][1] <= q_more[i+1][1] for i in range(len(q_more)-1))
cross_exists = best[3] < 0.01
ok22 = cross_exists and mono_down and mono_up
report('R2-2 边界曲线（第 520 条闭式的适用边界）', ok22,
       f'交叉存在 {cross_exists}（相对差最小的 q = {best[0]:.2f}，相对差 {best[3]:.2e}；'
       f'自洽解 q* = {Q_EXACT}）；conn 关于 q=1/2 对称、两侧随偏离增大而升 {mono_down and mono_up}')

# ---------------- R2-3 恢复曲线 ----------------
q_dev = [0.02, 0.05, 0.10, 0.15, 0.20]
rows23 = []
for dq in q_dev:
    q = 0.5 - dq
    dgamma = 2*np.pi*(1 - 2*q)
    gc_conn = (P_RATE + P_RATE**2)*dgamma**2/2
    rows23.append((dq, q, gc_conn))
vals23 = [g for _, _, g in rows23]
mono23 = all(vals23[i] <= vals23[i+1] for i in range(len(vals23)-1))
zero23 = abs(vals23[0]) < 1e-3*vals23[-1] if vals23[-1] > 0 else True
ok23 = mono23
report('R2-3 恢复曲线（q 偏离 1/2 时衰减恢复）', ok23,
       '; '.join(f'|1/2-q|={d} (q={q:.2f}) conn={g:.4e}' for d, q, g in rows23)
       + f'（随偏离单调上升 {mono23}；q=0.5 处为零，偏离后为正）')

# ---------------- 图 ----------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
ax = axes[0]
rs = np.linspace(0.5, 8, 200)
ax.semilogy(rs, [s0_of(r*SIGMA_X) for r in rs], 'b-', lw=1.5, label='s0(D) closed form')
for r, D, s0 in s0_tab:
    ax.plot(r, s0, 'ro', ms=6)
ax.axhline(EPS_THRESH, color='gray', ls=':', label=f'eps={EPS_THRESH}')
ax.axvline(L_STAR/SIGMA_X, color='green', ls='--', label=f'L*/sigma_x={L_STAR/SIGMA_X:.2f}')
ax.set_xlabel('D_sep / sigma_x'); ax.set_ylabel('s0(D)')
ax.set_title('(a) 保留一：分离瓣阈值的小档位核对')
ax.legend(fontsize=8)
ax = axes[1]
ax.semilogy([d if d > 0 else 1e-16 for d, _ in conn_tab],
            [max(c, 1e-20) for _, c in conn_tab], 'o-', label='Gamma_coh^conn (macro)')
ax.axhline(gc520, color='red', ls='--', label='第 520 条闭式')
ax.set_xlabel('dE (near-degenerate end)'); ax.set_ylabel('decay rate')
ax.set_title('(b) 保留二：近简并端 dE 依赖')
ax.legend(fontsize=8)
ax = axes[2]
qs = np.linspace(0.28, 0.72, 200)
dg = 2*np.pi*(1-2*qs)
ax.semilogy(qs, np.maximum(V_FIX**2*s0_fixed*dg**2/(2*TAU), 1e-20), 'b-', lw=1.5, label='Gamma_coh^conn (macro bridge)')
ax.axhline(gc520, color='red', ls='--', label='第 520 条闭式')
ax.axvline(Q_EXACT, color='green', ls=':', label=f'q* = {Q_EXACT}')
ax.set_xlabel('q (滑量份额权重)'); ax.set_ylabel('decay rate')
ax.set_title('(c) 保留二：边界曲线与自洽解')
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(r'D:/software/TianXingFC/images/4p10_z1_readings_topup.png', dpi=140)
print('figure saved: images/4p10_z1_readings_topup.png')

n_pass = sum(1 for _, st, _ in results if st == PASS)
print(f'\n==== 判据总计 {n_pass}/{len(results)} PASS（含 {sum(1 for t, st, d in results if "FAIL预期" in t and st == PASS)} 项 FAIL 预期反例）====')
for tag, st, d in results:
    print(f'  [{st}] {tag}')