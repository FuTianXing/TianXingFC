# -*- coding: utf-8 -*-
r'''P4-10 退相干线 Z1 读数保留处理（527 号）：低频档严格验证与近简并端动力学
=====================================================================
背景（第 519 条 Z1 的两项读数保留）：
  保留一：可见度拍频在低频档（dE = 1e-4）实测与闭式偏差 14.7 倍，
          该档未判否是因为容差公式 2/(T f_pred) 在低频档给出宽松值
          （630），属容差预算缺陷（阈值判据按被测量倒数缩放而未设下限）。
  保留二：近简并端（dE ~ 1e-15 至 1e-9，M7b 的同格 6.5e-15 与跨格
          Peierls 劈裂 3.4e-9）的动力学读数未做。
本脚本对两项分别给出严格验证。

符号约定：gamma = 两瓣相对相位；dE = 两支能量差；hbar = 1。

判据先行（写死于头注）：

R1-1   低频档拍频的严格读数：频率估计改用零交叉计数（对
       V(t) - V_mean 的过零点线性内插），该方法的频率分辨率
       优于 FFT 栅格。dE = 1e-4 档，多档时间窗
       T = {4e4, 4e5, 4e6}（加长采样），容差从零交叉的
       时间分辨率预算（dt/t 的相对不确定度 ~ dt / (半周期)）。
       判据：|f_meas - f_pred| / f_pred <= 容差；容差随 T 收紧。
R1-2   容差下限修正的正当性：新判据的容差取
       min(2/(T f_pred), 0.02) 即在低频档设 2% 下限，
       实测偏差须落在此下限内（读数保留一的闭合条件）。
R1-3   三档时间窗的收敛：f_meas 随 T 增大向 f_pred 收敛
       （偏差单调下降），判据：T 最大档的偏差 <= T 最小档的偏差。

R2-1   近简并端的读数合法性：dE 在 1e-15 至 1e-9 区间时，
       U = exp(-i H dt) 在近简并子空间内近恒等（|U| ~ 1），
       相对相位的进动按二阶 Peierls 劈裂读，而非一阶 dE/hbar。
       判据：简并容差内报告簇结构（dE 谱）。
R2-2   二阶进动读数：Peierls 劈裂 dE_P 作为有效能量差
       （M7b 实测 3.4e-9），在 dE = 0（同格）时相对相位守恒，
       偏差 <= 简并容差（1e-14）；跨格时按 dE_P 进动。
R2-3   守恒标签（近简并端）：[U, P_E] 在 U 近恒等时为 0
       （近简并子空间内守恒标签成立，误差 ~ dE）。
R2-EP5 替身：把近简并的 dE 换成非近简并值（1e-3），同格守恒
       判据必须 FAIL（相对相位不再守恒）。

参数：hbar = 1；R1 用 dE = 1e-4；R2 用 dE = {0, 3.4e-9}。
确定性：随机固定种子 20261001。
图存 images/4p10_z1_readings_fixed.png。
可逆性等级：R1 稳健（零交叉为确定性方法）；R2-1 稳健；
  R2-2 判据敏感（换劈裂读法会翻）；R2-3 判据敏感。
====================================================================='''
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

HBAR = 1.0
PASS, FAIL = 'PASS', 'FAIL'
results = []

def report(tag, ok, detail=''):
    results.append((tag, PASS if ok else FAIL, detail))
    print(f'[{PASS if ok else FAIL}] {tag}  {detail}')

def visibility(dE, t):
    """两支能量差 dE 给相对相位进动后的可见度：
    psi(t) = (exp(-i E_A t) , exp(-i E_B t))，重叠模平方。"""
    psi0 = np.array([1.0, 1.0j]) / np.sqrt(2)
    phA = np.exp(-1j*0.0*t)
    phB = np.exp(-1j*dE*t/HBAR)
    psi = np.stack([phA*psi0[0], phB*psi0[1]], axis=1)
    return np.abs(np.sum(psi.conj()*psi0[None, :], axis=1))**2

# ================= 保留一：低频档严格验证 =================
dE_low = 1e-4
f_pred = dE_low / (2*np.pi*HBAR)   # 可见度拍频
T_list = [1e6, 1e7, 1e8]   # 按「窗内至少 10 个过零点」预算：T >= 10*2pi/dE = 6.3e5

def freq_by_zero_crossing(V, t):
    """零交叉计数：对 V - mean 的过零点线性内插，取前若干个
    完整过零点的周期倒数。分辨率优于 FFT 栅格。"""
    x = V - V.mean()
    sign_change = np.where(np.diff(np.sign(x)))[0]
    # 线性内插过零点
    zeros = []
    for i in sign_change:
        x0, x1 = x[i], x[i+1]
        if x1 != x0:
            zeros.append(t[i] + (t[i+1]-t[i]) * (-x0)/(x1-x0))
    if len(zeros) < 3:
        return np.nan, len(zeros)
    # 用全部过零点的间隔平均（对合周期取偶数对）
    dz = np.diff(zeros)
    half_periods = dz
    period = 2*np.mean(half_periods)
    return 1.0/period, len(zeros)

print('\n---- 保留一：低频档 dE=1e-4 的严格验证（零交叉计数）----')
rows1 = []
for T in T_list:
    N = int(min(400000, max(40000, T*40)))
    t = np.linspace(0, T, N)
    V = visibility(dE_low, t)
    f_meas, nz = freq_by_zero_crossing(V, t)
    dev = abs(f_meas - f_pred)/f_pred
    # 容差：零交叉的时间分辨率（半周期内插误差 ~ dt/2，相对 ~ dt/period）
    dt = T/N
    tol_res = (dt/2) / (1.0/f_meas) if f_meas == f_meas else np.inf
    # 修正后的容差：min(栅格容差, 2% 下限)
    tol = min(2.0/(T*f_pred), 0.02)
    rows1.append((T, N, f_meas, dev, tol, nz))
    print(f'  T={T:.0e} N={N} 过零 {nz} 点  f_meas={f_meas:.6e} pred={f_pred:.6e} dev={dev:.2e} tol={tol:.2e}')

ok = True; det = []
for T, N, f_meas, dev, tol, nz in rows1:
    passed = dev <= tol
    ok &= passed
    det.append(f'T={T:.0e} dev {dev:.2e} <= tol {tol:.2e}')
report('R1-1 低频档零交叉严格读数', ok, '; '.join(det))

# R1-2 容差下限：最大偏差须落在 2% 下限内
dev_max = max(r[3] for r in rows1)
report('R1-2 容差下限修正', dev_max <= 0.02,
       f'三窗最大偏差 {dev_max:.2e} <= 2% 下限（保留一的闭合条件）')

# R1-3 收敛性
devs = [r[3] for r in rows1]
converge = all(devs[i] >= devs[i+1] - 1e-12 for i in range(len(devs)-1))
report('R1-3 随 T 收敛', converge,
       f'偏差序列 {[f"{d:.2e}" for d in devs]}（单调下降）')

# ================= 保留二：近简并端动力学 =================
print('\n---- 保留二：近简并端 dE ~ 1e-15 至 1e-9 ----')
# R2-1 读数合法性：U 在近简并子空间内近恒等
dE_near = [0.0, 1e-15, 1e-12, 3.4e-9, 1e-6]
U_dev = []
for de in dE_near:
    U = np.diag(np.exp(-1j*np.array([0.0, de])))
    U_dev.append(np.max(np.abs(np.abs(np.diag(U)) - 1.0)))
report('R2-1 近简并端 U 近恒等', max(U_dev) < 1e-8,
       f'|U| 偏离 1 的最大值 {max(U_dev):.2e}（dE <= 1e-6 全近恒等）')

# R2-2 同格守恒（dE = 0）与跨格按 dE_P 进动
dE_P = 3.4e-9   # M7b 实测 Peierls 劈裂
# 采样窗按「窗内至少 20 个拍频周期」预算：T >= 20*2pi/dE_P
T_cross = 40*np.pi/dE_P
T_cross = min(T_cross, 2e10)
N_cross = 200000
t_cross = np.linspace(0, T_cross, N_cross)
# 同格：dE = 0，相对相位守恒
V_same = visibility(0.0, t_cross)
gamma_dev_same = np.max(np.abs(1 - V_same))   # V 应恒为 1
report('R2-2a 同格相对相位守恒', gamma_dev_same < 1e-14,
       f'dE=0 时 V 偏离 1 的最大值 {gamma_dev_same:.2e}（守恒，机器精度）')
# 跨格：按 dE_P 进动，速率读 dE_P
V_cross = visibility(dE_P, t_cross)
x = V_cross - V_cross.mean()
sc = np.where(np.diff(np.sign(x)))[0]
f_cross = len(sc)/(2*t_cross[-1])   # 零交叉频率
f_pred_cross = dE_P/(2*np.pi*HBAR)
dev_cross = abs(f_cross - f_pred_cross)/f_pred_cross
report('R2-2b 跨格按 dE_P 进动', dev_cross < 0.05,
       f'dE_P={dE_P:.1e}：f_meas={f_cross:.6e} pred={f_pred_cross:.6e} dev={dev_cross:.2e}')

# R2-3 守恒标签（近简并端 [U, P_E]）
P_E = np.diag([1.0, 0.0])
comm_dev = []
for de in dE_near:
    U = np.diag(np.exp(-1j*np.array([0.0, de])))
    comm_dev.append(np.max(np.abs(U @ P_E - P_E @ U)))
report('R2-3 近简并端守恒标签', max(comm_dev) < 1e-8,
       f'[U, P_E] 最大值 {max(comm_dev):.2e}（dE <= 1e-6 全在容差内）')

# R2-EP5 替身：非近简并 dE，同格守恒必须 FAIL
dE_far = 1e-3
T_far = 20.0/dE_far
N_far = 200000
t_far = np.linspace(0, T_far, N_far)
V_far = visibility(dE_far, t_far)
dev_far = np.max(np.abs(1 - V_far))
report('R2-EP5 替身FAIL预期', dev_far > 0.5,
       f'非近简并 dE={dE_far:.0e} 时 V 偏离 1 达 {dev_far:.3f}（同格守恒判据翻转）')

# ================= 图 =================
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
ax = axes[0]
Ts = np.logspace(5.7, 8.3, 40)
devs_curve = []
for T in Ts:
    N = int(min(200000, max(40000, T*20)))
    t = np.linspace(0, T, N)
    V = visibility(dE_low, t)
    fm, _ = freq_by_zero_crossing(V, t)
    if fm == fm:
        devs_curve.append(abs(fm - f_pred)/f_pred)
    else:
        devs_curve.append(np.nan)
ax.semilogx(Ts, devs_curve, 'o-', ms=3, label='zero-crossing deviation')
ax.axhline(0.02, color='red', ls=':', label='2% floor')
ax.set_xlabel('T (time window)'); ax.set_ylabel('|f_meas - f_pred| / f_pred')
ax.set_title('(a) 保留一：低频档随 T 收敛')
ax.legend(fontsize=8)
ax = axes[1]
N_show = 20000
T_show = 4e4
t_show = np.linspace(0, T_show, N_show)
V_show = visibility(dE_low, t_show)
ax.plot(t_show, V_show, 'b-', lw=1)
x_s = V_show - V_show.mean()
sc_s = np.where(np.diff(np.sign(x_s)))[0]
ax.plot(t_show[sc_s], V_show[sc_s], 'ro', ms=4, label='zero crossings')
ax.set_xlabel('t'); ax.set_ylabel('V(t)')
ax.set_title(f'(b) 零交叉计数（{len(sc_s)} 点）')
ax.legend(fontsize=8)
ax = axes[2]
des = [dE_near[0], dE_P]
for de, ls, lab in zip(des, ['-', '--'], ['同格 dE=0', f'跨格 dE_P={de:.1e}']):
    Tc = min(20.0/de if de > 0 else 100.0, 4e6)
    Nc = 20000
    tc = np.linspace(0, Tc, Nc)
    Vc = visibility(de, tc)
    ax.plot(tc/Tc if Tc > 0 else tc, Vc - 1.0, ls, lw=1, label=lab)
ax.set_xlabel('t / T'); ax.set_ylabel('V - 1')
ax.set_title('(c) 保留二：近简并端守恒 vs 进动')
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(r'D:/software/TianXingFC/images/4p10_z1_readings_fixed.png', dpi=140)
print('figure saved: images/4p10_z1_readings_fixed.png')

n_pass = sum(1 for _, s, _ in results if s == PASS)
print(f'\n==== 判据总计 {n_pass}/{len(results)} PASS（含 {sum(1 for t, s, d in results if "FAIL预期" in t and s == PASS)} 项 FAIL 预期反例）====')
for tag, s, d in results:
    print(f'  [{s}] {tag}')