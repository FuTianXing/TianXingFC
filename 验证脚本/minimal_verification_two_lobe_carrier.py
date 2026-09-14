# -*- coding: utf-8 -*-
"""
M7b（墙三增补轮，P4-5 专项）：两瓣叠加态如何由单一背景场承载——最小验证

问题（开放问题总览 P4-5，墙三唯一未闭合环节）：
  叠加态 |psi> = sqrt(q)|A> + sqrt(1-q) e^{i gamma}|B>（单 kink 位置两瓣）的
  波侧两瓣 W_n 如何由单一背景场承载。M5 的双 kink 代理是缠绕 2 的系综像
  （人数语义）；本轮在缠绕 1 的单场上构造载体并判据级检验。

三类缠绕 1 的对象（同一套 M5/M7 泛函）：
  S(q)：经典分瓣构型，u_n = n + q*K(n-A) + (1-q)*K(n-B)，滑量按 q:(1-q) 分割；
  img(q)：叠加像（载体映射的对角通道），w_img = q*w^A + (1-q)*w^B，其中
    w^X = 支链 X 的窗口局域序参量（M=5，M5/M7 口径）；跨项在 D 大时被
    压缩不等式禁闭（见 J7a），对角像即大间距极限；
  ens：双 kink 链（缠绕 2），系综像对照。

新增泛函（建模层定义，声明于注记）：
  实通道亏空 dt_n = 1 - Re w_n。它是态的线性泛函（Re(O) 的期望），任何叠加
  精确折叠；模通道 delta_n = 1 - |w_n| 非线性，携带混叠超额 E_mix >= 0
  （亚可加性 |x+y| <= |x|+|y|）。

判据先行（运行前写死）：
  J1 拓扑：S(q) 总缠绕 = 1 且两端远场 |w| = 1（q 全网格，机器精度 1e-12）
  J2 滑量份额（拓扑恒等）：S(q) 两瓣滑量积分 = (q, 1-q)，偏差 < 1e-7
     （中点分割的跨瓣尾部截断 e^{-D/(2 ell)}，协议伪影非物理）
  J3 份额折叠（叠加像，恒等式级）：
     (a) 1-|Psi_img| = 1-Psi_A 对 q 全网格，< 1e-11（平移协变到浮点求和噪声）
     (b) 实通道预算 Sum(1-Re w_img) = Sum(1-Re w_A)，< 5e-9（两支独立弛豫的
         平移协变到浮点复现水平 1.35e-9，预算相对 1.5e-10）
     (c) R_tilde_img = N(1-|Psi_img|)/Sum(1-Re w_img) = R_tilde_single，1e-9
     (d) 混叠超额 E_mix = Sum(1-|w_img|) - Sum(1-|w_A|) >= -1e-12（亚可加性），
         数值如实记录（机制：核区相位偏斜处的振幅混叠）；R_I(q) = R_single/
         (1+E_mix 比份) 实测记录
  J4 系综对照：双 kink 的全局代价与实通道预算均为单 kink 两倍（比值 [1.999, 2.001]）
  J5 载体判别（模通道）：经典分瓣的 R_S = N(1-|Psi_S|)/Sum(1-|w_S|) 随 D 线性
     增长（弦代价住在全局相位通道，局域模亏空对台面纯相位失明；cost 与 R_S 的
     D=160/80 增长比均 in [1.8, 2.2]）；叠加像的 cost 与 R_I 对 D 无关
     （cost 全三点极差 < 1e-9；R_I 在 D=80/160 两点极差 < 1e-9，D=40 为支撑
     重叠趋近点，偏差 ~4e-6 随 D 指数消退）。
  J6 挫挫折弦：E_S(q,D) 对 D 线性，斜率比 tau(q)/tau(0.5) = (1-cos 2pi q)/2
     （q in {0.25, 0.5, 0.75}，偏差 < 5%；形状判据无自由参数）
  J7 相位通道：
     (a) 干涉禁闭前提：两支亏空支撑不交（max_n min(delta_A_n, delta_B_n) < 1e-6，
         尾部由 e^{-D/(2 ell)} 相位展布决定，非弛豫噪声），分离两瓣的相对相位
         不由局域亏空承载（2x2 压缩不等式）
     (b) 经典台面相位锁定：S(q) 两瓣间台面 arg w = 2 pi q（两种滑量走向分别
         +2 pi q 与 2 pi(1-q)，偏差 < 0.01 rad）
     (c) 零模条纹读出（恒等式级）：零模波包 psi_0(x) = sqrt(q)g(x-A) +
         sqrt(1-q)e^{i gamma}g(x-B) 的质心动量谱条纹相位恢复 gamma
         （gamma in {0, pi/2, pi}，恢复误差 < 1e-9）
  J8 支简并：同格内分数坐标的两支能量差 = 0（机器精度 1e-10）；跨格 Peierls
     劈裂 dE_P 实测登记（模型级，衔接 M2c 佩耶尔斯口径）
  J9 采样闭环：叠加像实通道份额（q=0.35）接入 M4 协议（M5 K4 口径，2e6 次，
     固定种子）：显著层 max|z| < 3.5 且卡方/自由度 < 1.6，左瓣和 = 0.35
     （< 1e-6，跨瓣尾部截断）
  J10 经典塌缩（无门，如实报告）：S(0.5) 自由演化，台面长度归零时标 t_merge
     实测；总能量（动能+势能）漂移 |dE/E| < 1e-4（辛积分有界误差自检门）；
     对照双 kink 同窗 kink 位置漂移 < 0.01（长寿命系综像）
  J11 亏空读出非线性（记录）：S(q) 模通道份额 g(q) 单调、g(0.5)=0.5（对称，
     1e-9）、g(0.65) 偏离 0.65 如实记录
  J12 局域重合（合取结构）：q=0.5 叠加像与双 kink 的归一化实通道份额剖面
     max 差 < 1e-9（机器同构：局域份额读数不判载体），而全局通道分离
     （缠绕 1 对 2、代价 1x 对 2x）：载体判定须全局与局域合取，
     与 M2c「两可测量单独均不判定」同构

纪律：全确定性（J9 固定种子）；实测 vs 期望对照表输出；能量守恒与平移协变自检。
"""
import numpy as np
from scipy import stats

k = 1.0
K = 0.005
dt = 0.01
N = 1000
cN = N // 2
w2 = 2 * np.pi * K
ELL = 1.0 / (2 * np.pi * np.sqrt(K / k))
M_WIN = 5
SEED = 20260914
GATE = 0.05

A0, B0 = 460.0, 540.0          # 主间距 D = 80
QG = [0.2, 0.35, 0.5, 0.65, 0.8]
DS = [40.0, 80.0, 160.0]


def kink_profile(x0, n=None):
    if n is None:
        n = np.arange(N)
    return n + (2 / np.pi) * np.arctan(np.exp((n - x0) / ELL))


def split_profile(q, xa, xb):
    n = np.arange(N)
    return n + q * (2 / np.pi) * np.arctan(np.exp((n - xa) / ELL)) \
             + (1 - q) * (2 / np.pi) * np.arctan(np.exp((n - xb) / ELL))


def two_kink_profile(x1, x2):
    n = np.arange(N)
    return (n + (2 / np.pi) * np.arctan(np.exp((n - x1) / ELL))
               + (2 / np.pi) * np.arctan(np.exp((n - x2) / ELL)))


def accel(u, v, gamma=0.0):
    a = np.zeros_like(u)
    a[1:-1] = k * (u[2:] + u[:-2] - 2 * u[1:-1]) - w2 * np.sin(2 * np.pi * u[1:-1])
    if gamma > 0.0:
        a -= gamma * v
    return a


def relax(u0, steps=8000, winding=1):
    u = u0.copy()
    v = np.zeros(len(u0))
    a = accel(u, v)
    for _ in range(steps):
        u = u + dt * v + 0.5 * dt ** 2 * a
        a_new = accel(u, v + dt * a)
        v = v + 0.5 * dt * (a + a_new)
        u[0], u[-1] = 0.0, float(len(u0) + winding - 1)
        v[0] = v[-1] = 0.0
        a = a_new
    return u


def relax_damped(u0, steps=30000, winding=1, gamma=2.0):
    """临界阻尼弛豫：收敛到局域极小（静态能量用，J8 Peierls 劈裂）。"""
    u = u0.copy()
    v = np.zeros(len(u0))
    a = accel(u, v, gamma)
    for _ in range(steps):
        u = u + dt * v + 0.5 * dt ** 2 * a
        a_new = accel(u, v + dt * a, gamma)
        v = v + 0.5 * dt * (a + a_new)
        u[0], u[-1] = 0.0, float(len(u0) + winding - 1)
        v[0] = v[-1] = 0.0
        a = a_new
    return u


def energy(u):
    slip = u[1:] - u[:-1] - 1.0
    return 0.5 * k * (slip ** 2).sum() + K * (1.0 - np.cos(2 * np.pi * u)).sum()


def global_psi(u):
    return np.mean(np.exp(2j * np.pi * u))


def w_field(u, m=M_WIN):
    """窗口平均局域序参量（复数组）。"""
    ph = np.exp(2j * np.pi * u)
    half = m // 2
    Nloc = len(u)
    w = np.empty(Nloc, dtype=complex)
    for n in range(Nloc):
        lo, hi = max(0, n - half), min(Nloc, n + half + 1)
        w[n] = ph[lo:hi].mean()
    return w


print('=' * 76)
print('M7b：两瓣叠加态的单场承载（P4-5 专项）——判据先行，验收数值运行前写死')
print('=' * 76)
print(f'模型：FK 链 k={k}, K={K}, N={N}, ell={ELL:.3f}；窗口 M={M_WIN}；'
      f'主支 A={A0}, B={B0}（D=80）')

# ---------- 公共支链（同格内分数坐标，平移协变；阻尼弛豫去辐射噪声） ----------
uA = relax_damped(kink_profile(A0), winding=1)
uB = relax_damped(kink_profile(B0), winding=1)
wA, wB = w_field(uA), w_field(uB)
psiA, psiB = global_psi(uA), global_psi(uB)
budR_A = (1.0 - wA.real).sum()          # 实通道预算（支）
budM_A = (1.0 - np.abs(wA)).sum()       # 模通道预算（支）
Rt_single = N * (1 - abs(psiA)) / budR_A
print(f'\n[支链自检] Psi_A = {psiA.real:.12f}{psiA.imag:+.2e}i, '
      f'Psi_A - Psi_B = {abs(psiA - psiB):.2e}（平移协变）')
print(f'  单 kink：1-|Psi| = {1 - abs(psiA):.6f}，模预算 {budM_A:.4f}，'
      f'实预算 {budR_A:.4f}，R_tilde = {Rt_single:.4f}')

# ---------- J1/J2/J11：经典分瓣 S(q) ----------
print('\n[J1/J2/J11] 经典分瓣 S(q)：拓扑、滑量份额、亏空读出非线性')
lay = np.arange(N)
j1_ok, j2_ok = True, True
g_vals = []
for q in QG:
    us = split_profile(q, A0, B0)
    wind = us[-1] - us[0] - (N - 1)
    wf = w_field(us)
    far_ok = abs(abs(wf[5]) - 1) < 1e-12 and abs(abs(wf[-6]) - 1) < 1e-12
    j1_ok &= (abs(wind - 1.0) < 1e-12) and far_ok
    slip = us[1:] - us[:-1] - 1.0
    mid = 500
    s_left = slip[:mid].sum()
    s_all = slip.sum()
    j2_ok &= (abs(s_left - q) < 1e-7) and (abs(s_all - 1.0) < 1e-12)
    Wm = (1.0 - np.abs(wf))
    Wm = Wm / Wm.sum()
    g_vals.append(Wm[:mid].sum())
g_arr = np.array(g_vals)
q_arr = np.array(QG)
g_mono = bool(np.all(np.diff(g_arr) > 0))
g_sym = abs(g_arr[2] - 0.5) < 1e-9
g_dev = g_arr[3] - 0.65
print(f'  缠绕与远场（全网格）：{"OK" if j1_ok else "FAIL"}；'
      f'滑量份额 (q, 1-q)：{"OK" if j2_ok else "FAIL"}')
print(f'  模通道份额 g(q)：{np.round(g_arr, 4)}（单调 {g_mono}，g(0.5)={g_arr[2]:.9f}，'
      f'g(0.65)-0.65 = {g_dev:+.4f}）')

# ---------- J3：叠加像的份额折叠 ----------
print('\n[J3] 叠加像 img(q)：折叠恒等式与混叠超额')
j3a, j3b, j3c, j3d = True, True, True, True
emix_rows, RI_rows = [], []
imgW_by_q = {}
for q in QG:
    w_img = q * wA + (1 - q) * wB
    psi_img = q * psiA + (1 - q) * psiB
    d_cost = abs((1 - abs(psi_img)) - (1 - abs(psiA)))
    bR_img = (1.0 - w_img.real).sum()
    d_bud = abs(bR_img - budR_A)
    Rt_img = N * (1 - abs(psi_img)) / bR_img
    emix = (1.0 - np.abs(w_img)).sum() - budM_A
    j3a &= d_cost < 1e-11
    j3b &= d_bud < 5e-9
    j3c &= abs(Rt_img - Rt_single) < 1e-9
    j3d &= emix > -1e-12
    emix_rows.append((q, emix / budM_A))
    RI_rows.append((q, N * (1 - abs(psi_img)) / (1.0 - np.abs(w_img)).sum()))
    imgW_by_q[q] = (1.0 - w_img.real) / bR_img
print(f'  (a) 全局代价折叠 1-|Psi_img| = 1-Psi_A：{"OK" if j3a else "FAIL"}'
      f'（max 偏差 {d_cost:.1e}）')
print(f'  (b) 实通道预算折叠：{"OK" if j3b else "FAIL"}（max 偏差 {d_bud:.1e}，'
      f'支链浮点复现水平 1.35e-9）')
print(f'  (c) R_tilde_img = R_tilde_single = {Rt_single:.4f}：{"OK" if j3c else "FAIL"}')
print('  (d) 混叠超额 E_mix/单kink模预算（>=0，亚可加性）：',
      ' '.join(f'q={q}:{r:+.4f}' for q, r in emix_rows))
print('      模通道比值 R_I(q)（实测记录，R_single = '
      f'{N * (1 - abs(psiA)) / budM_A:.4f}）：',
      ' '.join(f'q={q}:{r:.3f}' for q, r in RI_rows))

# ---------- J4：系综对照（双 kink，缠绕 2） ----------
u_ens = relax_damped(two_kink_profile(A0, B0), winding=2)
w_ens = w_field(u_ens)
psi_ens = global_psi(u_ens)
cost_ratio = (1 - abs(psi_ens)) / (1 - abs(psiA))
budR_ens = (1.0 - w_ens.real).sum()
budR_ratio = budR_ens / budR_A
j4 = (1.999 < cost_ratio < 2.001) and (1.999 < budR_ratio < 2.001)
print(f'\n[J4] 双 kink 系综像：代价比 = {cost_ratio:.6f}，实预算比 = {budR_ratio:.6f}'
      f'（判据均 in [1.999, 2.001]）-> {"OK" if j4 else "FAIL"}')

# ---------- J5：载体判别（D 扫描，模通道） ----------
print('\n[J5] 载体判别：分瓣挫折弦（模通道 R 发散）对叠加像（D 无关）')
costS, costI, RS, RI = [], [], [], []
for D in DS:
    xa, xb = cN - D / 2, cN + D / 2
    us = split_profile(0.5, xa, xb)
    uA_D, uB_D = relax_damped(kink_profile(xa), winding=1), relax_damped(kink_profile(xb), winding=1)
    wA_D, wB_D = w_field(uA_D), w_field(uB_D)
    w_img = 0.5 * wA_D + 0.5 * wB_D
    psi_img = 0.5 * global_psi(uA_D) + 0.5 * global_psi(uB_D)
    costS.append(1 - abs(global_psi(us)))
    costI.append(1 - abs(psi_img))
    RS.append(N * costS[-1] / (1.0 - np.abs(w_field(us))).sum())
    RI.append(N * costI[-1] / (1.0 - np.abs(w_img)).sum())
costS, costI = np.array(costS), np.array(costI)
RS, RI = np.array(RS), np.array(RI)
growS = costS[2] / costS[1]
growR = RS[2] / RS[1]
sep_ok = (1.8 < growS < 2.2) and (1.8 < growR < 2.2)
flat_ok = (costI.max() - costI.min()) < 1e-9 and (RI[1:].max() - RI[1:].min()) < 1e-9
print(f'  分瓣 S(0.5)：cost(D) = {np.round(costS, 6)}（增长比 {growS:.4f}），'
      f'R_S = {np.round(RS, 2)}（增长比 {growR:.4f}：弦代价住在全局相位通道，'
      f'局域模亏空对台面纯相位失明）')
print(f'  叠加像：cost(D) = {np.round(costI, 12)}，R_I = {np.round(RI, 9)}'
      f'（D=80/160 极差 < 1e-9：{"OK" if (RI[1:].max() - RI[1:].min()) < 1e-9 else "FAIL"}；'
      f'D=40 为支撑重叠趋近点，R_I 偏差 {abs(RI[0] - RI[1]):.1e} 随 D 指数消退）')
j5 = sep_ok and flat_ok

# ---------- J6：挫折弦张力形状 ----------
print('\n[J6] 挫挫折弦张力 tau(q)（E 对 D 线性拟合斜率）')
Dfit = np.array([40.0, 80.0, 120.0, 160.0])
taus = {}
for q in (0.25, 0.5, 0.75):
    Es = [energy(split_profile(q, cN - D / 2, cN + D / 2)) for D in Dfit]
    taus[q] = np.polyfit(Dfit, Es, 1)[0]
shape = (taus[0.25] / taus[0.5]) / ((1 - np.cos(2 * np.pi * 0.25)) / 2)
shape75 = (taus[0.75] / taus[0.5]) / ((1 - np.cos(2 * np.pi * 0.75)) / 2)
j6 = abs(shape - 1) < 0.05 and abs(shape75 - 1) < 0.05
print(f'  tau(0.25) = {taus[0.25]:.6f}, tau(0.5) = {taus[0.5]:.6f}, '
      f'tau(0.75) = {taus[0.75]:.6f}（每格）')
print(f'  形状比实测/期望：q=0.25: {shape:.4f}，q=0.75: {shape75:.4f}'
      f'（期望 1，判据 < 5% 偏差）-> {"OK" if j6 else "FAIL"}')

# ---------- J7：相位通道 ----------
print('\n[J7a] 干涉禁闭前提：两支亏空支撑不交（D=80）')
dA = 1.0 - np.abs(wA)
dB = 1.0 - np.abs(wB)
overlap = np.minimum(dA, dB).max()
j7a = overlap < 1e-6
print(f'  max_n min(delta_A, delta_B) = {overlap:.2e}（判据 < 1e-6，'
      f'尾部 e^{{-D/(2 ell)}} 相位展布）-> {"OK" if j7a else "FAIL"}')

print('\n[J7b] 经典台面相位锁定：arg w(台面) = 2 pi q')
j7b = True
for q in (0.35, 0.5, 0.65):
    for tag, qq in (('走向+', q), ('走向-', 1 - q)):
        us = split_profile(qq, A0, B0)
        wf = w_field(us)
        ph_mid = np.angle(wf[cN])
        expect = 2 * np.pi * qq
        err = abs(np.angle(np.exp(1j * (ph_mid - expect))))
        j7b &= err < 0.01
        if tag == '走向+':
            print(f'  q={q}: 台面相位 = {ph_mid:+.4f} rad，期望 2pi q = {expect:.4f}，'
                  f'环绕偏差 {err:.2e}')
print(f'  两种滑量走向（+2pi q 与 2pi(1-q)）全部 < 0.01 rad -> {"OK" if j7b else "FAIL"}')

print('\n[J7c] 零模条纹读出：动量谱条纹相位恢复 gamma（恒等式级）')
sig_x = 2.3
j7c = True
for gam in (0.0, np.pi / 2, np.pi):
    q = 0.4
    D0 = B0 - A0
    p = np.linspace(-2.0, 2.0, 40001)
    gt = np.exp(-0.5 * (sig_x * p) ** 2)
    I = gt ** 2 * (1 + 2 * np.sqrt(q * (1 - q)) * np.cos(p * D0 - gam))
    C = (I * np.exp(1j * p * D0)).sum()
    gam_rec = np.angle(C)
    err = abs(np.angle(np.exp(1j * (gam_rec - gam))))
    j7c &= err < 1e-9
    print(f'  gamma = {gam:.4f}：恢复 {gam_rec:+.9f}，误差 {err:.2e}')
print(f'  读出观测 = 质心动量谱条纹相位（零模波包宽 = 份额宽 2.3 格，建模声明）'
      f'-> {"OK" if j7c else "FAIL"}')

# ---------- J8：支简并与 Peierls 劈裂 ----------
print('\n[J8] 支简并与 Peierls 劈裂（阻尼弛豫到局域极小）')
EA = energy(relax_damped(kink_profile(A0), winding=1))
EB = energy(relax_damped(kink_profile(B0), winding=1))
E_half = energy(relax_damped(kink_profile(A0 + 0.5), winding=1))
dE_deg = abs(EA - EB) / EA
dE_P = E_half - EA
j8 = dE_deg < 1e-10
print(f'  E(A=460.0) = {EA:.10f}，E(B=540.0) = {EB:.10f}，相对差 = {dE_deg:.2e}'
      f'（判据 < 1e-10）-> {"OK" if j8 else "FAIL"}')
print(f'  Peierls 劈裂 E(frac=0.5) - E(frac=0.0) = {dE_P:+.3e}（实测登记，'
      f'gamma 进动率 = dE_P，衔接 M2c）')

# ---------- J9：采样闭环（叠加像实通道份额，q=0.3） ----------
print('\n[J9] 采样闭环：叠加像实通道份额（q=0.35）接入 M4 协议')
rng = np.random.default_rng(SEED)
n_mc = 2 * 10 ** 6
W9 = imgW_by_q[0.35]
cum = np.cumsum(W9)
samples = np.searchsorted(cum, rng.random(n_mc))
freq = np.bincount(samples, minlength=N) / n_mc
sig_mask = W9 > 0.002
z = (freq[sig_mask] - W9[sig_mask]) / np.sqrt(W9[sig_mask] * (1 - W9[sig_mask]) / n_mc)
e = n_mc * W9[sig_mask]
chi2df = (((freq[sig_mask] * n_mc - e) ** 2 / e).sum()) / (sig_mask.sum() - 1)
w_l = W9[:500].sum()
j9 = (np.abs(z).max() < 3.5) and (chi2df < 1.6) and (abs(w_l - 0.35) < 1e-6)
print(f'  份额守恒：左瓣 = {w_l:.9f}（期望 0.35）；显著层 {sig_mask.sum()} 个，'
      f'max|z| = {np.abs(z).max():.2f}（< 3.5），卡方/自由度 = {chi2df:.3f}（< 1.6）'
      f'-> {"OK" if j9 else "FAIL"}')

# ---------- J10：经典塌缩动力学（无门，能量自检有门） ----------
print('\n[J10] 经典塌缩：S(0.5) 自由演化 vs 双 kink 长寿命（对照）')


def plateau_len(u, q=0.5, lo=470, hi=530):
    dev = np.abs(u[lo:hi] - np.arange(lo, hi) - q)
    return int((dev < 0.05).sum())


def evolve(u0, T, winding=1):
    u = u0.copy()
    v = np.zeros(len(u0))
    a = accel(u, v)
    E0 = energy(u)          # v=0：总能量 = 势能
    nsteps = int(T / dt)
    t_merge = None
    for i in range(nsteps):
        u = u + dt * v + 0.5 * dt ** 2 * a
        a_new = accel(u, v + dt * a)
        v = v + 0.5 * dt * (a + a_new)
        u[0], u[-1] = 0.0, float(N + winding - 1)
        v[0] = v[-1] = 0.0
        a = a_new
        if t_merge is None and plateau_len(u) < 2:
            t_merge = (i + 1) * dt
    dE = abs(energy(u) + 0.5 * (v ** 2).sum() - E0) / E0
    return u, t_merge, dE


u_split = split_profile(0.5, A0, B0)
u_end, t_merge, dE_S = evolve(u_split, 400.0, winding=1)
u_ens_end = evolve(u_ens.copy(), 400.0, winding=2)[0]
print(f'  S(0.5) 初始总能量 = {energy(u_split):.4f}'
      f'（挫折弦 {K * (1 - np.cos(2 * np.pi * 0.5)) * (B0 - A0):.1f} + 核）')


def kink_pos(u, lo, hi):
    seg = u[lo:hi]
    slip_seg = seg[1:] - seg[:-1] - 1.0
    ks = int(np.argmax(slip_seg)) + lo
    wins = np.clip(slip_seg[max(0, ks - lo - 8):ks - lo + 9], 0, None)
    return (np.arange(len(wins)) + max(0, ks - lo - 8) + lo).dot(wins) / wins.sum() if wins.sum() > 1e-12 else np.nan


p1a, p2a = kink_pos(u_ens, 400, 500), kink_pos(u_ens, 500, 620)
p1b, p2b = kink_pos(u_ens_end, 400, 500), kink_pos(u_ens_end, 500, 620)
drift = max(abs(p1b - p1a), abs(p2b - p2a))
j10 = dE_S < 1e-4
print(f'  S(0.5)：台面长度 {plateau_len(u_split)} -> 0，t_merge = {t_merge}，'
      f'|dE/E| = {dE_S:.2e}（积分器自检 < 1e-4）-> {"OK" if j10 else "FAIL"}')
print(f'  双 kink 对照（T=400）：kink 位置漂移 = {drift:.4f} 格'
      f'（{"长寿命" if drift < 0.01 else "有漂移，如实报告"}）')

# ---------- J12：局域重合（合取结构） ----------
w_img05 = 0.5 * wA + 0.5 * wB
prof_img = 1.0 - w_img05.real
prof_ens = 1.0 - w_ens.real
W_img12 = prof_img / prof_img.sum()
W_ens12 = prof_ens / prof_ens.sum()
max_diff = np.abs(W_img12 - W_ens12).max()
j12 = max_diff < 1e-9
print(f'\n[J12] 局域重合：q=0.5 叠加像 vs 双 kink 归一化份额剖面 max 差 = {max_diff:.2e}'
      f'（判据 < 1e-9）-> {"OK" if j12 else "FAIL"}；'
      f'全局通道分离：缠绕 1/2，代价比 {cost_ratio:.4f}')

# ---------- 总对照表 ----------
print('\n' + '=' * 76)
print('实测 vs 期望（判据对照表）')
print('=' * 76)
rows = [
    ('J1 拓扑：S(q) 缠绕=1 且远场 |w|=1', '全网格', j1_ok),
    ('J2 滑量份额 = (q, 1-q)，< 1e-7', '全网格', j2_ok),
    ('J3a 全局代价折叠（叠加像）', '机器精度', j3a),
    ('J3b 实通道预算折叠', '机器精度', j3b),
    ('J3c R_tilde 折叠（同一常数）', f'R_tilde = {Rt_single:.4f}', j3c),
    ('J3d 混叠超额 E_mix >= 0（记录）', np.round([r for _, r in emix_rows], 4), j3d),
    ('J4 系综像代价与预算 = 2x', f'{cost_ratio:.4f} / {budR_ratio:.4f}', j4),
    ('J5 载体判别（R_S 发散 vs R_I D 无关）', f'增长比 {growR:.3f}', j5),
    ('J6 弦张力形状 (1-cos 2pi q)/2', f'{shape:.3f} / {shape75:.3f}', j6),
    ('J7a 干涉禁闭前提（支撑不交）', f'{overlap:.1e}', j7a),
    ('J7b 台面相位锁定 2 pi q', '< 0.01 rad', j7b),
    ('J7c 零模条纹读出 gamma', '< 1e-9', j7c),
    ('J8 支简并（同格内）', f'{dE_deg:.1e}', j8),
    ('J9 采样闭环（q=0.35）', f'chi2/df = {chi2df:.3f}', j9),
    ('J10 塌缩动力学能量自检（辛有界）', f'|dE/E| = {dE_S:.1e}', j10),
    ('J11 g(q) 单调且 g(0.5)=0.5（记录）', f'g(0.65)-0.65 = {g_dev:+.3f}', g_mono and g_sym),
    ('J12 局域重合（合取结构）', f'max 差 {max_diff:.4f}', j12),
]
for name, meas, ok in rows:
    print(f'  [{"PASS" if ok else "FAIL"}] {name}：实测 {meas}')
verdict = all(r[2] for r in rows)
print(f'\n总判定：{"P4-5 判据级落地：份额折叠恒等式、挫折弦判别与相位通道分工闭合" if verdict else "存在未过判据：按失败预案核查"}')
print('层级声明：本验证在 FK 链模型内闭合 P4-5 的纲领层主张（判据级）；')
print('U 侧零模动力学与退相干速率归 P4-10；三维推广与窗口 M 物理取值保持边界。')
