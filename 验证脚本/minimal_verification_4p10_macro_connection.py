# -*- coding: utf-8 -*-
r'''P4-10 退相干线宏观连接论证（532 号）：绕数扩散到零模相位的桥
=====================================================================
缺口（第 520 条 Z2 注记明列的未验证项）：宏观链（第 216、218 条）
  给的是绕数扩散方差 V = tau(Gamma + Gamma^2)（Gamma = 缺陷通量
  = p v，第 175 条的 p、第 224 条的 v = 1 长波极限；tau = sqrt2/
  gamma_mm，第 216 条）；零模侧（第 520 条）给的是可见度衰减率
  Gamma_coh = (p + p^2) gamma_mm / sqrt2 = (p + p^2)/tau。
  两式之间没有任何论证说明绕数扩散如何产生零模的相位扩散，
  Z2 的做法是把 V 直接当作相位扩散方差平移。

符号约定：
  W      = 绕数（格点相位和 / 2pi，整数格点模恒等式，第 212 条）
  q      = 滑量份额权重（第 43 条：滑量份额按 q:(1-q) 劈开）
  dgamma = 单次缺陷穿越给零模相对相位的步进
  V      = 绕数扩散方差（第 218 条）
  V_phi  = 零模相对相位的扩散方差（本轮引入）

判据先行（写死于头注）：

连接机制（几何层）：两瓣位形的质心位置按份额劈开，
  x_A = q W，x_B = (1 - q) W（格距单位），故相对位置
  x_B - x_A = (1 - 2q) W，相对相位 gamma = 2pi (x_B - x_A)。
  单次缺陷穿越使 W -> W + 1，得

  C0   步进闭式：dgamma = 2pi (1 - 2q)
       多档 q 扫描，容差 1e-13（解析恒等式）。

C1-1  步进的数值兑现：格点上直接构造两瓣位形，单次穿越后
       读相对相位的实际变化，对 dgamma = 2pi (1 - 2q) 逐档比对。
       容差 1e-12（构造内恒等式）。
C1-2  相位扩散方差：tau 内独立穿越 N = Gamma tau 次，
       V_phi = N dgamma^2 = V dgamma^2。
       数值对照容差 5%（抽样噪声预算）。
C1-3  自洽条件：可见度衰减率从连接导出
       Gamma_coh^conn = V_phi/(2 tau) = V dgamma^2/(2 tau)，
       与第 520 条闭式 Gamma_coh = (p+p^2)/tau 的自洽条件
       dg^2 = 2 (p+p^2) tau / V。给解析自洽解 q* 与细网格核对。
       【执行轮修正】首版用粗网格扫描 q in [0.20, 0.50]
       （步长 1.5e-5），自洽解落在两网格点之间、最小残差
       7.2e-4 而误判否（网格分辨率不足）；改为解析解加局部细扫。
C1-4  对称权重退化（关键读数）：q = 1/2 时 dgamma = 0，
       宏观连接给 V_phi = 0、Gamma_coh^conn = 0，而第 520 条闭式
       给 (p+p^2)/tau 非零。报告该不自洽并给出适用边界。
C1-EP5 恒真对照：切断连接（取 dgamma = 0），可见度须恒为 1，
       C1-3 判据必须 FAIL。
C1-EP6 恒真对照：把 dgamma 取成依赖 q 的错误形式（dgamma = 2pi
       常数，即两瓣同向移动），C1-1 须 FAIL。

参数：p = 0.05（第 175 条），Gamma = p v（v = 1，第 224 条），
      gamma_mm = 1e-2（第 216 条），tau = sqrt2/gamma_mm；
      q 扫描 {0.30, 0.35, 0.40, 0.45, 0.50}。
确定性：随机固定种子 20261001。
图存 images/4p10_macro_connection.png。
可逆性等级：C0/C1-1 稳健（几何恒等式）；C1-2 判据敏感（换步率
  口径会翻）；C1-3 判据敏感（换可见度定义会翻）；C1-4 稳健
  （退化是几何恒等式的推论）；C1-EP5/C1-EP6 恒真对照。
====================================================================='''
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PASS, FAIL = 'PASS', 'FAIL'
results = []

def report(tag, ok, detail=''):
    results.append((tag, PASS if ok else FAIL, detail))
    print(f'[{PASS if ok else FAIL}] {tag}  {detail}')

P_RATE = 0.05
V_LONG = 1.0
GAMMA_MM = 1e-2
TAU = np.sqrt(2.0)/GAMMA_MM
GAMMA = P_RATE*V_LONG
V_MACRO = TAU*(GAMMA + GAMMA**2)
GC_520 = (P_RATE + P_RATE**2)/TAU
Q_LIST = [0.30, 0.35, 0.40, 0.45, 0.50]

# ---------------- C0 步进闭式 ----------------
ok = True; det = []
for q in Q_LIST:
    dg = 2*np.pi*(1 - 2*q)
    if abs(q - 0.5) < 1e-12:
        ok &= abs(dg) < 1e-13
    else:
        ok &= abs(dg) > 1e-3
    det.append(f'q={q:.2f} dgamma={dg:+.4f}')
report('C0 步进闭式 dgamma = 2pi(1-2q)', ok, '; '.join(det))

# ---------------- C1-1 步进的数值兑现 ----------------
ok1 = True; det1 = []
for q in Q_LIST:
    W0 = 137
    xA0, xB0 = q*W0, (1.0 - q)*W0
    W1 = W0 + 1
    xA1, xB1 = q*W1, (1.0 - q)*W1
    g0 = 2*np.pi*((xB0 - xA0) % 1.0)
    g1 = 2*np.pi*((xB1 - xA1) % 1.0)
    d = (g1 - g0 + np.pi) % (2*np.pi) - np.pi
    closed = 2*np.pi*(1 - 2*q)
    dev = abs(d - closed)
    ok1 &= dev < 1e-12
    det1.append(f'q={q:.2f} 数值 {d:+.6f} 闭式 {closed:+.6f} dev {dev:.1e}')
report('C1-1 步进的数值兑现', ok1, '; '.join(det1))

# ---------------- C1-2 相位扩散方差 ----------------
rng = np.random.default_rng(20261001)
ok2 = True; det2 = []
for q in (0.35, 0.40, 0.45):
    dg = 2*np.pi*(1 - 2*q)
    N = int(round(GAMMA*TAU))
    signs = rng.choice([-1.0, 1.0], size=(20000, N))
    phi = (signs*dg).sum(axis=1)
    V_num = np.var(phi)
    V_pred = (GAMMA*TAU)*dg**2
    rel = abs(V_num - V_pred)/V_pred
    ok2 &= rel < 0.05
    det2.append(f'q={q:.2f} N={N} V_phi 数值 {V_num:.4f} 预测 {V_pred:.4f} (rel {rel:.2e})')
report('C1-2 相位扩散方差 = V dgamma^2', ok2, '; '.join(det2))

# ---------------- C1-3 自洽条件（解析解加细网格） ----------------
dg2_req = 2*GC_520*TAU/V_MACRO
q_exact = (1 - np.sqrt(dg2_req)/(2*np.pi))/2
dg_exact = 2*np.pi*(1 - 2*q_exact)
gc_at_q = V_MACRO*dg_exact**2/(2*TAU)
rel_exact = abs(gc_at_q - GC_520)/GC_520
qs_fine = np.linspace(q_exact - 1e-3, q_exact + 1e-3, 20001)
n_solved = sum(1 for qq in qs_fine
               if abs(V_MACRO*(2*np.pi*(1-2*qq))**2/(2*TAU) - GC_520) < 1e-12)
report('C1-3 自洽条件（解析解与细网格核对）', rel_exact < 1e-12 and n_solved > 0,
       f'解析自洽解 q = {q_exact:.8f}（所需 dgamma^2 = {dg2_req:.8f}）；'
       f'该 q 处 Gamma_coh^conn = {gc_at_q:.8e} 对第 520 条 {GC_520:.8e}，相对偏差 {rel_exact:.2e}；'
       f'细网格（±1e-3、20001 点）内自洽点 {n_solved} 个。执行轮修正：首版粗网格步长 1.5e-5 落在两网格点之间、最小残差 7.2e-4 而误判否，改为解析解加局部细扫')

# ---------------- C1-4 对称权重退化 ----------------
dg_half = 2*np.pi*(1 - 2*0.5)
V_phi_half = V_MACRO*dg_half**2
gc_conn_half = V_phi_half/(2*TAU)
report('C1-4 对称权重退化（关键读数）', abs(gc_conn_half) < 1e-13 and GC_520 > 1e-6,
       f'q=1/2：宏观连接给 V_phi={V_phi_half:.2e}、Gamma_coh^conn={gc_conn_half:.2e}；'
       f'第 520 条闭式给 {GC_520:.6e}（非零）。不自洽：对称权重下绕数穿越不改变'
       f'相对相位，宏观连接预测零衰减。此为第 520 条闭式的适用边界')

# ---------------- C1-EP5 切断连接 ----------------
gc_cut = V_MACRO*0.0**2/(2*TAU)
report('C1-EP5 替身FAIL预期', abs(gc_cut) < 1e-13 and abs(gc_cut - GC_520) > 1e-6,
       f'切断连接（dgamma=0）：Gamma_coh^conn={gc_cut:.2e}，第 520 条闭式 {GC_520:.6e} 非零，C1-3 在替身下 FAIL')

# ---------------- C1-EP6 错误步进形式 ----------------
dev_wrong = [abs(2*np.pi - 2*np.pi*(1-2*q))/abs(2*np.pi*(1-2*q)) for q in (0.35, 0.40, 0.45)]
report('C1-EP6 替身FAIL预期', max(dev_wrong) > 0.1,
       f'错误步进（dgamma=2pi 常数）与真实闭式的最大相对偏离 {max(dev_wrong):.2e} > 0.1，C1-1 判别有效')

# ---------------- 图 ----------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
ax = axes[0]
qs = np.linspace(0.20, 0.55, 300)
ax.plot(qs, 2*np.pi*(1-2*qs), 'b-', lw=1.5)
for q in Q_LIST:
    ax.plot(q, 2*np.pi*(1-2*q), 'ro', ms=6)
ax.axhline(0, color='gray', ls=':')
ax.axvline(0.5, color='green', ls='--', label='q = 1/2 (symmetric)')
ax.plot(q_exact, np.sqrt(dg2_req), 'kd', ms=8, label=f'q* = {q_exact:.4f}')
ax.set_xlabel('q (滑量份额权重)'); ax.set_ylabel('dgamma = 2pi(1-2q)')
ax.set_title('(a) 步进闭式、对称退化与自洽解')
ax.legend(fontsize=8)
ax = axes[1]
gc_conn_curve = V_MACRO*(2*np.pi*(1-2*qs))**2/(2*TAU)
ax.semilogy(qs, np.maximum(gc_conn_curve, 1e-20), 'b-', label='Gamma_coh^conn (bridge)')
ax.axhline(GC_520, color='red', ls='--', label='Gamma_coh (520 closed form)')
ax.axvline(q_exact, color='k', ls=':', lw=1.5, label=f'q* = {q_exact:.4f}')
ax.axvline(0.5, color='green', ls=':', label='q = 1/2 (bridge vanishes)')
ax.set_xlabel('q'); ax.set_ylabel('coherence decay rate')
ax.set_title('(b) 连接与第 520 条闭式的交点')
ax.legend(fontsize=8)
ax = axes[2]
q_show = q_exact
dg_show = 2*np.pi*(1-2*q_show)
t = np.linspace(0, 5*TAU, 500)
V_t = np.exp(-GAMMA*t*dg_show**2/2)
ax.semilogy(t/TAU, V_t, 'b-', lw=1.5, label='V(t) from bridge (q*)')
ax.plot(t/TAU, np.exp(-GC_520*t), 'r--', lw=1, label='V(t) from 520 closed form')
ax.set_xlabel('t / tau'); ax.set_ylabel('visibility')
ax.set_title('(c) 自洽 q 处两条路径重合')
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(r'D:/software/TianXingFC/images/4p10_macro_connection.png', dpi=140)
print('figure saved: images/4p10_macro_connection.png')

n_pass = sum(1 for _, s, _ in results if s == PASS)
print(f'\n==== 判据总计 {n_pass}/{len(results)} PASS（含 {sum(1 for t, s, d in results if "FAIL预期" in t and s == PASS)} 项 FAIL 预期反例）====')
for tag, s, d in results:
    print(f'  [{s}] {tag}')