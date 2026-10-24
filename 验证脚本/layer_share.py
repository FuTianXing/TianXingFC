# -*- coding: utf-8 -*-
"""
M5：层位份额语义的最小验证（测量纲领任务 2-2，W_n 物侧定义）v2
v1 三处构造修正：双 kink 缠绕边界公式（u[-1] = N + winding - 1，v1 误用 2N
  注入虚假缠绕）；键/格坐标统一（滑量生活在本键上，自然坐标 n+0.5）；滑量代理
  噪声门限（弛豫辐射的对称键噪声被 clip 削成正偏置，门限 5% 峰值抑制）。

定义（本脚本实现并检验）：
  位置基 = 法线轴层位 |n>（x = l_P·n，距离归粒定案）。
  位错与背景在第 n 层的耦合权重有两个一致的场侧代理：
    亏空深度 delta_n = 1 - |w_n|，w_n = 窗口平均的背景局域序参量
      （w_n = (1/M) Sum_{j in window(n)} exp(i 2 pi u_j)，M = 5 格点粗粒化）；
    滑量密度 sigma_n = u_{n+1} - u_n - 1（缠绕亏空的逐层分布，坐标 n+0.5，
    噪声门限 5% 峰值）。
  层位份额 W_n = raw_n / Sum_m raw_m（归一化），两种代理对照。
  概率幅 c_n = sqrt(W_n) exp(i Phi_n)，Phi_n = arg w_n。
  份额守恒 Sum W_n = 1 来自归一化；人数份额退位，相干份额上台。

判据先行（运行前写死）：
  K1 守恒：Sum W_n = 1 至机器精度（两种代理）
  K2 定位：亏空质心、滑量质心（统一为格点坐标）、kink 质心（键质心 + 0.5）
     两两偏差 < 0.1 格
  K3 尺度：W 的标准宽在 [1.5, 4.5] 格（~ kink 宽度 ell = 2.25，窗口展宽容许）
  K4 采样闭环：W_n（亏空代理）作角向份额接入 M4 协议（均匀快照 + 弧段读出），
     2e6 次位置采样频率 vs W_n：显著层（W_n > 0.002）|z| < 3.5，卡方/自由度 < 1.6
  K5 多缺陷：双 kink（间距 80 格，缠绕 2）W 双峰，两峰权重 (0.5, 0.5) +/- 0.02，
     峰质心 = 各 kink 位置 +/- 0.3 格
纪律：全确定性（MC 固定种子）；实测 vs 期望对照表输出。
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
SEED = 20260912
GATE = 0.05

def kink_profile(x0):
    n = np.arange(N)
    return n + (2 / np.pi) * np.arctan(np.exp((n - x0) / ELL))

def two_kink_profile(x1, x2):
    n = np.arange(N)
    return (n + (2 / np.pi) * np.arctan(np.exp((n - x1) / ELL))
               + (2 / np.pi) * np.arctan(np.exp((n - x2) / ELL)))

def accel(u, v):
    a = np.zeros_like(u)
    a[1:-1] = k * (u[2:] + u[:-2] - 2 * u[1:-1]) - w2 * np.sin(2 * np.pi * u[1:-1])
    return a

def relax(u0, steps=8000, winding=1):
    u = u0.copy()
    v = np.zeros(N)
    a = accel(u, v)
    for _ in range(steps):
        u = u + dt * v + 0.5 * dt ** 2 * a
        a_new = accel(u, v + dt * a)
        v = v + 0.5 * dt * (a + a_new)
        u[0], u[-1] = 0.0, float(N + winding - 1)   # 缠绕 W 的右边界：u_{N-1} = (N-1) + W
        v[0] = v[-1] = 0.0
        a = a_new
    return u

def center(u):
    w = u[1:] - u[:-1] - 1.0
    ks = int(np.argmax(w))
    lo, hi = max(0, ks - 8), min(len(w), ks + 9)
    ww = np.clip(w[lo:hi], 0.0, None)
    return (np.arange(lo, hi) * ww).sum() / ww.sum() if ww.sum() > 1e-12 else np.nan

def local_coherence(u, m=M_WIN):
    """窗口平均的局域序参量 |w_n|：背景局域相干强度（模型代理）。"""
    ph = np.exp(2j * np.pi * u)
    half = m // 2
    wabs = np.empty(N)
    for n in range(N):
        lo, hi = max(0, n - half), min(N, n + half + 1)
        wabs[n] = abs(ph[lo:hi].mean())
    return wabs

def share_from_raw(raw):
    return raw / raw.sum()

print('=' * 72)
print('M5 v2：层位份额语义（W_n 物侧定义）——判据先行，验收数值运行前写死')
print('=' * 72)
print(f'模型：FK 链 k={k}, K={K}, N={N}, ell={ELL:.3f}；粗粒化窗口 M={M_WIN}')
print('定义：W_n = 归一化耦合权重；代理甲 = 亏空深度 1-|w_n|；代理乙 = 滑量密度（门限 5%）')

# 单 kink：两种代理的 W_n（坐标统一：滑量与键质心换算到格点坐标 n+0.5）
u_rest = relax(kink_profile(float(cN)), winding=1)
wabs = local_coherence(u_rest)
W_d = share_from_raw(1.0 - wabs)
slip = u_rest[1:] - u_rest[:-1] - 1.0
slip = np.where(slip > GATE * slip.max(), slip, 0.0)      # 噪声门限
slip_site = np.append(slip, 0.0)                           # 键 n -> 格点坐标 n+0.5，末键归 0
W_s = share_from_raw(slip_site)
xc_bond = center(u_rest)
xc = xc_bond + 0.5                                         # 键质心换算到格点坐标
lay = np.arange(N)
lay_s = lay + 0.5
cenW_d = (lay * W_d).sum()
cenW_s = (lay_s * W_s).sum()
stdW_d = np.sqrt((lay ** 2 * W_d).sum() - cenW_d ** 2)
stdW_s = np.sqrt((lay_s ** 2 * W_s).sum() - cenW_s ** 2)
corr_core = np.corrcoef(W_d[cN - 20:cN + 20], W_s[1:][cN - 20:cN + 20])[0, 1]
d2 = max(abs(cenW_d - cenW_s), abs(cenW_d - xc), abs(cenW_s - xc))
print(f'\n[单 kink] 质心（格点坐标）：kink {xc:.3f}，亏空 {cenW_d:.3f}，滑量 {cenW_s:.3f}')
print(f'  宽度：亏空 {stdW_d:.3f}，滑量 {stdW_s:.3f}（ell = {ELL:.3f}）；核区相关系数 {corr_core:.4f}')

# K4 采样闭环：W_n（亏空代理）作角向份额 + M4 协议
rng = np.random.default_rng(SEED)
n_mc = 2 * 10 ** 6
cum = np.cumsum(W_d)
samples = np.searchsorted(cum, rng.random(n_mc))
freq = np.bincount(samples, minlength=N) / n_mc
sig_mask = W_d > 0.002
z = (freq[sig_mask] - W_d[sig_mask]) / np.sqrt(W_d[sig_mask] * (1 - W_d[sig_mask]) / n_mc)
e = n_mc * W_d[sig_mask]
chi2df = (((freq[sig_mask] * n_mc - e) ** 2 / e).sum()) / (sig_mask.sum() - 1)
print(f'\n[K4 采样闭环] 显著层 {sig_mask.sum()} 个，max|z| = {np.abs(z).max():.2f}（< 3.5），'
      f'卡方/自由度 = {chi2df:.3f}（< 1.6）')

# K5 双 kink（缠绕 2，边界公式修正）
x1, x2 = float(cN - 40), float(cN + 40)
u2 = relax(two_kink_profile(x1, x2), winding=2)
wabs2 = local_coherence(u2)
W2 = share_from_raw(1.0 - wabs2)
mid = int((x1 + x2) / 2) + 1
w1 = W2[:mid].sum()
w2h = W2[mid:].sum()
cen1 = (lay[:mid] * W2[:mid]).sum() / w1
cen2 = (lay[mid:] * W2[mid:]).sum() / w2h
print(f'\n[K5 双 kink] 两峰权重：{w1:.4f} / {w2h:.4f}（期望 0.5/0.5）；'
      f'峰质心 {cen1:.2f} / {cen2:.2f}（kink 位置 {x1:.1f} / {x2:.1f}）')

print('\n' + '=' * 72)
print('实测 vs 期望（判据对照表）')
print('=' * 72)
rows = [
    ('K1 守恒 Sum W_n = 1（两代理）',
     f'亏空 {W_d.sum():.12f}，滑量 {W_s.sum():.12f}',
     abs(W_d.sum() - 1) < 1e-12 and abs(W_s.sum() - 1) < 1e-12),
    ('K2 定位：三质心两两偏差 < 0.1',
     f'{d2:.4f}', d2 < 0.1),
    ('K3 宽度 [1.5, 4.5]（两代理）',
     f'亏空 {stdW_d:.3f}，滑量 {stdW_s:.3f}',
     1.5 < stdW_d < 4.5 and 1.5 < stdW_s < 4.5),
    ('K4 采样闭环 max|z| < 3.5', f'{np.abs(z).max():.2f}', np.abs(z).max() < 3.5),
    ('K4 采样闭环 卡方/自由度 < 1.6', f'{chi2df:.3f}', chi2df < 1.6),
    ('K5 双峰权重 0.5+/-0.02',
     f'{w1:.4f} / {w2h:.4f}', abs(w1 - 0.5) < 0.02 and abs(w2h - 0.5) < 0.02),
    ('K5 峰质心 = kink 位置 +/- 0.3',
     f'{abs(cen1 - x1):.3f} / {abs(cen2 - x2):.3f}',
     abs(cen1 - x1) < 0.3 and abs(cen2 - x2) < 0.3),
]
for name, meas, ok in rows:
    print(f'  [{"PASS" if ok else "FAIL"}] {name}：实测 {meas}')
verdict = all(r[2] for r in rows)
print(f'\n总判定：{"M5 判据全部通过——W_n 物侧定义成立，份额语义落地，人数退位" if verdict else "存在未过判据——按失败预案核查"}')
print('语义声明：位置概率的承载者 = 相干份额（位错-背景耦合按层分配），人数份额退居幕后')
