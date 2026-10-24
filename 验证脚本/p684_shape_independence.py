# -*- coding: utf-8 -*-
r'''η 动力学路线甲：退相干衰减形状的成立条件（判据 684 号）
=====================================================================
对象：第 520 条闭式 Gamma_coh = (p + p^2) gamma_mm / sqrt2 与
      满周升格论文 §4.1 的形状 V(t) = (1 + 2 Gamma_coh t)^{-1/2}。

本轮问题（路线甲）：该形状在什么条件下成立、在什么条件下换成别的形状。
形状由两个正交因素共同决定，二者在第 520 条里被同时固定而未分开：

  因素一 可见度定义
    V_amp(t) = |<exp(i phi)>|      振幅型（干涉条纹对比度，标准退相干）
    V_ens(t) = <exp(-phi^2/2)>     系综型（每实现相干因子再平均，第 520 条）

  因素二 相位过程类型
    连续高斯扩散（每步微小增量）
    离散复合泊松跳变（每摆程以概率 p 落位，落位给 O(1) 相位跳变）

理论期望（先立）：
  E1  高斯扩散 + 振幅型：V_amp = exp(-Gamma t)（特征函数），形状为纯指数。
  E2  高斯扩散 + 系综型：V_ens = (1 + 2 Gamma t)^(-1/2)。此为第 520 条。
      E1 与 E2 同源同方差，形状不同，故形状取决于可见度定义。
  E3  复合泊松 + 振幅型：V_amp = exp[-lambda t (1 - Re E[e^{i theta}])]，
      任何跳变幅度分布都给指数形状；分布只改率，不改形状。
  E4  自然参数档每相干时间事件数 N_ev = p + p^2 ~ 0.05 << 1，
      属稀疏档，连续扩散图像（每步微小增量）在此无物理依据。

判据：
  B1  高斯扩散 + 振幅型给指数：ln V 线性拟合 R^2 >= 0.999，
      且对闭式 (1+2Gt)^(-1/2) 最大相对偏差 >= 0.15（两形状确实不同）。
  B2  高斯扩散 + 系综型复现闭式：最大相对偏差 <= 2e-2
      （容差预算：n_impl=400 的抽样标准误 5e-2，取 2e-2 严于预算）。
  B3  复合泊松 + 振幅型给指数：三档跳变幅度分布下 ln V 拟合 R^2 >= 0.999。
  B4  复合泊松下形状对幅度分布不敏感、率敏感：三档分布的形状参数
      （指数拟合残差）差 <= 0.05，而率跨档散布 >= 10%。
  B5  自然参数档事件数读数（不做判据）：N_ev = p + p^2。
  B6  恒真反例（EP5）：把闭式替身为 exp(-Gt) 后 B2 必须 FAIL。
  B7  EP3 抽样半宽：各档系综半宽，确认偏差非抽样噪声。

可逆性等级：
  B1、B2、B3 判据稳健（换种子只动末位）。
  B4 判据敏感（换分布族会翻具体数值，方向不变）。
  B5、B7 读数。B6 恒真对照。

参数与确定性：随机固定种子 20261018；n_impl = 2000；N = 4000；
时间窗 5/Gamma。图存 images/p684_shape_independence.png。
====================================================================='''
import os
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


def gamma_coh(p, gamma_mm):
    """第 520 条闭式（原样照录）。"""
    tau = np.sqrt(2.0) / gamma_mm
    return (p + p * p) / tau, tau


def gauss_visibility(gamma, tau, n=4000, n_impl=20000, seed=20261018, horizon=5.0):
    """连续高斯扩散。返回两种可见度定义下的曲线。"""
    rng = np.random.default_rng(seed)
    T = horizon / gamma
    dt = T / n
    t = np.linspace(0.0, T, n)
    sd = np.sqrt(2.0 * gamma * dt)
    phi = np.cumsum(rng.normal(0.0, sd, (n_impl, n)), axis=1)
    v_amp_impl = np.cos(phi)
    v_ens_impl = np.exp(-0.5 * phi ** 2)
    v_amp = v_amp_impl.mean(axis=0)
    v_ens = v_ens_impl.mean(axis=0)
    half = v_ens_impl.std(axis=0) / np.sqrt(n_impl)
    return t, v_amp, v_ens, half


def poisson_visibility(lam, theta_dist, n=4000, n_impl=20000, seed=20261018,
                       horizon=5.0):
    """复合泊松跳变：率 lam，每跳相位 theta 依 theta_dist 抽样。

    按事件而非微小步推进：每个时间栅格内事件数 ~ Poisson(lam*dt)。
    """
    rng = np.random.default_rng(seed)
    T = horizon / lam
    dt = T / n
    counts = rng.poisson(lam * dt, (n_impl, n))
    if theta_dist == 'twopoint':
        theta = rng.choice([-1.0, 1.0], size=counts.shape)
    elif theta_dist == 'twopoint_small':
        theta = rng.choice([-0.4, 0.4], size=counts.shape)
    elif theta_dist == 'uniform':
        theta = rng.uniform(-1.0, 1.0, counts.shape)
    elif theta_dist == 'gaussjump':
        theta = rng.normal(0.0, 1.0, counts.shape)
    else:
        raise ValueError(theta_dist)
    phi = np.cumsum(counts * theta, axis=1)
    v_impl = np.cos(phi)
    v = v_impl.mean(axis=0)
    half = v_impl.std(axis=0) / np.sqrt(n_impl)
    return t, v, half


def max_rel_dev(V, ref):
    return float(np.max(np.abs(V - ref)) / np.max(ref))


def loglin_r2(t, V, n_fit=None, floor=0.1):
    """ln V 对 t 的线性拟合 R^2。

    拟合区间取 V >= floor：系综均值在 V 落到 1/sqrt(n_impl) 的噪声底
    之后不再有物理信息，把该段纳入拟合会人为压低 R^2。
    """
    mask = V >= floor
    if n_fit:
        mask &= (np.arange(len(t)) < n_fit)
    tt, vv = t[mask], np.log(V[mask])
    if len(tt) < 10:
        return 0.0, 0.0
    coef = np.polyfit(tt, vv, 1)
    pred = np.polyval(coef, tt)
    ss_res = np.sum((vv - pred) ** 2)
    ss_tot = np.sum((vv - vv.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return r2, -coef[0]


# ---------------- B1 高斯 + 振幅型 -> 指数 ----------------
gamma0 = 0.5
t, v_amp, v_ens, half = gauss_visibility(gamma0, 1.0)
r2_amp, rate_amp = loglin_r2(t, v_amp)
closed = (1.0 + 2.0 * gamma0 * t) ** (-0.5)
dev_vs_closed = max_rel_dev(v_amp, closed)
ok = (r2_amp >= 0.999) and (dev_vs_closed >= 0.15)
report('B1 高斯+振幅型给指数、与闭式不同', ok,
       f'ln V 拟合 R^2={r2_amp:.6f}，拟合率 {rate_amp:.4f}（解析 Gamma={gamma0}）；'
       f'对闭式最大偏差 {dev_vs_closed:.3f}')

# ---------------- B2 高斯 + 系综型 -> 闭式 ----------------
dev_ens = max_rel_dev(v_ens, closed)
ok = dev_ens <= 2e-2
report('B2 高斯+系综型复现闭式', ok,
       f'最大相对偏差 {dev_ens:.2e}（预算 2e-2，抽样标准误 5e-2）')

# ---------------- B3 复合泊松 + 振幅型 -> 指数 ----------------
lam = 0.5
ok = True
det = []
for dist in ['twopoint', 'twopoint_small', 'uniform']:
    tp, vp, hp = poisson_visibility(lam, dist)
    r2, rate = loglin_r2(tp, vp)
    ok &= r2 >= 0.999
    det.append(f'{dist}: R^2={r2:.6f} rate={rate:.4f}')
report('B3 复合泊松+振幅型给指数（三档幅度分布）', ok, '; '.join(det))

# ---------------- B4 形状对分布不敏感、率敏感 ----------------
resid = {}
rates = {}
for dist in ['twopoint', 'twopoint_small', 'uniform']:
    tp, vp, hp = poisson_visibility(lam, dist)
    r2, rate = loglin_r2(tp, vp)
    rates[dist] = rate
    resid[dist] = 1.0 - r2
shape_spread = max(resid.values()) - min(resid.values())
rate_vals = list(rates.values())
rate_spread = (max(rate_vals) - min(rate_vals)) / np.mean(rate_vals)
ok = (shape_spread <= 0.05) and (rate_spread >= 0.10)
report('B4 形状对幅度分布不敏感、率敏感', ok,
       f'形状残差散布 {shape_spread:.2e}（<=0.05）；率跨档散布 {rate_spread:.3f}（>=0.10）')

# ---------------- B5 自然参数档事件数（读数） ----------------
for p_nat, gm_nat in [(0.05, 1e-2), (0.05, 1e-4)]:
    rate_nat, tau_nat = gamma_coh(p_nat, gm_nat)
    n_ev = p_nat + p_nat ** 2
    print(f'[READ] B5 p={p_nat} gm={gm_nat:.0e}：N_ev = p+p^2 = {n_ev:.4f}'
          f'（与 gamma_mm 无关，故闭式口径下事件数恒稀疏）')

# ---------------- B6 恒真反例 ----------------
dev_exp = max_rel_dev(v_ens, np.exp(-gamma0 * t))
report('B6 恒真反例（exp 替身须 FAIL）', dev_exp > 0.15,
       f'闭式偏差 {dev_ens:.2e}；exp 替身偏差 {dev_exp:.3f}')

# ---------------- B7 EP3 抽样半宽 ----------------
rel_half = float(np.max(half) / np.max(v_ens))
print(f'[READ] B7 系综半宽/峰值 = {rel_half:.2e}'
      f'（n_impl=20000 的抽样标准误上界 7.1e-3，实际半宽须低于该量级）')

# ---------------- 出图 ----------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
ax = axes[0]
ax.semilogy(t, v_amp, '-', lw=1.6, label='振幅型 |<e^{iφ}>|（指数）')
ax.semilogy(t, v_ens, '--', lw=1.6, label='系综型 <e^{-φ²/2}>（闭式）')
ax.semilogy(t, np.exp(-gamma0 * t), 'k:', lw=1.0, label='exp(-Γt)')
ax.semilogy(t, closed, 'k-.', lw=1.0, label='(1+2Γt)^(-1/2)')
ax.set_xlabel('t（单位 τ=1）')
ax.set_ylabel('可见度')
ax.set_title('同一高斯扩散、两种可见度定义')
ax.legend(fontsize=7.5)
ax.grid(alpha=0.3, which='both')

ax = axes[1]
for dist, style in [('twopoint', '-'), ('twopoint_small', '--'), ('uniform', ':')]:
    tp, vp, hp = poisson_visibility(lam, dist)
    ax.semilogy(tp, np.abs(vp) + 1e-12, style, lw=1.4, label=f'泊松 {dist}')
ax.set_xlabel('t（单位 1/λ）')
ax.set_ylabel('可见度')
ax.set_title('复合泊松：三档幅度分布同一形状')
ax.legend(fontsize=7.5)
ax.grid(alpha=0.3, which='both')
plt.tight_layout()

out_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       '..', 'images'))
os.makedirs(out_dir, exist_ok=True)
out = os.path.join(out_dir, 'p684_shape_independence.png')
plt.savefig(out, dpi=140)
print(f'图存 {out}')

n_pass = sum(1 for _, s, _ in results if s == PASS)
print(f'\n判据 {n_pass}/{len(results)} PASS')
