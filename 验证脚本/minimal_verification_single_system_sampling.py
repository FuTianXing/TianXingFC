# -*- coding: utf-8 -*-
"""
M4：单系统 Born 采样与相位均匀性检验（测量纲领任务 2-1，裁决点 3 凭据）

判据先行（运行前写死）：
  J1 候选一（无共同时钟基准，先验纠缠整体相干的结构面）：系统节律与探测器触发
     节律无共同零点、频率比不可通约 -> 快照相位直方图均匀：36 分箱卡方/自由度
     < 1.6（约 p > 0.01），且差异度按 O(log n / n) 衰减
  J2 候选二（背景涨落信息层噪声）：触发时刻的背景相位噪声使快照相位分布趋于均匀：
     环绕正态的总变差距离 TVD(sigma_phi) 随 sigma_phi 单调下降，且在 sigma_phi
     <= 5 rad 内降到 0.05 以下（报告穿越点 sigma*）
  J3 Born 采样端到端：系统初相完全确定（phi0 = 0），触发抖动 sigma_phi = 3 rad，
     角向份额划分采样 2e6 次：频率与份额 w=(0.5,0.3,0.2) 的偏差 < 3sigma_stat
     + TVD(3)（统计与系统偏差分解），且 TVD(3) <= 0.05（容差判据）。
     【判据修正记录】初版误把"严格均匀"写成卡方 p>0.01（n=2e6）：任何有限 sigma
     的近似均匀都会被足够大的样本卡方拒绝，正确语言是 TVD 容差；p 值随 n 的衰减
     作为演示项输出（教训入验证注记过程记录）。
  J4 收敛律：频率误差按 n^(-1/2) 收敛（log-log 斜率 -0.5 +/- 0.15）；有限系综
     N 仅通过 w = N_a/N 的粒度（1/N）进入，不引入额外偏差
  J5 解析恒等式：均匀快照相位 + 角向份额划分 => P(a) = w_a（份额恒等式的单系统像）

假设层级声明（随裁决点 3 提交）：相位均匀性 = 条件性命题（无内禀零点 + 无共同时钟
  或背景相位噪声超过周期尺度），非独立新公设；两个候选机制分别对应结构面（Weyl
  等分布）与动力学面（噪声扩散）。
纪律：全确定性（固定种子）；解析与蒙特卡洛双通道；实测 vs 期望对照表输出。
"""
import numpy as np
from scipy import stats

SEED = 20260912
BINS = 36

def phase_hist_chi2(phases):
    """36 分箱均匀性卡方检验，返回 chi2/df。"""
    h, _ = np.histogram(np.mod(phases, 2 * np.pi), bins=BINS, range=(0, 2 * np.pi))
    e = h.sum() / BINS
    chi2 = ((h - e) ** 2 / e).sum()
    return chi2 / (BINS - 1), chi2

def wrapped_normal_tvd(sigma, n_grid=200000):
    """环绕正态 vs 均匀的总变差距离（解析数值积分）。"""
    x = np.linspace(0, 2 * np.pi, n_grid, endpoint=False)
    f = np.zeros_like(x)
    for k in range(-4, 5):
        f += np.exp(-((x + 2 * np.pi * k) ** 2) / (2 * sigma ** 2)) / (np.sqrt(2 * np.pi) * sigma)
    return 0.5 * np.abs(f - 1 / (2 * np.pi)).sum() * (2 * np.pi / n_grid)

def sector_sample(u, w):
    """角向份额划分：均匀相位 u 落入第 a 个弧段（弧宽 2*pi*w_a）。"""
    cum = np.cumsum(w)
    return np.searchsorted(cum, u * w.sum())

print('=' * 72)
print('M4：单系统 Born 采样与相位均匀性检验（判据先行，验收数值运行前写死）')
print('=' * 72)
print('假设：凝固时刻快照相位 ~ Uniform[0,2pi)；协议：角向份额划分 + 随机触发快照')
print('判据：J1 Weyl 等分布；J2 噪声扩散 TVD 递减且 sigma* <= 5；J3 Born 频率 3sigma；')
print('      J4 收敛斜率 -0.5+/-0.15；J5 解析恒等式 P(a)=w_a')

# J1 候选一：无共同时钟基准（频率比 = 黄金比，Weyl 等分布）
rng = np.random.default_rng(SEED)
ratio = (1 + 5 ** 0.5) / 2
n1 = 100000
ph1 = 2 * np.pi * ratio * np.arange(n1)
r1, chi1 = phase_hist_chi2(ph1)
disc = []
for n in (10 ** 3, 10 ** 4, 10 ** 5):
    h, _ = np.histogram(np.mod(2 * np.pi * ratio * np.arange(n), 2 * np.pi),
                        bins=BINS, range=(0, 2 * np.pi))
    disc.append(np.abs(h - h.sum() / BINS).sum() / (2 * h.sum()))
print(f'\n[J1 候选一 无共同时钟] n={n1}, chi2/df = {r1:.3f}（判据 < 1.6），'
      f'差异度 {disc[0]:.4f} -> {disc[2]:.4f}（n=1e3 -> 1e5，应下降）')

# J2 候选二：背景涨落相位噪声 -> 环绕正态 TVD
sigs = np.array([0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0])
tvds = np.array([wrapped_normal_tvd(s) for s in sigs])
mono = bool(np.all(np.diff(tvds) < 0))
sig_star_arr = sigs[tvds < 0.05]
sig_star = sig_star_arr[0] if sig_star_arr.size else float('inf')
if sig_star_arr.size:
    # 细化：粗网格（步长 0.5）只给首个穿越格点，二分收敛到 TVD = 0.05 真穿越点
    # 2026-09-13 评审复核：粗网格报 2.50，解析与细网格实值约 2.256
    lo, hi = sigs[tvds >= 0.05][-1], sig_star_arr[0]
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if wrapped_normal_tvd(mid) > 0.05:
            lo = mid
        else:
            hi = mid
    sig_star = 0.5 * (lo + hi)
print(f'\n[J2 候选二 背景涨落] TVD(sigma): '
      + ', '.join(f'{s:.1f}:{t:.4f}' for s, t in zip(sigs, tvds)))
print(f'  单调递减 = {mono}（判据 True）；TVD<0.05 穿越点 sigma* = {sig_star:.2f} rad（判据 <= 5）')

# J3 Born 采样端到端：定态初相 + 抖动触发 + 角向份额划分
w = np.array([0.5, 0.3, 0.2])
n3 = 2 * 10 ** 6
sig3 = 3.0
phi0 = 0.0
trig = phi0 + rng.normal(0.0, sig3, n3)          # 触发时刻的相位（定态 + 背景抖动）
out = sector_sample(np.mod(trig, 2 * np.pi) / (2 * np.pi) * w.sum(), w)
freq = np.bincount(out, minlength=3) / n3
sig_stat = np.sqrt(w * (1 - w) / n3)
tvd3 = wrapped_normal_tvd(sig3)
zscore = (freq - w) / sig_stat
ok3 = bool(np.all(np.abs(zscore) * sig_stat < 3 * sig_stat + tvd3))
h3, _ = np.histogram(np.mod(trig, 2 * np.pi), bins=BINS, range=(0, 2 * np.pi))
chi3 = ((h3 - h3.sum() / BINS) ** 2 / (h3.sum() / BINS)).sum()
p3 = stats.chi2.sf(chi3, BINS - 1)
print(f'\n[J3 Born 采样端到端] n={n3:.0e}, sigma_phi={sig3} rad（TVD={tvd3:.4f}，容差 0.05）')
for a in range(3):
    print(f'  结果 {a}: 频率 {freq[a]:.6f} vs 份额 {w[a]:.1f}，z = {zscore[a]:+.2f}')
print(f'  频率偏差均在 3sigma_stat + TVD 内 = {ok3}（系统偏差被 TVD 项完全解释）')
ok3b = bool(tvd3 <= 0.05)
print(f'  TVD({sig3}) <= 0.05 容差判据 = {ok3b}')
# 演示项：固定 sigma=3，卡方 p 值随 n 衰减（教训量化：大样本必拒近似均匀）
print('  [演示] 固定 sigma=3 的卡方均匀性 p 值随 n：')
p_row = []
for n_d in (10 ** 3, 10 ** 4, 10 ** 5, 10 ** 6, 2 * 10 ** 6):
    ph_d = phi0 + rng.normal(0.0, sig3, int(n_d))
    h_d, _ = np.histogram(np.mod(ph_d, 2 * np.pi), bins=BINS, range=(0, 2 * np.pi))
    chi_d = ((h_d - h_d.sum() / BINS) ** 2 / (h_d.sum() / BINS)).sum()
    p_d = stats.chi2.sf(chi_d, BINS - 1)
    p_row.append(p_d)
    print(f'    n={n_d:.0e}: p = {p_d:.4f}')

# J4 收敛律与有限 N
n_list = np.array([10 ** 2, 10 ** 3, 10 ** 4, 10 ** 5, 10 ** 6])
errs = []
for n in n_list:
    u = rng.random(n)
    o = sector_sample(u * w.sum(), w)
    errs.append(abs(np.mean(o == 1) - w[1]))
errs = np.array(errs)
slope = np.polyfit(np.log(n_list), np.log(errs + 1e-12), 1)[0]
wN = np.array([50, 30, 20]) / 100                  # 有限系综 N=100 的份额粒度
oN = sector_sample(rng.random(n3) * wN.sum(), wN)
fN = np.bincount(oN, minlength=3) / n3
zN = np.max(np.abs((fN - wN) / np.sqrt(wN * (1 - wN) / n3)))
print(f'\n[J4 收敛律] 斜率 = {slope:.3f}（判据 -0.5 +/- 0.15）；'
      f'有限 N=100 份额 (50,30,20)/100：最大 |z| = {zN:.2f}（无额外偏差）')

# J5 解析恒等式
print('\n[J5 解析恒等式] P(a) = 弧宽/2pi = w_a：份额恒等式的单系统像，恒等式级成立')

print('\n' + '=' * 72)
print('实测 vs 期望（判据对照表）')
print('=' * 72)
rows = [
    ('J1 候选一 Weyl 等分布 chi2/df < 1.6', f'{r1:.3f}', r1 < 1.6),
    ('J1 差异度随 n 下降', f'{disc[0]:.4f} -> {disc[2]:.4f}', disc[2] < disc[0]),
    ('J2 TVD 单调递减', f'{mono}', mono),
    (f'J2 TVD<0.05 穿越点 sigma* <= 5 rad', f'{sig_star:.2f}', sig_star <= 5.0),
    ('J3 Born 频率偏差 < 3sigma_stat + TVD', f'max|z|={np.max(np.abs(zscore)):.2f}, TVD={tvd3:.4f}', ok3),
    ('J3 TVD(3) <= 0.05 容差', f'{tvd3:.4f}', ok3b),
    ('J4 收敛斜率 -0.5+/-0.15', f'{slope:.3f}', abs(slope + 0.5) < 0.15),
    ('J4 有限 N 无额外偏差 max|z| < 3', f'{zN:.2f}', zN < 3),
]
for name, meas, ok in rows:
    print(f'  [{"PASS" if ok else "FAIL"}] {name}：实测 {meas}')
verdict = all(r[2] for r in rows)
print(f'\n总判定：{"M4 判据全部通过——相位均匀性以条件性命题形式成立，裁决点 3 凭据齐备" if verdict else "存在未过判据——按纲领失败预案核查"}')
print('层级声明：均匀性 = 条件性命题（无内禀零点 + 无共同时钟 / 背景噪声超过周期尺度），非独立新公设')
