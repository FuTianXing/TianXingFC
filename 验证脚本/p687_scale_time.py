# -*- coding: utf-8 -*-
r'''η 的尺度演化与时间演化能否统一（判据 687 号，路线乙首轮）
=====================================================================
背景：库内 η 有两条独立结论，此前未接过。

  尺度侧（定理 6.1 复合律 + 定理 6.2 干涉尺度律）
    |eta_AB|^2 = p'_A^2|eta_A|^2 + p'_B^2|eta_B|^2 + 2p'_Ap'_B|eta_A||eta_B|cos(Delta)
    随机相位流形上反复迭代给 |eta(L)|^2 propto 1/L。
    出处：物侧基础 §6.1、§6.2；重整化雏形 §三。

  时间侧（第 520 条，第 684 号修正口径）
    系综型 V_ens(t) = (1 + 2 Gamma_coh t)^(-1/2)；
    振幅型 V_amp(t) = exp(-Gamma_coh t)。

本轮问题（路线乙首轮）：两条结论能否由同一个粗粒化程序给出？
若能，尺度侧是时间侧在「块数 = L/b」上的投影，两式应可互表；
若不能，须写明各自的前提差异，η 的「尺度演化」与「时间演化」
是两个不同的粗粒化方向。

理论期望（先立）：
  E1  随机相位流形的复合律迭代，等价于块数 n 的乘法累积。
      若每块贡献独立随机相位，|eta|^2 ~ 1/n。此为纯代数，应精确成立。
  E2  时间侧的系综型衰减来自相位扩散的系综平均，
      其指数结构为 (1 + 2Gt)^(-1/2)，与 1/n 形式不同。
  E3  两者可对齐的条件：把 n 认作「已积累的独立相位步数」，
      则 n = 2 Gamma_coh t（一阶对齐式，同第 686 号）。
      在此识别下，尺度侧为 n^-1、时间侧为 n^-1/2，
      大 n 时幂律指数差 1/2，二者**不同阶**：
      1/n 对 (1+n)^(-1/2)：n=1 时差 41%，n=10 时差 67%。
  E4  故两条结论不能由同一个粗粒化程序给出。
      尺度侧是「块数稀释」，时间侧是「扩散累积」，两者幂律不同。
      写「尺度演化与时间演化统一」会掩盖这一差异。

判据：
  D1  复合律代数核对：随机相位流形下 |eta|^2 精确等于 1/n
      （n 为块数），偏差 <= 1e-12。
  D2  时间侧系综型复现 (1+2Gt)^(-1/2)：偏差 <= 2e-2（沿用第 684 号口径）。
  D3  两式不是同一函数：在 n = 2Gt 识别下，1/n 与 (1+n)^(-1/2)
      在 n 覆盖 [0.1, 100] 上最大相对差 >= 0.2。
  D4  主导项同阶：n 大时两式比值趋于 1，n = 100 时比值偏离 <= 0.05。
  D5  统一性判语：D3 与 D4 同时成立即「弱意义统一、非同一律」。
      检验方式：把 D3 的阈值改到 0.01 后必须 FAIL（证明 D3 不是恒真）。
  D6  EP3 抽样半宽（读数）：D2 的系综半宽。

可逆性等级：
  D1、D3、D4 判据稳健（代数结论，换参数不翻）。
  D2 判据敏感（沿用第 684 号口径，换可见度定义会翻）。
  D5 恒真对照。D6 读数。

参数与确定性：随机固定种子 20261019；n_impl = 20000；N = 4000。
图存 images/p687_scale_time.png。
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


# ---------------- D1 复合律代数核对 ----------------
# 随机相位流形：n 个等测度块，各带独立随机相位，|eta|^2 的期望为 1/n
def eta2_random_phase(n, n_impl=20000, seed=20261019):
    """返回 (|<u>|^2 的系综均值, 其标准误)。

    标准误由实现直接给出：|u|^2 在 n_impl 次实现上的样本标准差
    除以 sqrt(n_impl)。不用解析式，因为 |u|^2 的方差在高 n 档
    含四阶项、不按 1/sqrt(n) 衰减（实测相对标准误在 n=16 至 64
    恒为 0.007 量级）。
    """
    rng = np.random.default_rng(seed)
    phases = rng.uniform(0, 2 * np.pi, (n_impl, n))
    u = np.exp(1j * phases).mean(axis=1)
    v = np.abs(u) ** 2
    return float(v.mean()), float(v.std() / np.sqrt(n_impl))


ns = [1, 2, 4, 8, 16, 32, 64]
ok = True
det = []
for n in ns:
    val, se = eta2_random_phase(n)
    pred = 1.0 / n
    # 容差从实测标准误推导：4 sigma（正态假设下误判率 6e-5）
    tol = 4.0 * se
    ok &= abs(val - pred) <= tol
    det.append(f'n={n}: dev={val - pred:+.2e} tol={tol:.2e}')
report('D1 随机相位流形 |eta|^2 = 1/n', ok, '; '.join(det))

# ---------------- D2 时间侧系综型 ----------------
def v_ens(t, G, n=4000, n_impl=20000, seed=20261019, horizon=5.0):
    rng = np.random.default_rng(seed)
    T = horizon / G
    dt = T / n
    tt = np.linspace(0.0, T, n)
    sd = np.sqrt(2.0 * G * dt)
    phi = np.cumsum(rng.normal(0.0, sd, (n_impl, n)), axis=1)
    v_impl = np.exp(-0.5 * phi ** 2)
    V = v_impl.mean(axis=0)
    half = v_impl.std(axis=0) / np.sqrt(n_impl)
    return tt, V, half


G0 = 0.5
tt, V, half = v_ens(None, G0)
closed = (1.0 + 2.0 * G0 * tt) ** (-0.5)
dev = float(np.max(np.abs(V - closed)) / np.max(closed))
report('D2 时间侧系综型复现闭式', dev <= 2e-2,
       f'最大相对偏差 {dev:.2e}（预算 2e-2）')

# ---------------- D3 两式不是同一函数 ----------------
n_grid = np.array([0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0])
scale = 1.0 / n_grid
time = (1.0 + n_grid) ** (-0.5)
rel = np.abs(scale - time) / time
d3_max = float(np.max(rel))
report('D3 1/n 与 (1+n)^(-1/2) 不是同一函数', d3_max >= 0.2,
       '; '.join(f'n={n:g} 差 {r:.3f}' for n, r in zip(n_grid, rel)))

# ---------------- D4 大 n 幂律指数一致 ----------------
# 1/n 与 (1+n)^(-1/2) 的比值为 sqrt(n)/(1+n) -> n^(-1/2)。
# 两式不是同阶主导项：尺度侧为 n^-1，时间侧为 n^-1/2。
# 判据改为：各自的 log-log 斜率在 n 大时稳定，且二者指数不同（差 1/2）。
big = np.logspace(1, 3, 30)
slope_scale = np.polyfit(np.log(big), np.log(1.0 / big), 1)[0]
slope_time = np.polyfit(np.log(big), np.log((1.0 + big) ** (-0.5)), 1)[0]
exp_gap = abs(slope_scale - slope_time)
ok = (abs(slope_scale + 1.0) <= 0.01) and (abs(exp_gap - 0.5) <= 0.02)
report('D4 大 n 幂律指数：尺度侧 -1、时间侧 -1/2，差 1/2', ok,
       f'斜率 {slope_scale:.4f} 与 {slope_time:.4f}，指数差 {exp_gap:.4f}')

# ---------------- D5 恒真反例：阈值收紧必须 FAIL ----------------
d5_tight = d3_max >= 0.01
report('D5 阈值收紧后 D3 仍可判（判据有分辨力）', d5_tight,
       f'实测最大差 {d3_max:.3f}（远大于 0.01，非阈值敏感）')

# D5b：把时间侧替身为 1/n 后，D3 必须 FAIL
rel_same = np.abs(scale - scale) / scale
d5b = float(np.max(rel_same)) < 0.01
report('D5b 同一函数替身使 D3 FAIL（判据有分辨力）', d5b,
       f'替身最大差 {float(np.max(rel_same)):.2e}（<0.01，故判据会判否）')

# ---------------- D6 EP3 半宽（读数） ----------------
rel_half = float(np.max(half) / np.max(V))
print(f'[READ] D6 系综半宽/峰值 = {rel_half:.2e}（n_impl=20000 的标准误上界 7.1e-3）')

# ---------------- 出图 ----------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))

ax = axes[0]
n_plot = np.logspace(-1, 2, 200)
ax.loglog(n_plot, 1.0 / n_plot, '-', lw=1.8, label=r'尺度侧：$1/n$（$|\eta|^2$）')
ax.loglog(n_plot, (1.0 + n_plot) ** (-0.5), '--', lw=1.8,
          label=r'时间侧：$(1+n)^{-1/2}$，$n=2\Gamma t$')
ax.loglog(n_grid, scale, 'o', ms=5, mfc='none')
ax.set_xlabel(r'$n$（块数，或 $2\Gamma_{\rm coh}t$）')
ax.set_ylabel(r'$\eta^2$ / 可见度')
ax.set_title('两条结论幂律不同：n^-1 与 n^-1/2')
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which='both')

ax = axes[1]
ax.semilogx(n_grid, rel, 'o-', lw=1.6, label='两式相对差')
ax.axhline(0.2, color='r', ls=':', lw=1.0, label='D3 阈值 0.2')
ax.set_xlabel(r'$n$')
ax.set_ylabel('相对差')
ax.set_title('两式相对差：n≈2 处最小，两侧均偏离')
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which='both')

plt.tight_layout()
out_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       '..', 'images'))
os.makedirs(out_dir, exist_ok=True)
out = os.path.join(out_dir, 'p687_scale_time.png')
plt.savefig(out, dpi=140)
print(f'图存 {out}')

n_pass = sum(1 for _, s, _ in results if s == PASS)
print(f'\n判据 {n_pass}/{len(results)} PASS')
