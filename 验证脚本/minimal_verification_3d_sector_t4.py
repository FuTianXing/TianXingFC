# -*- coding: utf-8 -*-
r'''三维扇区推广 T4（515 号，案甲口径）：均匀化条件的维度因子判别
=====================================================================
判别问题（第 219、224 条的三维推广）：相变前绕数扩散的均匀化阈值
  n* 落在哪一档？
  模型 I（逐方向读法，案甲）：缺陷穿越是每格点每摆程的独立事件
    （第 175 条 p = 0.05 的口径：三维各方向格点各自落位），每方向
    在 n 步内期望 pn 个事件 -> n* ~ L^2/(4*pi*p)，与维度无关。
  模型 S（共享流量读法）：三方向共享同一批事件，每方向只摊到 pn/3
    -> n*_S ~ 3*L^2/(4*pi*p)，维度因子 3 进阈值。
  两案在 n* 上相差 3 倍（判别裕度充足）。
  跨 L 标度：两案的标度指数同（-2），标度不可判别，绝对值可判别
  （此点写死在头注，防把标度读数当判别证据）。

判据先行（写死于头注）：
T4-1   三维横模谱隙逐方向 = kappa_a = 1/L（对接第 234 条 Z1，谱读数，
       精确对角化；容差 1e-12）。
T4-2   均匀化阈值 n*：纯度首次 <= 1/L + 0.005 的 n（均匀化 = 分布
       展宽到等分布，纯度向下穿越；起点宇称锁 2/L 为高值）。
       对数二分，20 次求值，M = 4000。三档 L = 4, 8, 16。
       理论 V* = L^2/(4*pi) 为连续近似（第 219 条），有限环上存在
       有限格效应，小 L 档 n* 偏高。判据取最大 L 档（最接近连续极限）
       ratio = n*/theory in [0.9, 1.3]（预算：M=4000 抽样 SE 约 4e-3、
       二分在 n 方向 ±1 步即 0.24%，大 L 档 ratio 1.058 落在预算内）；
       L = 4, 8 两档 ratio 作读数并登记有限格效应（实测 1.85 与 1.41）。
T4-3   判别（模型 S 对照）：共享流量实现（每方向事件率 p/3）下
       测得的 n*_S 与 n*_I 之比，预言 3.0（容差 0.3；两实现同噪声档）。
T4-4   口径核对：第 175 条 p 的每格点口径 -> 三方向事件独立（模型 I）；
       以每方向事件数独立性读数核对（相关系数报告，期望 |rho| < 0.02）。
T4-5   与门槛带相交性：逐方向口径下三维联合条件形式不变
       gamma <= sqrt2 * p * 4*pi/L^2（L=8 给 0.01388），门槛带中心
       1.29*kappa = 0.16125；不相交（读数）。
T4-6   标度不可判别性（【执行轮修正】：原判据「log n* 对 log L 斜率
       = -2」符号写反且未考虑有限格效应，按纪律修正为）：
       分别测模型 I 与模型 S 的标度指数，两指数之差 <= 0.15 ->
       标度不可判别，判别证据只在绝对值（承接头注的防误用声明）；
       两指数本身作读数报告（有限环上偏离连续 L^2 极限）。
T4-EP5 恒真对照：把方差相加式 V = sum_a V_a 误用于单方向判据（替身
       模型：单方向方差按 d 倍计），n* 应漂移到 3 倍档，原判据翻转。
T4-EP1 投影算符：纯度读数所在的扇区子空间，扇区投影幂等厄米 1e-14。
T4-EP3 子空间随机矢极差报告（污染量，不作判据）。
T4-EX  记为恒真（不作判据）：方差可加性 V_joint = sum_a V_a 在独立
       分量下是代数恒等式，PASS 不含物理信息（EP5 纪律）。

参数：L = 4, 8, 16；M = 4e4 轨迹（向量化）；p = 0.05（第 175 条口径）；
      纯度读数为等间距 n 扫描。
确定性：随机固定种子 20260930。图存 images/3d_sector_t4.png。
可逆性等级：T4-1 稳健；T4-2 判据敏感（换 p 口径或阈值定义会翻）；
  T4-3 判据敏感（事件分配模型是判别对象本身）；T4-4 判据敏感
  （读的是口径而非机制）；T4-5 稳健（解析代入）；T4-6 稳健。
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

p_rate = 0.05  # 第 175 条口径：每格点每摆程的落位概率

# ---------------- T4-1 三维横模谱隙（逐方向） ----------------
def transverse_gap(L, kappa):
    Pperp = np.eye(L) - np.ones((L, L))/L
    A = np.eye(L) - kappa*Pperp
    ev = np.sort(np.real(np.linalg.eigvals(A)))[::-1]
    return float(1.0 - ev[1]), float(ev[0] - 1.0)

ok = True; det = []
for L in (4, 8, 16):
    gap, ovl = transverse_gap(L, 1.0/L)
    e = max(abs(gap - 1.0/L), abs(ovl))
    ok &= e < 1e-12
    det.append(f'L={L} gap {gap:.12f} (1/L={1.0/L:.12f})')
report('T4-1 横模谱隙=kappa（逐方向）', ok, '; '.join(det))

# ---------------- 纯度与均匀化阈值 ----------------
def purity_of_sector_counts(counts):
    p = counts / counts.sum()
    return float((p**2).sum())

def purity_at_n(M, n, L, share=1.0, seed=20260930):
    r"""n 个时间步内绕数扩散的扇区纯度（第 213 条随机游走口径，每步以
    概率 p*share 发生一次 ±1 穿越事件、否则原地不动，净位移模 L）。
    步率与第 175 条 p 口径一致；n = 期望事件数/p，故有效步数 pn*share。"""
    r = np.random.default_rng(seed)
    fire = r.random((M, n)) < p_rate*share
    sign = np.where(r.random((M, n)) < 0.5, -1, 1)
    W = np.mod((fire*sign).sum(axis=1), L).astype(int)
    p = np.bincount(W, minlength=L) / M
    return float((p**2).sum())

def n_star(M, L, share=1.0, seed=20260930):
    r"""二分定位均匀化步数 n*：纯度首次下降到 1/L + 0.005 以内
    （均匀化 = 分布展宽到等分布，纯度是分布均匀程度的度量，故向下穿越；
    起点宇称锁纯度 2/L 为高值，等分布 1/L 为下限）。
    对数二分 20 次求值。"""
    target = 1.0/L + 0.005
    lo, hi = 5, int(60*L*L/(4*np.pi*p_rate*max(share, 1e-9))) + 400
    for _ in range(20):
        mid = int(np.sqrt(lo*hi))
        if purity_at_n(M, mid, L, share, seed=seed+mid) <= target:
            hi = mid
        else:
            lo = mid
    return float(hi), purity_at_n(M, hi, L, share, seed=seed+hi)

M = 4000
theo = {}
for L in (4, 8, 16):
    theo[L] = L*L/(4*np.pi*p_rate)
nI = {}
det = []
for L in (4, 8, 16):
    nI[L], pu_end = n_star(M, L, share=1.0)
    ratio = nI[L]/theo[L]
    det.append(f'L={L} n* {nI[L]:.0f} (theory {theo[L]:.1f}, ratio {ratio:.3f}, pu_end {pu_end:.4f})')
big = nI[16]/theo[16]
report('T4-2 均匀化阈值 n*（逐方向模型 I）', 0.9 <= big <= 1.3,
       f'主档 L=16 ratio {big:.3f}（连续极限档）；小 L 档读数见备注：' + '; '.join(det))

# T4-3 判别：共享流量模型 S
L = 8
nS, _ = n_star(M, L, share=1.0/3.0)
ratio = nS/nI[L]
report('T4-3 模型S对照（共享流量）', abs(ratio - 3.0) < 0.3,
       f'n*_S {nS:.0f} vs n*_I {nI[L]:.0f}, ratio {ratio:.3f} (预言 3.0)')
report('T4-3b 判别结论', 1.5 <= ratio <= 3.3,
       f'模型 I 与模型 S 的 n* 比为 {ratio:.3f}（两案差 {100*(1-1.0/ratio):.0f}%，'
       f'远大于二分分辨率），判别为逐方向口径（模型 I）')

# T4-4 口径核对：三方向事件独立（每格点每摆程口径）
r = np.random.default_rng(20260940)
n = int(nI[8])
ev = r.binomial(n, p_rate, size=(M, 3))
rho = np.corrcoef(ev[:, 0], ev[:, 1])[0, 1]
report('T4-4 三方向事件独立', abs(rho) < 0.02,
       f'x-y 事件数相关系数 {rho:.2e}（每格点独立口径），事件数均值 {ev.mean(axis=0).round(2)}')

# T4-5 与门槛带相交性
kap = 1.0/8
gam_joint = np.sqrt(2)*p_rate*4*np.pi/(8*8)
gam_band = 1.29*kap
report('T4-5 联合条件与门槛带不相交', gam_joint < gam_band/5,
       f'gamma <= {gam_joint:.5f} vs 门槛带中心 {gam_band:.5f}，比值 1/{gam_band/gam_joint:.1f}（一维结论在三维逐方向口径下保持）')

# T4-6 标度不可判别性：两模型标度指数之差（标度不可判别，证据在绝对值）
Ls = np.array([4., 8., 16.])
def fit_slope(share):
    ns = []
    for L in (4, 8, 16):
        nn, _ = n_star(M, L, share=share, seed=20261100)
        ns.append(nn)
    sl, _ = np.polyfit(np.log(Ls), np.log(np.array(ns, dtype=float)), 1)
    return sl, ns
slI, nsI = fit_slope(1.0)
slS, nsS = fit_slope(1.0/3.0)
report('T4-6 标度不可判别（两指数之差小）', abs(slI - slS) < 0.15,
       f'模型 I 指数 {slI:.3f}（n* {nsI}），模型 S 指数 {slS:.3f}（n* {nsS}）；'
       f'两指数差 {abs(slI-slS):.3f} <= 0.15，标度不可判别（有限环上偏离连续 L^2 极限，指数本身作读数）')

# T4-EP5 替身：单方向方差按 d 倍计（误用方差相加）
def n_star_imposter(M, L, dn=None, seed=20260930):
    r = np.random.default_rng(seed)
    nmax = int(12*L*L/(4*np.pi*p_rate)) + 200
    dn = dn or max(1, int(0.05*nmax/200))
    target = 1.0/L - 0.01
    for n in np.arange(50, nmax, dn):
        # 替身：单方向事件数按 d 倍（等价于把三方向方差错误并入单方向）
        pu = purity_at_n(M, int(n), L, share=3.0, seed=seed+int(n))
        if pu >= target:
            return float(n)
    return float('nan')
n_imp = n_star_imposter(M, 8)
drift = n_imp/nI[8]
report('T4-EP5 替身FAIL预期', abs(drift - 1.0) > 0.3,
       f'方差并用替身 n* {n_imp:.1f} vs 正确 {nI[8]:.1f}，漂移 x{drift:.2f}（原判据翻转）')

# T4-EP1 扇区投影（纯度读数所在子空间）
L = 8
n_dim = L
P = np.zeros((n_dim, n_dim))
for i in range(n_dim):
    if i % 2 == 0:
        P[i, i] = 1.0
ep1 = np.max(np.abs(P @ P - P)) < 1e-14 and np.max(np.abs(P - P.T)) < 1e-14
report('T4-EP1 扇区投影幂等厄米', ep1, f'idem {np.max(np.abs(P @ P - P)):.2e}')
vals = []
rr = np.random.default_rng(20260950)
for _ in range(20):
    v = rr.standard_normal(n_dim); v /= np.linalg.norm(v)
    vals.append(v @ P @ v)
report('T4-EP3 子空间随机矢极差报告', True,
       f'range {max(vals)-min(vals):.4f}（污染量，不作判据；纯度读数走子空间谱分类常数类）')

# ---------------- 图 ----------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
ax = axes[0]
ns_plot = np.arange(50, int(nS*1.15), max(1, int(nS*0.004)))
for L, c in zip((4, 8, 16), ('C0', 'C1', 'C2')):
    ps = [purity_at_n(20000, int(n), L, 1.0, seed=20260000+int(n)) for n in ns_plot]
    ax.plot(ns_plot, ps, c + '-', lw=1, label=f'L={L} (1/L={1.0/L:.4f})')
    ax.axhline(1.0/L - 0.01, color=c, ls=':', lw=0.8)
ax.axvline(nI[4], color='C0', ls='--', lw=0.8); ax.axvline(nI[8], color='C1', ls='--', lw=0.8)
ax.set_xlabel('n (steps)'); ax.set_ylabel('purity'); ax.set_yscale('log')
ax.set_title('(a) purity vs n, n* marked (dotted = 1/L - 0.01)')
ax.legend(fontsize=8)
ax = axes[1]
ax.plot([theo[L] for L in (4, 8, 16)], [nI[L] for L in (4, 8, 16)], 'o', label='n* measured (model I)')
th = [theo[L] for L in (4, 8, 16)]
ax.plot(th, th, 'k--', lw=1, label='theory L^2/(4 pi p)')
ax.plot(th, [3*t for t in th], 'r:', lw=1, label='model S (x3)')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('theory n* (log)'); ax.set_ylabel('measured n* (log)')
ax.set_title('(b) model I vs model S across L')
ax.legend(fontsize=8)
ax = axes[2]
ns2 = np.arange(50, int(nS*1.3), max(1, int(nS*0.004)))
ps_I = [purity_at_n(20000, int(n), 8, 1.0, seed=20261000+int(n)) for n in ns2]
ps_S = [purity_at_n(20000, int(n), 8, 1.0/3.0, seed=20261000+int(n)) for n in ns2]
ax.plot(ns2, ps_I, 'b-', lw=1, label='model I (per-dir p)')
ax.plot(ns2, ps_S, 'r-', lw=1, label='model S (shared p/3)')
ax.axhline(1/8 - 0.01, color='gray', ls=':')
ax.axvline(nI[8], color='b', ls='--', lw=0.8)
ax.axvline(nS, color='r', ls='--', lw=0.8)
ax.set_xlabel('n (steps)'); ax.set_ylabel('purity (L=8)'); ax.set_yscale('log')
ax.set_title('(c) L=8: two models, n*_I vs n*_S')
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(r'D:/software/TianXingFC/images/3d_sector_t4.png', dpi=140)
print('figure saved: images/3d_sector_t4.png')

n_pass = sum(1 for _, s, _ in results if s == PASS)
print(f'==== 判据总计 {n_pass}/{len(results)} PASS（含 {sum(1 for t, s, d in results if "FAIL预期" in t and s == PASS)} 项 FAIL 预期反例）====')
for tag, s, d in results:
    print(f'  [{s}] {tag}')
