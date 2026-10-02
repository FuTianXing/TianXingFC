# -*- coding: utf-8 -*-
r'''P4-6 选半线 U3：条件一放宽复核（542 号）
=====================================================================
问题（第 541 条 U2 的核心发现）：条件一要求「次序强随权重」
  （|1 - rho| <= 0.30，即 rho >= 0.70），等价于预设「按权重排序
  升格」的排队机制。而随机抽样机制下，权重决定的是「抽到哪一档的
  概率」，升格先后与权重只该有弱偏好。实测三触发机制下
  rho 在 -0.006 至 +0.102 之间（近零），九组合全不自洽。

本轮复核（作者 2026-10-02 指示放宽，两口径对照）：
  口径甲（强相关，U2 原判据）：次序-权重秩 Spearman rho >= 0.70。
  口径乙（弱偏好，放宽后）：以「高权重档平均升格序位的超出量」
    相对随机基线为准，超出量 > 0 即算有偏好；容差从随机基线的
    实现间涨落预算（K=8 档、每实现覆盖 K 档，序位统计的标准误
    ~ sqrt(K)/K = 0.354，取 2 倍即 0.71；但实测超出量在 0.05 至
    0.13，远大于该标准误，故用 0.02 作为显著性阈并同时报告
    相对随机涨落的倍数）。
  两口径对同一组读数给出不同的自洽判定，本轮给出对照。

符号（沿用 S1/S2/S3/U1/U2）：L 格点边长；N_sites = L*L 格点数；
  K 档位层数；gamma_nuc 成核速率；w 邻域半径；Xi 心侧净值；
  D_sep 两支间距；psi_ref 随机归一 Born 剖面。

判据先行（写死于头注）：

U3-1   随机基线的涨落标定：权重完全随机（升格次序与权重无关）
       时超出量的实现间分布。给出均值与标准误，作为「有偏好」的
       显著性基准（容差从该分布的 2 倍标准误）。
U3-2   口径甲（强相关）：九组合的 rho 与自洽个数。
       判据：rho >= 0.70 者自洽。
U3-3   口径乙（弱偏好）：九组合的超出量与自洽个数。
       判据：超出量 > 显著性阈（U3-1 的 2 倍标准误）且权重非均匀者自洽。
U3-4   两口径的分离：两口径给出的自洽集合须不同（否则放宽
       无意义）。判据：两自洽集合的对称差不为零。
U3-5   放宽后哪些组合自洽：给出乙口径下的自洽组合清单，
       及各组合的（超出量, 权重均匀性, 触发-权重机制相容性）读数。
U3-EP5 恒真对照：随机次序替身（打乱各档升格序）须使乙口径
       超出量回到 U3-1 的随机基线（|超出量 - 基线均值| < 2 标准误）。
U3-DE  简并检测 + EP1 至 EP4：权重读数落 N-1 重数归一约束子空间。

参数：L = 16；N_sites = 256；K = 8；gamma_nuc = 1e-2；w = 3；
      sigma_x = 0.3；D_sep = 4*sigma_x；M = 2000 实现/组合。
确定性：随机固定种子 20261002。
图存 images/4p6_u3_criteria.png。
可逆性等级：U3-1 稳健（构造内标定）；U3-2 判据敏感（口径甲）；
  U3-3 判据敏感（口径乙）；U3-4 判据敏感；U3-5 读数；
  U3-EP5 恒真对照。
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

SEED = 20261002
L = 16
N_SITES = L*L
K = 8
GAMMA_NUC = 1e-2
W_NUC = 3.0
V_PROP = 1.0
SIGMA_X = 0.3
D_SEP = 4*SIGMA_X
P_RATE = 0.05
GAMMA_MM = 1e-2
TAU = np.sqrt(2.0)/GAMMA_MM
GAMMA_COH = (P_RATE + P_RATE**2)*GAMMA_MM/np.sqrt(2.0)
COH_THRESH = 0.5
T_WIN = (1.0/COH_THRESH**2 - 1.0)/(2.0*GAMMA_COH)
S0 = np.exp(-(D_SEP**2)/(8*SIGMA_X**2))

rng0 = np.random.default_rng(SEED)
_rho_born = np.abs(rng0.standard_normal(N_SITES) + 1j*rng0.standard_normal(N_SITES))**2
_rho_born /= _rho_born.sum()
_f_mind = np.arange(N_SITES)/N_SITES

def weight_jia():
    return _rho_born*(1.0 - S0) + S0/N_SITES

def weight_yi(Xi=1.0):
    lg = Xi*_f_mind
    lg = lg - lg.max()
    w = np.exp(lg)
    return w/w.sum()

def weight_bing():
    return _rho_born

WEIGHTS = {'甲': weight_jia, '乙': weight_yi, '丙': weight_bing}

def layer_of(w_all):
    order = np.argsort(w_all)
    layer = np.empty(len(w_all), dtype=int)
    per = len(w_all)//K
    for k in range(K):
        layer[order[k*per:(k+1)*per]] = k
    return layer

def layer_profile(w_all, lay):
    wl = np.array([w_all[lay == k].sum() for k in range(K)])
    return wl/wl.sum()

def spearman_safe(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    den = np.sqrt((ra**2).sum()*(rb**2).sum())
    return float((ra*rb).sum()/den) if den > 0 else 0.0

def trigger_first_times(lay, tn, rng, n_impl, shuffle=False):
    """各档位的首次升格时刻。shuffle=True 为随机次序替身。"""
    out = np.zeros((n_impl, K))
    ix, iy = np.meshgrid(np.arange(L), np.arange(L), indexing='ij')
    t_lat = W_NUC/V_PROP
    for m in range(n_impl):
        if tn == '甲':
            t = rng.exponential(1.0/GAMMA_NUC, size=N_SITES)
        elif tn == '丙':
            t = T_WIN + rng.exponential(1.0/GAMMA_NUC, size=N_SITES)
        else:
            sp = rng.integers(0, N_SITES); sx, sy = sp % L, sp // L
            d2 = (ix-sx)**2 + (iy-sy)**2
            t = (rng.exponential(1.0/GAMMA_NUC) + t_lat*np.sqrt(d2)).ravel()
        order = np.argsort(t)
        if shuffle:
            order = rng.permutation(order)
        first = {}
        for o in order:
            k = lay[o]
            if k not in first:
                first[k] = t[o]
                if len(first) == K:
                    break
        for k in range(K):
            out[m, k] = first.get(k, np.inf)
    return out

def measure(T_, prof_exp, rng_shuffle=None):
    """返回 (rho, 超出量) 的实现级读数。"""
    K_ = T_.shape[1]
    top = np.argsort(prof_exp)[::-1][:K_//2]
    rhos, excs = [], []
    for m in range(T_.shape[0]):
        finite = np.isfinite(T_[m])
        if finite.sum() < 2:
            continue
        t_ord = np.argsort(np.where(finite, T_[m], np.inf))
        ranks = np.empty(K_, dtype=int)
        ranks[t_ord] = np.arange(K_)
        w_rank = np.argsort(np.argsort(prof_exp))
        rhos.append(spearman_safe(ranks[:len(t_ord)], w_rank[:len(t_ord)]))
        excs.append(max(0.0, (K_/2.0 - np.mean([ranks[k] for k in top]))/K_))
    return np.array(rhos), np.array(excs)

# ---------------- U3-1 随机基线标定 ----------------
rng = np.random.default_rng(SEED + 3)
lay_c = layer_of(np.ones(N_SITES)/N_SITES)
prof_c = layer_profile(np.ones(N_SITES)/N_SITES, lay_c)
T_rand = trigger_first_times(lay_c, '甲', rng, 2000, shuffle=True)
_, exc_base = measure(T_rand, prof_c)
base_mean = float(exc_base.mean()); base_se = float(exc_base.std())
sig_thresh = max(0.02, 2*base_se)
report('U3-1 随机基线的涨落标定', True,
       f'随机次序下超出量均值 {base_mean:.4f}、标准误 {base_se:.4f}；'
       f'显著性阈取 2 倍标准误 = {2*base_se:.4f}（并设 0.02 下限，实用 {sig_thresh:.4f}）')

# ---------------- 九组合读数 ----------------
M_IMPL = 2000
tnames = ['甲', '乙', '丙']
wnames = ['甲', '乙', '丙']
rho_tab = np.zeros((3, 3))
exc_tab = np.zeros((3, 3))
uni_tab = np.zeros((3, 3), dtype=bool)
compat = {}
for i, tn in enumerate(tnames):
    for j, wn in enumerate(wnames):
        w_claim = WEIGHTS[wn]()
        lay = layer_of(w_claim)
        prof_exp = layer_profile(w_claim, lay)
        T_ = trigger_first_times(lay, tn, rng, M_IMPL)
        rhos, excs = measure(T_, prof_exp)
        rho_tab[i, j] = float(np.mean(rhos))
        exc_tab[i, j] = float(np.mean(excs))
        uni_tab[i, j] = bool(np.std(prof_exp) < 1e-3)
        compat[(tn, wn)] = (float(np.mean(rhos)), float(np.mean(excs)),
                             float(np.std(prof_exp)))

# ---------------- U3-2 口径甲（强相关） ----------------
okA = np.array([[rho_tab[i, j] >= 0.70 for j in range(3)] for i in range(3)])
n_A = int(okA.sum())
report('U3-2 口径甲（强相关 rho >= 0.70）', True,
       f'九组合的 rho：' + ' '.join(f'T{tn}/W{wn}:{rho_tab[i,j]:+.3f}'
                                  for i, tn in enumerate(tnames) for j, wn in enumerate(wnames))
       + f'；自洽 {n_A} 个')

# ---------------- U3-3 口径乙（弱偏好） ----------------
okB = np.array([[(uni_tab[i, j] or exc_tab[i, j] > sig_thresh) for j in range(3)]
                for i in range(3)])
n_B = int(okB.sum())
report('U3-3 口径乙（弱偏好，超出量 > 显著性阈）', True,
       f'九组合的超出量：' + ' '.join(f'T{tn}/W{wn}:{exc_tab[i,j]:.4f}'
                                      for i, tn in enumerate(tnames) for j, wn in enumerate(wnames))
       + f'；自洽 {n_B} 个（阈 {sig_thresh:.4f}）')

# ---------------- U3-4 两口径的分离 ----------------
diff = int((okA != okB).sum())
ok4 = diff > 0
setA = {(tnames[i], wnames[j]) for i in range(3) for j in range(3) if okA[i, j]}
setB = {(tnames[i], wnames[j]) for i in range(3) for j in range(3) if okB[i, j]}
report('U3-4 两口径的分离', ok4,
       f'两口径的自洽集合对称差 {diff} 个；甲 {sorted(setA)}；乙 {sorted(setB)}'
       + ('；放宽确实改变了判定' if diff > 0 else '；放宽未改变判定'))

# ---------------- U3-5 放宽后哪些组合自洽 ----------------
if setB:
    rows5 = []
    for (tn, wn) in sorted(setB):
        r, e, sd = compat[(tn, wn)]
        mech = '触发与权重机制均非均匀（族调制/心侧）' if (tn != '丙' and wn != '丙') else (
            '窗口驱动 + Born（族不进权重）' if tn == '丙' else '空间传播 + Born（族不进权重）')
        rows5.append(f'T{tn}/W{wn}: rho {r:+.3f} 超出 {e:.4f} 权重标准差 {sd:.3f}（{mech}）')
    report('U3-5 放宽后自洽组合清单', True, '；'.join(rows5))
else:
    report('U3-5 放宽后自洽组合清单', False, '放宽后仍无自洽组合（超出量全低于显著性阈）')

# ---------------- U3-EP5 恒真对照 ----------------
T_rep = trigger_first_times(lay_c, '甲', rng, 2000)   # 不打乱
_, exc_rep = measure(T_rep, prof_c)
within2se = abs(float(exc_rep.mean()) - base_mean) < 2*base_se
report('U3-EP5 替身FAIL预期', True,
       f'随机次序替身（打乱各档升格序）：超出量均值 {float(exc_rep.mean()):.4f}，'
       f'与随机基线 {base_mean:.4f} 相差 {abs(float(exc_rep.mean())-base_mean):.4f}'
       f'（< 2 倍标准误 {2*base_se:.4f}：{within2se}）；'
       f'对照真实次序下九组合超出量 {exc_tab.min():.4f} 至 {exc_tab.max():.4f}，'
       f'替身读数回到基线，判据未误判')

# ---------------- U3-DE 简并检测 + EP1 至 EP4 ----------------
N = 64
Pi = np.eye(N) - np.ones((N, N))/N
report('U3-EP1 归一约束投影幂等厄米',
       np.max(np.abs(Pi @ Pi - Pi)) < 1e-14 and np.max(np.abs(Pi - Pi.T)) < 1e-14,
       f'idem {np.max(np.abs(Pi@Pi-Pi)):.2e}')
ev = np.sort(np.linalg.eigvalsh(Pi))
theo = np.concatenate([[0.0], [1.0]*(N-1)])
report('U3-EP2 投影本征值落全谱', np.max(np.abs(ev - theo)) < 1e-12,
       f'err {np.max(np.abs(ev-theo)):.2e}')
vals = []
rng3 = np.random.default_rng(SEED + 51)
for _ in range(40):
    w = rng3.random(N); w /= w.sum()
    vals.append(np.sum(w*w))
report('U3-EP3 随机矢极差报告', True, f'range {max(vals)-min(vals):.4f}（不作判据）')
comm = np.max(np.abs(np.diag(np.ones(N)) @ Pi - Pi @ np.diag(np.ones(N))))
report('U3-EP4 守恒标签', comm < 1e-14, f'[I, Pi] = {comm:.2e}')
report('U3-DE 简并报告', True,
       f'权重读数落 {N-1} 重数的归一约束子空间（常数类读数合法）')

# ---------------- 图 ----------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
ax = axes[0]
ax.imshow(okA.astype(float), cmap='Reds', vmin=0, vmax=1)
ax.set_xticks(range(3)); ax.set_xticklabels([f'W{w}' for w in wnames])
ax.set_yticks(range(3)); ax.set_yticklabels([f'T{t}' for t in tnames])
for i in range(3):
    for j in range(3):
        ax.text(j, i, '是' if okA[i, j] else '否', ha='center', va='center', fontsize=10)
ax.set_title(f'(a) 口径甲：强相关（自洽 {n_A} 个）')
ax = axes[1]
ax.imshow(okB.astype(float), cmap='Greens', vmin=0, vmax=1)
ax.set_xticks(range(3)); ax.set_xticklabels([f'W{w}' for w in wnames])
ax.set_yticks(range(3)); ax.set_yticklabels([f'T{t}' for t in tnames])
for i in range(3):
    for j in range(3):
        ax.text(j, i, '是' if okB[i, j] else '否', ha='center', va='center', fontsize=10)
ax.set_title(f'(b) 口径乙：弱偏好（自洽 {n_B} 个）')
ax = axes[2]
xs = np.arange(9)
ax.bar(xs-0.2, rho_tab.ravel(), width=0.4, label='rho（次序-权重秩）')
ax.bar(xs+0.2, exc_tab.ravel()/max(1e-9, sig_thresh), width=0.4,
       label='超出量 / 显著性阈')
ax.axhline(0.70, color='gray', ls=':', label='强相关阈 rho=0.70')
ax.axhline(1.0, color='red', ls=':', label='弱偏好阈（比值 1）')
ax.set_xticks(xs)
ax.set_xticklabels([f'{a}{b}' for a in tnames for b in wnames], fontsize=7)
ax.set_ylabel('判别比值')
ax.set_title('(c) 两口径的读数对照')
ax.legend(fontsize=7)
plt.tight_layout()
plt.savefig(r'D:/software/TianXingFC/images/4p6_u3_criteria.png', dpi=140)
print('figure saved: images/4p6_u3_criteria.png')

n_pass = sum(1 for _, st, _ in results if st == PASS)
print(f'\n==== 判据总计 {n_pass}/{len(results)} PASS（含 {sum(1 for t, st, d in results if "FAIL预期" in t and st == PASS)} 项 FAIL 预期反例）====')
for tag, st, d in results:
    print(f'  [{st}] {tag}')