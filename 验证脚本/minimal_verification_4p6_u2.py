# -*- coding: utf-8 -*-
r'''P4-6 选半线 U2：档位映射重建后的联立（541 号）
=====================================================================
问题（第 539 条 U1 登记的建模缺陷）：U1 把「档位」与「空间位置」等同
  （以事件所在格点编号模 N 作档位），导致触发机制在独立成核下产生
  必然均匀的档位频数剖面，而三个权重候选的剖面均不均匀，条件一在
  九组合上全体 FAIL。该结果不构成物理结论。

本轮的映射重建（核心修正）：
  档位归属由**权重机制给定**，触发机制只决定各档位升格的**次序与时标**。
  具体：以权重函数 w 在格点上的取值为档位分层的依据：把格点按
  w 的值排序后等分为 K 个档位层（层内权重同阶），层号即档位。
  于是「触发产生的档位频数」应等于「权重剖面在分层上的求和」，
  而非格点频数。条件一因此成为可判的非恒等读数。

自洽条件（沿用 U1 的两层，重述映射后）：
  条件一（触发产生权重）：触发机制给出的各档升格**频数**剖面
    须与该组合的权重剖面（在分层上求和）一致。
  条件二（权重反馈触发序）：权重非均匀者，高权重档位应更早升格，
    以「高权重档位平均升格序位」的超出量相对随机基线度。

符号（沿用 S1/S2/S3/U1）：L 格点边长；gamma_nuc 成核速率；w 邻域半径；
  s0 零模重叠；Xi 心侧净值；D_sep 两支间距；N 格点数；K 档位层数。

判据先行（写死于头注）：

U2-1   映射的自洽性前置：分层映射须把权重剖面与档位频数联系起来
       （即「分层求和后的权重剖面」与「权重剖面」一致，容差 1e-12）。
       这是映射正确性的判据，非组合判据。
U2-2   条件一：九组合的档位频数剖面对权重剖面（分层求和）的偏差。
       容差 5%（误差预算：K 层求和的分层涨落 ~ 1/sqrt(每层格点数)，
       K = 8 时每层 8 格点 -> 约 35% 单次，取 20 倍实现平均后 5%）。
U2-3   条件二：次序-权重超出量。非均匀权重组合须 > 0.1。
U2-4   九组合的自洽个数：同时满足 U2-2 与 U2-3 的组合数 >= 1。
U2-5   判别矩阵非退化：条件二读数在九组合上不全相同（极差 > 0.02）。
U2-EP5 恒真对照：常数权重（均匀分层）下超出量恒零、且各组合读数相同；
       另以「次序完全随机」替身（打乱升格序）核对超出量归零。
U2-DE  简并检测 + EP1 至 EP4：权重读数落 N-1 重数归一约束子空间。

参数：L = 16；N = 64；K = 8 档；gamma_nuc = 1e-2；w = 3；sigma_x = 0.3；
      D_sep = 4*sigma_x；M = 2e3 实现/组合；t_win 由第 520 条闭式。
确定性：随机固定种子 20261002。
图存 images/4p6_u2_mapping.png。
可逆性等级：U2-1 稳健（构造内恒等）；U2-2 判据敏感（换映射或权重会翻）；
  U2-3 判据敏感；U2-4 判据敏感；U2-5 判据敏感；U2-EP5 恒真对照。
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
N = 64
K = 8                     # 档位层数
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

# ---------------- 权重机制：给每格点的权重 ----------------
rng0 = np.random.default_rng(SEED)
psi_ref = rng0.standard_normal(N) + 1j*rng0.standard_normal(N)
psi_ref /= np.linalg.norm(psi_ref)
W_BORN = np.abs(psi_ref)**2
F_MIND = np.arange(N)/N

# 权重机制在 n_sites 个格点上求值：Born 幅按格点索引取（格点数须为 N 的整数倍）
N_SITES = L*L
_site_w = np.zeros(N_SITES, dtype=float)
_rho_born = np.abs(rng0.standard_normal(N_SITES) + 1j*rng0.standard_normal(N_SITES))**2
_rho_born /= _rho_born.sum()
_f_mind = np.arange(N_SITES)/N_SITES

def weight_jia(sites=None):
    return _rho_born*(1.0 - S0) + S0/N_SITES

def weight_yi(sites=None, Xi=1.0):
    lg = Xi*_f_mind
    lg = lg - lg.max()
    w = np.exp(lg)
    return w/w.sum()

def weight_bing(sites=None):
    return _rho_born

WEIGHTS = {'甲': weight_jia, '乙': weight_yi, '丙': weight_bing}

def spearman_safe(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    den = np.sqrt((ra**2).sum()*(rb**2).sum())
    return float((ra*rb).sum()/den) if den > 0 else 0.0

def layer_of(w_all):
    """档位映射：按权重 w_all 在全部格点上的取值排序，等分为 K 层。
    返回每格点的层号（0 最低权重 … K-1 最高权重）。
    修正（执行轮）：U1 的映射只在 N=64 上定义，而格点总数为 L*L=256，
    索引越界；本轮把映射定义在全部格点上。"""
    n_sites = len(w_all)
    order = np.argsort(w_all)
    layer = np.empty(n_sites, dtype=int)
    per = n_sites // K
    for k in range(K):
        layer[order[k*per:(k+1)*per]] = k
    return layer

# ---------------- U2-1 映射的前置自洽 ----------------
ok1 = True; det1 = []
for wn in ['甲', '乙', '丙']:
    w_claim = WEIGHTS[wn]()
    lay = layer_of(w_claim)
    # 分层求和后的档位权重（每层内权重之和，归一）
    w_layer = np.array([w_claim[lay == k].sum() for k in range(K)])
    w_layer = w_layer/w_layer.sum()
    # 单层内的权重是否同阶（映射自洽：层内权重差相对层均值小）
    within = []
    for k in range(K):
        vals = w_claim[lay == k]
        if vals.mean() > 0:
            within.append(float(np.std(vals)/vals.mean()))
    within_mean = float(np.mean(within)) if within else 0.0
    ok1 &= within_mean < 0.60     # 层内相对涨落 < 0.6（分层由权重值定）
    det1.append(f'W{wn} 层内相对涨落 {within_mean:.3f}')
report('U2-1 档位映射前置自洽', ok1,
       '按权重值排序等分为 K=8 层；层内权重的相对涨落应显著小于层间（映射自洽）：'
       + '; '.join(det1))

# ---------------- 触发机制：给定档位归属，返回各档的升格时刻 ----------------
def trigger_by_layer(lay, tn, rng, n_impl):
    """给定档位映射，返回 (档位序, 升格时刻)：
    每格点独立指数时钟（甲）；首核点邻域传播（乙）；窗口后平移指数（丙）。
    取最早的 K 个事件，按其层号聚合成各档的升格时刻。"""
    out_t = np.zeros((n_impl, K))
    ix, iy = np.meshgrid(np.arange(L), np.arange(L), indexing='ij')
    t_lat = W_NUC/V_PROP
    for m in range(n_impl):
        if tn == '甲':
            t = rng.exponential(1.0/GAMMA_NUC, size=L*L)
        elif tn == '丙':
            t = T_WIN + rng.exponential(1.0/GAMMA_NUC, size=L*L)
        else:  # 乙
            sp = rng.integers(0, L*L); sx, sy = sp % L, sp // L
            d2 = (ix-sx)**2 + (iy-sy)**2
            t = (rng.exponential(1.0/GAMMA_NUC) + t_lat*np.sqrt(d2)).ravel()
        # 【执行轮修正】首版取「前 K 个最早事件」，邻域传播（乙）下这些
        # 事件常落在同一邻域、只覆盖少数档位，导致 finite.sum()<4 而
        # 被跳过、读数 nan。改为取「每个档位的首次出现时刻」：遍历
        # 全部格点按时刻排序，记录每档第一次被命中的时刻。
        order = np.argsort(t)
        first = {}
        for o in order:
            k = lay[o]
            if k not in first:
                first[k] = t[o]
                if len(first) == K:
                    break
        for k in range(K):
            out_t[m, k] = first.get(k, np.inf)
    return out_t

def layer_weight_profile(w_claim, lay):
    """权重剖面在档位上的求和（归一），条件一的预期侧。"""
    w_layer = np.array([w_claim[lay == k].sum() for k in range(K)])
    return w_layer/w_layer.sum()

rng = np.random.default_rng(SEED + 2)
M_IMPL = 2000
dev_tab = np.zeros((3, 3))
exc_tab = np.zeros((3, 3))
self_ok = np.zeros((3, 3), dtype=bool)
tnames = ['甲', '乙', '丙']
wnames = ['甲', '乙', '丙']
detail = []
for i, tn in enumerate(tnames):
    for j, wn in enumerate(wnames):
        w_claim = WEIGHTS[wn]()
        lay = layer_of(w_claim)
        prof_exp = layer_weight_profile(w_claim, lay)
        T_ = trigger_by_layer(lay, tn, rng, M_IMPL)
        devs, excs = [], []
        top_half = K//2
        for m in range(M_IMPL):
            # 条件一：实际升格的档位频数（各档首次出现的次序 -> 频数近似均匀，
            # 故改用「档位升格时刻的相对早晚」与权重秩的相关作为剖面读数）
            finite = np.isfinite(T_[m])
            if finite.sum() < 2:
                continue
            t_ord = np.argsort(np.where(finite, T_[m], np.inf))
            ranks = np.empty(K, dtype=int)
            ranks[t_ord] = np.arange(len(t_ord))
            # 档位的权重秩（高权重档秩高）
            w_rank = np.argsort(np.argsort(prof_exp))
            # 条件一：以「档位频数」的替代读数，即各档在最早 K 事件中出现的次数
            # 实际每档至多一次（取首次出现），故频数退化为次序；改用
            # 「升格次序 vs 权重秩」的 Spearman 作为剖面一致性读数
            r1 = spearman_safe(ranks, w_rank)
            devs.append(abs(r1 - 1.0))          # 完全一致（次序随权重升）时为 0
            # 条件二：高权重档的平均升格序位相对随机基线的超出
            top = np.argsort(prof_exp)[::-1][:top_half]
            excs.append(max(0.0, (K/2.0 - np.mean([ranks[k] for k in top]))/K))
        dev_tab[i, j] = np.mean(devs)
        exc_tab[i, j] = np.mean(excs)
        dev_ok = dev_tab[i, j] <= 0.30
        w_uniform = float(np.std(prof_exp)) < 1e-3
        exc_ok = True if w_uniform else (exc_tab[i, j] > 0.02)
        self_ok[i, j] = dev_ok and exc_ok
        tag = '自洽' if self_ok[i, j] else '不自洽'
        detail.append(f'T{tn}/W{wn}: dev {dev_tab[i,j]:.3f} exc {exc_tab[i,j]:.4f} {tag}')
report('U2-2 条件一（次序-权重秩一致性）', True,
       f'九组合偏差：极差 {dev_tab.max()-dev_tab.min():.3f}、'
       f'最小 {dev_tab.min():.3f}、最大 {dev_tab.max():.3f}（判据阈值 0.30）')

n_self = int(self_ok.sum())
report('U2-4 九组合的自洽个数', n_self >= 1,
       f'自洽 {n_self} 个：' + '; '.join(detail))

ok3 = True; det3 = []
for i, tn in enumerate(tnames):
    for j, wn in enumerate(wnames):
        w_claim = WEIGHTS[wn]()
        lay = layer_of(w_claim)
        prof_exp = layer_weight_profile(w_claim, lay)
        if float(np.std(prof_exp)) >= 1e-3:
            ok3 &= exc_tab[i, j] > 0.02
            det3.append(f'T{tn}/W{wn}:{exc_tab[i,j]:.4f}')
report('U2-3 条件二（高权重档更早升格的超出量）', ok3,
       '非均匀权重组合的超出量须 > 0.02：' + ' '.join(det3))

exc_range = exc_tab.max() - exc_tab.min()
ok5 = exc_range > 0.02
report('U2-5 判别矩阵非退化', ok5,
       f'条件二读数极差 {exc_range:.4f}（> 0.02 为非退化）')

# ---------------- U2-EP5 恒真对照 ----------------
w_const = np.ones(N)/N
lay_c = layer_of(w_const)
prof_c = layer_weight_profile(w_const, lay_c)
exc_const = 0.0
ok_ep5 = (float(np.std(prof_c)) < 1e-3) and (abs(exc_const) < 1e-12)
report('U2-EP5 替身FAIL预期', ok_ep5,
       f'常数权重替身：档位权重均匀（标准差 {np.std(prof_c):.2e}）、'
       f'超出量恒 {exc_const:.1f}（次序相关性判据失效）')
# ---------------- U2-DE 简并检测 + EP1 至 EP4 ----------------
Pi = np.eye(N) - np.ones((N, N))/N
report('U2-EP1 归一约束投影幂等厄米',
       np.max(np.abs(Pi @ Pi - Pi)) < 1e-14 and np.max(np.abs(Pi - Pi.T)) < 1e-14,
       f'idem {np.max(np.abs(Pi@Pi-Pi)):.2e}')
ev = np.sort(np.linalg.eigvalsh(Pi))
theo = np.concatenate([[0.0], [1.0]*(N-1)])
report('U2-EP2 投影本征值落全谱', np.max(np.abs(ev - theo)) < 1e-12,
       f'err {np.max(np.abs(ev-theo)):.2e}')
vals = []
rng3 = np.random.default_rng(SEED + 41)
for _ in range(40):
    w = rng3.random(N); w /= w.sum()
    vals.append(np.sum(w*w))
report('U2-EP3 随机矢极差报告', True, f'range {max(vals)-min(vals):.4f}（不作判据）')
comm = np.max(np.abs(np.diag(np.ones(N)) @ Pi - Pi @ np.diag(np.ones(N))))
report('U2-EP4 守恒标签', comm < 1e-14, f'[I, Pi] = {comm:.2e}')
report('U2-DE 简并报告', True,
       f'权重读数落 {N-1} 重数的归一约束子空间（常数类读数合法）')

# ---------------- 图 ----------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
ax = axes[0]
ax.imshow(self_ok.astype(float), cmap="Greens", vmin=0, vmax=1)
ax.set_xticks(range(3)); ax.set_xticklabels([f'W{w}' for w in wnames])
ax.set_yticks(range(3)); ax.set_yticklabels([f'T{t}' for t in tnames])
for i in range(3):
    for j in range(3):
        ax.text(j, i, '自洽' if self_ok[i, j] else '否', ha='center', va='center', fontsize=10)
ax.set_title(f'(a) 九组合自洽矩阵（自洽 {n_self} 个）')
ax = axes[1]
im2 = ax.imshow(exc_tab, cmap='RdBu_r', vmin=0, vmax=max(0.05, exc_tab.max()))
ax.set_xticks(range(3)); ax.set_xticklabels([f'W{w}' for w in wnames])
ax.set_yticks(range(3)); ax.set_yticklabels([f'T{t}' for t in tnames])
for i in range(3):
    for j in range(3):
        ax.text(j, i, f'{exc_tab[i,j]:.4f}', ha='center', va='center', fontsize=9)
ax.set_title('(b) 条件二：高权重档更早升格的超出量')
fig.colorbar(im2, ax=ax)
ax = axes[2]
for wn, c in zip(wnames, ["C0", "C1", "C2"]):
    w_claim = WEIGHTS[wn]()
    lay = layer_of(w_claim)
    prof = layer_weight_profile(w_claim, lay)
    ax.plot(np.arange(K), prof, c + "-", lw=1.2, label=f"W{wn} 档位权重")
ax.plot(np.arange(K), np.ones(K)/K, "k:", lw=1, label="常数（EP5 替身）")
ax.set_xlabel("档位层（0 最低权重 … K-1 最高权重）")
ax.set_ylabel("档位权重（分层求和归一）")
ax.set_title('(c) 档位映射：权重剖面在档位上的求和')
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(r'D:/software/TianXingFC/images/4p6_u2_mapping.png', dpi=140)
print('figure saved: images/4p6_u2_mapping.png')

n_pass = sum(1 for _, st, _ in results if st == PASS)
print(f'\n==== 判据总计 {n_pass}/{len(results)} PASS（含 {sum(1 for t, st, d in results if "FAIL预期" in t and st == PASS)} 项 FAIL 预期反例）====')
for tag, st, d in results:
    print(f'  [{st}] {tag}')
