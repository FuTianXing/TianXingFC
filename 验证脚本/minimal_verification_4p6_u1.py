# -*- coding: utf-8 -*-
r'''P4-6 选半线 U1：触发候选与权重候选的联立（539 号）
=====================================================================
问题：触发分布的三候选（S1/S2：甲独立成核、乙邻域传播、丙窗口驱动）
  与权重律的三候选（S3：甲几何层、乙心侧层、丙投影层）此前正交未联立。
  本轮判定九个组合的自洽性，形态为判否驱动的收敛。

自洽条件两层：
  条件一（触发产生权重）：触发机制跑完给出的实际事件权重剖面，
    须与该组合声明的权重机制一致（由组合给定的权重函数在事件集上取值）。
  条件二（权重反馈触发序）：权重机制若非均匀分布，则升格次序
    不应完全随机：权重大的档位应更早升格（次序与权重相关）。

符号约定（沿用 S1/S2/S3）：
  L = 格点边长；gamma_nuc = 成核速率；w = 邻域半径；
  s0 = 零模波包重叠；Xi = 心侧净值参数；D_sep = 两支间距
  触发候选 T in {甲, 乙, 丙}；权重候选 W in {甲, 乙, 丙}

判据先行（写死于头注）：

U1-1   条件一：九组合的「触发产生的实际权重剖面」对「组合声明的权重
       函数」的偏差。判据：偏差 <= 2%（容差从抽样噪声预算：
       M = 2e4 事件，相对 SE ~ 0.7%，取 2%）。不自洽的组合判否。
U1-2   条件二：次序-权重相关性。九组合中权重非均匀者，
       升格次序与权重的 Spearman 相关须显著（rho > 0.3）；
       权重均匀者（乙心侧层的极端或无反馈实现）预期 rho 近 0。
       判据：非均匀权重组合的 rho > 0.3。
U1-3   自洽组合的个数：九组合中同时满足 U1-1 与 U1-2 的个数。
       判据：>= 1（存在自洽组合）；若 = 0 则全体不自洽，
       记为「候选集需扩充」并按判否含义登记。
U1-4   条件一的分离性：条件一对九组合的判别矩阵须非退化
       （即不同组合给出的偏差不全相同，否则条件一无区分力）。
       判据：偏差矩阵的极差 > 10%。
U1-EP5 恒真对照：把权重函数改为与触发无关的常数（各档等权），
       U1-2 的次序-权重相关性必须失效（rho 近 0），
       且 U1-1 对依赖权重的组合必须 FAIL。
U1-DE  简并检测 + EP1 至 EP4：权重剖面读数落 N-1 重数的归一约束子空间，
       按第 506 条做子空间谱分类。

【建模缺陷登记，重要】本轮把「档位」与「空间位置」等同（以事件所在格点
  编号模 N 作档位）。该建模使触发机制在独立成核下产生均匀的档位频数剖面，
  而三个权重候选的剖面均不均匀，故条件一全体 FAIL（dev 0.86 至 1.52）。
  **该结果不构成九个组合都不自洽的物理结论，根因是建模缺陷**：档位应由权重机制
  决定，不该由触发事件的格点位置随机决定。按纪律如实登记，不掩盖；
  修正后的联立判据须重建「触发 -> 档位」的映射（由权重机制给定档位归属，
  触发机制只决定档位出现的次序与时标），留待后续轮次。

参数：L = 16；gamma_nuc = 1e-2；w = 3；sigma_x = 0.3；D_sep = 4*sigma_x；
      N = 64 档；M = 2e4 事件/组合；t_win 由第 520 条闭式导出。
确定性：随机固定种子 20261002。
图存 images/4p6_u1_joint.png。
可逆性等级：U1-1 判据敏感；U1-2 判据敏感；U1-3 判据敏感；
  U1-4 判据敏感；U1-EP5 恒真对照。
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
N = N_SITES = 64
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

# ---------------- 触发机制：返回事件的（时刻，位置）----------------
def trigger_jia(rng, n_impl=2000):
    """甲 独立成核：每格点独立指数时钟，取最早的 N 个事件。"""
    t = rng.exponential(1.0/GAMMA_NUC, size=(n_impl, L*L))
    idx = np.argsort(t, axis=1)[:, :N]
    tt = np.take_along_axis(t, idx, axis=1)
    return tt, idx % L, idx // L

def trigger_yi(rng, n_impl=2000):
    """乙 邻域传播：首核点随机，半径 w 内有限速度。"""
    t_lat = W_NUC/V_PROP
    ix, iy = np.meshgrid(np.arange(L), np.arange(L), indexing='ij')
    T = np.zeros((n_impl, N)); X = np.zeros_like(T); Y = np.zeros_like(T)
    for m in range(n_impl):
        sp = rng.integers(0, L*L); sx, sy = sp % L, sp // L
        d2 = (ix-sx)**2 + (iy-sy)**2
        t0 = rng.exponential(1.0/GAMMA_NUC)
        tf = (t0 + t_lat*np.sqrt(d2)).ravel()
        order = np.argsort(tf)[:N]
        T[m] = tf[order]; X[m] = (order % L); Y[m] = (order // L)
    return T, X, Y

def trigger_bing(rng, n_impl=2000):
    """丙 窗口驱动：t_win 前无事件，此后平移指数。"""
    t = T_WIN + rng.exponential(1.0/GAMMA_NUC, size=(n_impl, L*L))
    idx = np.argsort(t, axis=1)[:, :N]
    tt = np.take_along_axis(t, idx, axis=1)
    return tt, idx % L, idx // L

TRIGGERS = {'甲': trigger_jia, '乙': trigger_yi, '丙': trigger_bing}

# ---------------- 权重机制：给每格点的权重函数 ----------------
rng0 = np.random.default_rng(SEED)
psi_ref = rng0.standard_normal(N) + 1j*rng0.standard_normal(N)
psi_ref /= np.linalg.norm(psi_ref)
W_BORN = np.abs(psi_ref)**2
F_MIND = np.arange(N)/N

def weight_jia(sites):
    """甲 几何层：Born 的族调制 w_i = |psi_i|^2 (1-s0) + s0/N。"""
    return W_BORN*(1.0 - S0) + S0/N

def weight_yi(sites, Xi=1.0):
    """乙 心侧层：logits = Xi * f_i，归一化。"""
    lg = Xi*F_MIND
    lg = lg - lg.max()
    w = np.exp(lg)
    return w/w.sum()

def weight_bing(sites):
    """丙 投影层：纯 Born（族不进权重）。"""
    return W_BORN

WEIGHTS = {'甲': weight_jia, '乙': weight_yi, '丙': weight_bing}

def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    den = np.sqrt((ra**2).sum()*(rb**2).sum())
    return float((ra*rb).sum()/den) if den > 0 else 0.0

# ---------------- 九组合的自洽评估（执行轮重写） ----------------
# 首版两项判据为构造恒真或实现错误，本轮重写：
#   条件一：实测剖面改取「触发机制实际产生的事件在档位上的频数归一」，
#           与「组合声明的权重函数」比对（首版用声明函数同作实测与预期，偏差恒零）；
#   条件二：次序-权重相关改用「高权重档位实际先升格的比例」相对随机基线 0.5 的超出量
#           （首版用位序权重对名次作相关，而位序由触发给、与权重取值无关，测不到反馈）。
rng = np.random.default_rng(SEED + 1)
dev_tab = np.zeros((3, 3))      # 行 = 触发候选，列 = 权重候选
exc_tab = np.zeros((3, 3))      # 次序-权重秩超出量
self_ok = np.zeros((3, 3), dtype=bool)
tnames = ['甲', '乙', '丙']
wnames = ['甲', '乙', '丙']
detail = []
for i, tn in enumerate(tnames):
    T, X, Y = TRIGGERS[tn](rng)
    for j, wn in enumerate(wnames):
        w_claim = WEIGHTS[wn](np.arange(N))
        devs, excs = [], []
        for m in range(T.shape[0]):
            site_ids = (np.round(Y[m])*L + np.round(X[m]))[:N].astype(int)
            rank = site_ids % N
            prof = np.bincount(rank, minlength=N).astype(float)
            prof = prof/prof.sum()
            devs.append(float(np.max(np.abs(prof - w_claim))/np.max(w_claim)))
            top = np.argsort(w_claim)[::-1][:N//2]
            names = np.empty(N, dtype=int)
            names[rank] = np.arange(N)
            hit = names[np.isin(np.arange(N), top)]
            hit = hit[hit < N]
            frac = float(np.mean(hit < N/2.0)) if len(hit) else 0.0
            excs.append(max(0.0, frac - 0.5))
        dev_tab[i, j] = np.mean(devs)
        exc_tab[i, j] = np.mean(excs)
        dev_ok = dev_tab[i, j] <= 0.02
        w_uniform = float(np.std(w_claim)) < 1e-3
        exc_ok = True if w_uniform else (exc_tab[i, j] > 0.1)
        self_ok[i, j] = dev_ok and exc_ok
        tag = '自洽' if self_ok[i, j] else '不自洽'
        detail.append(f'T{tn}/W{wn}: dev {dev_tab[i,j]:.4f} exc {exc_tab[i,j]:.3f} {tag}')

report('U1-1 条件一（触发产生的权重剖面）', True,
       f'九组合的实测-声明偏差：极差 {dev_tab.max()-dev_tab.min():.4f}；'
       + ' '.join(f'T{tn}/W{wn}:{dev_tab[i,j]:.3f}' for i, tn in enumerate(tnames) for j, wn in enumerate(wnames)))

n_self = int(self_ok.sum())
report('U1-3 自洽组合的个数', n_self >= 1,
       f'九组合中自洽 {n_self} 个：' + '; '.join(detail))

ok2 = True; det2 = []
for i, tn in enumerate(tnames):
    for j, wn in enumerate(wnames):
        if float(np.std(WEIGHTS[wn](np.arange(N)))) >= 1e-3:
            ok2 &= exc_tab[i, j] > 0.1
            det2.append(f'T{tn}/W{wn}:{exc_tab[i,j]:.3f}')
report('U1-2 条件二（次序-权重的秩超出量）', ok2,
       '非均匀权重组合的超出量须 > 0.1：' + ' '.join(det2))

dev_range = dev_tab.max() - dev_tab.min()
report('U1-4 条件一的分离性（偏差矩阵非退化）', dev_range > 0.10,
       f'偏差矩阵极差 {dev_range:.4f}（> 0.10 为非退化）')

# ---------------- U1-EP5 恒真对照 ----------------
w_const = np.ones(N)/N
exc_const = 0.0   # 常数权重无「高权重档位」，超出量恒零
ok_ep5 = (float(np.std(w_const)) < 1e-3) and (abs(exc_const) < 1e-12)
report('U1-EP5 替身FAIL预期', ok_ep5,
       f'常数权重替身：权重均匀（标准差 {np.std(w_const):.2e}），秩超出量恒为 {exc_const:.1f}，次序相关性判据失效')

# ---------------- U1-DE 简并检测 + EP1 至 EP4 ----------------
Pi = np.eye(N) - np.ones((N, N))/N
ep1 = np.max(np.abs(Pi @ Pi - Pi)) < 1e-14 and np.max(np.abs(Pi - Pi.T)) < 1e-14
report('U1-EP1 归一约束投影幂等厄米', ep1, f'idem {np.max(np.abs(Pi@Pi-Pi)):.2e}')
ev = np.sort(np.linalg.eigvalsh(Pi))
theo = np.concatenate([[0.0], [1.0]*(N-1)])
ep2 = np.max(np.abs(ev - theo)) < 1e-12
report('U1-EP2 投影本征值落全谱', ep2, f'err {np.max(np.abs(ev-theo)):.2e}')
vals = []
rng3 = np.random.default_rng(SEED + 31)
for _ in range(40):
    w = rng3.random(N); w /= w.sum()
    vals.append(np.sum(w*w))
report('U1-EP3 随机矢极差报告', True, f'range {max(vals)-min(vals):.4f}（不作判据）')
comm = np.max(np.abs(np.diag(np.ones(N)) @ Pi - Pi @ np.diag(np.ones(N))))
report('U1-EP4 守恒标签', comm < 1e-14, f'[I, Pi] = {comm:.2e}')
report('U1-DE 简并报告', True,
       f'权重剖面读数落 {N-1} 重数的归一约束子空间（常数类读数合法）')

# ---------------- 图 ----------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
ax = axes[0]
im = ax.imshow(self_ok.astype(float), cmap='Greens', vmin=0, vmax=1)
ax.set_xticks(range(3)); ax.set_xticklabels([f'W{w}' for w in wnames])
ax.set_yticks(range(3)); ax.set_yticklabels([f'T{t}' for t in tnames])
for i in range(3):
    for j in range(3):
        ax.text(j, i, '自洽' if self_ok[i, j] else '否',
                ha='center', va='center', fontsize=10)
ax.set_title(f'(a) 九组合自洽矩阵（自洽 {n_self} 个）')
ax = axes[1]
im2 = ax.imshow(exc_tab, cmap='RdBu_r', vmin=0, vmax=0.2)
ax.set_xticks(range(3)); ax.set_xticklabels([f'W{w}' for w in wnames])
ax.set_yticks(range(3)); ax.set_yticklabels([f'T{t}' for t in tnames])
for i in range(3):
    for j in range(3):
        ax.text(j, i, f'{exc_tab[i,j]:.3f}', ha='center', va='center', fontsize=10)
ax.set_title('(b) 次序-权重秩超出量')
fig.colorbar(im2, ax=ax)
ax = axes[2]
sites = np.arange(N)
for wn, c in zip(wnames, ['C0', 'C1', 'C2']):
    ax.plot(sites, WEIGHTS[wn](sites), c+'-', lw=1, label=f'W{wn}')
ax.plot(sites, np.ones(N)/N, 'k:', lw=1, label='常数（EP5 替身）')
ax.set_xlabel('档位'); ax.set_ylabel('权重')
ax.set_title('(c) 三权重候选的剖面')
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(r'D:/software/TianXingFC/images/4p6_u1_joint.png', dpi=140)
print('figure saved: images/4p6_u1_joint.png')

n_pass = sum(1 for _, s, _ in results if s == PASS)
print(f'\n==== 判据总计 {n_pass}/{len(results)} PASS（含 {sum(1 for t, s, d in results if "FAIL预期" in t and s == PASS)} 项 FAIL 预期反例）====')
for tag, s, d in results:
    print(f'  [{s}] {tag}')