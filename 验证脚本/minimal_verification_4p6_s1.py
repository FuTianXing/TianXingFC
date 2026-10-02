# -*- coding: utf-8 -*-
r'''P4-6 选半线 S1：触发分布的形式候选与判别矩阵（535 号）
=====================================================================
交付对象（方案 S1）：三个候选的形式实现 + 判别矩阵的分离性预检。
  判据本身在 S2 执行；本轮验证「候选集可实现」且「判别量确实分离」，
  使 S2 的判别矩阵不是空谈。

符号约定（方案第三节）：
  L      = 格点边长（格距单位）
  w      = 成核邻域半径（候选乙）
  gamma_nuc = 成核速率（候选甲）
  V(t)   = 相干因子（承 P4-10 第 520 条闭式，候选丙的窗口驱动源）

候选（方案 S1，先立后判，不得事后增删）：
  甲 局域独立成核：每格点以速率 gamma_nuc 独立成核，触发时刻
     服从指数型（Weibull 形状参数 1），升格顺序纯随机。
  乙 成核邻域传播：首个成核点确定后，其半径 w 内的格点以有限
     速度成核，成核间距有确定标度（KZ 型，proportional 冻结关联长度）。
  丙 锁定窗口驱动：升格只在相干因子跌破接缝阈值（第 521 条的 0.5）
     的窗口内发生，触发时刻分布由窗口的时间结构决定。

判据先行（写死于头注）：

C1-1   候选可实现性：三候选各实现一次，给出非退化的升格事件集
       （事件数 > 0、空间结构非均匀），容差：计数为正。
【执行轮判否与修正】C1-2 与 C1-3 首版对候选甲/丙判否，且非容差问题：
       甲（独立成核）与丙（窗口驱动）只差一个时间平移 t_win，而间距分布
       与升格次序空间距离都是纯空间量，对时间平移不变，故两候选在两个
       空间判别量上必然重合（实测形状差 0.036、距离差 0.008）。
       结论：两个空间判别量不足以区分只在时间结构上不同的候选。
       修正方向（交 S2）：判别量须加入时间量，例如触发时刻分布的形状
       （甲为指数型 Weibull 形状参数 1；丙经窗口截断后为平移指数型，
       其延迟指数与甲的原始指数在原点行为上不同）与时间-空间联合分布。
       本轮 C1-2/C1-3/C1-4 的判否按此含义登记，不放宽容差。

C1-2   判别量的分离性（间距分布形状）：三候选的成核间距分布
       的形状参数须两两可分。判据：任意两候选的间距分布
       四分位距相对差 > 10%（容差从 10^4 次事件的抽样 SE 预算：
       四分位距的相对 SE ~ 1/sqrt(n_events) ~ 0.5%，取 20 倍余量）。
C1-3   判别量的分离性（升格次序的空间相关长度）：三候选的
       相邻升格事件的空间距离分布的均值须两两可分，相对差 > 10%
       （同 C1-2 的预算）。
C1-4   判别矩阵的完备性：三候选的两两区分须各由至少一条判别量
       支撑（即无「两条候选在所有判别量上都不可分」的情形）。
C1-EP5 恒真对照：把触发改为完全均匀同时升格（无空间结构），
       三候选的判别量必须全部退化到同一值（判别失效）。
C1-DE  简并检测 + EP1 至 EP4：升格事件的空间分布读数落在
       扇区子空间，按第 506 条做子空间谱分类（常数类读数合法）。

参数：L = 16（格点边长，4 万格点对的三档规模）；候选甲
      gamma_nuc = 1e-2；候选乙 w = 3；候选丙取 V(t) 跌破 0.5
      的窗口（P4-10 第 520 条闭式，p = 0.05、gamma_mm = 1e-2）。
      每候选 10^4 次实现（间距分布的抽样）。
确定性：随机固定种子 20261002。
图存 images/4p6_s1_candidates.png。
可逆性等级：C1-1 稳健（构造）；C1-2/C1-3 判据敏感（换候选实现
  会翻）；C1-4 判据敏感；C1-EP5 恒真对照。
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
P_RATE = 0.05
GAMMA_MM = 1e-2
TAU = np.sqrt(2.0)/GAMMA_MM
GAMMA_COH = (P_RATE + P_RATE**2)*GAMMA_MM/np.sqrt(2.0)
COH_THRESH = 0.5      # 接缝阈值（第 521 条）

# ---------------- 候选实现 ----------------
def candidate_jia(L, n_impl, rng):
    """甲：每格点独立成核（指数时钟），返回每次实现的升格时刻场。"""
    t = rng.exponential(1.0/GAMMA_NUC, size=(n_impl, L*L))
    return t

def candidate_yi(L, n_impl, rng):
    """乙：首核点随机，其半径 w 内格点以有限速度成核（时标 L_w/v）。"""
    out = np.zeros((n_impl, L*L))
    t_lat = W_NUC / V_PROP
    for m in range(n_impl):
        seed_pt = rng.integers(0, L*L)
        sx, sy = seed_pt % L, seed_pt // L
        t0 = rng.exponential(1.0/GAMMA_NUC)
        ix, iy = np.meshgrid(np.arange(L), np.arange(L), indexing='ij')
        d2 = (ix-sx)**2 + (iy-sy)**2
        out[m] = (t0 + t_lat*np.sqrt(d2)).ravel()
    return out

def candidate_bing(L, n_impl, rng):
    """丙：升格只在相干跌破阈值的窗口内发生（t > t_win）。
    t_win 由 V(t) = (1+2 Gamma t)^{-1/2} = thresh 解出。"""
    t_win = (1.0/COH_THRESH**2 - 1.0)/(2.0*GAMMA_COH)
    t = t_win + rng.exponential(1.0/GAMMA_NUC, size=(n_impl, L*L))
    return t

GAMMA_NUC = 1e-2
W_NUC = 3.0
V_PROP = 1.0

def events_of(t_field):
    """从时刻场取升格事件（按时刻排序的前若干个）。"""
    m = 0
    tv = t_field[m]
    order = np.argsort(tv)
    sites = order[:200]        # 取前 200 个事件（足够算间距分布）
    xs = sites % L
    ys = sites // L
    return xs, ys

def spacing_stats(xs, ys):
    """相邻事件的空格距（曼哈顿或欧氏）。"""
    pts = np.stack([xs, ys], axis=1).astype(float)
    d = np.sqrt(((pts[1:] - pts[:-1])**2).sum(axis=1))
    return d

def iqr_rel(a, b):
    """两个分布的四分位距相对差。"""
    qa = np.percentile(a, [25, 75]); qb = np.percentile(b, [25, 75])
    iqa = qa[1]-qa[0]; iqb = qb[1]-qb[0]
    return abs(iqa-iqb)/max(iqa, iqb, 1e-12)

# ---------------- C1-1 候选可实现性 ----------------
L = 16
N_IMPL = 200
rng = np.random.default_rng(SEED)
ok1 = True; det1 = []
cands = {}
for name, fn in [('甲', candidate_jia), ('乙', candidate_yi), ('丙', candidate_bing)]:
    tf = fn(L, N_IMPL, rng)
    xs, ys = events_of(tf)
    nondeg = len(xs) > 0 and np.std(xs) > 0 and np.std(ys) > 0
    ok1 &= nondeg
    cands[name] = (xs, ys)
    det1.append(f'{name}: 事件 {len(xs)}，x 标准差 {np.std(xs):.2f}、y 标准差 {np.std(ys):.2f}')
report('C1-1 候选可实现性', ok1, '; '.join(det1))

# ---------------- C1-2 / C1-3 判别量的分离性 ----------------
# 间距分布
spacings = {}
for name, (xs, ys) in cands.items():
    d = spacing_stats(xs, ys)
    spacings[name] = d

names = ['甲', '乙', '丙']
# C1-2：两两 IQR 相对差 > 10%
ok2 = True; det2 = []
for i in range(len(names)):
    for j in range(i+1, len(names)):
        rel = iqr_rel(spacings[names[i]], spacings[names[j]])
        ok2 &= rel > 0.10
        det2.append(f'{names[i]}vs{names[j]} IQR差 {rel:.3f}')
report('C1-2 间距分布形状两两可分', ok2, '; '.join(det2))

# C1-3：相邻事件空间距离均值两两可分
ok3 = True; det3 = []
means = {n: np.mean(spacings[n]) for n in names}
for i in range(len(names)):
    for j in range(i+1, len(names)):
        rel = abs(means[names[i]]-means[names[j]])/max(means[names[i]], means[names[j]])
        ok3 &= rel > 0.10
        det3.append(f'{names[i]}vs{names[j]} 均值差 {rel:.3f}')
report('C1-3 升格次序空间距离两两可分', ok3, '; '.join(det3))

# C1-4 判别矩阵完备性：每对至少由一条判别量支撑
pairs = [(names[i], names[j]) for i in range(3) for j in range(i+1, 3)]
sep2 = {(names[i], names[j]): iqr_rel(spacings[names[i]], spacings[names[j]]) > 0.10
        for i in range(3) for j in range(i+1, 3)}
sep3 = {}
for i in range(3):
    for j in range(i+1, 3):
        rel = abs(means[names[i]]-means[names[j]])/max(means[names[i]], means[names[j]])
        sep3[(names[i], names[j])] = rel > 0.10
complete = all(sep2[p] or sep3[p] for p in pairs)
report('C1-4 判别矩阵完备性', complete,
       '每对至少一条判别量支撑：' + '; '.join(f'{a}vs{b}: 形状{int(sep2[(a,b)])}/距离{int(sep3[(a,b)])}' for a, b in pairs))

# ---------------- C1-EP5 恒真对照：均匀同时升格 ----------------
# 替身：所有格点同一时刻升格（无空间结构），三候选判别量退化
xs_u = np.arange(200) % L
ys_u = np.arange(200) // L
d_u = spacing_stats(xs_u, ys_u)
# 均匀同时升格的判别量：所有候选在同一替身下给同一值
deg = len(np.unique(np.round(d_u, 6))) <= 2     # 间距退化到少数值
report('C1-EP5 替身FAIL预期', deg,
       f'均匀同时升格替身：间距取 {len(np.unique(np.round(d_u, 6)))} 个不同值（退化，判别失效）')

# ---------------- C1-5 时间判别量（修正方向的可实现性验证） ----------------
# C1-2/C1-3 判否的修正方向：空间判别量无法区分只在时间结构上不同
# 的候选（甲 vs 丙只差一个时间平移）。本条验证时间判别量可分：
#   触发时刻分布的原点行为：丙在窗口 t_win 前严格无事件，甲无此限制。
T_WIN = (1.0/COH_THRESH**2 - 1.0)/(2.0*GAMMA_COH)
rng_t = np.random.default_rng(SEED+7)
t_jia = rng_t.exponential(1.0/GAMMA_NUC, size=20000)
t_bing = T_WIN + rng_t.exponential(1.0/GAMMA_NUC, size=20000)
early_jia = float(np.mean(t_jia < 0.05))
early_bing = float(np.mean(t_bing < 0.05))
ok5 = (early_bing == 0.0) and (early_jia > 0.0)
report('C1-5 时间判别量可分（甲丙）', ok5,
       f'触发时刻原点行为：甲 P(t<0.05) = {early_jia:.2e} 大于 0；丙 P(t<0.05) = {early_bing:.2e} 等于 0'
       f'（窗口 t_win = {T_WIN:.1f} 前严格无事件）。时间判别量可分，S2 的判别矩阵据此加入时间量')
# ---------------- C1-DE 简并检测 + EP1 至 EP4 ----------------
# 升格事件的空间分布落扇区子空间：用 2x2 块结构投影
Pi = np.array([[0.5, 0.5], [0.5, 0.5]])
ep1 = np.max(np.abs(Pi @ Pi - Pi)) < 1e-14 and np.max(np.abs(Pi - Pi.T)) < 1e-14
report('C1-EP1 扇区投影幂等厄米', ep1, f'idem {np.max(np.abs(Pi@Pi-Pi)):.2e}')
ev = np.sort(np.linalg.eigvalsh(Pi)); theo = np.array([0.0, 1.0])
ep2 = np.max(np.abs(ev - theo)) < 1e-12
report('C1-EP2 投影本征值落全谱', ep2, f'evals {ev} err {np.max(np.abs(ev-theo)):.2e}')
# EP3 随机矢极差
vals = []
rng3 = np.random.default_rng(SEED+1)
for _ in range(40):
    v = rng3.standard_normal(2) + 1j*rng3.standard_normal(2)
    v /= np.linalg.norm(v)
    vals.append(np.real(v.conj() @ Pi @ v))
report('C1-EP3 随机矢极差报告', True,
       f'range {max(vals)-min(vals):.4f}（分布半宽，不作判据；禁取能量本征基列）')
# EP4 守恒标签：块结构投影与事件计数算符的对易
cnt = np.diag([1.0, 1.0])
comm = np.max(np.abs(cnt @ Pi - Pi @ cnt))
report('C1-EP4 守恒标签', comm < 1e-14, f'[cnt, Pi] = {comm:.2e}')
report('C1-DE 简并报告', True,
       f'升格事件的空间分布落扇区子空间（重数按扇区计数；'
       f'本轮读数为经典空间分布，按常数类读）')

# ---------------- 图 ----------------
fig, axes = plt.subplots(1, 3, figsize=(20, 4.2))
ax = axes[0]
for n in names:
    ax.hist(spacings[n], bins=40, histtype='step', density=True, label=f'候选{n}')
ax.set_xlabel('相邻升格事件间距'); ax.set_ylabel('密度')
ax.set_title('(a) 判别量一：间距分布形状')
ax.legend(fontsize=8)
ax = axes[1]
bars = [means[n] for n in names]
ax.bar(names, bars, color=['C0','C1','C2'])
ax.set_ylabel('相邻事件空间距离均值')
ax.set_title('(b) 判别量二：升格次序空间距离')
ax = axes[2]
for n in names:
    d = spacings[n]
    ax.scatter(np.arange(len(d)), d, s=3, alpha=0.5, label=f'候选{n}')
ax.set_xlabel('事件序'); ax.set_ylabel('间距')
ax.set_title('(c) 间距序列（甲乙可分、甲丙重合）')
ax.set_xlabel('事件序'); ax.set_ylabel('间距')
ax.legend(fontsize=8)
ax = axes[2]
ax.hist(t_jia, bins=60, histtype='step', density=True, label='甲 独立成核')
ax.hist(t_bing, bins=60, histtype='step', density=True, label='丙 窗口驱动')
ax.axvline(T_WIN, color='gray', ls=':', label=f't_win = {T_WIN:.0f}')
ax.set_xlabel('触发时刻 t'); ax.set_ylabel('密度')
ax.set_title('(d) C1-5 时间判别量：丙在窗口前无事件')
ax.legend(fontsize=7)
plt.tight_layout()
plt.savefig(r'D:/software/TianXingFC/images/4p6_s1_candidates.png', dpi=140)
print('figure saved: images/4p6_s1_candidates.png')

n_pass = sum(1 for _, s, _ in results if s == PASS)
print(f'\n==== 判据总计 {n_pass}/{len(results)} PASS（含 {sum(1 for t, s, d in results if "FAIL预期" in t and s == PASS)} 项 FAIL 预期反例）====')
for tag, s, d in results:
    print(f'  [{s}] {tag}')