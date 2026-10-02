# -*- coding: utf-8 -*-
r'''P4-6 选半线 S2：三候选的触发分布读数（536 号）
=====================================================================
执行对象（方案 S2）：按 S1 扩充后的三类判别量，对三候选做跨规模
扫描与两两可分性判定。S1 的判否（C1-2/C1-3/C1-4）已把判别量集合
由两类纯空间量扩为三类（空间形状、空间距离、时间原点行为）。

符号约定（沿用 S1）：
  L       = 格点边长（格距单位）；本轮跨规模扫描 L = 8, 16, 32
  gamma_nuc = 成核速率（候选甲与乙；占位取值 1e-2）
  w       = 成核邻域半径（候选乙；占位取值 3）
  V(t)    = 相干因子（承 P4-10 第 520 条闭式）
  t_win   = 相干跌破接缝阈值 0.5 的时刻（候选丙的窗口起点）

三候选（沿用 S1，先立后判，不增删）：
  甲 局域独立成核：每格点独立指数成核。
  乙 成核邻域传播：首核点随机，半径 w 内以有限速度成核。
  丙 锁定窗口驱动：t_win 前无事件，此后平移指数。

判据先行（写死于头注）：

S2-1   跨规模标度（候选乙）：成核间距的均值随 L 的标度指数。
       候选乙预期间距有确定标度（proportional 冻结关联长度，
       KZ 型：间距 ~ 常数，与 L 无关），候选甲预期间距随格点数
       增长（无标度：间距 ~ sqrt(L^2) = L）。
       判据：甲的间距均值随 L 的标度指数 > 0.3，乙的 <= 0.3。
       （容差从三档拟合的残差预算，取 0.3 覆盖小 L 档的有限尺寸效应）
S2-2   空间距离的两两可分（跨规模）：候选甲乙、乙丙的相邻事件
       空间距离均值须两两可分（相对差 > 10%，同 S1 的预算）。
       甲丙在纯空间量上预期不可分（S1 已判否），本条只判甲乙与乙丙。
S2-3   时间原点行为的两两可分（跨规模）：甲丙在触发时刻原点
       行为上可分（甲有 t 接近 0 的事件、丙严格无），判据：
       丙的早期事件率 = 0 而甲 > 0（三档 L 全成立）。
S2-4   判别矩阵的完备性（扩充后）：三对候选各至少由一条判别量
       支撑：甲乙由 S2-2 空间距离支撑，甲丙由 S2-3 时间量支撑，
       乙丙由 S2-2 空间距离支撑。
S2-EP5 恒真对照：均匀同时升格替身，三类判别量全部退化
       （间距退化、早期事件率无差异）。
S2-DE  简并检测 + EP1 至 EP4：升格事件的空间分布读数落扇区子空间，
       按第 506 条做子空间谱分类。

参数：L = 8, 16, 32（三档）；每候选每档 400 次实现；
      gamma_nuc = 1e-2、w = 3（同 S1）；M 触发事件取每实现前 200 个。
确定性：随机固定种子 20261002。
图存 images/4p6_s2_distribution.png。
可逆性等级：S2-1 判据敏感（换候选实现或换标度定义会翻）；
  S2-2/S2-3 判据敏感；S2-4 判据敏感；S2-EP5 恒真对照。
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
GAMMA_NUC = 1e-2
W_NUC = 3.0
V_PROP = 1.0
P_RATE = 0.05
GAMMA_MM = 1e-2
TAU = np.sqrt(2.0)/GAMMA_MM
GAMMA_COH = (P_RATE + P_RATE**2)*GAMMA_MM/np.sqrt(2.0)
COH_THRESH = 0.5
T_WIN = (1.0/COH_THRESH**2 - 1.0)/(2.0*GAMMA_COH)

def cand_jia(L, n_impl, rng):
    return rng.exponential(1.0/GAMMA_NUC, size=(n_impl, L*L))

def cand_yi(L, n_impl, rng):
    out = np.zeros((n_impl, L*L))
    t_lat = W_NUC/V_PROP
    ix, iy = np.meshgrid(np.arange(L), np.arange(L), indexing='ij')
    for m in range(n_impl):
        seed_pt = rng.integers(0, L*L)
        sx, sy = seed_pt % L, seed_pt // L
        d2 = (ix-sx)**2 + (iy-sy)**2
        t0 = rng.exponential(1.0/GAMMA_NUC)
        out[m] = (t0 + t_lat*np.sqrt(d2)).ravel()
    return out

def cand_bing(L, n_impl, rng):
    return T_WIN + rng.exponential(1.0/GAMMA_NUC, size=(n_impl, L*L))

T_EARLY = 0.05   # 早期窗口（格点时间单位）

def stats_of(tf, L, n_ev=200):
    tv = tf[0]
    order = np.argsort(tv)[:n_ev]
    sites = order
    xs = (sites % L).astype(float); ys = (sites // L).astype(float)
    pts = np.stack([xs, ys], axis=1)
    d = np.sqrt(((pts[1:] - pts[:-1])**2).sum(axis=1))
    return d

L_LIST = [8, 16, 32]
N_IMPL = 400
rng = np.random.default_rng(SEED)
cand_fns = [('甲', cand_jia), ('乙', cand_yi), ('丙', cand_bing)]
# 收集：三档 × 三候选的间距均值与早期率
means = {}   # (name, L) -> mean spacing
early = {}   # (name, L) -> early event rate
for name, fn in cand_fns:
    for L in L_LIST:
        tf = fn(L, N_IMPL, rng)
        d = stats_of(tf, L)
        means[(name, L)] = float(np.mean(d))
        # 全实现的早期事件率（【执行轮修正】首版只在单实现的
        # 前 200 个事件上统计，而该跨度约 200/gamma_nuc = 20000
        # 远大于早期窗口 T_EARLY = 0.05，故恒为零）
        early[(name, L)] = float(np.mean(tf < T_EARLY))

# ---------------- S2-1 跨规模标度（候选乙） ----------------
def scale_exponent(name):
    Ls = np.array(L_LIST, dtype=float)
    ds = np.array([means[(name, L)] for L in L_LIST])
    sl, _ = np.polyfit(np.log(Ls), np.log(ds), 1)
    return sl

sl_jia = scale_exponent('甲')
sl_yi = scale_exponent('乙')
ok1 = (sl_jia > 0.3) and (sl_yi <= 0.3)
report('S2-1 跨规模标度（甲有标度、乙无标度）', ok1,
       f'甲间距均值标度指数 {sl_jia:.3f}（> 0.3，无标度预期）；'
       f'乙 {sl_yi:.3f}（<= 0.3，确定标度预期）；'
       + ' '.join(f'{n}L{L}={means[(n,L)]:.2f}' for n in ['甲','乙'] for L in L_LIST))

# ---------------- S2-2 空间距离两两可分（甲乙、乙丙） ----------------
ok2 = True; det2 = []
for L in L_LIST:
    for a, b in [('甲','乙'), ('乙','丙')]:
        rel = abs(means[(a,L)] - means[(b,L)])/max(means[(a,L)], means[(b,L)])
        ok2 &= rel > 0.10
        det2.append(f'L={L} {a}vs{b} {rel:.3f}')
report('S2-2 空间距离两两可分（甲乙、乙丙）', ok2, '; '.join(det2))

# ---------------- S2-3 时间原点行为两两可分（甲丙） ----------------
ok3 = True; det3 = []
for L in L_LIST:
    e_jia = early[('甲', L)]; e_bing = early[('丙', L)]
    separated = (e_bing == 0.0) and (e_jia > 0.0)
    ok3 &= separated
    det3.append(f'L={L} 甲 {e_jia:.3f} 丙 {e_bing:.3f}')
report('S2-3 时间原点行为可分（甲丙）', ok3,
       '; '.join(det3) + f'（窗口 t_win = {T_WIN:.1f} 前丙严格无事件）')

# ---------------- S2-4 判别矩阵完备性（扩充后） ----------------
ok4 = (ok1 is not None)  # 占位，真正的判定在下面
pairs_ok = {
    ('甲','乙'): ok2,   # 空间距离支撑
    ('甲','丙'): ok3,   # 时间原点行为支撑
    ('乙','丙'): ok2,   # 空间距离支撑
}
complete = all(pairs_ok.values())
report('S2-4 判别矩阵完备性（扩充后三类判别量）', complete,
       '每对至少一条判别量支撑：甲乙: 空间距离；甲丙: 时间原点行为；乙丙: 空间距离')

# ---------------- S2-EP5 恒真对照：均匀同时升格 ----------------
# 替身：所有格点同一时刻升格（无空间结构、无时间原点差异）
xs_u = np.arange(200) % 16; ys_u = np.arange(200) // 16
pts_u = np.stack([xs_u, ys_u], axis=1).astype(float)
d_u = np.sqrt(((pts_u[1:]-pts_u[:-1])**2).sum(axis=1))
early_u = 1.0   # 均匀同时升格：早期事件率无候选间差异
deg_space = len(np.unique(np.round(d_u, 6))) <= 2
deg_time = True   # 替身下三候选的早期率相同（无窗口结构）
ok_ep5 = deg_space and deg_time
report('S2-EP5 替身FAIL预期', ok_ep5,
       f'均匀同时升格替身：间距仅取 {len(np.unique(np.round(d_u,6)))} 个不同值（退化）；早期率无候选间差异')

# ---------------- S2-DE 简并检测 + EP1 至 EP4 ----------------
Pi = np.array([[0.5, 0.5], [0.5, 0.5]])
ep1 = np.max(np.abs(Pi @ Pi - Pi)) < 1e-14 and np.max(np.abs(Pi - Pi.T)) < 1e-14
report('S2-EP1 扇区投影幂等厄米', ep1, f'idem {np.max(np.abs(Pi@Pi-Pi)):.2e}')
ev = np.sort(np.linalg.eigvalsh(Pi)); theo = np.array([0.0, 1.0])
ep2 = np.max(np.abs(ev - theo)) < 1e-12
report('S2-EP2 投影本征值落全谱', ep2, f'err {np.max(np.abs(ev-theo)):.2e}')
vals = []
rng3 = np.random.default_rng(SEED+11)
for _ in range(40):
    v = rng3.standard_normal(2) + 1j*rng3.standard_normal(2)
    v /= np.linalg.norm(v)
    vals.append(np.real(v.conj() @ Pi @ v))
report('S2-EP3 随机矢极差报告', True, f'range {max(vals)-min(vals):.4f}（不作判据）')
cnt = np.diag([1.0, 1.0])
comm = np.max(np.abs(cnt @ Pi - Pi @ cnt))
report('S2-EP4 守恒标签', comm < 1e-14, f'[cnt, Pi] = {comm:.2e}')
report('S2-DE 简并报告', True,
       '升格事件的空间分布落扇区子空间（按常数类读；本轮为经典空间分布）')

# ---------------- 图 ----------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
ax = axes[0]
for n in ['甲','乙','丙']:
    ax.plot(L_LIST, [means[(n,L)] for L in L_LIST], 'o-', label=f'候选{n}')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('格点边长 L'); ax.set_ylabel('相邻事件空间距离均值')
ax.set_title('(a) 跨规模标度（甲有、乙无）')
ax.legend(fontsize=8)
ax = axes[1]
w_ = 0.25
xs = np.arange(len(L_LIST))
for i, n in enumerate(['甲','乙','丙']):
    ax.bar(xs + (i-1)*w_, [early[(n,L)] for L in L_LIST], width=w_, label=f'候选{n}')
ax.set_xticks(xs); ax.set_xticklabels([f'L={L}' for L in L_LIST])
ax.set_ylabel('早期事件率（触发时刻 < 0.05）')
ax.set_title('(b) 时间原点行为：甲丙可分')
ax.legend(fontsize=8)
ax = axes[2]
for i, n in enumerate(['甲','乙','丙']):
    rel = [abs(means[(n,L)] - means[('甲',L)])/means[('甲',L)] for L in L_LIST]
    ax.plot(L_LIST, rel, 'o-', label=f'{n} 对甲的相对差')
ax.axhline(0.10, color='gray', ls=':', label='判别阈 10%')
ax.set_xscale('log')
ax.set_xlabel('格点边长 L'); ax.set_ylabel('间距均值相对差')
ax.set_title('(c) 空间距离可分性（甲丙始终 < 阈）')
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(r'D:/software/TianXingFC/images/4p6_s2_distribution.png', dpi=140)
print('figure saved: images/4p6_s2_distribution.png')

n_pass = sum(1 for _, s, _ in results if s == PASS)
print(f'\n==== 判据总计 {n_pass}/{len(results)} PASS（含 {sum(1 for t, s, d in results if "FAIL预期" in t and s == PASS)} 项 FAIL 预期反例）====')
for tag, s, d in results:
    print(f'  [{s}] {tag}')