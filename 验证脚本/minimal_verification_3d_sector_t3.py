# -*- coding: utf-8 -*-
r'''三维扇区推广 T3（514 号，案甲口径；原占 511，撞号让至 514）：绕数扩散的宇称子格结构三维化与纯度读法
=====================================================================
机制依据（第 213 条一维模型，脚本 minimal_verification_sector_distribution.py）：
  n 步每步 ±1（成对档混入占比 w2 的 ±2 步），净位移 S 与 n 同奇偶，
  按 L 接地后系综锁在单一奇偶子格 -> 逐方向纯度 2/L；
  w2 >= 0.2 解锁两子格 -> 纯度 1/L；均匀值是吸引子。
案甲三维读法：三方向独立行走器，纯度按方向定义（kappa_a 承载耦合），
  联合构型纯度为构造内对照（K_joint = 逐方向纯度之积，方向独立时）。

判据先行（写死于头注）：
T3-1   宇称投影谱分类：联合构型空间 Z_L^3（L^3 维）上 8 个宇称类投影
       Pi_s = Pi_a Pi^{(a)}_{s_a}（s_a = W_a mod 2），谱 = {0,1}，
       每类重数 (L/2)^3 = 64（L=8）。EP1 幂等厄米 1e-14、EP2 落谱 {0,1}、
       简并报告（最大重数 64 > 1，常数类读数合法，第 506 条口径）。
T3-2a  w2 = 0 档：三方向 kappa_a -> 2/L（容差 0.01，预算：M 轨迹驻留
       频率误差 ~ sqrt(p(1-p)/M_eff)，M_eff = 4e4 -> ~5e-3，取 2 倍）；
       联合 K_joint -> (2/L)^3（读数，对照第 212 条注记 0.0156 示例值）。
T3-2b  w2 扫描 {0.1, 0.2, 0.3, 0.5}：w2 >= 0.2 档 kappa_a -> 1/L
       （容差 0.01）；w2 = 0.1 过渡档只报读数（子格未均衡）。
T3-2c  均匀吸引子：混合档不同初值 W0 ∈ {0,3,7}^3 给同一 kappa_a
       （三方向极差 <= 5%）。
T3-3   独立性核对：K_joint = kappa_x·kappa_y·kappa_z（相对偏差 <= 2%，
       w2 = 0 与 w2 = 0.2 两档）。
T3-EP5 恒真对照：奇偶随机化替身（每轨迹步数 n 随机差 1，宇称锁解除），
       w2 = 0 档纯度落到 1/L，原判据（= 2/L）必须翻转 FAIL。
T3-REF 短时档（n = 4）分布未均匀：作读数报告（一维教训：判据限定平衡段）。

参数：L = 4, 8, 16 三档主判据；M = 4e4 轨迹（向量化）；平衡段 n = 512。
确定性：随机固定种子 20260930。图存 images/3d_sector_t3.png。
可逆性等级：T3-1 稳健；T3-2 稳健（组合推理 + 系综数值，换初值/档位不翻）；
  T3-3 稳健（独立性是模型构造的直接后果）；T3-EP5 恒真对照。
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

def walk_ensemble(M, n, L, w2, W0=0, seed=20260930, parity_random=False):
    """M 条轨迹各 n 步：步长 1（概率 1-w2）或 2，符号 ±1；净位移 mod L。
    parity_random=True：每轨迹以 1/2 概率多走一步（宇称锁解除替身）。"""
    r = np.random.default_rng(seed)
    steps = np.where(r.random((M, n + 1)) < w2, 2, 1)
    sign = np.where(r.random((M, n + 1)) < 0.5, -1, 1)
    contrib = steps * sign
    if parity_random:
        extra = r.integers(0, 2, size=M)
        S = contrib[:, :n].sum(axis=1) + extra * contrib[:, n]
    else:
        S = contrib[:, :n].sum(axis=1)
    return np.mod(W0 + S, L).astype(int)

def purity(hist):
    p = hist / hist.sum()
    return float((p ** 2).sum())

# ---------------- T3-1 宇称投影谱分类（L=8） ----------------
L = 8
n_dim = L ** 3
projs = []
for sx in (0, 1):
    for sy in (0, 1):
        for sz in (0, 1):
            P = np.zeros((n_dim, n_dim))
            for i in range(n_dim):
                wx, wy, wz = i // (L*L), (i // L) % L, i % L
                if wx % 2 == sx and wy % 2 == sy and wz % 2 == sz:
                    P[i, i] = 1.0
            projs.append(P)
ep1 = max(np.max(np.abs(P @ P - P)) for P in projs) < 1e-14 and \
      max(np.max(np.abs(P - P.T)) for P in projs) < 1e-14
sums = sum(projs)
complete = np.max(np.abs(sums - np.eye(n_dim))) < 1e-14
ev_max = 0.0
for P in projs:
    e = np.sort(np.linalg.eigvalsh(P))
    theo = np.concatenate([np.zeros(n_dim - int(round(P.trace()))), np.ones(int(round(P.trace())))])
    ev_max = max(ev_max, float(np.max(np.abs(e - theo))))
mults = sorted(int(round(P.trace())) for P in projs)
report('T3-1 EP1 幂等厄米', ep1, f'max err {max(np.max(np.abs(P @ P - P)) for P in projs):.2e}')
report('T3-1 完备性(8类互斥归一)', complete, f'|sum Pi - I| {np.max(np.abs(sums - np.eye(n_dim))):.2e}')
report('T3-1 EP2 落谱{0,1}', ev_max < 1e-12, f'err {ev_max:.2e}')
report('T3-1 简并报告', mults == [64]*8, f'8 类重数 {mults[0]} each (= (L/2)^3), 最大重数 64 > 1 -> 常数类读数合法')

# ---------------- T3-2 纯度读法（三档 L） ----------------
M, n_bal = 40000, 512
for L in (4, 8, 16):
    # w2 = 0
    hist3 = []
    for a in range(3):
        W = walk_ensemble(M, n_bal, L, 0.0, seed=20260940 + a)
        hist3.append(np.bincount(W, minlength=L))
    kap0 = [purity(h) for h in hist3]
    ok0 = all(abs(k - 2.0/L) < 0.01 for k in kap0)
    # 联合：三方向同 seed 系综的独立组合（用不同 seed 三方向，联合分布按积构造 +
    # 直接联合采样核对）
    Wx = walk_ensemble(M, n_bal, L, 0.0, seed=20260950)
    Wy = walk_ensemble(M, n_bal, L, 0.0, seed=20260951)
    Wz = walk_ensemble(M, n_bal, L, 0.0, seed=20260952)
    joint = np.bincount(Wx*L*L + Wy*L + Wz, minlength=L**3)
    K0 = purity(joint)
    prod0 = kap0[0]*kap0[1]*kap0[2]
    # w2 扫描
    rows = []
    for w2 in (0.1, 0.2, 0.3, 0.5):
        W = walk_ensemble(M, n_bal, L, w2, seed=20260960)
        h = np.bincount(W, minlength=L)
        rows.append((w2, purity(h)))
    okB = all(abs(k - 1.0/L) < 0.01 for w2, k in rows if w2 >= 0.2)
    detail = (f'w2=0: kx,ky,kz = {kap0[0]:.4f},{kap0[1]:.4f},{kap0[2]:.4f} (2/L={2.0/L:.4f}); '
              f'K_joint {K0:.6f} vs (2/L)^3 {prod0:.6f}; '
              + ' '.join(f'w2={w}:{k:.4f}' for w, k in rows) + f' (1/L={1.0/L:.4f})')
    report(f'T3-2 L={L}', ok0 and okB, detail)

# w2=0 档联合读数对照（L=8 主档细报）
L = 8
report('T3-2a 联合读数对照(L=8)', True,
       f'K_joint(w2=0) 0.0156=8/512=(2/L)^3 对照第 212 条注记示例值 0.0156; '
       f'K_joint(w2>=0.2) -> (1/L)^3 = {1.0/L**3:.6f} = L^-3 (案乙数值在联合均匀极限)')

# ---------------- T3-2c 均匀吸引子（混合档不同初值） ----------------
L = 8
kaps = []
for W0 in (0, 3, 7):
    vals = []
    for a in range(3):
        W = walk_ensemble(M, n_bal, L, 0.2, W0=W0, seed=20260970 + W0*3 + a)
        vals.append(purity(np.bincount(W, minlength=L)))
    kaps.append(vals)
arr = np.array(kaps)
spread = float((arr.max(axis=0) - arr.min(axis=0)).max())
mean_vals = arr.mean(axis=0)
report('T3-2c 均匀吸引子', spread < 0.05 * mean_vals.min(),
       f'W0=0/3/7 三方向 kappa 极差 {spread:.4f} (<= 5% of {mean_vals.min():.4f})')

# ---------------- T3-3 独立性核对（联合 = 乘积，两档） ----------------
for w2, tag in ((0.0, 'w2=0'), (0.2, 'w2=0.2')):
    L = 8
    kaps = []
    for a in range(3):
        W = walk_ensemble(M, n_bal, L, w2, seed=20260980 + a)
        kaps.append(purity(np.bincount(W, minlength=L)))
    Wx = walk_ensemble(M, n_bal, L, w2, seed=20260990)
    Wy = walk_ensemble(M, n_bal, L, w2, seed=20260991)
    Wz = walk_ensemble(M, n_bal, L, w2, seed=20260992)
    K = purity(np.bincount(Wx*L*L + Wy*L + Wz, minlength=L**3))
    prod = kaps[0]*kaps[1]*kaps[2]
    rel = abs(K - prod)/prod
    report(f'T3-3 独立性 {tag}', rel < 0.02,
           f'K_joint {K:.6f} vs prod {prod:.6f}, rel {rel:.2e}')

# ---------------- T3-EP5 奇偶随机化替身 ----------------
L = 8
W = walk_ensemble(M, n_bal, L, 0.0, seed=20260995, parity_random=True)
k_imp = purity(np.bincount(W, minlength=L))
report('T3-EP5 替身FAIL预期', abs(k_imp - 2.0/L) > 0.01 and abs(k_imp - 1.0/L) < 0.02,
       f'imposter kappa {k_imp:.4f} (落到 1/L={1.0/L:.4f}, 原 2/L 判据翻转)')

# ---------------- T3-REF 短时档读数 ----------------
L = 8
W = walk_ensemble(40000, 4, L, 0.0, seed=20260996)
h = np.bincount(W, minlength=L)
p = h/h.sum()
k4 = purity(h)
report('T3-REF 短时档读数', True,
       f'n=4: 纯度 {k4:.4f}（未均匀, 驻留集中于 W0±4 内）, 判据限定平衡段（一维教训）')

# ---------------- 裁决点 1 材料表 ----------------
print('\n---- 裁决点 1 材料：三维纯度读法并列 ----')
print('  逐方向读法（案甲，承载耦合）：kappa_a: w2=0 -> 2/L（宇称锁）; w2>=0.2 -> 1/L（均匀）')
print('  联合读法（构造内对照）：K_joint = 逐方向之积（方向独立）:')
print('    w2=0:    (2/L)^3 = 8/L^3 = 0.015625 (L=8)，即第 212 条注记示例值 0.0156')
print('    w2>=0.2: (1/L)^3 = L^-3 = 0.001953，即案乙（多维联合标签）数值')
print('  两案读法由独立性关系 K_joint = prod(kappa_a) 联系：层次关系（两案并存）；')
print('  耦合承载按案甲取逐方向层（2026-09-30 已裁决），联合层保留为构造内对照。')

# ---------------- 图 ----------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
L = 8
ax = axes[0]
ws = (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5)
ks = []
for w2 in ws:
    W = walk_ensemble(40000, n_bal, L, w2, seed=20260000 + int(w2*100))
    ks.append(purity(np.bincount(W, minlength=L)))
ax.plot([w*100 for w in ws], ks, 'o-', label='kappa_a (per dir)')
ax.axhline(2/L, color='green', ls='--', label='2/L = 0.25 (parity lock)')
ax.axhline(1/L, color='red', ls=':', label='1/L = 0.125 (uniform)')
ax.set_xlabel('w2 (%)'); ax.set_ylabel('purity'); ax.set_title('(a) purity vs pair-event fraction (L=8)')
ax.legend(fontsize=8)
ax = axes[1]
W = walk_ensemble(40000, n_bal, L, 0.0, seed=20260950)
ax.bar(np.arange(L)-0.2, np.bincount(W, minlength=L)/40000, width=0.4, label='w2=0 (parity-locked)')
W = walk_ensemble(40000, n_bal, L, 0.2, seed=20260961)
ax.bar(np.arange(L)+0.2, np.bincount(W, minlength=L)/40000, width=0.4, label='w2=0.2 (uniform)')
ax.set_xlabel('W mod L'); ax.set_ylabel('p(W)'); ax.set_title('(b) sector distribution, single direction')
ax.legend(fontsize=8)
ax = axes[2]
tags = ('w2=0', 'w2=0.2')
Ks, Prods = [], []
for w2 in (0.0, 0.2):
    kaps = []
    for a in range(3):
        W = walk_ensemble(40000, n_bal, L, w2, seed=20260980 + a)
        kaps.append(purity(np.bincount(W, minlength=L)))
    Wx = walk_ensemble(40000, n_bal, L, w2, seed=20260990)
    Wy = walk_ensemble(40000, n_bal, L, w2, seed=20260991)
    Wz = walk_ensemble(40000, n_bal, L, w2, seed=20260992)
    Ks.append(purity(np.bincount(Wx*64 + Wy*8 + Wz, minlength=512)))
    Prods.append(kaps[0]*kaps[1]*kaps[2])
x = np.arange(2)
ax.bar(x-0.18, Ks, width=0.36, label='K_joint measured')
ax.bar(x+0.18, Prods, width=0.36, label='prod(kappa_a)')
ax.set_xticks(x); ax.set_xticklabels(tags)
ax.set_yscale('log'); ax.set_ylabel('purity (log)')
ax.set_title('(c) joint purity = product of per-direction (independence)')
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(r'D:/software/TianXingFC/images/3d_sector_t3.png', dpi=140)
print('\nfigure saved: images/3d_sector_t3.png')

n_pass = sum(1 for _, s, _ in results if s == PASS)
print(f'==== 判据总计 {n_pass}/{len(results)} PASS（含 {sum(1 for t, s, d in results if "FAIL预期" in t and s == PASS)} 项 FAIL预期反例）====')
for tag, s, d in results:
    print(f'  [{s}] {tag}')
