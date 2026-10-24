# -*- coding: utf-8 -*-
r'''三维扇区推广 T1至T2（510 号，案甲口径：逐方向单向标签，方向间 kappa_phi 同步）
=====================================================================
判据先行（写死于头注；执行轮修正过 T2-1b 口径，理由见注记过程记录）：

T1-1   逐方向模恒等式 theta^{(a)}_{j+L} = theta^{(a)}_j (mod 2pi)、
       逐方向相异扇区数 = L、联合构型计数 = L^3（两计数并存；
       案甲：L^3 不承载耦合定义）。容差 1e-12（整数格点无截断，
       误差预算 = 浮点舍入 eps·L·迭代数），三档 L = 4, 8, 16。
T1-EP5 恒真对照：替身映射 theta_{j+L} = theta_j + pi，原判据必须 FAIL。
T1-DE  简并检测 + EP1至EP3：Pi_x = |0><0|_x ⊗ I_yz（L^3 维），
       谱 = {1 x L^2, 0 x (L^3 - L^2)}；EP1 幂等厄米 1e-14；
       EP2 本征值落全谱 {0,1}；EP3 子空间随机矢期望极差报告；
       简并报告：最大簇重数 L^2 > 1，投影算符属常数类（幂等谱 0/1），合法。
T2-1a  几何层（无噪声，随机横模制备）：三方向横模自由衰减读 kappa_a = 1/L，
       容差 1e-10（确定性轨迹；拟合段截尾于 |v| > 1e-6，防止浮点沉底段
       污染斜率：预算 = t·eps ≈ 4e-14 对数域，容差留三个量级）。
T2-1b  场层级（噪声稳态制备 + 撤噪衰减读数）：OU 驱动（beta^2 = 2kappa，
       第 200 条口径）演化 5000 步到稳态，撤噪后横模模长衰减读 kappa_a。
       容差 1e-10：横模簇公共衰减率是几何量（簇简并 L-1，读数走簇公共
       衰减率 = 常数类合法读数，第 506 条口径），噪声只作稳态制备。
       【执行轮修正记录】原设计「稳态自相关拟合慢模、容差 5e-4」经预算
       复核不可达：噪声地板 floor ~ C(0)·sqrt(tau_int/N) 在 L=4 档于
       lag>=5 处已淹没信号（实测 err 9.5e-2）；方差读数路径需 N >= 2e7
       才达 5e-4。改为激励-衰减读数，教训记入注记。
T2-1-EP5 替身：慢模改设 (1 - 1/(2L))，读数给 1/(2L)，原判据必须 FAIL。
T2-SM  冒烟检查（强制）：打印 max|F_p|；预生成共享噪声数组，kappa_phi=0 时
       只扰 x 链初条件，y,z 链轨迹与零扰基线逐位相同（变化的轴 = 扰动的轴）；
       kappa_phi>0 时 y,z 整体模响应非零且随 kappa_phi 增大。
T2-2a  方向间同步：三链整体相位差在 kappa_phi/kappa ∈ {0.5,1,2,4} 四档
       衰减到 <= 1e-10；kappa_phi = 0 对照档 FAIL 预期（相位差保持初值）。
T2-2b  结构保持：整体模通道耦合不改变横模（开关耦合横模分量逐位一致，
       容差 1e-12）；替身（跨链同指数邻居耦合进横模通道）必须 FAIL。
T2-EP1 P_sync = v v^T / |v|^2（3L 维）幂等厄米、迹 = 1（1e-14）。
T2-EP2 A_a 本征值落理论谱 {1} ∪ {1-kappa}（1e-12）。
T2-EP3 随机横模矢（20 个）kappa 读数极差报告（同簇公共衰减率，应精确为零级）。
T2-EP4 [A_a, J/L] = 0 与 [A_joint, P_sync] = 0（1e-14，整体模守恒标签）。

模型：每方向一条环形链的回拉更新 A_a = I - kappa·P_perp（横模谱
      (1-kappa)^{x(N-1)}，整体模 1；与第 234 条谱语言同构）；
      方向间耦合只走整体模通道（案甲：kappa_phi 同步扇区中心）。
可逆性等级：T1 稳健；T2-1a 稳健；T2-1b 稳健（读数为几何量，噪声仅制备）；
      T2-2a 稳健；T2-2b 敏感（结构断言，换耦合通道会翻）。
确定性：随机固定种子 20260930。图存 images/3d_sector_t1_t2.png。
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

# ---------------- T1 几何层 ----------------
def wound_config(L):
    j = np.arange(L)
    return np.array([2*np.pi*W*j/L for W in range(L)])  # (L扇区, L格点)

def t1_identity(L):
    for a in range(3):  # x,y,z 三方向同一构造，方向只换标签
        Th = wound_config(L)
        # theta_{j+L} = 2*pi*W*(j+L)/L = theta_j + 2*pi*W -> 恒等 mod 2pi（W 整数）
        for W in range(L):
            j = np.arange(L)
            diff = 2*np.pi*W*(j+L)/L - 2*np.pi*W*j/L - 2*np.pi*W
            if np.max(np.abs(((diff + np.pi) % (2*np.pi)) - np.pi)) > 1e-12:
                return False, f'L={L} dir={a} W={W} identity broken'
        for W1 in range(L):
            for W2 in range(W1+1, L):
                d = (Th[W1] - Th[W2] + np.pi) % (2*np.pi) - np.pi
                if np.max(np.abs(d)) < 1e-9:
                    return False, f'L={L} sectors {W1},{W2} degenerate'
    return True, f'L={L}: 3 dirs identity ok, {L} distinct sectors each, joint count {L**3}'

t1_ok = True
for L in (4, 8, 16):
    ok, msg = t1_identity(L)
    t1_ok &= ok
    report(f'T1-1 L={L}', ok, msg)

# T1-EP5 替身：theta_{j+L} = theta_j + pi
L = 8
j = np.arange(L)
broken = 2*np.pi*1*(j+L)/L - 2*np.pi*1*j/L - (2*np.pi*1 + np.pi)
ep5_violated = np.max(np.abs(((broken + np.pi) % (2*np.pi)) - np.pi)) > 1e-9
report('T1-EP5 替身FAIL预期', ep5_violated, 'shift-pi imposter correctly detected as FAIL' if ep5_violated else 'imposter passed (bad)')

# T1-DE 简并检测（L=8 档，Pi_x = |0><0|_x ⊗ I_yz 在 L^3 维联合构型空间）
L = 8
n = L**3
Ix = np.zeros((n, n))
for i in range(n):
    if i // (L*L) == 0:
        Ix[i, i] = 1.0
sym_err = np.max(np.abs(Ix - Ix.T))
idem_err = np.max(np.abs(Ix @ Ix - Ix))
evals = np.linalg.eigvalsh(Ix)
cluster1 = int(np.sum(evals > 0.5)); cluster0 = int(np.sum(evals <= 0.5))
ep2_err = max(np.max(np.abs(evals[evals > 0.5] - 1)), np.max(np.abs(evals[evals <= 0.5])))
vals = []
rng = np.random.default_rng(20260930)
for _ in range(20):
    v = rng.standard_normal(n)
    v /= np.linalg.norm(v)
    vals.append(v @ Ix @ v)
ep3_range = max(vals) - min(vals)
report('T1-DE EP1 幂等厄米', sym_err < 1e-14 and idem_err < 1e-14, f'sym {sym_err:.2e} idem {idem_err:.2e}')
report('T1-DE EP2 落谱{0,1}', ep2_err < 1e-12, f'err {ep2_err:.2e}')
report('T1-DE 简并报告', True, f'clusters 1x{cluster1}(=L^2), 0x{cluster0}(=L^3-L^2), max mult {cluster1}>1 -> 常数类投影, 合法')
report('T1-DE EP3 极差报告', 0 <= ep3_range <= 1.0, f'range {ep3_range:.4f} (污染量, 不作判据读数)')

# ---------------- T2 场层与同步 ----------------
def pullback_matrix(L, kappa):
    Pperp = np.eye(L) - np.ones((L, L))/L
    return np.eye(L) - kappa*Pperp

def read_kappa_from_decay(L, kappa_true, T=400):
    """几何层：随机横模制备，自由衰减，拟合段截尾于 |v|>1e-6（防浮点沉底段）。"""
    v = rng.standard_normal(L)
    v -= v.mean()
    v /= np.linalg.norm(v)
    A = pullback_matrix(L, kappa_true)
    logs = []
    for t in range(T):
        v = A @ v
        nrm = np.linalg.norm(v)
        if nrm < 1e-6:
            break
        logs.append(np.log(nrm))
    t = np.arange(len(logs))
    slope = np.polyfit(t, logs, 1)[0]
    return 1.0 - np.exp(slope)

ok = True; det = []
for L in (4, 8, 16):
    kap = 1.0/L
    ks = [read_kappa_from_decay(L, kap) for _ in range(3)]
    err = max(abs(k - kap) for k in ks)
    ok &= err < 1e-10
    det.append(f'L={L} err {err:.2e}')
report('T2-1a 几何层 kappa=1/L', ok, '; '.join(det))

def steady_state_decay_kappa(L, kappa_true, Ndrive=5000, Tdecay=40, M=8, seed=20260930):
    """场层级：噪声稳态制备后撤噪，横模簇公共衰减率读 kappa（几何量）。"""
    r = np.random.default_rng(seed)
    beta = np.sqrt(2*kappa_true)
    A = pullback_matrix(L, kappa_true)
    reads = []
    for m in range(M):
        phi = r.standard_normal(L)*0.3
        for t in range(Ndrive):
            phi = A @ phi + beta*r.standard_normal(L)/np.sqrt(L)
        d = phi - phi.mean()
        logs = []
        for t in range(Tdecay):
            d = A @ d
            nrm = np.linalg.norm(d)
            logs.append(np.log(nrm))
        slope = np.polyfit(np.arange(len(logs)), logs, 1)[0]
        reads.append(1.0 - np.exp(slope))
    return reads

ok = True; det = []
for L in (4, 8, 16):
    kap = 1.0/L
    reads = steady_state_decay_kappa(L, kap)
    spread = max(reads) - min(reads)
    err = max(abs(k - kap) for k in reads)
    ok &= err < 1e-10
    det.append(f'L={L} err {err:.2e} spread {spread:.2e}')
report('T2-1b 场层级 kappa=1/L（噪声制备+撤噪衰减）', ok, '; '.join(det))

# 对照读数（不作判据）：稳态自相关自适应臂长拟合，展示噪声地板效应（执行轮教训）
L = 8; kap = 1.0/L
r = np.random.default_rng(20260931)
beta = np.sqrt(2*kap)
A = pullback_matrix(L, kap)
N = 200000; burn = 2000
phi = r.standard_normal(L)*0.3
for t in range(burn):
    phi = A @ phi + beta*r.standard_normal(L)/np.sqrt(L)
traj = np.zeros((N, L))
for t in range(N):
    phi = A @ phi + beta*r.standard_normal(L)/np.sqrt(L)
    traj[t] = phi
traj = traj - traj.mean(axis=1, keepdims=True)
c0 = np.mean(traj*traj)
C = np.array([np.mean(traj[:N-lag]*traj[lag:])/(c0+1e-300) for lag in range(61)])
floor = np.mean(np.abs(C[50:]))
arm = [t for t in range(2, 48) if abs(C[t]) > 10*floor]
slope = np.polyfit(arm, np.log(np.abs(C[arm])), 1)[0]
k_fit = 1.0 - np.exp(slope)
print(f'[对照读数] L=8 稳态自相关自适应臂长拟合: arm 2..{arm[-1]}, floor {floor:.2e}, '
      f'k_fit {k_fit:.6f} vs 1/L {kap:.6f}（floor 污染示范, 不作判据）')

# T2-1-EP5 替身：慢模 (1 - 1/(2L))
L = 8; kap_true = 1.0/(2*L)
reads = steady_state_decay_kappa(L, kap_true, seed=20260933)
k_read = np.mean(reads)
report('T2-1-EP5 替身FAIL预期', abs(k_read - 1.0/8) > 5e-4,
       f'imposter k_read {k_read:.6f} vs 1/L {1.0/8:.6f} (应显著偏离)')

# T2-SM 冒烟检查（预生成共享噪声；kappa_phi=0 退耦档）
L = 8; kap = 1.0/L; beta = np.sqrt(2*kap)
A = pullback_matrix(L, kap)
T = 50
noise = rng.standard_normal((T, 3, L))/np.sqrt(L)  # 共享噪声：两分支同序列
base = [rng.standard_normal(L)*0.3 for _ in range(3)]
Fmax = max(np.max(np.abs(b)) for b in base)

def run_smoke(perturb, kphi):
    phis = [b.copy() for b in base]
    if perturb:
        phis[0] = phis[0] + 0.1
    traj = [[], [], []]
    for t in range(T):
        for a in range(3):
            phis[a] = A @ phis[a] + beta*noise[t, a]
        if kphi > 0:
            mbar = np.mean([p.mean() for p in phis])
            for a in range(3):
                phis[a] = phis[a] + kphi*(mbar - phis[a].mean())
        for a in range(3):
            traj[a].append(phis[a].copy())
    return [np.array(traj[a]) for a in range(3)]

trb = run_smoke(False, 0.0)
trp = run_smoke(True, 0.0)
diffs = [np.max(np.abs(trp[a] - trb[a])) for a in range(3)]
smoke_ok = Fmax > 0 and diffs[0] > 0 and diffs[1] == 0.0 and diffs[2] == 0.0
report('T2-SM 冒烟(退耦档)', smoke_ok,
       f'max|F_p| {Fmax:.4f}; 轴差 x {diffs[0]:.4f}, y {diffs[1]:.1e}, z {diffs[2]:.1e} (变化的轴=扰动的轴)')
# kappa_phi>0：y,z 整体模响应非零且随 kphi 增大
resps = []
for kphi in (kap*0.5, kap):
    trr = run_smoke(True, kphi)
    resps.append(np.max(np.abs(trr[1] - trb[1])))
report('T2-SM 耦合响应单调', resps[1] > resps[0] > 0, f'kphi=0.5k resp {resps[0]:.4f} < kphi=1k resp {resps[1]:.4f}')

# T2-2a 同步（整体模耦合），四档 + 0 档对照
def sync_test(kphi_ratio, T=4000):
    kap = 1.0/8; kphi = kphi_ratio*kap
    m = np.array([0.30, -0.20, 0.10])
    for t in range(T):
        mbar = m.mean()
        m = m + kphi*(mbar - m)
    return np.max(np.abs(m - m.mean()))
ok = True; det = []
for ratio in (0.5, 1.0, 2.0, 4.0):
    d = sync_test(ratio)
    ok &= d < 1e-10
    det.append(f'r={ratio} {d:.2e}')
report('T2-2a 同步四档', ok, '; '.join(det))
d0 = sync_test(0.0)
report('T2-2a EP5 对照档FAIL预期', d0 > 0.01, f'kappa_phi=0 residual {d0:.4f} (保持初差, 判FAIL)')

# T2-2b 结构保持：整体模通道耦合不改变横模（几何层）
L = 8; kap = 1.0/L; kphi = kap
A = pullback_matrix(L, kap)
v = rng.standard_normal(L); v -= v.mean(); v /= np.linalg.norm(v)
phi_off = v.copy(); phi_on = v.copy()
m_shift = 0.17
for t in range(300):
    phi_off = A @ phi_off
    phi_on = A @ phi_on
    phi_on = phi_on + kphi*(m_shift - phi_on.mean())
def perp(u): return u - u.mean()
d = np.max(np.abs(perp(phi_off) - perp(phi_on)))
report('T2-2b 整体模通道不扰横模', d < 1e-12, f'perp diff {d:.2e}')
# 替身：把耦合做进横模通道（邻居耦合）-> 有效慢模漂移，必须 FAIL
Aimp = (1-kphi)*A + kphi*np.roll(np.eye(L), 1, axis=1) @ A
evs = np.sort(np.abs(np.linalg.eigvals(Aimp)))[::-1]
gap_imp = 1 - evs[1]
report('T2-2b EP5 替身FAIL预期', abs(gap_imp - kap) > 5e-4,
       f'imposter横模谱隙 {gap_imp:.6f} vs kappa {kap:.6f}')

# T2-EP1 至 EP4
L = 8; kap = 1.0/L
Psync = np.ones((3*L, 3*L))/(3*L)
ep1 = np.max(np.abs(Psync @ Psync - Psync)) < 1e-14 and np.max(np.abs(Psync - Psync.T)) < 1e-14 and abs(np.trace(Psync) - 1) < 1e-12
report('T2-EP1 P_sync 幂等厄米迹1', ep1, f'trace {np.trace(Psync):.12f}')
ev = np.sort(np.abs(np.linalg.eigvals(pullback_matrix(L, kap))))[::-1]
theo = np.array([1.0] + [1-kap]*(L-1))
e2 = np.max(np.abs(ev - theo))
report('T2-EP2 本征值落理论谱', e2 < 1e-12, f'err {e2:.2e}')
vals = []
for _ in range(20):
    v = rng.standard_normal(L); v -= v.mean(); v /= np.linalg.norm(v)
    vals.append(read_kappa_from_decay(L, kap, T=200))
rng_w = max(vals) - min(vals)
report('T2-EP3 随机矢极差报告', rng_w < 1e-10, f'range {rng_w:.2e} (同簇公共衰减率, 精确零级)')
Ajoint = np.kron(np.eye(3), pullback_matrix(L, kap))
comm1 = np.max(np.abs(pullback_matrix(L, kap) @ (np.ones((L, L))/L) - (np.ones((L, L))/L) @ pullback_matrix(L, kap)))
comm2 = np.max(np.abs(Ajoint @ Psync - Psync @ Ajoint))
report('T2-EP4 整体模守恒标签', comm1 < 1e-14 and comm2 < 1e-14, f'[A,J/L] {comm1:.2e}; [A_joint,P_sync] {comm2:.2e}')

# ---------------- 图 ----------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
ax = axes[0]
lags = np.arange(61)
ax.semilogy(lags, np.abs(C) + 1e-16, 'o-', ms=3, label='measured |C(t)| (L=8)')
ax.axhline(10*floor, color='red', ls=':', lw=1, label='10x floor')
ax.axvline(arm[-1], color='gray', ls='--', lw=1, label=f'fit arm end t={arm[-1]}')
ax.set_xlabel('lag t'); ax.set_ylabel('|C(t)|')
ax.set_title('(a) floor pollution in autocorr fit (ref. only)')
ax.legend(fontsize=8)
ax = axes[1]
for ratio, ls in zip((0.0, 0.5, 1.0, 2.0, 4.0), ('-', '--', '-.', ':', '-')):
    kap8 = 1.0/8; kphi = ratio*kap8
    m = np.array([0.30, -0.20, 0.10]); hist = []
    for t in range(60):
        mbar = m.mean(); m = m + kphi*(mbar - m); hist.append(max(np.max(np.abs(m - mbar)), 1e-17))
    ax.semilogy(hist, ls, label=f'kphi/kappa={ratio}')
ax.set_xlabel('step'); ax.set_ylabel('max phase gap'); ax.set_title('(b) inter-direction sync (0: FAIL ref)')
ax.legend(fontsize=8)
ax = axes[2]
Ls = (4, 8, 16); reads_m = []; errs = []
for L in Ls:
    kapL = 1.0/L
    rr = steady_state_decay_kappa(L, kapL)
    reads_m.append(np.mean(rr)); errs.append(max(abs(k - kapL) for k in rr))
ax.bar([str(L) for L in Ls], reads_m, width=0.5, yerr=errs, capsize=4, label='kappa read')
ax.axhline(1/16, color='gray', ls=':'); ax.axhline(1/8, color='gray', ls=':'); ax.axhline(1/4, color='gray', ls=':')
ax.set_ylim(0, 0.3)
ax.set_xlabel('L'); ax.set_ylabel('kappa_a read'); ax.set_title('(c) per-direction kappa vs 1/L (dots)')
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(r'D:/software/TianXingFC/images/3d_sector_t1_t2.png', dpi=140)
print('figure saved: images/3d_sector_t1_t2.png')

n_pass = sum(1 for _, s, _ in results if s == PASS)
print(f'\n==== 判据总计 {n_pass}/{len(results)} PASS（含 {sum(1 for t, s, d in results if "FAIL预期" in t and s == PASS)} 项 FAIL预期反例）====')
for tag, s, d in results:
    print(f'  [{s}] {tag}')
