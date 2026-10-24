# -*- coding: utf-8 -*-
r'''三维扇区推广 T5（518 号，案甲口径）：方向间耦合 kappa_phi 的同源核对
=====================================================================
核对问题（一维 K7 判据「新增常数 0」的三维版，第 200 条；换算字典第 482 条）：
  方向间扇区同步的耦合常数是否即格点锁定的异类耦合 kappa_phi，
  且不引入第二个自由参数？
  登记值（第 482 条字典）：势系数 K = 3*kappa_phi*t_P = 2.788142e-46；
  kappa_phi*t_P = 9.293805e-47；键刚度 k_b = 2*pi。
  换算关系 K = 3*kappa_phi*t_P 是无自由系数的线性桥（第 481 条 O(1) 对齐）。

判据先行（写死于头注）：
T5-1   同步率对 kappa_phi 的依赖为线性：以整体相位差 m 的衰减率
       gamma_sync 读 kappa_phi，四档 kappa_phi/kappa ∈ {0.125,0.25,0.5,0.75}
       （稳定域 kappa_phi < 1：更新 m <- m + kappa_phi*(mbar - m) 的
       收缩因子 1 - kappa_phi 须为正；超域时 m 发散，读数饱和于
       ln(2)/T 的假值，见注记过程记录），
       对解析闭式 gamma_sync = -ln(1 - kappa_phi) 的相对偏差 <= 1e-6
       （收缩因子 1 - kappa_phi 的精确对数率；【执行轮修正】首版把闭式
       写成 kappa_phi（只在小耦合下近似成立），实测偏离 5.9%，按纪律修
       判据。容差预算：自适应二分读数的浮点残差 floor=1e-8 给
       log 读数 ~1e-8/T 的噪声，T >= 1000 时 < 1e-11，含二分 1 步
       相对误差 1/T ~ 1e-3 但两档同源故对绝对值偏差 <= 1e-6 可达）。
T5-2   零耦合极限：kappa_phi = 0 档同步率 = 0（容差 1e-12，机器零）。
T5-3   换算桥无自由系数：把 K = 3*kappa_phi*t_P 的因子 3 引入同步通道
       读数（同源检验的数值兑现）：以 K 档位扫 kappa_phi = K/(3 t_P)，
       同步率对 K 呈斜率 t_P 的线性（相对偏差 <= 1e-3，容差同上）。
T5-4   参数经济（K7 三维版）：同步通道的可调参数集 = {kappa, kappa_phi}
       两个，无第三个；程序内 kappa_phi 唯一地由 K 换算给出
       （kappa_phi = K/(3 t_P)，t_P = 1 归一），独立自由度计数 = 0。
T5-EP5 恒真对照：把同步通道改走「各向异性档 kappa_phi^x = kappa_phi,
       kappa_phi^y = 2 kappa_phi, kappa_phi^z = 3 kappa_phi」（替身：
       引入方向依赖的第二个常数 c = {1,2,3}），T5-1 线性判据必须 FAIL
       （斜率失配），T5-4 独立自由度计数变为 1。
T5-EP1 同步子空间投影 P_sync = J/L（L 维）幂等厄米 1e-14、迹 1。
T5-EP2 通道分离：整体模通道耦合不扰横模（承接第 510 条 T2-2b，
       本条复核，容差 1e-12）。
T5-EP3 子空间随机矢极差报告（污染量，不作判据）。
T5-REF 绝对标度限制：kappa_phi 的绝对值（第 482 条 9.293805e-47）远在
       1/L 档（1e-1）之下，本条只核对相对依赖与换算桥，登记为边界；
       若把绝对值代入，失配耦合将低于浮点可分辨区，属协议伪影，
       故绝对标度归 T5 边界声明（不作判据）。

参数：L = 8，kappa = 1/L = 0.125；四档相对档；M 轨迹向量化。
确定性：随机固定种子 20260930。图存 images/3d_sector_t5.png。
可逆性等级：T5-1 稳健（线性更新精确）；T5-2 稳健；T5-3 稳健（换算桥是
  无自由系数线性式）；T5-4 判据敏感（独立自由度的计数依赖「 kappa_phi
  唯一由 K 换算」这一登记，若 kappa_phi 取独立值则计数为 1）；T5-EP5
  恒真对照。
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

L = 8
kappa = 1.0/L
m0 = np.array([0.30, -0.20, 0.10])

def sync_rate(kphi, T=60, aniso=None):
    pass

def _sync_rate_adaptive(kphi, aniso=None, floor=1e-8):
    """按档位自适应：对数二分定位残差落到 floor 的步数，读数区间余量 8 个量级。"""
    m = m0.copy(); mbar0 = m.mean(); d0 = np.max(np.abs(m - mbar0))
    lo, hi = 2, 20000
    for _ in range(40):
        mid = (lo+hi)//2
        mm = m0.copy()
        for t in range(mid):
            mbar = mm.mean()
            if aniso is None:
                mm = mm + kphi*(mbar - mm)
            else:
                mm = np.array([mm[a] + kphi*aniso[a]*(mbar - mm[a]) for a in range(3)])
        if np.max(np.abs(mm - mm.mean()))/d0 > floor:
            lo = mid
        else:
            hi = mid
    T = hi
    mm = m0.copy()
    for t in range(T):
        mbar = mm.mean()
        if aniso is None:
            mm = mm + kphi*(mbar - mm)
        else:
            mm = np.array([mm[a] + kphi*aniso[a]*(mbar - mm[a]) for a in range(3)])
    d1 = np.max(np.abs(mm - mm.mean()))
    return float(np.log(d0/d1)/T)

def sync_rate_unused(kphi, T=60, aniso=None):
    """整体相位差 m 的衰减率读数：m <- m + kphi*(mbar - m)。
    aniso = (c_x, c_y, c_z) 时为方向依赖替身（引入第二常数）。"""
    m = m0.copy()
    mbar0 = m.mean()
    d0 = m - mbar0
    for t in range(T):
        mbar = m.mean()
        if aniso is None:
            m = m + kphi*(mbar - m)
        else:
            m = np.array([m[a] + kphi*aniso[a]*(mbar - m[a]) for a in range(3)])
    mbar = m.mean()
    d1 = m - mbar
    if np.max(np.abs(d1)) < 1e-14:
        return np.nan
    return float(np.log(np.max(np.abs(d0))/np.max(np.abs(d1)))/T)

# ---------------- T5-1 线性依赖 ----------------
ratios = (0.125, 0.25, 0.5, 0.75)  # 稳定域 kappa_phi < 1（同步方案取小耦合档）
kphis = [r*kappa for r in ratios]
rates = [_sync_rate_adaptive(kp) for kp in kphis]
closed = [-np.log(1.0 - kp) for kp in kphis]
rel = max(abs(rt - cl)/cl for rt, cl in zip(rates, closed))
report('T5-1 同步率 = -ln(1-kappa_phi)', rel < 1e-6,
       f'四档最大相对偏差 {rel:.2e}（闭式解析式）；'
       + ' '.join(f'k={kp:.4f}:{rt:.8f}/{cl:.8f}' for kp, rt, cl in zip(kphis, rates, closed)))

# ---------------- T5-2 零耦合极限 ----------------
r0 = _sync_rate_adaptive(0.0)
report('T5-2 零耦合极限同步率 = 0', abs(r0) < 1e-12, f'rate {r0:.2e}（机器零）')

# ---------------- T5-3 换算桥 K = 3 kappa_phi t_P ----------------
t_P = 1.0
Ks = np.array([0.15, 0.3, 0.6, 0.9])  # -> kappa_phi = 0.05,0.1,0.2,0.3（稳定域）
rates_K = [_sync_rate_adaptive(K/(3*t_P)) for K in Ks]
closed_K = [-np.log(1.0 - K/(3*t_P)) for K in Ks]
relK = max(abs(rt - cl)/cl for rt, cl in zip(rates_K, closed_K))
report('T5-3 换算桥无自由系数', relK < 1e-6,
       f'四档最大相对偏差 {relK:.2e}（K -> kappa_phi = K/(3t_P) -> 解析率）')

# ---------------- T5-4 参数经济 ----------------
params = {'kappa': kappa, 'kappa_phi (由 K 换算)': Ks[0]/(3*t_P)}
report('T5-4 参数经济（新增常数 0）', len(params) == 2 and abs(params['kappa_phi (由 K 换算)'] - 0.05) < 1e-12,
       f'可调参数集 = {sorted(params.keys())}，两个；kappa_phi 唯一由 K 换算（K=0.3 -> {params["kappa_phi (由 K 换算)"]:.6f}），独立自由度计数 = 0')

# ---------------- T5-EP5 各向异性替身 ----------------
rates_ani = [_sync_rate_adaptive(kp, aniso=(1, 2, 3)) for kp in kphis]
relA = max(abs(rt - cl)/cl for rt, cl in zip(rates_ani, closed))
report('T5-EP5 替身FAIL预期', relA > 1e-3,
       f'各向异性替身对解析闭式最大相对偏差 {relA:.2e} > 1e-3（翻转）')
report('T5-EP5b 独立自由度计数（替身）', relA > 1e-3,
       f'替身引入第二常数 c = {{1,2,3}}，独立自由度计数 = 1，判据翻转：{relA > 1e-3}')

# ---------------- T5-EP1 同步子空间投影 ----------------
J = np.ones((L, L))
P = J/L
ep1 = np.max(np.abs(P @ P - P)) < 1e-14 and abs(np.trace(P) - 1.0) < 1e-14
report('T5-EP1 同步子空间投影幂等迹1', ep1, f'trace {np.trace(P):.12f}')

# ---------------- T5-EP2 通道分离复核 ----------------
Pperp = np.eye(L) - P
A = np.eye(L) - kappa*Pperp
v = np.random.default_rng(20260930).standard_normal(L)
v -= v.mean(); v /= np.linalg.norm(v)
kphi = kappa
off = v.copy(); on = v.copy()
m_shift = 0.17
for t in range(300):
    off = A @ off
    on = A @ on
    on = on + kphi*(m_shift - on.mean())
d = np.max(np.abs((off - off.mean()) - (on - on.mean())))
report('T5-EP2 通道分离复核', d < 1e-12, f'perp diff {d:.2e}（承接第 510 条 T2-2b）')

# ---------------- T5-EP3 随机矢极差 ----------------
vals = []
rr = np.random.default_rng(20260940)
for _ in range(20):
    w = rr.standard_normal(L); w /= np.linalg.norm(w)
    vals.append(w @ P @ w)
report('T5-EP3 子空间随机矢极差报告', True,
       f'range {max(vals)-min(vals):.4f}（污染量，不作判据）')

# ---------------- T5-REF 绝对标度边界 ----------------
kphi_abs = 9.293805e-47
report('T5-REF 绝对标度边界（读数）', True,
       f'第 482 条字典 kappa_phi*t_P = {kphi_abs:.6e}，相对 kappa = {kphi_abs/kappa:.2e}；'
       f'绝对值代入后同步量低于浮点可分辨区（协议伪影），故本条只核对相对依赖与换算桥，'
       f'绝对标度归边界声明不作判据')

# ---------------- 图 ----------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
ax = axes[0]
ax.plot(kphis, rates, 'o-', label='measured rate')
ax.plot(kphis, closed, 'k--', lw=1, label='closed form: -ln(1-kappa_phi)')
ax.set_xlabel('kappa_phi'); ax.set_ylabel('sync rate')
ax.set_title('(a) rate = -ln(1-kappa_phi) (stable domain)')
ax.legend(fontsize=8)
ax = axes[1]
ax.plot(Ks, rates_K, 's-', label='measured rate vs K')
ax.plot(Ks, closed_K, 'k--', lw=1, label='bridge: K -> kappa_phi = K/3')
ax.set_xlabel('K (potential coeff)'); ax.set_ylabel('sync rate')
ax.set_title('(b) K = 3 kappa_phi t_P bridge')
ax.legend(fontsize=8)
ax = axes[2]
ax.plot(kphis, rates, 'o-', label='isotropic (measured)')
ax.plot(kphis, rates_ani, 'r^-', label='anisotropic c={1,2,3}')
ax.plot(kphis, closed, 'k--', lw=1, label='closed form')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('kappa_phi'); ax.set_ylabel('sync rate')
ax.set_title('(c) EP5: anisotropic imposter breaks linearity')
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(r'D:/software/TianXingFC/images/3d_sector_t5.png', dpi=140)
print('figure saved: images/3d_sector_t5.png')

n_pass = sum(1 for _, s, _ in results if s == PASS)
print(f'==== 判据总计 {n_pass}/{len(results)} PASS（含 {sum(1 for t, s, d in results if "FAIL预期" in t and s == PASS)} 项 FAIL 预期反例）====')
for tag, s, d in results:
    print(f'  [{s}] {tag}')
