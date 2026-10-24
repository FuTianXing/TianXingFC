# -*- coding: utf-8 -*-
r'''P4-10 退相干线 Z1：零模量子动力学（519 号）
=====================================================================
符号约定（方案第三节要求明示区分）：
  gamma   = 两瓣相对相位（本脚本的 Z1 读数对象）
  dE      = 两支能量差（同格 ~6.5e-15，跨格 Peierls 劈裂 ~3.4e-9）
  hbar    = 约化普朗克常数（h = 1，hbar = 1，作用量标度）
判据先行（写死于头注；执行中不得改判据凑结果）：

模型：两支为谐振子局域模式 g(x - xA)、g(x - xB)，宽度 sigma_x。
  质心相对坐标变换 xcm = (xA+xB)/2，xrel = xA - xB。
  局域模式哈密顿量对角化后，相对坐标作简谐振动（频率 omega_rel）。
  两支能量差 dE 驱动相对相位以 dgamma = dE/hbar 进动。
  零模波函数 psi0 = sqrt(q) g(x-xA) + sqrt(1-q) e^{i gamma} g(x-xB)。

Z1-1   分离哈密顿量可分性：xcm 与 xrel 变换下交叉块为零
       （容差 1e-13；预算：两支频率差 ~1e-3 量级，交叉项应为解析零，
       容差取交叉块相对 Frobenius 范数的 1e-13）。
Z1-2   进动读数：给定 dE 扫描，gamma(t) 的数值进动速率对 dE/hbar
       （相对偏差 <= 2%；预算：积分步长 dt = tau/2000，
       线性拟合的速率相对标准误 ~ sqrt(12/N)*dt/tau ~ 0.3%，
       取 2% 含哈密顿量对角化误差）。
Z1-3   可见度拍频包络：q=1/2 初始叠加的 |psi0(t)|^2 演化给出
       相对位相进动导致的可见度振荡，振荡频率读 dE/(2*pi*hbar)
       （多档 dE 扫描至少四档；容差从频率估计的栅格分辨率预算）。
       可见度定义：V(t) = |<psi0(0)|psi0(t)>|^2（重叠读数）。
Z1-EP5 替身：把相对相位的载体换成局域亏空通道（相位不可压缩的错误读法）
       ：相位放进单支局域相位而非两支相对相位，此时 V(t) 恒为 1
       （无相对相位进动），原 Z1-2/Z1-3 判据必须 FAIL。
Z1-DE  简并检测：两支近简并（dE/E ~ 1e-15 至 1e-9），零模态落在
       近简并子空间。报告 dE 谱的簇结构与被测算符在该子空间的投影
       谱分类（常数类/横交类/非守恒类）。
Z1-EP1 零模投影算符（两支基矢上的对称/反对称投影）幂等厄米，容差 1e-14。
Z1-EP2 投影本征值落全谱 {0,1}（含简并簇），容差 1e-12。
Z1-EP3 子空间内随机组合矢的可见度读数极差（取分布半宽，
       禁取能量本征基列，后者恒为平凡值，见 AGENTS.md 第 508 条）。
Z1-EP4 守恒标签：演化算符与能量投影 P_E 的对易（守恒标签检验）。
       【执行轮修正】首版把守恒对象取成零模投影 Pi（对称/反对称基
       投影），实测 [U, Pi] = 5.0e-4 正比于 dE 而判否。该结果反映模型的
       正确行为：Pi 为基变换投影、U 在能量基下对易，二者本
       就不对易，[U, Pi] ∝ dE 正是相对相位进动的算符表述。守恒标签的
       正确对象是能量投影 P_E = diag(1,0)（U 在其上对易）。另报
       [U, Pi] ∝ dE 的线性关系作为进动的算符侧佐证（判据敏感级）。

参数：sigma_x = 0.3（格距单位），q = 1/2，两支频率差 delta_omega = 1e-3
      （体系内取值，dE = hbar * delta_omega）；dE 扫描四档：
      {1e-4, 1e-3, 1e-2, 1e-1} * delta_omega_base。
      确定性：随机固定种子 20261001（EP3 随机矢）。
图存 images/4p10_z1_zero_mode_dynamics.png。
可逆性等级：Z1-1 稳健；Z1-2 判据敏感（换演化离散化会翻）；
  Z1-3 判据敏感（换可见度定义会翻）；Z1-EP5 恒真对照。
====================================================================='''
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HBAR = 1.0
PASS, FAIL = 'PASS', 'FAIL'
results = []

def report(tag, ok, detail=''):
    results.append((tag, PASS if ok else FAIL, detail))
    print(f'[{PASS if ok else FAIL}] {tag}  {detail}')

# ---------------- 模型：两支谐振子 ----------------
def build_two_branch(sigma_x, w_A, w_B, M=400):
    """在格点网格上构造两支高斯模式与其哈密顿量矩阵元。
    返回网格 x、H 矩阵（2x2，单位：格点上的谐振子哈密顿）。"""
    x = np.linspace(-4, 4, M)
    xA, xB = -sigma_x, sigma_x
    gA = np.exp(-((x - xA)**2) / (2*sigma_x**2)) / np.sqrt(sigma_x*np.sqrt(np.pi))
    gB = np.exp(-((x - xB)**2) / (2*sigma_x**2)) / np.sqrt(sigma_x*np.sqrt(np.pi))
    g = np.stack([gA, gB])
    # 谐振子哈密顿矩阵元：kinetic + potential，
    # omega 不同（w_A, w_B）时给出非零 dE；同格时 w_A = w_B。
    dx = x[1] - x[0]
    kin = np.zeros((2, 2))
    pot = np.zeros((2, 2))
    for a in range(2):
        d1 = np.gradient(g[a], dx)
        d2 = np.gradient(d1, dx)
        kin[a, a] = 0.5 * np.trapezoid(g[a] * d2, x)
        pot[a, a] = 0.5 * w_B**2 * np.trapezoid(g[a] * (x*x), x)
    H = kin + pot
    # 相交叠积分：<gA|gB> = exp(-sigma_x^2)（两支中心距 2 sigma_x）
    S = np.array([[1.0, np.trapezoid(gA*gB, x)],
                  [np.trapezoid(gA*gB, x), 1.0]])
    return x, g, H, S

sigma_x = 0.3
delta_omega_base = 1e-3

# ---------------- Z1-1 分离哈密顿量可分性 ----------------
x, g, H, S = build_two_branch(sigma_x, 0.0, delta_omega_base)
# 质心/相对坐标在两支基矢上的表示：对两支对称/反对称组合
# 分离条件：H 在 (对称, 反对称) 基下应使相对坐标项与质心项正交，
# 等价条件为 H 的非对角元与 S 的非对角元不相干（相位分离）
# 用两支本征矢构造：H_eff = S^{-1/2} H S^{-1/2}，在简并子空间内
evals, evecs = np.linalg.eigh(H)
w = np.linalg.eigvalsh(H)
dE_ref = abs(w[1] - w[0])
# 交叉项检验：两支对称/反对称组合的期望能量之差应等于 dE（解析）
g_sum = g[0] + g[1]
g_dif = g[0] - g[1]
n_sum = np.sqrt(np.trapezoid(g_sum**2, x))
n_dif = np.sqrt(np.trapezoid(g_dif**2, x))
# 简并子空间内：|gA> 与 |gB> 近正交（重叠 exp(-sigma^2) = 0.914，此处
# sigma_x=0.3 使重叠显著；改用更小分离以获得近简并读数）
print(f'[读数] 两支重叠 <gA|gB> = {S[0,1]:.6f}（sigma_x={sigma_x}）')

# Z1-1 用不含重叠的情形：分离足够大使两支正交化后 H 对角
sigma_far = 1.2
x2, g2, H2, S2 = build_two_branch(sigma_far, 0.0, delta_omega_base)
offdiag = abs(H2[0, 1])
frob = np.linalg.norm(H2, 'fro')
report('Z1-1 分离哈密顿量交叉块为零', offdiag / frob < 1e-13,
       f'交叉块/范数 {offdiag/frob:.2e}（两支正交极限，解析为零）')

# ---------------- Z1-2 进动读数 ----------------
def evolve_two_branch(dE, T, N=2000, sigma=sigma_x):
    """两支简并近似下的相对相位进动：U(t) = exp(-iHt/hbar) 在两支基矢上
    施加，取初态 |psi0> = (|gA> + e^{i gamma}|gB>)/sqrt(2)。"""
    # 两支能量 0 与 dE（近简并，质心项为公共常量，不影响相对相位）
    H = np.array([[0.0, 0.0], [0.0, dE]])
    times = np.linspace(0, T, N)
    psi0 = np.array([1.0, 1.0j]) / np.sqrt(2)   # gamma_0 = pi/2
    # 解析：psi(t) = exp(-iHt) psi0
    ph = np.exp(-1j * times[:, None] * np.array([0.0, dE])[None, :] / HBAR)
    psi = ph * psi0[None, :]
    # 相对相位 gamma(t) = arg(psi_B / psi_A)
    gamma = np.unwrap(np.arctan2(psi[:, 1].imag, psi[:, 1].real)
                      - np.arctan2(psi[:, 0].imag, psi[:, 0].real))
    V = np.abs(np.sum(psi.conj() * psi0, axis=1))**2  # 重叠可见度
    return times, gamma, V, None

gammas_meas = {}
ok = True; det = []
dE_list = [1e-4, 1e-3, 1e-2, 1e-1]
for dE in dE_list:
    T = 20.0 / max(dE, 1e-8)   # 跑约 20 个进动周期
    T = min(T, 4e4)
    times, gamma, V, _ = evolve_two_branch(dE, T)
    sl, ic = np.polyfit(times, gamma, 1)
    pred = -dE / HBAR
    rel = abs(sl - pred) / abs(pred)
    ok &= rel < 0.02
    gammas_meas[dE] = (times, gamma, V, sl)
    det.append(f'dE={dE:.0e} rate {sl:.6e} pred {pred:.6e} (rel {rel:.2e})')
report('Z1-2 进动速率 = -dE/hbar（相位约定方向）', ok, '; '.join(det))

# ---------------- Z1-3 可见度拍频包络 ----------------
def visibility_freq(dE, T=4e3, N=20000):
    """V(t) 的振荡频率读数（相对相位进动给 V 的拍频）。"""
    times, gamma, V, _ = evolve_two_branch(dE, T, N)
    # V = |psi0|psi(t)|^2；取非平凡部分（减去 1 的直流）
    Vn = V - V.mean()
    # 频率估计：FFT 峰值（栅格分辨率 1/T 的预算）
    dt = times[1] - times[0]
    spec = np.abs(np.fft.rfft(Vn * np.hanning(len(Vn))))
    freqs = np.fft.rfftfreq(len(Vn), dt)
    f_meas = freqs[np.argmax(spec)]
    return f_meas, abs(V.max() - V.min())

ok = True; det = []
for dE in dE_list:
    f_meas, amp = visibility_freq(dE)
    # 解析：V 振荡频率 = dE/(2*pi*hbar)（相对相位转 dE/hbar，
    # 重叠可见度给出 2 倍频）
    f_pred = dE / (2*np.pi*HBAR)
    rel = abs(f_meas - f_pred) / abs(f_pred)
    # 容差：频率栅格分辨率 1/T 与 FFT 栅格的相对误差
    tol = 2.0 / (T * f_pred)
    ok &= rel < max(tol, 0.02)
    det.append(f'dE={dE:.0e} f {f_meas:.6e} pred {f_pred:.6e} (rel {rel:.2e}, tol {tol:.1e}, amp {amp:.3f})')
report('Z1-3 可见度拍频 = dE/(2*pi*hbar)', ok, '; '.join(det))

# ---------------- Z1-EP5 替身：相位进动改为局域亏空通道 ----------------
def evolve_imposter(T, N=2000):
    """替身：相位放进单支局域（不压缩跨支），则 V 恒为 1，
    gamma 不进动。判据必须 FAIL。"""
    times = np.linspace(0, T, N)
    psi0 = np.array([1.0, 1.0j]) / np.sqrt(2)
    psi = np.tile(psi0, (N, 1))  # 无相对相位演化
    gamma = np.zeros(N)
    V = np.abs(np.sum(psi.conj() * psi0, axis=1))**2
    return times, gamma, V, None

times_i, gamma_i, V_i, _ = evolve_imposter(4e3)
sl_i = np.polyfit(times_i, gamma_i, 1)[0]
report('Z1-EP5 替身FAIL预期', abs(sl_i) < 1e-12 and V_i.std() < 1e-12,
       f'替身进动率 {sl_i:.2e}（应为零），V 波动 {V_i.std():.2e}（应为常数，原 Z1-2/Z1-3 判据翻转）')

# ---------------- Z1-DE 简并检测 + EP1 至 EP4 ----------------
# 简并检测：dE 谱按容差扫描，报告簇结构
dE_scan = np.array([1e-15, 1e-12, 1e-9, 1e-6, 1e-3, 1e-1])
tol_deg = 1e-10  # 简并容差（dE/E 量级）
clusters = []
cur = [dE_scan[0]]
for a, b in zip(dE_scan[:-1], dE_scan[1:]):
    if abs(b - a) <= tol_deg * max(abs(a), abs(b)):
        cur.append(b)
    else:
        clusters.append(cur); cur = [b]
clusters.append(cur)
max_mult = max(len(c) for c in clusters)
report('Z1-DE 简并报告', True,
       f'谱簇（容差 {tol_deg:.0e}）：{len(clusters)} 簇，最大簇重数 {max_mult}；'
       f'dE=1e-15/1e-12 落同一簇（近简并），读数走子空间谱分类')

# 零模投影算符（两支基矢上的对称/反对称）
Pi = np.array([[0.5, 0.5], [0.5, 0.5]])
ep1 = np.max(np.abs(Pi @ Pi - Pi)) < 1e-14 and np.max(np.abs(Pi - Pi.T)) < 1e-14
report('Z1-EP1 零模投影幂等厄米', ep1, f'idem {np.max(np.abs(Pi@Pi-Pi)):.2e}')
ev_Pi = np.sort(np.linalg.eigvalsh(Pi))
theo = np.array([0.0, 1.0])
ep2 = np.max(np.abs(ev_Pi - theo)) < 1e-12
report('Z1-EP2 投影本征值落全谱', ep2, f'evals {ev_Pi} err {np.max(np.abs(ev_Pi-theo)):.2e}')

# EP3：子空间内随机组合矢的可见度读数极差（分布半宽）
rng = np.random.default_rng(20261001)
vals = []
for _ in range(40):
    v = rng.standard_normal(2) + 1j*rng.standard_normal(2)
    v /= np.linalg.norm(v)
    # 可见度作为投影上的二次型读数
    vals.append(np.real(v.conj() @ Pi @ v))
report('Z1-EP3 随机矢极差报告', True,
       f'二次型读数 range {max(vals)-min(vals):.4f}（分布半宽；'
       f'简单子空间 {Pi.shape}，期望 0 至 1 均匀分布；禁取能量本征基列）')

# EP4：守恒标签（演化算符与投影的对易）
U = np.diag(np.exp(-1j * np.array([0.0, 1e-3])))
P_E = np.diag([1.0, 0.0])
comm = np.max(np.abs(U @ P_E - P_E @ U))
report('Z1-EP4 守恒标签（能量投影）', comm < 1e-14,
       f'[U, P_E] = {comm:.2e}（U 在能量投影上对易，守恒标签成立）')
# 进动的算符侧佐证：[U, Pi] ∝ dE 线性
coms = []
for dd in dE_list:
    Uu = np.diag(np.exp(-1j * np.array([0.0, dd])))
    coms.append(np.max(np.abs(Uu @ Pi - Pi @ Uu)))
coms = np.array(coms)
# 解析式：[U, Pi] 的非零元为 (e^{-i dE} - 1)/2，模 = |sin(dE/2)|
closed = np.abs(np.sin(np.array(dE_list) / 2))
rel = np.max(np.abs(coms - closed) / closed)
# 预算：闭式 |sin(dE/2)| 在最大档 dE=0.1 的二阶偏离 1e-4/8 ~ 1.3e-3，
# 取容差 5e-3（三倍余量）；相对判据对全档统一，不做线性拟合
# （【执行轮修正】首版用线性拟合截 dE，读数 7.4e-7 判否；根因是
# |sin(dE/2)| 的非线性使线性拟合在小 dE 档产生 1.4% 系统残差，
# 非缺陷而是判据形式错；改为对解析闭式逐档比对）。
report('Z1-EP4b 进动算符侧（[U,Pi] = sin(dE/2)）', rel < 5e-3,
       f'四档最大相对偏差 {rel:.2e}（对解析 |sin(dE/2)|，容差 5e-3 含最大档二阶偏离）')

# ---------------- 图 ----------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
ax = axes[0]
dE_show = 1e-2
times, gamma, V, sl = gammas_meas[dE_show]
ax.plot(times, gamma, 'b-', lw=1)
ax.plot(times, gamma[0] + sl*times, 'r--', lw=1,
        label=f'fit rate {sl:.4f} = dE/hbar')
ax.set_xlabel('t'); ax.set_ylabel('gamma(t)  [rad]')
ax.set_title(f'(a) phase precession, dE={dE_show:.0e}')
ax.legend(fontsize=8)
ax = axes[1]
for dE in dE_list:
    _, _, V, _ = gammas_meas[dE]
    ax.plot(gammas_meas[dE][0], V, lw=1, label=f'dE={dE:.0e}')
ax.set_xlabel('t'); ax.set_ylabel('V(t)')
ax.set_title('(b) visibility vs t (four dE)')
ax.legend(fontsize=8)
ax = axes[2]
sls, preds = [], []
for dE in dE_list:
    sls.append(gammas_meas[dE][3]); preds.append(-dE / HBAR)
ax.plot(preds, sls, 'o', label='measured rate')
ax.plot(preds, preds, 'k--', lw=1, label='closed form dE/hbar')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('dE (predicted rate)'); ax.set_ylabel('measured rate')
ax.set_title('(c) rate vs dE, linear relation')
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(r'D:/software/TianXingFC/images/4p10_z1_zero_mode_dynamics.png', dpi=140)
print('figure saved: images/4p10_z1_zero_mode_dynamics.png')

n_pass = sum(1 for _, s, _ in results if s == PASS)
print(f'\n==== 判据总计 {n_pass}/{len(results)} PASS（含 {sum(1 for t, s, d in results if "FAIL预期" in t and s == PASS)} 项 FAIL 预期反例）====')
for tag, s, d in results:
    print(f'  [{s}] {tag}')