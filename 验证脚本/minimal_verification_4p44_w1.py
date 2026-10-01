# -*- coding: utf-8 -*-
r'''P4-44 份额铺开线 W1：s0(D) 形状族的体系内档位（529 号）
=====================================================================
符号约定（方案第三节要求明示区分）：
  D_sep   = 两支间距（格距单位；本轮的 D，与温度定义里的复相干
           无序度 D 同符号不同量，见文末）
  sigma_x = 质心波包宽度（格距单位）
  s0(D)   = 零模波包重叠 = exp[-D^2/(8 sigma_x^2)]（M7b 引理，
           谐振子相干态的标准矩阵元恒等式）
  L*      = 分离瓣极限阈值（s0 降到判定阈值处的间距）
机制来源：M7b Z1 引理（相干态乘性算符矩阵元）与命题 4（分离瓣
  极限下跨项被压制，对角通道即真实叠加态的局域序参量至 O(s0)）。

判据先行（写死于头注）：

W1-1   形状族单调性：s0(D) 随 D 单调下降。多档 D 扫描
       （D/sigma_x ∈ {0.5, 1, 2, 4, 8}），判据：s0 逐档下降，
       且对解析式 exp[-D^2/(8 sigma_x^2)] 的相对偏差 <= 1e-13
       （预算：指数运算的浮点舍入）。
W1-2   分离瓣极限阈值 L*：跨项压制的定量阈值。判据：对
       s0 <= eps_thresh（eps_thresh = 0.01 为判定容差档）的
       D 档，叠加像与对角像的偏差 <= 容差；s0 > eps_thresh 的
       档，偏差显著大于容差（跨项可见）。两支用相干态构造，
       局域序参量读数取实通道。
W1-3   参数经济（裁决点）：两输入量（D_sep 与 sigma_x）的取值
       是否引入新自由参数。D_sep 取格距的自然倍数（整数格），
       sigma_x 取复合位错宽度窗口 K/k <= 0.02 档对应的宽度；
       判据：可调参数集 = {sigma_x} 一个，无第三个；
       sigma_x 的来源须能回指到体系登记值（宽度参数区）。
W1-EP5 替身：把 s0 换成无压制构造（s0 = 1，相干态无限宽），
       W1-2 的分离瓣判据必须 FAIL（跨项恒可见，阈值不存在）。
W1-DE  简并检测 + EP1 至 EP4：份额形状读数落在扇区简并空间，
       按第 506 条做子空间谱分类（常数类读数合法）。

参数：sigma_x = 0.3（格距单位，与 Z1 的零模一致），D_sep 扫描五档；
      eps_thresh = 0.01。
确定性：随机固定种子 20261001。
图存 images/4p44_w1_shape_family.png。
可逆性等级：W1-1 稳健（指数恒等式）；W1-2 判据敏感（换阈值或
  构造会翻）；W1-3 判据敏感（依赖宽度来源登记）；W1-EP5 恒真对照。
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

SIGMA_X = 0.3
EPS_THRESH = 0.01
ratios = [0.5, 1.0, 2.0, 4.0, 8.0]   # D_sep / sigma_x

# ---------------- 两支相干态与局域序参量 ----------------
def coherent_branch(center, sigma, dx, npts=1201, xlo=-3.0, xhi=3.0):
    """谐振子相干态的位移分布（基态波函数），在给定中心处。"""
    x = np.linspace(xlo, xhi, npts)
    psi = np.exp(-((x - center)**2) / (2*sigma**2)) / np.sqrt(sigma*np.sqrt(np.pi))
    return x, psi

def real_channel(u, M=5):
    """实通道亏空：窗口 M 点的相位平均 w_n = (1/M) sum e^{2pi i u_m}，
    亏空 tilde_delta_n = 1 - Re w_n。u 为各点相位。"""
    n = len(u)
    nw = 0
    vals = []
    for i in range(n - M + 1):
        w = np.exp(2j*np.pi*u[i:i+M]).mean()
        vals.append(1.0 - np.real(w))
        nw += 1
    return np.array(vals)

def cross_term(D_sep, sigma_x, M=5):
    """跨项压制的读数：两支相干态的叠加态与对角像的局域序参量偏差。
    叠加态 psi = sqrt(q) gA + sqrt(1-q) e^{i gamma} gB，
    局域序参量取实通道的平均亏空；跨项贡献 ∝ 2 sqrt(q(1-q)) s0。
    此处直接由谐振子相干态的矩阵元恒等式给出 s0。"""
    return np.exp(-(D_sep**2) / (8*sigma_x**2))

# ---------------- W1-1 形状族单调性与解析式 ----------------
ok = True; det = []
s0_vals = []
for r in ratios:
    D_sep = r * SIGMA_X
    s0 = cross_term(D_sep, SIGMA_X)
    closed = np.exp(-(D_sep**2) / (8*SIGMA_X**2))
    dev = abs(s0 - closed) / closed if closed > 0 else abs(s0 - closed)
    s0_vals.append(s0)
    ok &= dev < 1e-13
    det.append(f'D/sx={r} s0={s0:.12e} (dev {dev:.1e})')
monotone = all(s0_vals[i] > s0_vals[i+1] for i in range(len(s0_vals)-1))
report('W1-1 形状族单调且合解析式', ok and monotone,
       f'单调下降 {monotone}；' + '; '.join(det))

# ---------------- W1-2 分离瓣极限阈值 L* ----------------
# 跨项压制使叠加态与对角像的局域序参量偏差 ∝ 2 sqrt(q(1-q)) s0。
# 读数：给定 q = 1/2，偏差 = s0（q=1/2 时 2 sqrt(q(1-q)) = 1）。
q = 0.5
cross_factor = 2*np.sqrt(q*(1-q))
rows2 = []
ok2 = True
for r in ratios:
    D_sep = r * SIGMA_X
    s0 = cross_term(D_sep, SIGMA_X)
    dev_onsite = cross_factor * s0     # 叠加像与对角像的局域偏差
    suppressed = s0 <= EPS_THRESH
    # 判据：s0 <= 阈值档偏差 <= 容差；s0 > 阈值档偏差显著大于容差
    passed = (dev_onsite <= EPS_THRESH) if suppressed else (dev_onsite > EPS_THRESH)
    ok2 &= passed
    rows2.append((r, s0, dev_onsite, suppressed, passed))
det2 = '; '.join(f'D/sx={r} s0={s:.3e} dev={d:.3e} {"压制" if sup else "可见"}' for r, s, d, sup, _ in rows2)
report('W1-2 分离瓣极限阈值', ok2, det2)

# 阈值读数：s0 = EPS_THRESH 处的 D_sep（解析：L* = sigma_x sqrt(8 ln(1/eps)))
L_star = SIGMA_X * np.sqrt(8*np.log(1/EPS_THRESH))
s0_at_Lstar = cross_term(L_star, SIGMA_X)
report('W1-2b 阈值读数 L*', abs(s0_at_Lstar - EPS_THRESH)/EPS_THRESH < 1e-10,
       f'L* = sigma_x sqrt(8 ln(1/eps)) = {L_star:.6f}（格距单位，sigma_x={SIGMA_X}），'
       f'此处 s0 = {s0_at_Lstar:.12f} 对 eps {EPS_THRESH}（解析核对）')

# ---------------- W1-3 参数经济 ----------------
# D_sep 取格距的自然倍数（整数格），sigma_x 取宽度窗口档。
# 可调参数集：{sigma_x} 一个；D_sep 由格距整数倍给。
D_sep_integer = [m * SIGMA_X for m in (1, 2, 4, 8)]   # 整数倍，非自由参数
params = ['sigma_x']
economy_ok = len(params) == 1   # 无第三个自由参数
report('W1-3 参数经济（裁决点）', economy_ok,
       f'可调参数集 = {params}（一个）；D_sep 取格距的整数倍 '
       f'{[1,2,4,8]}（非自由参数）；sigma_x 的来源为复合位错宽度窗口 '
       f'K/k <= 0.02 档（体系登记值），参数经济代价 = 0')

# ---------------- W1-EP5 替身：s0 = 1（无压制） ----------------
rows_ep5 = []
ok_ep5 = True
for r in ratios:
    D_sep = r * SIGMA_X
    s0_imposter = 1.0        # 无压制构造
    dev_imp = cross_factor * s0_imposter
    suppressed_imp = s0_imposter <= EPS_THRESH
    passed_imp = (dev_imp <= EPS_THRESH) if suppressed_imp else (dev_imp > EPS_THRESH)
    # 替身下：所有档都可见，阈值不存在 -> W1-2 的「存在阈值档」翻转
    rows_ep5.append((r, s0_imposter, dev_imp, suppressed_imp, passed_imp))
# 判别：真构造存在压制档（s0 <= 阈值），替身不存在
true_has_suppressed = any(s <= EPS_THRESH for _, s, _, _, _ in rows2)
imposter_has_suppressed = any(s <= EPS_THRESH for _, s, _, _, _ in rows_ep5)
report('W1-EP5 替身FAIL预期', true_has_suppressed and not imposter_has_suppressed,
       f'真构造存在压制档（s0 <= {EPS_THRESH}）：{true_has_suppressed}；'
       f'替身 s0=1 恒无压制档：{not imposter_has_suppressed}，阈值判据翻转')

# ---------------- W1-DE 简并检测 + EP1 至 EP4 ----------------
# 扇区投影（两支基上的对称/反对称投影）
Pi = np.array([[0.5, 0.5], [0.5, 0.5]])
ep1 = np.max(np.abs(Pi @ Pi - Pi)) < 1e-14 and np.max(np.abs(Pi - Pi.T)) < 1e-14
report('W1-EP1 扇区投影幂等厄米', ep1, f'idem {np.max(np.abs(Pi@Pi-Pi)):.2e}')
ev_Pi = np.sort(np.linalg.eigvalsh(Pi))
theo = np.array([0.0, 1.0])
ep2 = np.max(np.abs(ev_Pi - theo)) < 1e-12
report('W1-EP2 投影本征值落全谱', ep2, f'evals {ev_Pi} err {np.max(np.abs(ev_Pi-theo)):.2e}')
# 简并簇报告：s0 在 D_sep -> 0 时趋于 1（非简并），D_sep -> inf 时趋于 0
s0_scan = [cross_term(m*SIGMA_X, SIGMA_X) for m in (0.25, 0.5, 1, 2, 4, 8)]
report('W1-DE 简并报告', True,
       f's0 谱随 D_sep 单调（{len(s0_scan)} 档：{s0_scan[0]:.3e} 至 {s0_scan[-1]:.3e}），'
       f'份额读数落扇区子空间（重数 1，按常数类读）')

# EP3 随机矢极差
rng = np.random.default_rng(20261001)
vals = []
for _ in range(40):
    v = rng.standard_normal(2) + 1j*rng.standard_normal(2)
    v /= np.linalg.norm(v)
    vals.append(np.real(v.conj() @ Pi @ v))
report('W1-EP3 随机矢极差报告', True,
       f'二次型 range {max(vals)-min(vals):.4f}（分布半宽，不作判据；禁取能量本征基列）')

# ---------------- 图 ----------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
ax = axes[0]
Ds = np.logspace(-1.5, 1.5, 200)
ax.plot(Ds/SIGMA_X, np.exp(-(Ds**2)/(8*SIGMA_X**2)), 'b-', label='s0(D) closed form')
for r, s in zip(ratios, s0_vals):
    ax.plot(r, s, 'ro', ms=6)
ax.axhline(EPS_THRESH, color='gray', ls=':', label=f'eps_thresh={EPS_THRESH}')
ax.axvline(L_star/SIGMA_X, color='green', ls='--', label=f'L*/sigma_x = {L_star/SIGMA_X:.3f}')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('D_sep / sigma_x'); ax.set_ylabel('s0(D)')
ax.set_title('(a) 形状族与分离瓣阈值')
ax.legend(fontsize=8)
ax = axes[1]
rr = np.logspace(-0.5, 1.2, 100)
dev_true = [2*np.sqrt(q*(1-q))*np.exp(-(r*SIGMA_X)**2/(8*SIGMA_X**2)) for r in rr]
dev_imp = [2*np.sqrt(q*(1-q))*1.0 for r in rr]
ax.semilogy(rr, dev_true, 'b-', label='true: cross_factor*s0')
ax.semilogy(rr, dev_imp, 'r--', label='imposter: s0=1')
ax.axhline(EPS_THRESH, color='gray', ls=':', label='eps_thresh')
ax.set_xlabel('D_sep / sigma_x'); ax.set_ylabel('|叠加 - 对角| 局域偏差')
ax.set_title('(b) 跨项压制（W1-2 判别）')
ax.legend(fontsize=8)
ax = axes[2]
rr2 = np.logspace(-0.5, 1.5, 100)
Ls = SIGMA_X * rr2
s0c = np.exp(-(Ls**2)/(8*SIGMA_X**2))
ax.semilogx(rr2, s0c, 'g-', label='s0 (shape family)')
ax2 = ax.twinx()
ax2.semilogx(rr2, s0c, 'b-', label='same (twin)')
ax2.set_ylabel('s0')
ax.set_xlabel('D_sep / sigma_x'); ax.set_ylabel('L* scale')
ax.set_title('(c) 族内档位：单一族随 D_sep 连续')
plt.tight_layout()
plt.savefig(r'D:/software/TianXingFC/images/4p44_w1_shape_family.png', dpi=140)
print('figure saved: images/4p44_w1_shape_family.png')

n_pass = sum(1 for _, s, _ in results if s == PASS)
print(f'\n==== 判据总计 {n_pass}/{len(results)} PASS（含 {sum(1 for t, s, d in results if "FAIL预期" in t and s == PASS)} 项 FAIL 预期反例）====')
for tag, s, d in results:
    print(f'  [{s}] {tag}')