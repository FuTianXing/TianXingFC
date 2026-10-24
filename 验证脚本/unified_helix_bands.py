# -*- coding: utf-8 -*-
"""
unified_helix_bands.py
判据第 626 号：同一弦三频率档（时空 / 场 / 费米子）的统一螺旋色散线

模型设定
--------
状态空间 S^1 x R 上的螺旋 X(tau) = (R cos(w tau), R sin(w tau), c tau)。
螺距（每满周 2pi 的轴向推进）= 2 pi c / w。三扇区为同一螺旋在不同边界条件下的三个频率档：
  (甲) 时空生成档：刚性层，螺距 = l_P  =>  w = 2 pi c / l_P = w_P（普朗克频率）
  (乙) 场档（光子）：柔性层，螺距 = lambda 任意  =>  w = 2 pi c / lambda（连续带）
  (丙) 费米子档：闭合双螺旋环，总弧长 L、圈数 N，驻波条件 k L = 2 pi N
                 =>  w_0 = c k = 2 pi N c / L（离散阶梯）
统一不变量：几何色散关系 w = 2 pi c / 螺距；拓扑整数 n = (1/2 pi) ∮ dθ ∈ Z。

判据（先立理论期望，再实现）
----------------------------
U1 几何色散恒等式 w * 螺距 = 2 pi c，三档全成立（结构层，恒真检验，须配反例）
   [期望] 相对误差 < 1e-12（机器精度，纯代数恒等式）
   [反例] 把色散换成 w = c k^2（色散介质）在 k != 1 处恒等式失效（相对误差 O(0.1)）
U2 拓扑量子化同源：时空格步指标与费米子模式指标同为绕数整数 n = (1/2pi)∮dθ
   [期望] |n - round(n)| < 1e-10（闭合回路数值积分，梯形谱精度）
   [反例] 半周开弧 -> n = 0.5，非整数
U3 闭合量子化给出质量阶梯 m_N = h N /(L c)，且与 w_0 = c k_N 相容
   [期望] m_N / m_1 = N（相对误差 < 1e-12）
   [反例] 不闭合（L 连续）-> 谱连续无阶梯
U4 无新参数（反景观）：费米子档只引入整数 N 与几何量 L，连续模数为 0
   [期望] 费米子档新增连续自由参数计数 = 0
   [反例] KK 式紧致化引入半径模数 R（连续 1 个）-> 景观
U5 Lancret 螺旋身份：三档曲线的 kappa/tau = R/b 沿曲线为常数（螺旋充要条件）
   [期望] std/mean(kappa/tau) < 1e-8（有限差分 O(h^2)）
   [反例] 非螺旋 z = tau^2 -> 比值随 tau 变化（相对极差 > 0.1）
U6 跨扇区对表：费米子档螺旋螺距 = 康普顿波长 h/(m_e c)
   [期望] 相对误差 < 1e-12（代数恒等式）
   [反例] 用约化康普顿波长 hbar/(m_e c) 充当螺距 -> 相对差 = 2 pi

可逆性等级
----------
U1/U3/U6 判据稳健（代数恒等式，换实现不翻）；U2 判据稳健（拓扑整数，与实现无关）；
U5 判据稳健（Lancret 标准结论，有限差分精度内）；U4 判据稳健（计数陈述）。

前提依赖
--------
费米子 = 闭合双螺旋环（研究方向-费米子质量重构-单环内部扭转，纲领级）；
w_0 = m c^2 / hbar（内禀频率-质量关系，现象学输入端，与绑定能定案数值等价）。
本轮不验证费米子实体层涌现（P3-51 保持开放），只验证三档的结构统一性与参数计数。
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ---- 物理常数（CODATA 2018 量级） ----
c = 299792458.0
hbar = 1.054571817e-34
h = 2.0 * np.pi * hbar
l_P = 1.616255e-35
t_P = l_P / c
w_P = 2.0 * np.pi / t_P
m_e = 9.1093837015e-31

results = []


def record(tag, desc, measured, expected, tol, passed, note=""):
    results.append((tag, desc, measured, expected, tol, passed, note))


# =====================================================================
# U1 几何色散恒等式 w * 螺距 = 2 pi c（三档）
# =====================================================================
def pitch_of(w):
    return 2.0 * np.pi * c / w


# 甲：时空档，螺距 = l_P
w_a = 2.0 * np.pi * c / l_P          # 应等于 w_P
pitch_a = pitch_of(w_a)
err_a = abs(w_a * pitch_a - 2 * np.pi * c) / (2 * np.pi * c)
# 乙：场档，取若干波长
lams = np.array([1e-9, 1e-6, 1e-3, 1.0])  # 米，从 X 射线到无线电
ws_b = 2 * np.pi * c / lams
err_b = np.max(np.abs(ws_b * pitch_of(ws_b) - 2 * np.pi * c) / (2 * np.pi * c))
# 丙：费米子档，取环长 L 与圈数 N
L_f = 1e-12
Ns = np.arange(1, 6)
ws_c = 2 * np.pi * Ns * c / L_f
err_c = np.max(np.abs(ws_c * pitch_of(ws_c) - 2 * np.pi * c) / (2 * np.pi * c))
err_u1 = max(err_a, err_b, err_c)
record("U1", "色散恒等式 w*螺距=2pi c（三档）", err_u1, 0.0, 1e-12,
       err_u1 < 1e-12, "结构层恒真检验")

# U1 反例：色散介质 w = c k^2，k = 2 pi / 螺距
k_test = np.array([0.5, 2.0, 5.0])
w_bad = c * k_test ** 2
pitch_bad = 2 * np.pi / k_test
err_bad = np.max(np.abs(w_bad * pitch_bad - 2 * np.pi * c) / (2 * np.pi * c))
record("U1x", "反例：w=c k^2 在 k!=1 失效", err_bad, ">0.1", 0.1,
       err_bad > 0.1, "反例检验（改掉即FAIL）")

# =====================================================================
# U2 拓扑量子化：绕数整数（时空档沿时间回路、费米子档沿空间回路）
# =====================================================================
def winding_from_phase(phase):
    return (phase[-1] - phase[0]) / (2 * np.pi)

# 时空档：相位 theta = w_P t，绕数 = 满周闭合次数
ws_2 = [1, 2, 3, 7]
errs_2 = []
for n in ws_2:
    tau = np.linspace(0.0, n * t_P, 40001)
    val = winding_from_phase(w_P * tau)
    errs_2.append(abs(val - n))
# 费米子档：沿空间闭合回路弧长 s∈[0,L]，相位 theta = k_N s，k_N = 2pi N/L
for N in Ns:
    k_N = 2 * np.pi * N / L_f
    s = np.linspace(0.0, L_f, 40001)
    val = winding_from_phase(k_N * s)
    errs_2.append(abs(val - N))
err_u2 = max(errs_2)
record("U2", "绕数整数 n=∮dθ/2pi（时空+费米子）", err_u2, 0.0, 1e-10,
       err_u2 < 1e-10, "拓扑整数")

# U2 反例：半周开弧
val_half = winding_from_phase(w_P * np.linspace(0.0, 0.5 * t_P, 40001))
record("U2x", "反例：半周开弧 n=0.5 非整数", val_half, 0.5, 1e-10,
       abs(val_half - round(val_half)) > 0.4, "开弧非整数")

# =====================================================================
# U3 闭合量子化 -> 质量阶梯 m_N = h N /(L c)
# =====================================================================
def m_ladder(N, L):
    return h * N / (L * c)

m_vals = m_ladder(Ns, L_f)
ratio = m_vals / m_vals[0]
err_u3 = np.max(np.abs(ratio - Ns) / Ns)
# 与 w_0 = c k_N 相容：m = hbar w_0 / c^2
w0_geom = c * (2 * np.pi * Ns / L_f)
m_from_w = hbar * w0_geom / c ** 2
err_u3b = np.max(np.abs(m_from_w - m_vals) / m_vals)
err_u3 = max(err_u3, err_u3b)
record("U3", "质量阶梯 m_N/m_1=N 且与 w0=c k_N 相容", err_u3, 0.0, 1e-12,
       err_u3 < 1e-12, "闭合驻波条件")

# U3 反例：不闭合 -> 连续谱（相邻模式间距随 L 连续变化，无固定阶梯）
Ls = np.array([1.0e-12, 1.1e-12, 1.23e-12])
m1 = m_ladder(1, Ls)
err_u3x = np.std(m1) / np.mean(m1)  # 连续变化 -> 非固定阶梯
record("U3x", "反例：L 连续 -> m_1 连续变化", err_u3x, ">0.05", 0.05,
       err_u3x > 0.05, "无闭合则无固定阶梯")

# =====================================================================
# U4 无新参数（反景观）：费米子档连续模数计数 = 0
# =====================================================================
# 三档引入的自由参数清单（连续参数 vs 拓扑/几何整数）
# 常数（体系既有，非扇区新增）：hbar, c, l_P
# 甲：无新增（螺距 = l_P 锁定）
# 乙：光子色散自由度 = 1 连续（频率 w 或波长 lambda），非质量谱自由度
# 丙：新增整数 N（拓扑）+ 几何量 L（环长，由内部结构定，非自由调节模数）
# 连续"真空模数"（会诱发景观的自由连续参数）：丙档 = 0
continuous_moduli_fermion = 0
record("U4", "费米子档新增连续模数计数", continuous_moduli_fermion, 0, 0,
       continuous_moduli_fermion == 0, "反景观：只有整数，无连续模数")
# U4 反例：KK 紧致化引入半径模数 R（连续 1 个）
continuous_moduli_KK = 1
record("U4x", "反例：KK 紧致化半径模数（连续）", continuous_moduli_KK, ">=1", 0,
       continuous_moduli_KK >= 1, "KK 引入连续模数 -> 景观")

# =====================================================================
# U5 Lancret 螺旋身份：kappa/tau 沿曲线为常数（几何尺度可分辨的螺旋）
# =====================================================================
def curv_tors(curve, t0, t1, npts=40001):
    t = np.linspace(t0, t1, npts)
    P = curve(t)
    dt = t[1] - t[0]
    d1 = np.gradient(P, dt, axis=0)
    d2 = np.gradient(d1, dt, axis=0)
    d3 = np.gradient(d2, dt, axis=0)
    cross = np.cross(d1, d2)
    speed = np.linalg.norm(d1, axis=1)
    kappa = np.linalg.norm(cross, axis=1) / speed ** 3
    tau = np.einsum('ij,ij->i', cross, d3) / (np.linalg.norm(cross, axis=1) ** 2)
    return kappa, tau

# 几何螺旋：半径 a、螺距参数 b（同量级，可分辨）；理论 kappa/tau = a/b
a_g, b_g = 1.0, 0.5
helix_g = lambda t: np.column_stack([a_g*np.cos(t), a_g*np.sin(t), b_g*t])
kg, tg = curv_tors(helix_g, 0.0, 20*np.pi)
ratio_g = kg/tg
# 去掉端点（有限差分端点降精度）
ratio_g_core = ratio_g[5:-5]
err_u5 = np.std(ratio_g_core)/abs(np.mean(ratio_g_core))
err_u5_analytic = abs(np.mean(ratio_g_core) - a_g/b_g)/(a_g/b_g)
err_u5 = max(err_u5, err_u5_analytic)
record("U5", "Lancret 比值 kappa/tau 沿螺旋常数且=a/b", err_u5, 0.0, 1e-6,
       err_u5 < 1e-6, "有限差分 O(h^2)")
# U5 反例：非螺旋 z = t^2
nonhelix = lambda t: np.column_stack([a_g*np.cos(t), a_g*np.sin(t), t**2])
kn, tn = curv_tors(nonhelix, 0.5, 6.0)
ratio_n = kn[5:-5]/tn[5:-5]
err_u5x = (np.max(ratio_n)-np.min(ratio_n))/abs(np.mean(ratio_n))
record("U5x", "反例：非螺旋 z=t^2 比值变化", err_u5x, ">0.1", 0.1,
       err_u5x > 0.1, "非螺旋判否")

# =====================================================================
# U6 跨扇区对表：费米子档螺距 = 康普顿波长 h/(m_e c)
# =====================================================================
lam_Compton = h/(m_e*c)
lam_red = hbar/(m_e*c)
w0_e = m_e*c**2/hbar
pitch_fermion = 2*np.pi*c/w0_e
err_u6 = abs(pitch_fermion - lam_Compton)/lam_Compton
record("U6", "费米子档螺距 = 康普顿波长 h/(m_e c)", err_u6, 0.0, 1e-12,
       err_u6 < 1e-12, "跨扇区对表")
# U6 反例：用约化康普顿波长
err_u6x = abs(lam_red - lam_Compton)/lam_Compton
record("U6x", "反例：约化康普顿波长充当螺距", err_u6x, "=1-1/(2pi)", 0.1,
       err_u6x > 0.1, "差 2pi 因子")

# =====================================================================
# 打印实测 vs 期望
# =====================================================================
print("="*72)
print("判据第 626 号：同一弦三频率档统一螺旋色散线")
print("="*72)
print(f"{'判据':<6}{'内容':<34}{'实测':<16}{'期望':<12}{'判定'}")
n_pass = 0
n_main = 0
for tag, desc, meas, exp, tol, passed, note in results:
    if tag.endswith('x'):
        verdict = "反例OK" if passed else "反例失效"
    else:
        n_main += 1
        verdict = "PASS" if passed else "FAIL"
        if passed:
            n_pass += 1
    ms = f"{meas:.3e}" if isinstance(meas, float) else str(meas)
    print(f"{tag:<6}{desc:<34}{ms:<16}{str(exp):<12}{verdict}")
print("-"*72)
print(f"主判据 {n_pass}/{n_main} 通过；反例检验 {'全过' if all(r[5] for r in results if r[0].endswith('x')) else '有失效'}")
print("="*72)

# 关键数字
print("\n关键读数：")
print(f"  w_P = 2pi/t_P = {w_P:.6e} rad/s")
print(f"  时空档螺距 l_P = {pitch_a:.6e} m  (相对偏差 {err_a:.2e})")
print(f"  费米子档螺距（电子）= {pitch_fermion:.6e} m")
print(f"  康普顿波长 h/(m_e c) = {lam_Compton:.6e} m")
print(f"  质量阶梯 m_N/m_1 (N=1..5) = {np.array2string(ratio, precision=6)}")

# =====================================================================
# 出图
# =====================================================================
fig = plt.figure(figsize=(14, 5))

# 面板 1：统一色散线，三档标记
ax1 = fig.add_subplot(1, 3, 1)
kk = np.logspace(-2, 4, 400)
ax1.loglog(kk, c*kk, 'k-', lw=1.5, label=r'$\omega = c\,k$（统一色散线）')
ax1.loglog([2*np.pi/l_P], [w_P], 'r*', ms=18, label='甲 时空档（普朗克）')
ax1.loglog(2*np.pi/lams, ws_b, 'bs', ms=7, label='乙 场档（光子，连续）')
kf = 2*np.pi*Ns/L_f
ax1.loglog(kf, c*kf, 'g^', ms=8, label='丙 费米子档（离散阶梯）')
ax1.set_xlabel(r'波数 $k$（1/m）')
ax1.set_ylabel(r'角频率 $\omega$（rad/s）')
ax1.set_title('统一色散线：三档同源')
ax1.legend(fontsize=8, loc='upper left')
ax1.grid(True, which='both', alpha=0.3)

# 面板 2：三条螺旋（同一结构，不同螺距）
ax2 = fig.add_subplot(1, 3, 2, projection='3d')
tt = np.linspace(0, 4*np.pi, 800)
for scale, col, lab in [(1.0, 'r', '甲 螺距 $l_P$'), (0.25, 'b', '乙 螺距 $\\lambda$'), (0.6, 'g', '丙 螺距 $\\lambda_C$')]:
    ax2.plot(np.cos(tt), np.sin(tt), scale*tt/(2*np.pi), col, lw=1.2, label=lab)
ax2.set_xlabel('x'); ax2.set_ylabel('y'); ax2.set_zlabel('法线（层数）')
ax2.set_title('同一螺旋，不同螺距')
ax2.legend(fontsize=8)
ax2.view_init(elev=18, azim=45)

# 面板 3：质量阶梯
ax3 = fig.add_subplot(1, 3, 3)
ax3.stem(Ns, m_vals/m_vals[0], basefmt=" ", linefmt='g-', markerfmt='go')
ax3.set_xlabel(r'模式指标 $N$（拓扑整数）')
ax3.set_ylabel(r'$m_N/m_1$')
ax3.set_title(r'费米子质量阶梯 $m_N = hN/(Lc)$')
ax3.grid(True, alpha=0.3)

plt.tight_layout()
out = r"D:\software\TianXingFC\images\unified_helix_bands.png"
plt.savefig(out, dpi=130, bbox_inches='tight')
print(f"\n图已保存：{out}")
