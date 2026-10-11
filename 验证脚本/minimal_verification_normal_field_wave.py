# -*- coding: utf-8 -*-
"""
最小验证：法线场动力学健康性（张量桥 b 探路）

背景：分工双场方案（法线场 = 引力专用，2026-09-10 作者指正）的引力扇区动力学 =
碰撞写入（每拍注入 + 邻居抄写平均扩散）加惯性项。本脚本检验三个健康性判据：

判据 A（DC 扇区）：写入 + 抄写平均 + 惯性项的固定点仍为离散泊松稳态（惯性项不破坏
   引力记录扇区），稳态 1/r。
判据 B（波动扇区）：惯性 + 弹性（无扩散）给出无色散波动，相速度 c_s = sqrt(beta)
   = 1 格/拍 = c（取 beta = 1，l_P/t_P 单位），峰幅不衰减。
判据 C（同源判据，核心）：扩散系数 alpha 同时控制 DC 稳态收敛速率与波动阻尼率。
   实测两通道率：gamma_wave = alpha(1-cos k)/2（单层阻尼），gamma_DC = alpha(1-cos k_min)。
   理论比率 (k/k_min)^2/2 与 alpha 无关；期望结论：单一 alpha 无法同时服务两通道，
   波动扇区需独立于写-扩散的弹性刚度（双参数化）。
判据 D（模式谱）：朴素读法（每分量标量弹性 / SO(3) 值场）的传播模数与 GR 期望
   （3+1 维 TT 双偏振）的差距，量化"多余模"并定位到约束（度规识别 / 3+1 铰链）。

边界声明：本脚本验证数学层（机制翻译为方程后的动力学性质），不验证本体层（写入
幅度与 m/m_P 的对应、倾斜方向结构、约束的物理来源）。判据 D 只做约束计数演示，
不证明法线场动力学在连续极限自动给出 TT 约束。
"""
import numpy as np

R_REPORT = []
print("=" * 72)
print("最小验证：法线场动力学健康性（张量桥 b 探路）")
print("=" * 72)

# ----------------------------------------------------------------------
# A. DC 扇区：写入 + 抄写平均 + 惯性 -> 离散泊松稳态
# ----------------------------------------------------------------------
print("\n[A] DC 扇区：惯性项不破坏引力记录稳态")
N = 101
Np = N + 2                    # 填充数组域（两端 Dirichlet 边界，与迭代同一域）
c0 = N // 2
A1 = 2.0 * np.eye(Np) - np.eye(Np, k=1) - np.eye(Np, k=-1)
A1[0, :] = 0.0; A1[0, 0] = 1.0
A1[-1, :] = 0.0; A1[-1, -1] = 1.0
b1 = np.zeros(Np); b1[c0] = 1.0
theta_exact = np.linalg.solve(A1, b1)   # 与迭代同域、同源位，边界同为 0

alpha_dc = 0.9
w_dc = 0.05
s_eff = 2.0 * w_dc / alpha_dc

def run_dc(mu, T, alpha=alpha_dc, w=w_dc):
    """狄利克雷边界 + 每拍注入 + 抄写平均 + 惯性"""
    u = np.zeros(Np)             # u[0], u[-1] = 边界（0）
    u_prev = np.zeros(Np)
    for _ in range(T):
        mid = u[1:-1]
        neigh = 0.5 * (u[2:] + u[:-2])
        unew = mid + alpha * (neigh - mid) + mu * (mid - u_prev[1:-1])
        unew[c0 - 1] += w      # 注入位点：内部数组下标 c0-1 = 填充下标 c0
        u_prev[1:-1] = mid
        u[1:-1] = unew
    return u[1:-1]

for mu in (0.0, 0.5, 0.9):
    T = 30000
    th = run_dc(mu, T)
    err = np.max(np.abs(th - s_eff * theta_exact[1:-1])) / s_eff
    R_REPORT.append(("A", mu, err))
    print(f"  mu = {mu:.1f}：稳态最大相对偏差 = {err:.3e}（期望 < 1e-4）")

# ----------------------------------------------------------------------
# B. 波动扇区：惯性 + 弹性（无扩散），beta = 1 -> 无色散，c_s = 1 = c
# ----------------------------------------------------------------------
print("\n[B] 波动扇区：beta = 1 时色散关系 omega(k) = k（无色散），c_s = 1 = c")
print("  解析期望：omega(k) = 2 arcsin(sqrt(beta) sin(k/2))；beta = 1 -> omega = k")

Nw = 4000
T_wave = 1500
n0 = 400
sigma = 40.0
k0 = 0.5
x = np.arange(Nw)
init = np.exp(-0.5 * ((x - n0) / sigma) ** 2) * np.cos(k0 * (x - n0))

th2 = init.copy()
th2_prev = np.roll(init, -1)    # psi(x,-1)=f(x+1)：右行波包（左行初速会回绕）
for _ in range(T_wave):
    th_new = 2.0 * th2 - th2_prev + (np.roll(th2, 1) + np.roll(th2, -1) - 2.0 * th2)
    th2_prev, th2 = th2, th_new
peak_now = np.argmax(th2)
c_s_meas = (peak_now - n0) / T_wave
amp = np.max(th2)
R_REPORT.append(("B", c_s_meas, amp))
print(f"  右行波包峰位移 {peak_now - n0} 格 / {T_wave} 拍 -> c_s 实测 = {c_s_meas:.8f}（期望 1.0）")
print(f"  峰幅 = {amp:.8f}（期望 1.0；beta=1 无色散无阻尼，波包保形传播）")

# 平面波频散实测：单站时间序列 FFT
k_test = 0.5
Tp = 4096
site = 500
u_site = np.zeros(Tp)
th3 = np.sin(k_test * x)
th3_prev = np.roll(th3, -1)     # 右行平面波：单频信号
for t in range(Tp):
    th_new = 2.0 * th3 - th3_prev + (np.roll(th3, 1) + np.roll(th3, -1) - 2.0 * th3)
    th3_prev, th3 = th3, th_new
    u_site[t] = th3[site]
spec = np.abs(np.fft.rfft(u_site))
freqs = np.fft.rfftfreq(Tp)
peak_f = freqs[np.argmax(spec[1:]) + 1]
omega_meas = 2.0 * np.pi * peak_f
R_REPORT.append(("B2", k_test, omega_meas))
print(f"  平面波 k = {k_test}：omega 实测 = {omega_meas:.6f}（期望 = k = {k_test}；"
      f"FFT 分辨 2pi/{Tp} = {2 * np.pi / Tp:.5f}）")

# ----------------------------------------------------------------------
# C. 同源判据：扩散系数 alpha 同时管 DC 收敛与波动阻尼
# ----------------------------------------------------------------------
print("\n[C] 同源判据：单一扩散系数能否同时服务 DC 松弛与波动弱阻尼？")
ks = np.pi / 8.0
k_min = np.pi / (N + 1)         # 填充域最低模（Dirichlet，Np = N+2 点）

def wave_damp(alpha, T=3000, Nc=2000):
    th = np.sin(ks * np.arange(Nc))
    th_prev = np.roll(th, -1)   # 右行单频波：能量无驻波振荡，纯指数衰减
    E = np.zeros(T)
    for t in range(T):
        th_new = 2.0 * th - th_prev + (np.roll(th, 1) + np.roll(th, -1) - 2.0 * th)
        th_new = th_new + alpha * ((np.roll(th_new, 1) + np.roll(th_new, -1)) * 0.5 - th_new)
        th_prev, th = th, th_new
        E[t] = np.sum(th * th)
    E0 = E[0]
    valid = np.where(E > 1e-8 * E0)[0]      # 只拟合指数区，避开浮点地板
    if len(valid) < 100:
        valid = np.arange(min(100, T))
    y = np.log(E[valid])
    return -0.5 * np.polyfit(valid, y, 1)[0]   # 幅度阻尼率 gamma

def dc_relax(alpha):
    """最低模松弛率 gamma_DC：初始 sin(k_min x)，纯扩散无源"""
    xin = np.arange(1, Np - 1)
    u = np.zeros(Np); u[1:-1] = np.sin(k_min * xin)
    E0 = np.sum(u[1:-1] ** 2)
    gd_th = alpha * (1.0 - np.cos(k_min))
    T = max(2000, int(6.0 / gd_th) + 1)
    E = np.zeros(T)
    for t in range(T):
        mid = u[1:-1]
        unew = mid + alpha * (0.5 * (u[2:] + u[:-2]) - mid)
        u[1:-1] = unew
        E[t] = np.sum(u[1:-1] ** 2)
    valid = np.where(E > 1e-8 * E0)[0]
    if len(valid) < 100:
        valid = np.arange(min(100, T))
    y = np.log(E[valid])
    return -0.5 * np.polyfit(valid, y, 1)[0]   # 幅度松弛率 gamma_DC

omega_k = ks
theory_ratio = (ks / k_min) ** 2 / 2
print(f"  测试波数 k = {ks:.4f}，omega = {omega_k:.4f}")
print(f"  理论：gamma_wave = alpha(1-cos k)/2（单层阻尼），gamma_DC = alpha(1-cos k_min)")
print(f"  比率与 alpha 无关：gamma_wave/gamma_DC = (k/k_min)^2/2 = {theory_ratio:.0f}")
print("  alpha     gamma_wave/omega   gamma_DC 实测   比率 gw/gd   判定")
ok_any = False
ratios = []
for alpha in (0.01, 0.02, 0.05, 0.1, 0.5):
    gw = wave_damp(alpha)
    gd = dc_relax(alpha)
    ratio = gw / gd
    ratios.append(ratio)
    gw_over_w = gw / omega_k
    flag = "通过" if (gw_over_w < 0.01 and ratio < 1.0) else ""
    if flag:
        ok_any = True
    R_REPORT.append(("C", alpha, gw_over_w, gd, ratio))
    print(f"  {alpha:6.3f}   {gw_over_w:14.4f}   {gd:11.3e}   {ratio:9.1f}   {flag}")

ratio_mean = float(np.mean(ratios))
verdict_c = ("存在单参数区同时满足两判据" if ok_any
             else f"不存在：波动阻尼与 DC 松弛共用同一扩散系数，实测比率 {ratio_mean:.0f} "
                  f"与 alpha 无关（理论 {(ks / k_min) ** 2 / 2:.0f}）。DC 定型需 ~1/gamma_DC 拍，"
                  f"期间波数 k 的波动衰减至 e^-{ratio_mean:.0f} 量级。波动扇区需独立于写-扩散的"
                  f"弹性刚度（双参数化），此为本判据的结论")
print(f"  [C] 判定：{verdict_c}")

# ----------------------------------------------------------------------
# D. 模式谱：约束计数，朴素读法 vs GR 期望
# ----------------------------------------------------------------------
print("\n[D] 模式谱：朴素读法的传播模数 vs GR 期望")
print("  3D 对称张量 h_ij 有 6 分量；TT 约束（无迹 + 横波）选出 2 个传播模 = GR 引力子")
print("  朴素读法（SO(3) 值场逐分量标量弹性）：3 个分量 = 3 支传播模 > 2")

# 傅里叶空间约束计数：对每个 k != 0，TT 条件 h_ii = 0（1 条）、k^j h_ij = 0（3 条）
# 6 - 1 - 3 = 2（对 k != 0；约束在傅里叶空间线性独立）
# 数值核对：构造约束矩阵，计算零空间维数
kv = np.array([0.3, 0.5, 0.7])
def tt_count(k):
    # h: 对称 3x3 -> 6 维向量 (h11,h22,h33,h12,h13,h23)
    # 无迹: h11+h22+h33 = 0
    # 横波: k1 h11 + k2 h12 + k3 h13 = 0
    #       k1 h12 + k2 h22 + k3 h23 = 0
    #       k1 h13 + k2 h23 + k3 h33 = 0
    C = np.array([
        [1, 1, 1, 0, 0, 0],
        [k[0], 0, 0, k[1], k[2], 0],
        [0, k[1], 0, k[0], 0, k[2]],
        [0, 0, k[2], 0, k[0], k[1]],
    ])
    # 零空间维数 = 6 - rank(C)
    r = np.linalg.matrix_rank(C)
    return 6 - r
for k in (np.array([0.3, 0.5, 0.7]), np.array([1.0, 0.2, 0.1])):
    cnt = tt_count(k)
    R_REPORT.append(("D", tuple(np.round(k, 2)), cnt))
    print(f"  k = {np.round(k,2)}：TT 传播模数 = {cnt}（期望 2，GR 引力子）")

print(f"  [D] 判定：朴素读法（3 支）含多余模；TT 约束（2 支）须由度规识别 / 3+1 铰链"
      f"在连续极限给出，此为张量桥 c 的交付物")

# ----------------------------------------------------------------------
# 总判定
# ----------------------------------------------------------------------
print("\n" + "=" * 72)
print("总判定")
print("=" * 72)
print("  A（DC 稳态含惯性）：" + ("通过" if all(e < 1e-4 for _, mu, e in [r for r in R_REPORT if r[0] == 'A']) else "未通过"))
print(f"  B（c_s = c，无色散无阻尼）：c_s = {c_s_meas:.6f}，峰幅 = {amp:.6f}")
print(f"  C（扩散-波动同源）：{verdict_c}")
print("  D（模式谱）：约束计数 2 = GR 引力子；朴素读法含多余模，约束为张量桥 c 交付物")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    mus = [r[1] for r in R_REPORT if r[0] == 'A']
    errs = [r[2] for r in R_REPORT if r[0] == 'A']
    ax[0].semilogy(mus, errs, "o-")
    ax[0].set_xlabel("惯性系数 mu"); ax[0].set_ylabel("DC 稳态相对偏差")
    ax[0].set_title("A: 惯性项不破坏引力记录稳态")
    alphas = [r[1] for r in R_REPORT if r[0] == 'C']
    ratios = [r[4] for r in R_REPORT if r[0] == 'C']
    ax[1].plot(alphas, ratios, "o-", label="gamma_wave/gamma_DC 实测")
    ax[1].axhline((ks / k_min) ** 2 / 2, ls="--", color="r", label="理论 (k/k_min)^2/2")
    ax[1].set_xlabel("扩散系数 alpha"); ax[1].set_ylabel("波动阻尼率 / DC 松弛率")
    ax[1].legend()
    ax[1].set_title("C: 比率与 alpha 无关，单参数区不存在")
    fig.tight_layout()
    fig.savefig(r"D:\software\TianXingFC\images\minimal_verification_normal_field_wave.png", dpi=140)
    print("\n图已存 images/minimal_verification_normal_field_wave.png")
except Exception as e:
    print("\n[绘图跳过]", e)
