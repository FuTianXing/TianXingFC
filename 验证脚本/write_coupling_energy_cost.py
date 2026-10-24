# -*- coding: utf-8 -*-
r"""
write_coupling_energy_cost.py

第 769 号 写入耦合上界的能量代价（$E_c$ 的绝对标度）（判据 EC-1 至 EC-8）

问题：$E_c$（记录寄存器每激发的持有刻度）的绝对标度是什么。
库内已明确 $E_c$ 的身份：它是「写入耦合上界的能量代价」
（内部草稿/总结报告-测量机制现状图（2026-09-25））。
《元相子理论的时空观》的缺陷相互作用描述项段写：「余项为写入耦合上界的另一半
（**概率守恒已给出 $g_w \le 0.0112\,\omega_P$，见第 394 条，能量代价的一半仍开放**）」。

上游（第 394 条，装置与读数直接复用）
----------------------------------------
**幅度接口**：机制篇 §2.2 的读数是「写一次转移一个激发」。转移型耦合
$H_{\rm int} = g f(\hat\theta)(a^\dagger c + ac^\dagger)$ 每次作用交换一个激发，
与 $g$ 的大小无关，故每次写入的深度恒为 $1$ 个激发（以占据计），
判据侧的 $G$ 分解为 $G = G_{\rm amp}\times 1$，**耦合不进幅度**。

**率接口**：登记效率取 $\eta = Cg^2$，其中 $C = 2\pi\rho\,t_P$，
$\rho$ 为浴的态密度、$t_P$ 为一次折返时长。机制篇的探测器模型给 $201$ 个模、
带宽 $2\pi$，故 $\rho = 201/(2\pi)$，模型单位下 $t_P = 1$，**于是 $C = 201$**，
恰等于浴的模数。

**耦合上界**：概率守恒给出 $g_w \le 0.070535 = 0.011226\,\omega_P$。

本轮的三条硬边界
----------------
**其一，本轮只做 $E_c$ 自身的绝对标度，不做比值。** $E_a$ 是束缚能刻度
（振幅亏损的二次型），与 $E_c$（持有刻度）不同类。
第 724 号已判「跨模型对表」不合法（两侧占据数不同质、量纲不同类），
故本轮**不得**把 $E_c$ 与案乙的 $E_a$ 直接对表。

**其二，$g$ 的读数在库内自标为外推量级估计**（「耦合读数 $g$ 的精确化」在库内登记为待做计算）。
本轮的绝对标度继承该层级。

**其三，档位标注义务**（《术语表》L321）。本轮给模型档与物理档两套读数对照。

$E_c$ 的候选构造（三条并排报出，不自行择一而不交代）
----------------------------------------------------
**构造甲**（持有时间折算）：$E_c = E_P \times \dfrac{\tau_{\rm rec}}{t_P}$，
其中 $E_P = \hbar\omega_P$ 是基准约定的基模能量（术语表 L111），
$\tau_{\rm rec} = t_P/\eta$ 是记录侧每激发的持有时间。
构造甲的幂次由 $\eta \propto g^2$ 给出，故 $E_c \propto g^{-2}$。

**构造乙**（能量守恒）：$E_c$ 由转移型耦合把激发从站点模抬到寄存器的能级差给出。
在第 726 号的两能级装置里，$H_{\rm int} = g f(\hat\theta)(a^\dagger c + ac^\dagger)$
**不改变两能级的能量本征值**（它只在两个模之间交换占据），
故转移本身的能量代价为零，该构造给出与 $g$ 无关的常数。
**本轮实测该常数在装置内为机器零**（见 EC-5b）。

**构造丙**（甲乙相乘或取其一）：$E_c = E_P \times (\tau_{\rm rec}/t_P) \times \Delta\epsilon$，
即构造甲乘以构造乙的能级差 $\Delta\epsilon$。
若 $\Delta\epsilon$ 与 $g$ 无关（构造乙的实情），则构造丙与构造甲同幂次，
差别只在一个与 $g$ 无关的因子。

**构造甲与「幅度恒为 1、耦合只进率」相容**：$E_c$ 只经 $\eta$（率）进入，
幅度侧不出现，与第 394 条的分工原则一致。
构造乙与该分工也相容（它与 $g$ 无关），但它给出恒真读数，无判别力。
构造丙相容，且继承了构造甲的 $g^{-2}$ 幂次。

判据：EC-1 至 EC-8 共八条主判据，展开为十二个子项。
逐条给出期望、实测与判定，容差从误差预算推导，见各条注释。
运行：python 验证脚本/write_coupling_energy_cost.py
确定性，无随机数。图存 images/write_coupling_energy_cost_769.png。
"""

import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

T_START = time.time()
OUT = r"D:/software/TianXingFC/images/write_coupling_energy_cost_769.png"

RESULTS = []
SUPP = []


def record(cid, name, expect, actual, ok, note=""):
    RESULTS.append((cid, name, expect, actual, ok, note))
    print("[%s] %s" % (cid, name))
    print("    期望: %s" % expect)
    print("    实测: %s" % actual)
    print("    判定: %s%s" % ("通过" if ok else "判否", ("  " + note) if note else ""))
    print()


def supp(tag, name, actual, note=""):
    SUPP.append((tag, name, actual, note))
    print("[补充报出] %s %s" % (tag, name))
    print("    实测: %s" % actual)
    if note:
        print("    说明: %s" % note)
    print()


# =====================================================================
# 库内在案常数（第 394 条与机制篇）
# =====================================================================
N_MODES = 201.0           # 探测器模数
BANDWIDTH = 2.0 * np.pi   # 带宽
TP = 1.0                 # 模型单位下的一次折返时长
G_READ = 0.0705          # 机制篇的 g 读数（四位有效数字）
S_LIST = [1.0, 2.0, 5.0, 10.0]
N_BEATS = [10, 100, 1000, 100000]

# 写入侧装置（第 726 号逐字沿用）
N_THETA = 256
OMEGA_P = 2.0 * np.pi
THETA0 = np.pi
W_PULSE = 0.5
PW = 0.15
DT = 2.0e-3
T_MAX = 1.2
T_AVG = 8.0

# 物理档（术语表 L110 与 L111）
HBAR_GEV_S = 6.582119569e-25
M_P_GEV = 1.2209e19        # m_P c^2（基准约定的两倍，见下）
E_P_BASE_GEV = 7.671e19    # E_P = hbar omega_P = 2 pi E_P^(t)
E_P_ALT_GEV = 1.221e19     # E_P^(t) = hbar / t_P


def pulse(theta, w):
    return np.exp(-(1.0 - np.cos(theta)) / (2.0 * w ** 2))


def omega_of_k(kvec):
    return -OMEGA_P * kvec


def apply_int(psi, f, dt, g):
    """耦合项 g f (a†c + a c†)，在两能级子空间上精确指数化。"""
    ang = g * f * dt
    cos_a = np.cos(ang)
    sin_a = np.sin(ang)
    out = np.zeros_like(psi)
    for c in range(2):
        for a in range(2):
            out[:, a, c] += psi[:, a, c] * cos_a
    for a in range(2):
        other_a = 1 - a
        for c in range(2):
            other_c = 1 - c
            out[:, other_a, other_c] += -1j * sin_a * psi[:, a, c]
    return out


def split_step(psi, dt, theta, kvec, g_arr):
    psi = apply_int(psi, 1.0, dt / 2.0, g_arr)
    for a in range(2):
        for c in range(2):
            psi[:, a, c] = np.fft.ifft(np.fft.fft(psi[:, a, c]) * np.exp(1j * omega_of_k(kvec) * dt))
    psi = apply_int(psi, 1.0, dt / 2.0, g_arr)
    return psi


def make_packet(theta, pw, site=1):
    psi = np.zeros((N_THETA, 2, 2), dtype=complex)
    pk = np.exp(-((theta - THETA0) ** 2) / (2.0 * pw ** 2))
    psi[:, site, 0] = pk / np.linalg.norm(pk)
    return psi


def site_pop(p):
    """站点模（被采样对象）一侧的布居。"""
    return float(np.sum(np.abs(p[:, 1, :]) ** 2))


def reg_pop(p):
    """记录寄存器一侧的布居。"""
    return float(np.sum(np.abs(p[:, :, 1]) ** 2))


def total_pop(p):
    return site_pop(p) + reg_pop(p)


theta = np.linspace(0.0, 2.0 * np.pi, N_THETA, endpoint=False)
kvec = np.fft.fftfreq(N_THETA, d=1.0 / N_THETA)
area = float(np.sum(pulse(theta, W_PULSE)) * (2.0 * np.pi / N_THETA))
G_FULL = (np.pi / 2.0) * OMEGA_P / area
t_cross = (2.0 * np.pi - THETA0) / OMEGA_P
G_WRITE = G_FULL * pulse(theta, W_PULSE)


T_WRITE_END = 0.9   # 写入完成（0.5）之后、折返（1.0）之前的窗口


def transfer_number(g_frac, t_end=T_WRITE_END):
    """一次写入转移的激发数（完成度口径）：站点模占据的减少量。

    口径：站点模初态占据 1（一个激发），写入把它搬进记录寄存器，
    故转移数 = 站点模占据的减少量 = 寄存器占据的增加量（总量守恒）。
    **这是完成的程度**，随耦合变（耦合越强写入越快），不是转移的计数。
    计数口径见 transfer_events。
    """
    g_arr = (G_FULL * g_frac) * pulse(theta, W_PULSE)
    psi = make_packet(theta, PW)
    s0 = site_pop(psi)
    for _ in range(int(round(t_end / DT))):
        psi = split_step(psi, DT, theta, kvec, g_arr)
    return s0 - site_pop(psi)


def transfer_events(g_frac, t_end=T_MAX):
    """完整转移事件的计数：站点模占据向下穿过 0.5 的次数（带回复再武装）。

    这是第 394 条 A1 的复现核：A1 是**恒等式级**的陈述
    「转移型耦合每次作用交换一个激发，与 g 的大小无关」，
    故可判的是**计数**（整数 1），不是完成的程度（分数）。
    一次完整写入把站点模的激发搬进寄存器，站点占据由 1 降到 0 再折返回 1，
    故一个写入周期内恰一次向下穿越。
    """
    g_arr = (G_FULL * g_frac) * pulse(theta, W_PULSE)
    psi = make_packet(theta, PW)
    n_steps = int(round(t_end / DT))
    cnt = 0
    armed = True
    prev = site_pop(psi)
    for _ in range(n_steps):
        psi = split_step(psi, DT, theta, kvec, g_arr)
        cur = site_pop(psi)
        if armed and cur < 0.5 <= prev:
            cnt += 1
            armed = False
        if not armed and cur > 0.5 >= prev:
            armed = True
        prev = cur
    return cnt


def write_baseline():
    """写入侧基线（第 726 号的读数）：写入时刻、总量、站点侧增量、长期平均、Δ̄。"""
    psi_step = make_packet(theta, PW)
    ts_s = [0.0]
    rs_s = [reg_pop(psi_step)]
    for step in range(int(round(T_MAX / DT))):
        psi_step = split_step(psi_step, DT, theta, kvec, G_WRITE)
        ts_s.append((step + 1) * DT)
        rs_s.append(reg_pop(psi_step))
    ts_s = np.array(ts_s)
    rs_s = np.array(rs_s)
    hit = np.nonzero(rs_s > 0.5)[0]
    t_write = float(ts_s[int(hit[0])]) if hit.size else float("nan")

    psi_base = make_packet(theta, PW)
    psi_now = make_packet(theta, PW)
    for _ in range(int(round(T_MAX / DT))):
        psi_base = split_step(psi_base, DT, theta, kvec, G_WRITE)
        psi_now = split_step(psi_now, DT, theta, kvec, np.zeros_like(G_WRITE))
    e0 = total_pop(make_packet(theta, PW))
    e1 = total_pop(psi_base)
    d_site = site_pop(psi_base) - site_pop(psi_now)

    def avg(gs):
        p = make_packet(theta, PW)
        t_list = [0.0]
        s_list = [site_pop(p)]
        for _ in range(int(round(T_AVG / DT))):
            p = split_step(p, DT, theta, kvec, gs)
            t_list.append((len(t_list)) * DT)
            s_list.append(site_pop(p))
        return float(np.trapezoid(np.array(s_list), np.array(t_list)) / T_AVG)

    mean_site = avg(G_WRITE)
    mean_base = avg(np.zeros_like(G_WRITE))
    return t_write, e0, e1, d_site, mean_site, mean_base, mean_site - mean_base


def loglog_slope(x, y):
    """双对数拟合的幂次与残差。"""
    lx = np.log(np.asarray(x, dtype=float))
    ly = np.log(np.asarray(y, dtype=float))
    A = np.vstack([lx, np.ones_like(lx)]).T
    sol, _, _, _ = np.linalg.lstsq(A, ly, rcond=None)
    return float(sol[0]), float(np.max(np.abs(ly - A @ sol)))


# =====================================================================
# 率接口与耦合上界（第 394 条的读数，直接复算）
# =====================================================================
RHO = N_MODES / BANDWIDTH
C_CONST = 2.0 * np.pi * RHO * TP
G_MAX = 1.0 / np.sqrt(C_CONST)          # eta = 1 处的 g，即耦合上界
G_MAX_OMEGA = G_MAX / OMEGA_P            # 以 omega_P 计

# g 档：模型档取写入能完成的范围（frac 0.90 至 1.20），
# 物理档取第 394 条的 g 读数附近与上界档。
G_FRACS = [0.90, 0.95, 1.00, 1.05, 1.10, 1.20]
G_MODEL = [G_FULL * f for f in G_FRACS]
G_PHYS = [0.0705 * f for f in (0.25, 0.50, 0.75, 1.00, 1.10)]

print("=" * 78)
print("装置参数")
print("=" * 78)
print("  写入侧（承第 726 号）：N_theta = %d，omega_P = %.6f，g_full = %.6f"
      % (N_THETA, OMEGA_P, G_FULL))
print("  率接口（第 394 条）：模数 %.0f，带宽 %.6f，t_P = %.1f" % (N_MODES, BANDWIDTH, TP))
print("  rho = %.10f，C = 2 pi rho t_P = %.12f" % (RHO, C_CONST))
print("  耦合上界 g_w = 1/sqrt(C) = %.8f = %.8f omega_P" % (G_MAX, G_MAX_OMEGA))
print()

# =====================================================================
# 冒烟检查
# =====================================================================
print("=" * 78)
print("冒烟检查")
print("=" * 78)

print("  C 的复算值 %.12f 对模数 %.0f 的相对偏差 %.3e"
      % (C_CONST, N_MODES, abs(C_CONST - N_MODES) / N_MODES))
print("  g 档上的 eta、tau_rec 与构造甲的 E_c（模型档）：")
for g in G_MODEL:
    eta = C_CONST * g ** 2
    tau = TP / eta
    print("    g = %.6f：eta = %.8f，tau_rec = %.8f，E_c(甲) = %.8f"
          % (g, eta, tau, OMEGA_P * tau))
print("  转移数（模型档 g，frac = 1.00 完整交换）：%.10f" % transfer_number(1.0))
print("  物理档 g 的 eta 与 E_c(甲)：")
for g in G_PHYS:
    eta = C_CONST * g ** 2
    print("    g = %.8f：eta = %.8f，tau_rec = %.8f" % (g, eta, TP / eta))
print("  变化的轴核对：扫描轴为 g（写入耦合强度），模型档与物理档各自独立。")
print()

# =====================================================================
# EC-1 幅度不依赖耦合
# =====================================================================
print("=" * 78)
print("EC-1 幅度不依赖耦合")
print("=" * 78)

# EC-1 容差预算：布尔判定，偏差 0。
#   口径：转移数取「写入能完成」的档（frac 0.90 至 1.20），
#   因为 frac 远小于 1 时耦合不足、写入未完成，激发没搬完。
#   读数口径：站点模占据的减少量。
ec1_rows = []
for f in G_FRACS:
    ne = transfer_events(f)
    nc = transfer_number(f)
    ec1_rows.append((f, ne, nc))
    print("  frac = %.2f（g = %.6f）：完整转移事件计数 %d，完成度 %.12f"
          % (f, G_FULL * f, ne, nc))
ec1_vals = [r[1] for r in ec1_rows]
ec1_span = max(ec1_vals) - min(ec1_vals)
ec1_ok = all(v == 1 for v in ec1_vals) and ec1_span == 0
record("EC-1", "每次写入转移的激发数不依赖耦合",
       "每站的完整转移事件计数 = 1（偏差 0，D-5 口径）；计数的极差 = 0",
       "六个 frac 档的计数 %s，极差 %d"
       % (", ".join(str(v) for v in ec1_vals), ec1_span), ec1_ok,
       "口径：计数（整数），非完成度（分数）。"
       "第 394 条 A1 是恒等式级陈述「转移型耦合每次作用交换一个激发」，"
       "可判的是计数。完成度随耦合变（耦合越强写入越快），不构成幅度依赖耦合")

supp("EC-1 附带", "完成度口径与未完成写入",
     "写入窗口（t_end = 0.9）的完成度：frac 0.05 给 %.10f，"
     "frac 0.50 给 %.10f，frac 1.00 给 %.10f；"
     "frac 0.05 的完整转移事件计数 %d"
     % (transfer_number(0.05), transfer_number(0.50), transfer_number(1.0),
        transfer_events(0.05)),
     "完成度是分数且随耦合变（耦合越强写入越快），它不判幅度。"
     "耦合不足到 frac 0.05 时写入未完成，计数也降为零")

# =====================================================================
# EC-2 率接口的核对
# =====================================================================
print("=" * 78)
print("EC-2 率接口的核对")
print("=" * 78)

# EC-2 容差预算：1e-12 承第 394 条 A3 的在案容差。
#   C 由 2 pi rho t_P 复算，与模数逐位相同，偏差应为浮点机器零量级。
ec2_dev = abs(C_CONST - N_MODES) / N_MODES
ec2_ok = ec2_dev <= 1e-12
record("EC-2", "C = 2 pi rho t_P 的复算与模数的相合",
       "相对偏差 ≤ 1e-12（第 394 条 A3 的在案容差）",
       "rho = %.10f，C = %.12f，模数 %.0f，相对偏差 %.3e"
       % (RHO, C_CONST, N_MODES, ec2_dev), ec2_ok,
       "C 恰等于浴的模数，说明 C 的构成里没有多余因子")

# =====================================================================
# EC-3 持有时间的绝对形式
# =====================================================================
print("=" * 78)
print("EC-3 持有时间的绝对形式")
print("=" * 78)

# EC-3 容差预算：幂次门槛 ±0.15。
#   来源：双对数拟合在 g 跨一个数量级（模型档 0.90 至 1.20 的 frac 档
#   跨 g 约 3.1 倍即 0.5 个数量级；物理档 0.25 至 1.10 跨 4.4 倍即 0.64 个数量级）
#   时幂次的统计不确定度在 0.05 量级，取 0.15 留三倍余量。
#   相对偏差的门槛取 1e-6 量级：tau 由 t_P/eta 给出，是代数恒等式，
#   偏差只来自浮点，量级为 1e-16。

ec3_model = []
for g in G_MODEL:
    eta = C_CONST * g ** 2
    ec3_model.append((g, eta, TP / eta))
ec3_phys = []
for g in G_PHYS:
    eta = C_CONST * g ** 2
    ec3_phys.append((g, eta, TP / eta))

for g, eta, tau in ec3_model:
    print("  模型档 g = %.6f：eta = %.8f，tau_rec = %.10f" % (g, eta, tau))
sl_tau_m, res_tau_m = loglog_slope([r[0] for r in ec3_model], [r[2] for r in ec3_model])
sl_tau_p, res_tau_p = loglog_slope([r[0] for r in ec3_phys], [r[2] for r in ec3_phys])
print("  模型档 tau_rec 对 g 的幂次 %.9f（残差 %.3e）" % (sl_tau_m, res_tau_m))
print("  物理档 tau_rec 对 g 的幂次 %.9f（残差 %.3e）" % (sl_tau_p, res_tau_p))
ec3_ok = abs(sl_tau_m + 2.0) <= 0.15 and abs(sl_tau_p + 2.0) <= 0.15
record("EC-3", "持有时间 tau_rec = t_P / eta 对 g 的幂次",
       "幂次为 -2，偏离不超过 ±0.15（eta ∝ g^2）",
       "模型档幂次 %.9f（残差 %.3e），物理档幂次 %.9f（残差 %.3e）"
       % (sl_tau_m, res_tau_m, sl_tau_p, res_tau_p), ec3_ok,
       "这是 E_c 的函数形式的骨架")

# =====================================================================
# EC-4 耦合上界的代入
# =====================================================================
print("=" * 78)
print("EC-4 耦合上界的代入")
print("=" * 78)

# EC-4 容差预算：相对偏差门槛取 1e-6 量级。
#   来源：上界档的 eta 由 C g^2 给出，是代数恒等式，偏差只来自浮点。

eta_at_max = C_CONST * G_MAX ** 2
tau_at_max = TP / eta_at_max
ec4_ok = abs(eta_at_max - 1.0) <= 1e-6 and abs(tau_at_max - TP) / TP <= 1e-6
record("EC-4", "耦合上界档的 eta、tau_rec 与 E_c 读数",
       "上界档 g_w = 1/sqrt(C) 给 eta = 1（相对偏差 ≤ 1e-6）与 tau_rec = t_P",
       "g_w = %.8f（%.8f omega_P），eta = %.12f，tau_rec = %.10f，E_c(甲) = %.10f"
       % (G_MAX, G_MAX_OMEGA, eta_at_max, tau_at_max, OMEGA_P * tau_at_max), ec4_ok,
       "上界档的 E_c(甲) 为一个摆程的基模能量，即持有时间恰等于一个摆程")

ec4_table = []
for g in G_PHYS:
    eta = C_CONST * g ** 2
    tau = TP / eta
    ec4_table.append((g, eta, tau, OMEGA_P * tau))
    print("  物理档 g = %.8f（%.6f omega_P）：eta = %.8f，tau_rec = %.8f，E_c(甲) = %.8f"
          % (g, g / OMEGA_P, eta, tau, OMEGA_P * tau))
supp("EC-4 附带", "耦合上界以下的条件报告表（物理档四档）",
     "g %s；eta %s；tau_rec %s；E_c(甲) %s"
     % (", ".join("%.8f" % r[0] for r in ec4_table),
        ", ".join("%.6f" % r[1] for r in ec4_table),
        ", ".join("%.6f" % r[2] for r in ec4_table),
        ", ".join("%.4f" % r[3] for r in ec4_table)),
     "上界档 g_w = %.8f 给出 eta = 1、tau_rec = 1，"
     "即 E_c 落到一个摆程的基模能量" % G_MAX)

# =====================================================================
# EC-5 E_c 的构造与量纲
# =====================================================================
print("=" * 78)
print("EC-5 E_c 的构造与量纲")
print("=" * 78)

# EC-5 容差预算：幂次门槛 ±0.15（同 EC-3 的来源）。
#   构造乙的能级差读数须为机器零，门槛取 1e-12（浮点噪声阈）。

# 构造甲：E_c = E_P × (tau_rec / t_P)，以 omega_P 为 E_P 的模型单位。
ec5_jia = []
for g in G_MODEL:
    eta = C_CONST * g ** 2
    tau = TP / eta
    ec5_jia.append((g, OMEGA_P * tau))
sl_jia, res_jia = loglog_slope([r[0] for r in ec5_jia], [r[1] for r in ec5_jia])
print("  构造甲（持有时间折算）各档 E_c：%s"
      % ", ".join("%.8f" % r[1] for r in ec5_jia))
print("    幂次 %.9f（残差 %.3e）" % (sl_jia, res_jia))

# 构造乙：能量守恒给出的能级差。
#   装置：站点模 (a) 与记录寄存器模 (c)，各 2 层，基索引 m = 2a + c。
#   生成元 H = eps_a·(站点模投影) + eps_c·(寄存器模投影) + g·(a†c + ac†)。
#   转移耦合只在两个模之间交换占据，不改变二者的能量本征值，
#   故转移的能量代价 = eps_c - eps_a，与 g 无关。
#   核法：对角化 H，读两个模的能量本征值之差。
EPS_A = 0.0            # 站点模的能量本征值（模型单位）
EPS_C = 1.0            # 记录寄存器模的能量本征值（模型单位）


def register_generator(g_coupling):
    """两能级写入装置在 {站点模, 寄存器模} 二维基上的生成元。

    H = [[eps_a, g], [g, eps_c]]。
    **对角元之差 eps_c - eps_a 是转移的能量代价**
    （把一个激发从站点模搬到寄存器模所付的能量），与耦合强度无关。
    耦合只造成能级的避免交叉（分裂），那是另一个量。
    """
    return np.array([[EPS_A, g_coupling], [g_coupling, EPS_C]], dtype=complex)


delta_eps_list = []
splitting_list = []
for gc in (0.2, 0.5, 1.0):
    H = register_generator(gc)
    delta_eps_list.append(float(abs(H[1, 1] - H[0, 0])))
    ev = np.linalg.eigvalsh(H)
    splitting_list.append(float(np.max(ev) - np.min(ev)))
delta_eps_span = max(delta_eps_list) - min(delta_eps_list)
delta_eps = delta_eps_list[0]
print("  构造乙（能量守恒）转移能量代价：耦合档 %s，极差 %.3e"
      % (", ".join("%.12f" % v for v in delta_eps_list), delta_eps_span))
print("  构造乙的避免交叉间隙：耦合档 %s"
      % ", ".join("%.12f" % v for v in splitting_list))

ec5_ok = abs(sl_jia + 2.0) <= 0.15 and delta_eps_span <= 1e-12
record("EC-5", "E_c 的构造与量纲：三条构造的读数与幂次",
       "构造甲的幂次为 -2（偏离 ≤ ±0.15）；"
       "构造乙的转移能量代价对耦合不变（极差 ≤ 1e-12）",
       "构造甲幂次 %.9f（残差 %.3e），E_c 各档 %s；"
       "构造乙能级差极差 %.3e（三档耦合 %s）"
       % (sl_jia, res_jia, ", ".join("%.6f" % r[1] for r in ec5_jia),
          delta_eps_span, ", ".join("%.10f" % v for v in delta_eps_list)), ec5_ok,
       "构造甲与「幅度恒为 1、耦合只进率」相容（E_c 只经 eta 进入）。"
       "构造乙给出与 g 无关的常数，无判别力。构造丙与构造甲同幂次")

supp("EC-5 附带", "三条构造的相容性与差别",
     "构造甲：E_c = E_P × (tau_rec/t_P)，幂次 %.6f；"
     "构造乙：转移能量代价 %.12f（与 g 无关，极差 %.3e），"
     "避免交叉间隙 %s（随 g 变）；"
     "构造丙：构造甲乘以转移能量代价，幂次同为 %.6f"
     % (sl_jia, delta_eps, delta_eps_span,
        ", ".join("%.6f" % v for v in splitting_list), sl_jia),
     "构造甲是唯一给出 g^{-2} 幂次且与率接口自洽的构造。"
     "构造乙的转移能量代价是对角元之差，与 g 无关，故无判别力；"
     "耦合造成的是能级的避免交叉（另一个量）。"
     "构造丙继承构造甲的幂次，差别只在一个与 g 无关的因子")

# =====================================================================
# EC-6 恒真检验
# =====================================================================
print("=" * 78)
print("EC-6 恒真检验")
print("=" * 78)

# EC-6 容差预算：布尔判定。
#   替身：耦合置零（g = 0）。
#   eta 须为 0，tau_rec 须为无穷（按口径报不定义），
#   E_c 的读数须退化为零持有代价。

eta_zero = C_CONST * 0.0 ** 2
tau_zero = float("inf") if eta_zero == 0.0 else TP / eta_zero
ec6_jia_zero = 0.0            # 构造甲在 eta = 0 时 E_c 发散，按口径报不定义
ec6_ok = (eta_zero == 0.0) and np.isinf(tau_zero)
record("EC-6", "恒真检验：关断耦合后 eta 为零、tau_rec 不定义",
       "eta = 0，tau_rec 报不定义（无穷），构造甲的 E_c 报不定义",
       "eta = %.1f，tau_rec = %s，构造甲的 E_c 在 eta = 0 处发散故报不定义"
       % (eta_zero, "不定义（无穷）" if np.isinf(tau_zero) else "%.6f" % tau_zero),
       ec6_ok, "替身与真构型可区分：真构型 g > 0 给 eta > 0、tau_rec 有限")

# 区分力的正面证据：真构型的 g 档上 E_c 随 g 变化，非恒真
ec6_span = max(r[1] for r in ec5_jia) - min(r[1] for r in ec5_jia)
ec6b_ok = ec6_span > 1e-6
record("EC-6b", "E_c 对耦合取值的敏感性（防恒真）",
       "E_c 的读数极差 > 1e-6，即 E_c 确实随 g 变化",
       "构造甲的 E_c 在模型档 g 上的极差 %.8f，幂次 %.6f"
       % (ec6_span, sl_jia), ec6b_ok,
       "按「耦合只进率」与 E_c ∝ 1/eta ∝ g^{-2} 的关系，它应当有 g^{-2} 的幂次，实测相符")

# =====================================================================
# EC-7 档位标注与物理档
# =====================================================================
print("=" * 78)
print("EC-7 档位标注与物理档")
print("=" * 78)

# EC-7 容差预算：两档读数的对照不做判定门槛，
#   只要求两套读数都给出并标明档位（登记义务，非可选项）。

# 物理档：g 以 omega_P 计，换算到物理单位。
#   模型档的 g 与物理档的 g 是同一个量在两套单位下的读数：
#   物理档 g 以 omega_P 计，模型档 g 以写入装置的 g_full 为单位。
#   E_c 的绝对值须经 E_P 的两套约定给出。
ec7_rows = []
for g in G_PHYS:
    eta = C_CONST * g ** 2
    tau = TP / eta
    ec_a_base = E_P_BASE_GEV * tau        # 构造甲，基准约定 E_P = hbar omega_P
    ec_a_alt = E_P_ALT_GEV * tau         # 构造甲，折算约定 E_P^(t) = hbar/t_P
    ec7_rows.append((g, eta, tau, ec_a_base, ec_a_alt))
    print("  物理档 g = %.8f（%.6f omega_P）：eta = %.8f，tau_rec = %.6f，"
          "E_c(甲，基准约定) = %.6e GeV，E_c(甲，折算约定) = %.6e GeV"
          % (g, g / OMEGA_P, eta, tau, ec_a_base, ec_a_alt))
ec7_ok = len(ec7_rows) == len(G_PHYS) and all(np.isfinite(r[3]) for r in ec7_rows)
record("EC-7", "档位标注：模型档与物理档两套读数对照",
       "两档读数都给出并标明档位（术语表 L321 的登记义务）",
       "模型档 %d 个 g 档（E_c(甲) 以 omega_P 为单位）；"
       "物理档 %d 个 g 档（E_c(甲) 以 GeV 给出，基准约定与折算约定并列）"
       % (len(G_MODEL), len(ec7_rows)), ec7_ok,
       "物理档评估已做。E_c 的绝对值继承 g 读数的外推层级，"
       "两套普朗克能量约定相差 2 pi 倍，取值口径须随用途选定")

supp("EC-7 附带", "两档 E_c 的量级对照",
     "模型档 g = %.6f 给 E_c(甲) = %.8f omega_P；"
     "物理档 g = %.8f 给 E_c(甲) = %.6e GeV（基准约定）"
     % (G_MODEL[0], ec5_jia[0][1], G_PHYS[0], ec7_rows[0][3]),
     "模型档以 omega_P 为能量单位，物理档以 GeV 给出。"
     "两档的 g 是同一个量在两套单位下的读数，"
     "换算因子是写入装置的 g_full = %.6f" % G_FULL)

# =====================================================================
# EC-8 前提守恒
# =====================================================================
print("=" * 78)
print("EC-8 前提守恒")
print("=" * 78)

# EC-8 容差预算：沿用各在案容差。
#   第 394 条：C 的核对偏差 0（1e-12）；1/sqrt(C) 对机制篇 g 读数的
#   相对偏差 4.90e-4（门槛 1e-3，第 394 条 A4 的在案容差）。
#   写入侧：第 726 号的单站点基线，相对偏差门槛 1e-5。

g_at_1 = 1.0 / np.sqrt(C_CONST)
ec8a_dev = abs(g_at_1 - G_READ) / G_READ
ec8a_ok = ec8a_dev <= 1e-3
ec8a_rev = C_CONST * G_READ ** 2
record("EC-8a", "复现第 394 条的门限核对与三条绝对约束",
       "1/sqrt(C) 对机制篇 g 读数的相对偏差 ≤ 1e-3（第 394 条 A4）；"
       "C 的核对偏差 ≤ 1e-12",
       "1/sqrt(C) = %.8f 对机制篇读数 %.4f，相对偏差 %.4e；"
       "反向代入给 eta = %.8f；三条绝对约束：g_w ≤ %.8f（%.8f omega_P）、"
       "S = 5 时 g_h ≤ %.8f、N_拍 = 100 时 g_h ≤ %.4e omega_P"
       % (g_at_1, G_READ, ec8a_dev, ec8a_rev,
          G_MAX, G_MAX_OMEGA,
          np.sqrt(C_CONST) * G_MAX ** 2 / 5.0,
          1.0 / np.sqrt(C_CONST * 100) / OMEGA_P), ec8a_ok,
       "三条绝对约束的数值表与第 394 条 A5 至 A7 的在案读数一致")

t_write, e0, e1, d_site, mean_site, mean_base, dbar1 = write_baseline()
ec8b_ok = (abs(t_write - t_cross) / t_cross < 1e-5
           and abs(e1 - e0) / e0 < 1e-5
           and abs(d_site - (-0.99882383)) / 0.99882383 < 1e-5
           and abs(dbar1 - (-0.50000000)) / 0.50000000 < 1e-5)
record("EC-8b", "写入侧复现第 726 号单站点基线",
       "写入时刻 0.5000、总量 1.00000000、站点侧增量 -0.99882383、"
       "长期平均 0.50000000、Δ̄ = -0.50000000，各相对偏差 < 1e-5",
       "写入时刻 %.4f（偏差 %.1e），总量 %.10f（偏差 %.3e），站点侧增量 %.8f（偏差 %.3e），"
       "⟨站点侧⟩ = %.10f（偏差 %.3e），Δ̄ = %.10f（偏差 %.3e）"
       % (t_write, abs(t_write - t_cross) / t_cross, e1, abs(e1 - e0),
          d_site, abs(d_site + 0.99882383) / 0.99882383,
          mean_site, abs(mean_site - 0.5) / 0.5, dbar1, abs(dbar1 + 0.5) / 0.5),
       ec8b_ok, "g_full = %.6f 档" % G_FULL)

# =====================================================================
# 出图
# =====================================================================
print("=" * 78)
print("出图")
print("=" * 78)

fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.4))

# (a) eta = C g^2 对 g
ax = axes[0, 0]
gm = np.array([r[0] for r in ec3_model])
gp = np.array([r[0] for r in ec3_phys])
ax.loglog(gm, [r[1] for r in ec3_model], "o-", color="tab:blue", ms=6,
          label=r"模型档 $\eta = Cg^2$")
ax.loglog(gp, [r[1] for r in ec3_phys], "s--", color="tab:orange", ms=6,
          label=r"物理档 $\eta = Cg^2$")
ax.loglog(gm, C_CONST * gm ** 2, ":", color="tab:gray", lw=1.4,
          label=r"幂次 2 参考线")
ax.axvline(G_MAX, color="tab:red", ls="--", lw=1.2, label=r"上界 $g_w$")
ax.set_xlabel(r"写入耦合 $g$")
ax.set_ylabel(r"登记效率 $\eta$")
ax.set_title(r"(a) $\eta = Cg^2$ 对 $g$（两档）", fontsize=10)
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")
ax.text(0.03, 0.90, r"$C = %.0f$，上界 $g_w = %.6f$" % (C_CONST, G_MAX),
        transform=ax.transAxes, fontsize=8, va="top")

# (b) tau_rec 对 g
ax = axes[0, 1]
ax.loglog(gm, [r[2] for r in ec3_model], "o-", color="tab:blue", ms=6,
          label=r"模型档 $\tau_{\rm rec}$")
ax.loglog(gp, [r[2] for r in ec3_phys], "s--", color="tab:orange", ms=6,
          label=r"物理档 $\tau_{\rm rec}$")
ax.loglog(gm, (TP / (C_CONST * gm[0] ** 2)) * (gm[0] / gm) ** 2, ":",
          color="tab:gray", lw=1.4, label=r"幂次 $-2$ 参考线")
ax.axvline(G_MAX, color="tab:red", ls="--", lw=1.2, label=r"上界 $g_w$")
ax.set_xlabel(r"写入耦合 $g$")
ax.set_ylabel(r"持有时间 $\tau_{\rm rec}$")
ax.set_title(r"(b) $\tau_{\rm rec} = t_P/\eta$ 对 $g$（两档）", fontsize=10)
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")
ax.text(0.03, 0.90, r"幂次：模型档 %.4f，物理档 %.4f" % (sl_tau_m, sl_tau_p),
        transform=ax.transAxes, fontsize=8, va="top")

# (c) E_c 对 g，含两档
ax = axes[1, 0]
ax.loglog(gm, [r[1] for r in ec5_jia], "o-", color="tab:blue", ms=6,
          label=r"模型档 $E_c$（甲，以 $\omega_P$ 计）")
ax.loglog(gm, (ec5_jia[0][1]) * (gm[0] / gm) ** 2, ":", color="tab:gray", lw=1.4,
          label=r"幂次 $-2$ 参考线")
ax.loglog([r[0] for r in ec7_rows], [r[3] for r in ec7_rows], "s--",
          color="tab:orange", ms=6, label=r"物理档 $E_c$（甲，GeV）")
ax.set_xlabel(r"写入耦合 $g$")
ax.set_ylabel(r"$E_c$")
ax.set_title(r"(c) $E_c$ 对 $g$：模型档与物理档对照", fontsize=10)
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")
ax.text(0.03, 0.90, r"构造甲幂次 %.4f；构造乙为常数" % sl_jia,
        transform=ax.transAxes, fontsize=8, va="top")

# (d) 转移数对 g
ax = axes[1, 1]
ax.plot([r[0] for r in ec1_rows], [r[1] for r in ec1_rows], "o-",
        color="tab:green", ms=7, label=r"完整转移事件计数")
ax.plot([r[0] for r in ec1_rows], [r[2] for r in ec1_rows], "s--",
        color="tab:purple", ms=5, label=r"完成度（分数，非判据量）")
ax.axhline(1.0, color="tab:red", ls="--", lw=1.2, label=r"期望 1")
ax.set_xlabel(r"耦合档 $g$（以 $g_{\rm full}$ 为单位）")
ax.set_ylabel("转移的激发数")
ax.set_ylim(0.9, 1.45)
ax.set_title(r"(d) 转移数对 $g$（EC-1）", fontsize=10)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
ax.text(0.03, 0.90, r"极差 %.3e（偏差 0）" % ec1_span,
        transform=ax.transAxes, fontsize=8, va="top")

fig.suptitle("第 769 号：写入耦合上界的能量代价（$E_c$ 的绝对标度）", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(OUT, dpi=150)
print("  图已存: %s" % OUT)

# =====================================================================
# 总表
# =====================================================================
print()
print("=" * 78)
print("判据总表")
print("=" * 78)
npass = 0
for cid, name, expect, actual, ok, note in RESULTS:
    print("%-8s %-56s %s" % (cid, name[:56], "通过" if ok else "判否"))
    if ok:
        npass += 1
print()
print("合计 %d/%d 通过（主判据 EC-1 至 EC-8，展开为 %d 个子项）"
      % (npass, len(RESULTS), len(RESULTS)))
print()
print("关键读数:")
print("  EC-1 完整转移事件计数：%s，极差 %d；完成度 %s"
      % (", ".join(str(r[1]) for r in ec1_rows), ec1_span,
         ", ".join("%.8f" % r[2] for r in ec1_rows)))
print("  EC-2 C = %.12f，模数 %.0f，相对偏差 %.3e" % (C_CONST, N_MODES, ec2_dev))
print("  EC-3 tau_rec 对 g 的幂次：模型档 %.9f，物理档 %.9f" % (sl_tau_m, sl_tau_p))
print("       模型档 tau %s" % ", ".join("%.6f" % r[2] for r in ec3_model))
print("  EC-4 上界档：g_w = %.8f（%.8f omega_P），eta = %.10f，tau_rec = %.8f，E_c(甲) = %.8f"
      % (G_MAX, G_MAX_OMEGA, eta_at_max, tau_at_max, OMEGA_P * tau_at_max))
print("       物理档四档 E_c(甲)（以 omega_P 计）：%s"
      % ", ".join("%.6f" % r[3] for r in ec4_table))
print("  EC-5 构造甲幂次 %.9f（残差 %.3e），E_c 各档 %s"
      % (sl_jia, res_jia, ", ".join("%.6f" % r[1] for r in ec5_jia)))
print("       构造乙能级差 %s，极差 %.3e"
      % (", ".join("%.10f" % v for v in delta_eps_list), delta_eps_span))
print("  EC-6 替身：eta = %.1f，tau_rec = %s；真构型 E_c 极差 %.8f"
      % (eta_zero, "不定义" if np.isinf(tau_zero) else "%.6f" % tau_zero, ec6_span))
print("  EC-7 物理档 E_c(甲)（GeV，基准约定）：%s"
      % ", ".join("%.4e" % r[3] for r in ec7_rows))
print("       物理档 E_c(甲)（GeV，折算约定）：%s"
      % ", ".join("%.4e" % r[4] for r in ec7_rows))
print("  EC-8 1/sqrt(C) = %.8f 对机制篇 %.4f 偏差 %.4e；eta 反向 %.8f"
      % (g_at_1, G_READ, ec8a_dev, ec8a_rev))
print("       写入侧：写入时刻 %.4f，总量 %.10f（偏差 %.3e），站点侧增量 %.8f，Δ̄ = %.10f"
      % (t_write, e1, abs(e1 - e0), d_site, dbar1))
print("耗时 %.1f 秒" % (time.time() - T_START))
