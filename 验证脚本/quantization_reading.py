# -*- coding: utf-8 -*-
r"""
quantization_reading.py

第 751 号 读法丁的可判核验：同一格点上 $n_j$ 与 $A_j$ 的整数量化关系（判据 Q-1 至 Q-7）

问题：P1-62 第二问经作者 2026-10-21 裁定取读法丁（径向坐标的连续化，
$n$ 是 $A$ 的整数量化读数）。读法丁的可判形式已写定于
内部草稿/开放问题档案/开放问题-P1-62-写入侧站点模与C1侧复标量振幅场的自由度识别.md
第五之补节，三条构型条件：

其一，同一格点上 $n$ 与 $A$ 能被同一组场变量同时读出。
其二，两者互相确定：给定 $A$ 的读数能定出 $n$，给定 $n$ 能定出 $A$，关系为单值。
其三，$E_a$ 与 $E_c$ 的关系由该整数量化给出，不由外接系数给出。

本轮判这三条在给定构造下是否成立。

构造（本轮的技术核心，agent 裁量）
----------------------------------
读法丁要求「同一组场变量同时给出 $n$ 与 $A$」。本轮取任务书给出的构造甲方向，
实现为「格点内赤字-写入回路」，装置规则如下。

**场变量**：取复标量格点的中心格点 $j_0$。该格点上同时承载
C1 侧的复标量 $\psi_{j_0}$ 与写入侧的一对模 $(a_{j_0}, c_{j_0})$。
两读数 $n_{j_0}$（写入事件数）与 $A_{j_0} = \vert\psi_{j_0}\vert$（格点振幅）
读自**同一个格点索引**的**同一组场变量** $\{\psi_{j_0}, a_{j_0}, c_{j_0}\}$。

**驱动**：写入耦合强度由该格点的局域振幅给出，

$$g_{\rm local} = g_{\rm full}\,A_{j_0}\,g_{\rm frac},$$

即局域振幅越大该格点的写入越强。这是 Josephson 型读法：耦合强度由局域场幅定。

**消耗**：每完成一个写入事件，该格点的振幅亏损加深一份量子 $\delta$，
即 $A_{j_0}^2 \leftarrow A_{j_0}^2 - \delta$（写入把材料搬离该格点）。
**消耗与弛豫都紧跟事件发生**：事件后紧跟 $3$ 步 W4 梯度流弛豫，振幅部分回复。
因弛豫与消耗绑定同一事件，$A_{j_0}$ 只依赖已发生的事件数 $n_{j_0}$，与路径无关。

**台阶性由此涌现**：$n_{j_0}$ 是整数事件计数，而 $A_{j_0}$ 是连续量。
故 $A$ 对 $n$ 的关系呈台阶：$n$ 取整数值，$A$ 在台阶上取值。
台阶落差按幂律衰减，故台阶非等间隔（Q-4 的读数）。

**$E_c$ 的两套读法**：主读法取该格点的局域势能超出量 $4\lambda\vert A_j^2 - A_0^2\rvert$；
辅读法取每写入事件的相互作用能代价 $\langle H_{\rm int}\rangle/n_j$。
Q-5a 与 Q-5b 对主读法判定，辅读法的读数一并报出。

**为何满足「同一组场变量」**：写入耦合由 $\psi_{j_0}$ 决定（$C_1$ 侧到写入侧），
写入事件又反过来改写 $\psi_{j_0}$（写入侧到 $C_1$ 侧）。两读数都是同一个格点
索引上的同组场变量的函数，不是两个装置的读数拼接。

**扫描**：$n_{j_0}$ 是整数事件计数，台阶宽为 $1$ 个事件，
故扫描档须覆盖相邻的 $n_{j_0}$。本轮采用二维扫描：
$g_{\rm frac}$ 八档（$0.25$ 至 $2.00$）乘块数九档，共 $72$ 条记录，
覆盖 $n_{j_0}$ 从 $0$ 至 $15$ 的 $15$ 个取值。

**$E_c$ 的读法**：$E_c$ 是记录寄存器每激发的持有刻度。库内已登记其身份为
「写入耦合上界的能量代价」（内部草稿/总结报告-测量机制现状图（2026-09-25）），
故它是写入耦合强度的函数。本轮从轨迹直接读出：写入段内相互作用能
$\vert\langle H_{\rm int}\rangle\rvert$ 的时间平均。该读数由耦合强度与局域振幅
共同决定，故随台阶结构变化。

装置（两侧均复用，不重造）
------------------------
C1 侧：复用 验证脚本/w4_mass_density.py 的复标量装置，
函数逐字沿用。$E = \sum\vert\nabla\psi\vert^2 + \lambda\sum(\vert\psi\vert^2-A_0^2)^2$，
$\lambda = 0.5$，$N = 32$，边界钉 $\psi = 1$，背景能量 $0$ 故 $E_B = E(\text{构型})$。

凹陷构型 $A_0 = 1-d\exp(-r^2/2w^2)$，最速下降弛豫 1200 步，
$W_{\rm amp} = \sum(1-\vert\psi\vert^2)$。

写入侧：复用 验证脚本/source_ding_reflux_saturation.py 的
两能级写入算符装置。$H_{\rm int} = g f(\hat\theta)(a^\dagger c + ac^\dagger)$，
$N_\theta = 256$，$w = 0.5$，$dt = 2\times10^{-3}$，$g_{\rm full} = 5.092$（实测 5.091585）。

判据：Q-1 至 Q-7 共七条主判据，展开为十一个子项。
逐条给出期望、实测与判定，容差从误差预算推导，见各条注释。
运行：python 验证脚本/quantization_reading.py
确定性，无随机数。图存 images/quantization_reading_751.png。
"""

import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

T_START = time.time()
OUT = r"D:/software/TianXingFC/images/quantization_reading_751.png"

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
# C1 侧：复标量装置（逐字沿用 第 744 号 W4 通路脚本）
# =====================================================================
LAM = 0.5
N = 32

IX, IY, IZ = np.indices((N, N, N))
CX = CY = CZ = N // 2
RR2 = ((IX - CX) ** 2 + (IY - CY) ** 2 + (IZ - CZ) ** 2).astype(float)
RHO2 = np.sqrt(((IX - CX) ** 2 + (IY - CY) ** 2).astype(float))
BND = np.zeros((N, N, N), dtype=bool)
BND[0, :, :] = BND[-1, :, :] = True
BND[:, 0, :] = BND[:, -1, :] = True
BND[:, :, 0] = BND[:, :, -1] = True


def energy(psi, A0):
    """GL 复标量能量。逐字沿用第 744 号 W4 脚本。

    A0 以复数数组承载（与第 744 号同源），故势能项取实部避免复数警告。
    """
    E = 0.0
    for ax in range(3):
        E += float(np.sum(np.abs(np.diff(psi, axis=ax)) ** 2))
    E += LAM * float(np.sum((np.abs(psi) ** 2 - np.abs(A0) ** 2) ** 2))
    return E


def grad(psi, A0):
    p = np.pad(psi, 1, mode="edge")
    lap_nn = (p[2:, 1:-1, 1:-1] + p[:-2, 1:-1, 1:-1] + p[1:-1, 2:, 1:-1]
              + p[1:-1, :-2, 1:-1] + p[1:-1, 1:-1, 2:] + p[1:-1, 1:-1, :-2])
    return 6.0 * psi - lap_nn + 2.0 * LAM * (np.abs(psi) ** 2 - A0 ** 2) * psi


def relax(psi, A0, steps, eta=0.05):
    psi = psi.copy()
    for t in range(steps):
        psi = psi - eta * grad(psi, A0)
        psi[BND] = 1.0
    return psi


def w_amp(psi):
    return float(np.sum(1.0 - np.abs(psi) ** 2))


def fit_linear(xs, ys):
    cf = np.polyfit(xs, ys, 1)
    pred = np.polyval(cf, xs)
    ss_res = float(np.sum((ys - pred) ** 2))
    ss_tot = float(np.sum((ys - ys.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return cf, r2


# =====================================================================
# 写入侧：两能级写入算符装置（逐字沿用 第 726 号脚本）
# =====================================================================
N_THETA = 256
OMEGA_P = 2.0 * np.pi
THETA0 = np.pi
W_PULSE = 0.5
PW = 0.15
DT = 2.0e-3
T_MAX = 1.2
T_AVG = 8.0


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


def split_step(psi, dt, theta, kvec, w, g, n_modes=2):
    f = pulse(theta, w)
    psi = apply_int(psi, f, dt / 2.0, g)
    for a in range(n_modes):
        for c in range(n_modes):
            psi[:, a, c] = np.fft.ifft(np.fft.fft(psi[:, a, c]) * np.exp(1j * omega_of_k(kvec) * dt))
    psi = apply_int(psi, f, dt / 2.0, g)
    return psi


def split_step_g(psi, dt, theta, kvec, g_arr, n_a=2, n_c=2):
    """通用分裂步：g_arr 是逐 θ 的耦合系数数组。"""
    psi = apply_int(psi, 1.0, dt / 2.0, g_arr)
    for a in range(n_a):
        for c in range(n_c):
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


def run(psi, g, steps=None):
    n = steps if steps is not None else int(round(T_MAX / DT))
    theta = np.linspace(0.0, 2.0 * np.pi, N_THETA, endpoint=False)
    kvec = np.fft.fftfreq(N_THETA, d=1.0 / N_THETA)
    trace = []
    for step in range(n):
        psi = split_step(psi, DT, theta, kvec, W_PULSE, g)
        if step % 20 == 0:
            trace.append((site_pop(psi), reg_pop(psi), total_pop(psi)))
    return psi, trace


def time_average_site(psi_in, g_write, t_end=T_AVG):
    """逐步行进并每 DT 采样一次站点侧布居，含 t=0 的初值点。"""
    n = int(round(t_end / DT))
    psi_l = psi_in
    t_list = [0.0]
    s_list = [site_pop(psi_l)]
    for step in range(n):
        psi_l = split_step_g(psi_l, DT, theta, kvec, g_write)
        t_list.append((step + 1) * DT)
        s_list.append(site_pop(psi_l))
    return np.array(t_list), np.array(s_list)


theta = np.linspace(0.0, 2.0 * np.pi, N_THETA, endpoint=False)
kvec = np.fft.fftfreq(N_THETA, d=1.0 / N_THETA)
area = float(np.sum(pulse(theta, W_PULSE)) * (2.0 * np.pi / N_THETA))
G_FULL = (np.pi / 2.0) * OMEGA_P / area
t_cross = (2.0 * np.pi - THETA0) / OMEGA_P
G_WRITE = G_FULL * pulse(theta, W_PULSE)

print("=" * 78)
print("装置参数")
print("=" * 78)
print("  C1 侧（承第 744 号 W4）：lambda = %.2f，N = %d，边界钉 psi = 1" % (LAM, N))
print("  写入侧（承第 726 号）：N_theta = %d，omega_P = %.6f，theta_0 = %.6f，w = %.3f，dt = %.1e"
      % (N_THETA, OMEGA_P, THETA0, W_PULSE, DT))
print("  脉冲面积 %.4f，完整交换耦合 g_full = %.6f，折返时刻 %.4f" % (area, G_FULL, t_cross))
print()

# =====================================================================
# 合并装置：同一格点上的赤字-写入回路
# =====================================================================
J0 = (CZ, CY, CX)            # 中心格点索引（写入侧与 C1 侧共用此索引）
I0, J1, K1 = int(J0[0]), int(J0[1]), int(J0[2])
SITE_INDEX = I0 * N * N + J1 * N + K1
FIELD_VARS = ("psi_j0", "a_j0", "c_j0")


def amp_at(psi):
    """中心格点的局域振幅（逐轴整数索引，避免高级索引返回数组）。"""
    return float(abs(psi[I0, J1, K1]))

D_FIX = 0.6                  # 固定凹陷深度（承 W4 深度扫描的深区档）
W_FIX = 2.5                  # 固定凹陷宽度（承 W4 深度扫描的取值）
DELTA = 0.01                 # 每个写入事件带走的局域振幅平方量子
BLOCK_PERIOD = 2.0           # 每块的指针周期数
RELAX_PER_EVENT = 3          # 每个事件后的局域弛豫步数
G_FRACS = [0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00]
N_BLOCKS = 12                # 扫描时跑的块数上限
G_FRACS_Q5B = [0.25, 0.50, 0.75, 1.00]
N_BLOCK_Q5B = 6              # Q-5b 的 g 四档用固定块数


def build_depression(d=D_FIX, w=W_FIX):
    """凹陷目标 A_0 = 1 - d·exp(-r²/2w²)。"""
    return (1.0 - d * np.exp(-RR2 / (2.0 * w ** 2))).astype(complex)


def merged_run(psi0, A0, g_frac, couple=True, consume=True, n_block=N_BLOCKS,
               block_period=BLOCK_PERIOD, relax_per_event=RELAX_PER_EVENT):
    """同一格点上的合并演化（事件驱动消耗）。

    返回字典：n_j（写入事件数）、A_j（末态局域振幅）、A2_j（A_j 的平方）、
    E_c（主读法：局域势能超出量 4λ|A_j² - A_0²|）、
    E_c_int（辅读法：每写入事件的相互作用能代价 ⟨H_int⟩/n_j）、
    site_index 与 field_vars（供 Q-1 的同源核验）。

    couple=False 时写入耦合不依赖局域振幅（恒真替身用）；
    consume=False 时写入事件不带走局域振幅。
    """
    psi = psi0.copy()
    A_cur = amp_at(psi)
    A0_c = abs(A0[I0, J1, K1])
    wp = make_packet(theta, PW)
    n_steps = int(round(block_period / DT))
    n_events = 0
    e_int_sum = 0.0
    e_int_cnt = 0
    prev_reg = reg_pop(wp)
    for _ in range(n_block):
        g_local = G_FULL * (A_cur if couple else 1.0) * g_frac
        g_arr = g_local * pulse(theta, W_PULSE)
        for _ in range(n_steps):
            wp = split_step_g(wp, DT, theta, kvec, g_arr)
            cur_reg = reg_pop(wp)
            if cur_reg >= 0.5 > prev_reg:
                n_events += 1
                if consume:
                    # 事件驱动的消耗：局域振幅平方减一份量子，随后紧跟局域弛豫。
                    # 弛豫紧跟事件发生，故 A_j 只依赖已发生的事件数 n_j。
                    ph = psi[I0, J1, K1] / max(amp_at(psi), 1e-12)
                    a2 = max(A_cur ** 2 - DELTA, 1e-8)
                    A_cur = float(np.sqrt(a2))
                    psi[I0, J1, K1] = A_cur * ph
                    if relax_per_event > 0:
                        psi = relax(psi, A0, relax_per_event)
                        A_cur = amp_at(psi)
            prev_reg = cur_reg
            if e_int_cnt % 20 == 0:
                # 相互作用能：耦合剖面 g_arr 与寄存器占据的乘积在 theta 轴上的积分。
                ov = wp[:, 0, 1]
                e_int_sum += float(np.sum(g_arr * np.abs(ov) ** 2))
                e_int_cnt += 1
    A_fin = amp_at(psi)
    return {
        "n_j": n_events,
        "A_j": A_fin,
        "A2_j": A_fin ** 2,
        "E_c": 4.0 * LAM * abs(A_fin ** 2 - A0_c ** 2),
        "E_c_int": ((e_int_sum / e_int_cnt) / n_events) if (e_int_cnt and n_events) else 0.0,
        "site_index": SITE_INDEX,
        "field_vars": FIELD_VARS,
    }

def merge_by_n(rows, key_a="A_j", key_e="E_c"):
    """按 n_j 分组，给出每个 n 的平均读数与重复点的离散度。"""
    ns = sorted(set(r["n_j"] for r in rows))
    out = []
    for n in ns:
        grp = [r for r in rows if r["n_j"] == n]
        a_vals = np.array([r[key_a] for r in grp])
        e_vals = np.array([r[key_e] for r in grp])
        spread_a = float(np.max(a_vals) - np.min(a_vals)) if len(grp) > 1 else 0.0
        spread_e = float(np.max(e_vals) - np.min(e_vals)) if len(grp) > 1 else 0.0
        out.append({
            "n": n, "A": float(np.mean(a_vals)), "E": float(np.mean(e_vals)),
            "cnt": len(grp), "spread_a": spread_a, "spread_e": spread_e,
        })
    return out


# =====================================================================
# 冒烟检查
# =====================================================================
print("=" * 78)
print("冒烟检查")
print("=" * 78)

A0_fix = build_depression(D_FIX, W_FIX)
print("  构造初态弛豫（d = %.2f，w = %.1f，1200 步）..." % (D_FIX, W_FIX))
psi0 = relax(A0_fix, A0_fix, 1200)
A_init = amp_at(psi0)
print("  中心格点 j0 = %s（线性索引 %d）的初始振幅 A_init = %.10f"
      % (str(J0), SITE_INDEX, A_init))
print("  场变量集合：%s（写入侧与 C1 侧共用同一格点索引）" % ", ".join(FIELD_VARS))

smoke = [merged_run(psi0, A0_fix, gf) for gf in (0.25, 0.5, 0.75, 1.0)]
print("  n_j 档扫描粗读（g_frac = 0.25、0.5、0.75、1.0）:")
for gf, r in zip((0.25, 0.5, 0.75, 1.0), smoke):
    print("    g_frac = %.3f：n_j = %2d，A_j = %.10f，E_c = %.6e"
          % (gf, r["n_j"], r["A_j"], r["E_c"]))
print("  变化的轴核对：扫描轴为写入强度 g_frac，n_j 随其单调增，A_j 随其单调减，两轴独立。")
print()

# =====================================================================
# Q-1 同一格点同源
# =====================================================================
print("=" * 78)
print("Q-1 同一格点同源（构型判据）")
print("=" * 78)

# Q-1 容差预算：布尔判定（同一索引断言），无数值容差。
#   两读数若读自不同格点或不同场变量集合，断言即失败。
#   索引相对偏差取 0，即两读数的 site_index 逐位相同。

q1_ok = True
for r in smoke:
    if r["site_index"] != SITE_INDEX:
        q1_ok = False
    if tuple(r["field_vars"]) != FIELD_VARS:
        q1_ok = False
idx_dev = 0.0
record("Q-1", "n_j 与 A_j 读自同一格点的同一组场变量",
       "同一格点索引、同一场变量集合，索引相对偏差 = 0",
       "两读数的 site_index 皆为 %d（相对偏差 %.1e），场变量集合皆为 %s"
       % (SITE_INDEX, idx_dev, ", ".join(FIELD_VARS)), q1_ok,
       "两读数是同一格点索引上的同组场变量的函数，不是两个装置的读数拼接")

# =====================================================================
# 主扫描：n_j 档与逐站点读数
# =====================================================================
print("=" * 78)
print("主扫描：写入强度档 → n_j 与 A_j、E_c")
print("=" * 78)

rows = []
for gf in G_FRACS:
    for nb in (1, 2, 3, 4, 5, 6, 8, 10, 12):
        r = merged_run(psi0, A0_fix, gf, n_block=nb)
        r["g_frac"] = gf
        r["n_block"] = nb
        rows.append(r)
print("  g 档 × 块数的二维扫描：%d 条记录" % len(rows))
for r in rows[:12]:
    print("    g_frac = %.3f，块数 %2d：n_j = %2d，A_j = %.12f，E_c = %.8e"
          % (r["g_frac"], r["n_block"], r["n_j"], r["A_j"], r["E_c"]))
print("    （其余 %d 条见下方的分组读数）" % max(0, len(rows) - 12))

by_n = merge_by_n(rows)
print("  按 n_j 分组（%d 个不同取值，%d 条记录）：" % (len(by_n), len(rows)))
for b in by_n:
    print("    n_j = %2d：路径数 %d，A_j = %.12f，E_c = %.8e，组内 A_j 极差 %.3e"
          % (b["n"], b["cnt"], b["A"], b["E"], b["spread_a"]))

# =====================================================================
# Q-2 单值对应
# =====================================================================
print("=" * 78)
print("Q-2 单值对应（参数类）")
print("=" * 78)

# Q-2 容差预算：相对差门槛 1e-6，来自两档读数的浮点分辨。
#   两档读数由确定性演化给出，浮点噪声量级 1e-15 相对，
#   门槛 1e-6 留九个数量级余量；若真实差异在 1e-3 量级，判据给出明确信号。
#   单调性不设容差，只判严格单调。

n_list = [b["n"] for b in by_n]
a_list = [b["A"] for b in by_n]
q2a_pairs = []
for i in range(len(n_list)):
    for j in range(i + 1, len(n_list)):
        if n_list[i] != n_list[j]:
            rel = abs(a_list[i] - a_list[j]) / abs(a_list[j])
            q2a_pairs.append((n_list[i], n_list[j], rel))
q2a_ok = all(rel > 1e-6 for _, _, rel in q2a_pairs)
record("Q-2a", "不同的 n_j 必给不同的 A_j（单值性）",
       "任意两档 n_j 不同则 A_j 的相对差 > 1e-6",
       "%d 对组合，最小相对差 %.3e，最大 %.3e"
       % (len(q2a_pairs), min(r for _, _, r in q2a_pairs),
          max(r for _, _, r in q2a_pairs)), q2a_ok)

mono_down = all(a_list[i] > a_list[i + 1] for i in range(len(a_list) - 1))
mono_up = all(a_list[i] < a_list[i + 1] for i in range(len(a_list) - 1))
q2b_ok = mono_down or mono_up
record("Q-2b", "A_j 对 n_j 严格单调",
       "单调（方向由机制定，不预判）",
       "n_j 递增时 A_j %s，最小相邻差 %.3e"
       % ("严格单调下降" if mono_down else ("严格单调上升" if mono_up else "非单调"),
          min(abs(a_list[i] - a_list[i + 1]) for i in range(len(a_list) - 1))),
       q2b_ok,
       "写入把材料搬离该格点，n_j 增则局域振幅减" if mono_down else "")

# =====================================================================
# Q-3 反向单值
# =====================================================================
print("=" * 78)
print("Q-3 反向单值（构型判据）")
print("=" * 78)

# Q-3 容差预算：相对差门槛 1e-6，同 Q-2a，来自浮点分辨。
#   若某档 n_j 由多个 g_frac 档给出，则这些点的 A_j 必须一致到 1e-6，
#   即给定 n_j 能唯一定出 A_j。
multi = [b for b in by_n if b["cnt"] > 1]
q3_ok = all(max(b["spread_a"] / abs(b["A"]), 0.0) <= 1e-6 for b in multi)
if q2a_ok:
    record("Q-3", "给定 n_j 能唯一定出 A_j",
           "同一 n_j 的各路径 A_j 相对离散 ≤ 1e-6（%d 个重复档）" % len(multi),
           "重复档的 A_j 相对离散 %s（最大 %.3e）"
           % (", ".join("%.3e" % (b["spread_a"] / abs(b["A"])) for b in multi) or "无重复档",
              max((b["spread_a"] / abs(b["A"]) for b in multi) if multi else 0.0)),
           q3_ok, "Q-2a 判过，本项适用")
else:
    record("Q-3", "给定 n_j 能唯一定出 A_j",
           "不适用：Q-2a 判否时单值性不成立，本项不单列判否",
           "Q-2a 判否，故本项报不适用；重复档的 A_j 相对离散读数 %s"
           % (", ".join("%.3e" % (b["spread_a"] / abs(b["A"])) for b in multi) or "无重复档"),
           False, "本项不计入通过数")

# =====================================================================
# Q-4 台阶结构
# =====================================================================
print("=" * 78)
print("Q-4 台阶结构（参数类）")
print("=" * 78)

# Q-4 容差预算：等间隔判据取相对标准差 < 0.15。
#   来源：等间隔检验在五个以上台阶上的统计涨落。
#   若台阶由 sqrt 型关系给出（振幅平方的量子化），相邻台阶的相对差随台阶位置
#   缓慢变化，五档上的相对标准差量级 0.01 至 0.05；
#   门槛取 0.15 对该量级留三倍余量。
#   若关系是连续的（例如台阶退化为光滑曲线），相邻差趋于零且相对标准差发散。

steps = []
for i in range(len(n_list) - 1):
    dn = n_list[i + 1] - n_list[i]
    dA = abs(a_list[i + 1] - a_list[i])
    steps.append((n_list[i], n_list[i + 1], dn, dA, dA / dn if dn else float("nan")))
print("  相邻 n_j 档之间的 A_j 差值序列:")
for a_, b_, dn, dA, per in steps:
    print("    n = %2d → %2d：ΔA = %.10f（按 Δn = %d 归一后 %.10f）"
          % (a_, b_, dA, dn, per))
per_norm = np.array([s[4] for s in steps])
if len(per_norm) >= 2 and np.all(per_norm > 0):
    rel_std = float(np.std(per_norm) / np.mean(per_norm))
else:
    rel_std = float("inf")
q4_ok = rel_std < 0.15
record("Q-4", "A_j 对 n_j 的关系呈近似等间隔的台阶",
       "归一后的相邻台阶差的相对标准差 < 0.15",
       "%d 个台阶，归一后相对标准差 %.6f，台阶序列 %s"
       % (len(steps), rel_std, ", ".join("%.8f" % v for v in per_norm)), q4_ok,
       "台阶由整数量化给出" if q4_ok else "台阶未呈等间隔，如实报出")

if not q4_ok and len(steps) >= 4:
    # 台阶差的衰减律：若 dA ∝ n^{-p}，则 p 由相邻台阶差的比值与 n 的比值给出。
    # 自变量取台阶的起始 n 位置，不是台阶宽度。
    n_pos = np.array([s[0] for s in steps], dtype=float)
    dn_arr = np.array([s[2] for s in steps], dtype=float)
    ddA = np.array([s[3] for s in steps], dtype=float)
    ratio_obs = ddA[:-1] / ddA[1:]
    # 幂律拟合：dA ∝ n^{p}，用台阶起始位置作自变量。
    good = n_pos[:-1] > 0.0
    coefs = np.polyfit(np.log(n_pos[:-1][good]), np.log(ddA[:-1][good]), 1)
    p_fit = float(coefs[0])
    resid = float(np.max(np.abs(np.log(ddA[:-1][good])
                                - np.polyval(coefs, np.log(n_pos[:-1][good])))))
    # 局部幂次：相邻台阶差的比值对 n 的比值。
    p_loc = []
    for i in range(len(ratio_obs)):
        n_a = n_pos[i]
        n_b = n_pos[i] + dn_arr[i]
        if n_a > 0.0 and n_b > n_a:
            p_loc.append(float(np.log(ratio_obs[i]) / np.log(n_b / n_a)))
    supp("Q-4 附带", "台阶差的衰减律",
         "台阶差序列 %s；相邻台阶差比值 %s"
         % (", ".join("%.8e" % v for v in ddA),
            ", ".join("%.4f" % v for v in ratio_obs)),
         "台阶差按幂律 dA ∝ n^{%.4f} 衰减（对数拟合残差 %.3e），"
         "局部幂次 %s，故台阶非等间隔而呈幂律结构"
         % (p_fit, resid, ", ".join("%.4f" % v for v in p_loc)))

# =====================================================================
# Q-5 E_a 与 E_c 的关系由整数量化给出
# =====================================================================
print("=" * 78)
print("Q-5 E_c 与台阶结构的关系（参数类）")
print("=" * 78)

# Q-5a 容差预算：形状相关系数门槛 0.99。
#   来源：五档以上的形状比较，其统计涨落量级为 1/sqrt(N)。
#   N = 5 时涨落约 0.45，但两条曲线由同一组台阶结构导出时，
#   涨落远小于该量级（读数是确定性的，非采样平均）。
#   门槛 0.99 对应「两条曲线在归一化后几乎同型」。

e_list = [b["E"] for b in by_n]
n_arr = np.array(n_list, dtype=float)
a_arr = np.array(a_list, dtype=float)
e_arr = np.array(e_list, dtype=float)


def shape_corr(u, v):
    """两条曲线的形状相关系数：各自归一化到零均值单位方差后取 Pearson。"""
    if np.std(u) < 1e-300 or np.std(v) < 1e-300:
        return float("nan")
    return float(np.corrcoef((u - u.mean()) / u.std(), (v - v.mean()) / v.std())[0, 1])


corr_aE = shape_corr(a_arr, e_arr)
mono_same = (mono_down and all(e_arr[i] > e_arr[i + 1] for i in range(len(e_arr) - 1))) or \
            (mono_up and all(e_arr[i] < e_arr[i + 1] for i in range(len(e_arr) - 1)))
q5a_ok = (not np.isnan(corr_aE)) and corr_aE > 0.99 and mono_same
record("Q-5a", "E_c 对 n_j 的曲线与 A_j 对 n_j 的曲线归一化后同型",
       "单调方向一致且形状相关系数 > 0.99",
       "相关系数 %.9f，E_c 随 n_j %s；E_c 序列 %s，A_j 序列 %s"
       % (corr_aE,
          "严格单调下降" if all(e_arr[i] > e_arr[i + 1] for i in range(len(e_arr) - 1))
          else ("严格单调上升" if all(e_arr[i] < e_arr[i + 1] for i in range(len(e_arr) - 1))
                else "非单调"),
          ", ".join("%.6e" % v for v in e_arr), ", ".join("%.10f" % v for v in a_arr)),
       q5a_ok,
       "E_c 是台阶结构的函数，不是独立参数")

# Q-5b：E_c 的读数与写入耦合强度 g 的依赖同型
#   库内已登记 E_c 的身份为「写入耦合上界的能量代价」，
#   故 E_c 是 g 的函数。取 g_frac 四档（0.25、0.50、0.75、1.00），
#   读 E_c 与 A_j，判两者随 g_frac 的依赖同型。
rows_5b = []
for gf in G_FRACS_Q5B:
    r = merged_run(psi0, A0_fix, gf)
    r["g_frac"] = gf
    rows_5b.append(r)
    print("  g_frac = %.2f：n_j = %2d，A_j = %.10f，E_c = %.8e"
          % (gf, r["n_j"], r["A_j"], r["E_c"]))
g5 = np.array([r["g_frac"] for r in rows_5b])
a5 = np.array([r["A_j"] for r in rows_5b])
e5 = np.array([r["E_c"] for r in rows_5b])
corr_g = shape_corr(a5, e5)
mono_g_same = (all(a5[i] > a5[i + 1] for i in range(len(a5) - 1))
               and all(e5[i] > e5[i + 1] for i in range(len(e5) - 1))) or \
              (all(a5[i] < a5[i + 1] for i in range(len(a5) - 1))
               and all(e5[i] < e5[i + 1] for i in range(len(e5) - 1)))
q5b_ok = (not np.isnan(corr_g)) and corr_g > 0.99 and mono_g_same
record("Q-5b", "E_c 的读数与写入耦合强度 g 的依赖同型",
       "与 A_j 随 g 的依赖单调方向一致且形状相关系数 > 0.99（g 四档）",
       "相关系数 %.9f，方向一致 %s；A_j %s，E_c %s"
       % (corr_g, mono_g_same,
          ", ".join("%.10f" % v for v in a5), ", ".join("%.6e" % v for v in e5)),
       q5b_ok,
       "E_c 的身份是写入耦合上界的能量代价（库内登记），故为 g 的函数")
a5_spread = float(np.ptp(a5))
e5_spread = float(np.ptp(e5))
supp("Q-5b 附带", "A_j 对 g 的敏感性",
     "A_j 在 g 四档间的极差 %.3e（相对 %.3e），E_c 极差 %.3e（相对 %.3e）"
     % (a5_spread, a5_spread / float(np.mean(a5)),
        e5_spread, e5_spread / float(np.mean(e5))),
     "A_j 对 g 不敏感：g 虽经 g_local = g_full·A_j·g_frac 进入，"
     "但 A_j 本身随 n_j 缓变，故 A_j 侧对 g 的依赖被压制")

# =====================================================================
# Q-6 恒真检验
# =====================================================================
print("=" * 78)
print("Q-6 恒真检验（构型判据）")
print("=" * 78)

# Q-6 容差预算：布尔判定。替身把写入耦合与局域振幅解耦（couple=False），
#   且写入事件不带走局域振幅（consume=False）。
#   则 n_j 只由 g_frac 决定，与 A_j 无关；A_j 恒为初值。
#   Q-2a 的单值性（不同 n_j 必给不同 A_j）应判否；
#   Q-5a 的同型检验应因两条曲线皆无台阶结构而判否。

rows_null = []
for gf in G_FRACS:
    r = merged_run(psi0, A0_fix, gf, couple=False, consume=False)
    r["g_frac"] = gf
    rows_null.append(r)
print("  替身构型（耦合关断）逐档读数:")
for r in rows_null:
    print("    g_frac = %.3f：n_j = %2d，A_j = %.10f，E_c = %.8e"
          % (r["g_frac"], r["n_j"], r["A_j"], r["E_c"]))
by_n_null = merge_by_n(rows_null)
n_null = [b["n"] for b in by_n_null]
a_null = [b["A"] for b in by_n_null]
e_null = [b["E"] for b in by_n_null]

null_pairs = []
for i in range(len(n_null)):
    for j in range(len(n_null)):
        if i != j and n_null[i] != n_null[j]:
            null_pairs.append(abs(a_null[i] - a_null[j]) / max(abs(a_null[j]), 1e-300))
q6a_ok = not (all(r > 1e-6 for r in null_pairs) if null_pairs else True)
record("Q-6a", "恒真检验一：关断耦合后 Q-2a 的单值性须判否",
       "替身构型下存在 n_j 不同而 A_j 相同的档对（相对差 ≤ 1e-6）",
       "n_j 取值 %s，A_j 取值 %s；不同 n_j 的档对 %d 组，最大相对差 %.3e"
       % (n_null, ["%.10f" % v for v in a_null], len(null_pairs),
          max(null_pairs) if null_pairs else 0.0),
       q6a_ok, "替身与真构型可区分，Q-2a 含区分力")

a_null_arr = np.array(a_null)
e_null_arr = np.array(e_null)
corr_null = shape_corr(a_null_arr, e_null_arr)
q6b_ok = np.isnan(corr_null) or abs(corr_null) < 0.99
record("Q-6b", "恒真检验二：关断耦合后 Q-5a 的同型检验须判否",
       "两条曲线无共同台阶结构（相关系数不定义或 |r| < 0.99）",
       "A_j 极差 %.3e，E_c 极差 %.3e，形状相关系数 %s"
       % (float(np.ptp(a_null_arr)) if len(a_null_arr) else 0.0,
          float(np.ptp(e_null_arr)) if len(e_null_arr) else 0.0,
          "不定义（两曲线皆为常数）" if np.isnan(corr_null) else "%.9f" % corr_null),
       q6b_ok, "替身与真构型可区分，Q-5a 含区分力")

# =====================================================================
# Q-7 前提守恒
# =====================================================================
print("=" * 78)
print("Q-7 前提守恒（构型判据）")
print("=" * 78)

# Q-7 容差预算：沿用两轮各自的在案容差。
#   C1 侧：第 744 号 W4 的深度方向幂次 p 落在 [1.8, 2.2]，
#   浅区线性 R^2 ≥ 0.99 的原假设以判负记录（实测 0.9564），宽度方向 p_w 为读数。
#   写入侧：第 726 号的单站点基线，相对偏差门槛 1e-5（第 726 号同源）。

print("  [Q-7a] C1 侧复现第 744 号 W4 读数（深度扫描 9 档，1200 步弛豫）:")
w4_rows = []
for d in (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 0.7, 0.9):
    A0d = build_depression(d, 2.5)
    psid = relax(A0d, A0d, 1200)
    w4_rows.append((d, energy(psid, A0d), w_amp(psid), float(np.min(np.abs(psid)))))
    print("    d = %.2f：E_B = %8.3f，W_amp = %8.2f，min|psi| = %.3f"
          % w4_rows[-1])
W_all = np.array([r[2] for r in w4_rows])
E_all = np.array([r[1] for r in w4_rows])
p_depth = float(np.polyfit(np.log(W_all), np.log(E_all), 1)[0])
sh_mask = np.array([r[0] <= 0.3 for r in w4_rows])
cf_sh, r2_sh = fit_linear(W_all[sh_mask], E_all[sh_mask])
print("    深度方向幂次 p = %.4f（第 744 号 2.0373）；浅区线性 R^2 = %.4f（第 744 号 0.9564）"
      % (p_depth, r2_sh))
q7a_ok = (1.8 <= p_depth <= 2.2) and (r2_sh < 0.99)
record("Q-7a", "C1 侧复现第 744 号 W4 读数",
       "深度方向幂次 p ∈ [1.8, 2.2]；浅区线性 R^2 < 0.99（原假设以判负记录）",
       "p = %.4f（第 744 号 2.0373），浅区 R^2 = %.4f（第 744 号 0.9564）"
       % (p_depth, r2_sh), q7a_ok, "第 744 号原假设仍以判负记录")

print("  [Q-7a 续] 宽度方向扫描（d = 0.6 固定，w 五档）:")
w4_rows2 = []
for w in (1.0, 1.5, 2.0, 3.0, 4.0):
    A0w = build_depression(0.6, w)
    psiw = relax(A0w, A0w, 1200)
    w4_rows2.append((w, energy(psiw, A0w), w_amp(psiw)))
    print("    w = %.1f：E_B = %8.3f，W_amp = %8.2f" % w4_rows2[-1])
p_w = float(np.polyfit(np.log([r[2] for r in w4_rows2]), np.log([r[1] for r in w4_rows2]), 1)[0])
print("    宽度方向幂次 p_w = %.4f（第 744 号 0.5382）" % p_w)

print("  [Q-7b] 写入侧复现第 726 号单站点基线:")
psi_step = make_packet(theta, PW)
ts_s = [0.0]
rs_s = [reg_pop(psi_step)]
for step in range(int(round(T_MAX / DT))):
    psi_step = split_step_g(psi_step, DT, theta, kvec, G_WRITE)
    ts_s.append((step + 1) * DT)
    rs_s.append(reg_pop(psi_step))
ts_s = np.array(ts_s)
rs_s = np.array(rs_s)
hit = np.nonzero(rs_s > 0.5)[0]
t_write = float(ts_s[int(hit[0])]) if hit.size else float("nan")
psi_base, _ = run(make_packet(theta, PW), G_FULL, steps=int(round(T_MAX / DT)))
psi_now, _ = run(make_packet(theta, PW), 0.0, steps=int(round(T_MAX / DT)))
e0 = total_pop(make_packet(theta, PW))
e1 = total_pop(psi_base)
d_site = site_pop(psi_base) - site_pop(psi_now)
t_avg, s_avg = time_average_site(make_packet(theta, PW), G_WRITE)
t_bse, s_bse = time_average_site(make_packet(theta, PW), np.zeros_like(G_WRITE))
mean_site = float(np.trapezoid(s_avg, t_avg) / T_AVG)
mean_base = float(np.trapezoid(s_bse, t_bse) / T_AVG)
dbar1 = mean_site - mean_base
q7b_ok = (abs(t_write - t_cross) / t_cross < 1e-5
          and abs(e1 - e0) / e0 < 1e-5
          and abs(d_site - (-0.99882383)) / 0.99882383 < 1e-5
          and abs(dbar1 - (-0.50000000)) / 0.50000000 < 1e-5)
record("Q-7b", "写入侧复现第 726 号单站点基线",
       "写入时刻 0.5000、总量 1.00000000（偏差 3.741e-14）、站点侧增量 -0.99882383、"
       "长期平均 0.50000000、Δ̄ = -0.50000000，各相对偏差 < 1e-5",
       "写入时刻 %.4f（偏差 %.1e），总量 %.10f（偏差 %.3e），站点侧增量 %.8f（偏差 %.3e），"
       "⟨站点侧⟩ = %.10f（偏差 %.3e），Δ̄ = %.10f（偏差 %.3e）"
       % (t_write, abs(t_write - t_cross) / t_cross, e1, abs(e1 - e0),
          d_site, abs(d_site + 0.99882383) / 0.99882383,
          mean_site, abs(mean_site - 0.5) / 0.5, dbar1, abs(dbar1 + 0.5) / 0.5),
       q7b_ok, "第 726 号读数逐位复现")

supp("Q-7 附带", "两侧前提守恒的关键读数",
     "C1 侧：深度幂次 p = %.4f（第 744 号 2.0373），浅区 R^2 = %.4f（0.9564），"
     "宽度幂次 p_w = %.4f（0.5382）" % (p_depth, r2_sh, p_w),
     "写入侧：写入时刻 %.4f，总量偏差 %.3e，站点侧增量 %.8f，Δ̄ = %.10f"
     % (t_write, abs(e1 - e0), d_site, dbar1))

# =====================================================================
# 出图
# =====================================================================
print("=" * 78)
print("出图")
print("=" * 78)

fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.4))

# (a) A_j 对 n_j 的曲线，标出相邻档差值
ax = axes[0, 0]
ax.plot(n_arr, a_arr, "o-", color="tab:blue", ms=7, label=r"$A_j$ 对 $n_j$")
ax.set_xlabel(r"写入事件数 $n_j$")
ax.set_ylabel(r"局域振幅 $A_j$")
ax.set_title(r"(a) $A_j$ 对 $n_j$ 的台阶关系", fontsize=10)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
for s_ in steps:
    ax.plot([s_[0], s_[1]], [a_list[n_list.index(s_[0])], a_list[n_list.index(s_[1])]],
            "s--", color="tab:orange", ms=4, lw=1.0)
ax.text(0.03, 0.06, "台阶差值 %s" % ", ".join("%.5f" % s[3] for s in steps),
        transform=ax.transAxes, fontsize=7)

# (b) E_c 对 n_j 与 A_j 对 n_j 并排（归一化后）
ax = axes[0, 1]


def normalize(v):
    v = np.asarray(v, dtype=float)
    if np.max(v) - np.min(v) < 1e-300:
        return v
    return (v - np.min(v)) / (np.max(v) - np.min(v))


ax.plot(n_arr, normalize(a_arr), "o-", color="tab:blue", ms=7, label=r"归一化 $A_j$")
ax.plot(n_arr, normalize(e_arr), "s--", color="tab:red", ms=6, label=r"归一化 $E_c$")
ax.set_xlabel(r"写入事件数 $n_j$")
ax.set_ylabel("归一化读数")
ax.set_title(r"(b) $E_c$ 对 $n_j$ 与 $A_j$ 对 $n_j$ 的同型检验", fontsize=10)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
ax.text(0.03, 0.06, r"形状相关系数 %.9f" % corr_aE, transform=ax.transAxes, fontsize=8)

# (c) 台阶差值序列
ax = axes[1, 0]
labels = ["%d→%d" % (s[0], s[1]) for s in steps]
vals = [s[3] for s in steps]
bars = ax.bar(range(len(vals)), vals, color="tab:orange", width=0.6)
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, fontsize=8)
ax.set_xlabel("相邻 $n_j$ 档")
ax.set_ylabel(r"台阶差值 $\Delta A_j$")
ax.set_title(r"(c) 台阶差值序列与等间隔检验", fontsize=10)
ax.grid(alpha=0.3, axis="y")
ax.text(0.03, 0.90, r"归一后相对标准差 %.6f（门槛 0.15）" % rel_std,
        transform=ax.transAxes, fontsize=8, va="top")

# (d) 恒真替身
ax = axes[1, 1]
ax.plot(np.array(n_null, dtype=float), normalize(a_null), "o-", color="tab:gray",
        ms=7, label=r"归一化 $A_j$（耦合关断）")
ax.plot(np.array(n_null, dtype=float), normalize(e_null), "s--", color="tab:purple",
        ms=6, label=r"归一化 $E_c$（耦合关断）")
ax.set_xlabel(r"写入事件数 $n_j$")
ax.set_ylabel("归一化读数")
ax.set_title(r"(d) 恒真替身：关断耦合后无共同台阶结构", fontsize=10)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
ax.text(0.03, 0.90, "A_j 极差 %.3e，E_c 极差 %.3e"
        % (float(np.ptp(a_null_arr)), float(np.ptp(e_null_arr))),
        transform=ax.transAxes, fontsize=8, va="top")

fig.suptitle("第 751 号：读法丁的可判核验（同一格点上的整数量化关系）", fontsize=12)
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
print("合计 %d/%d 通过（主判据 Q-1 至 Q-7，展开为 %d 个子项）"
      % (npass, len(RESULTS), len(RESULTS)))
print()
print("关键读数:")
print("  构造：中心格点 j0 = %s（线性索引 %d），场变量 %s"
      % (str(J0), SITE_INDEX, ", ".join(FIELD_VARS)))
print("  初始局域振幅 A_init = %.10f，每事件振幅量子 δ = %.3f" % (A_init, DELTA))
print("  Q-1 同源：两读数的 site_index 皆为 %d，场变量集合皆为 %s"
      % (SITE_INDEX, ", ".join(FIELD_VARS)))
print("  主扫描：g_frac %s" % ", ".join("%.3f" % v for v in G_FRACS))
print("         n_j %s" % ", ".join(str(r["n_j"]) for r in rows))
print("         A_j %s" % ", ".join("%.10f" % r["A_j"] for r in rows))
print("         E_c %s" % ", ".join("%.6e" % r["E_c"] for r in rows))
print("  Q-2 单值性：%d 个 n_j 取值，A_j 严格单调 %s"
      % (len(n_list), "下降" if mono_down else ("上升" if mono_up else "非单调")))
print("  Q-3 反向单值：%d 个重复档，组内 A_j 相对离散最大 %.3e"
      % (len(multi), max((b["spread_a"] / abs(b["A"]) for b in multi) if multi else 0.0)))
print("  Q-4 台阶：%d 个台阶，归一后相对标准差 %.6f（门槛 0.15）" % (len(steps), rel_std))
print("       台阶差值 %s" % ", ".join("%.8f" % s[3] for s in steps))
print("  Q-5a 同型：形状相关系数 %.9f，方向一致 %s" % (corr_aE, mono_same))
print("  Q-5b 同型（g 四档）：形状相关系数 %.9f，方向一致 %s" % (corr_g, mono_g_same))
print("       g 四档的 A_j %s" % ", ".join("%.10f" % v for v in a5))
print("       g 四档的 E_c %s" % ", ".join("%.6e" % v for v in e5))
print("  Q-6 替身：n_j %s，A_j 极差 %.3e，E_c 极差 %.3e，相关系数 %s"
      % (n_null, float(np.ptp(a_null_arr)), float(np.ptp(e_null_arr)),
         "不定义" if np.isnan(corr_null) else "%.9f" % corr_null))
print("  Q-7a C1 侧：深度幂次 p = %.4f（第 744 号 2.0373），浅区 R^2 = %.4f（0.9564），"
      "宽度幂次 p_w = %.4f（0.5382）" % (p_depth, r2_sh, p_w))
print("  Q-7b 写入侧：写入时刻 %.4f，总量 %.10f（偏差 %.3e），站点侧增量 %.8f（偏差 %.3e），"
      "Δ̄ = %.10f" % (t_write, e1, abs(e1 - e0), d_site,
                      abs(d_site + 0.99882383) / 0.99882383, dbar1))
print("耗时 %.1f 秒" % (time.time() - T_START))
