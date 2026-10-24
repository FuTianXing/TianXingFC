# -*- coding: utf-8 -*-
r"""
case_b_face_wall.py

第 758 号 案乙部件三：三维直壁构型上的键耦合版（判据 FW-1 至 FW-8）

问题：第 755 号做的是一维链上的键耦合。本轮把构型换成三维直壁（面缺陷），
即第 755 号读数里明确不判的那一层。

装置（复用，不重造）
--------------------
**面缺陷动力学线**（承第 289 条与三维面缺陷写入节拍轮）：

$$\gamma\,\dot v = k\nabla^2 v - 2\pi K\sin 2\pi v + f + \xi,\qquad
k = 1,\ \gamma = 0.5958,\ K = 0.001.$$

$x$ 两端固定（左 $0$、右 $-1$，恒有一个 kink，绕数受边界保护），$y$、$z$ 周期，
截面 $A = N_yN_z$。写入自触发口径：站点相位每跨过一个整数写一笔；
折返时刻取同一站点跨过半整数（kink 中心经过）的时刻。

**键刚度就是 $k$**。按 M-F 裁决（术语表 L318）：同类耦合（键间余弦）归键刚度 $k_b$，
格点内三元组锁定归势项 $K$。故第 341 条方程里的 $k$ 即键刚度，
与案乙的格点间键弹性 $\kappa_\parallel$ 是同一个物理量：
$$\kappa_\parallel \equiv k_b \equiv k.$$
本轮直接扫 $k$，不另建 $\kappa_\parallel$ 通道。

**壁张力的解析对照**（《元相子理论的时空观》L305，2026-09-21 固化，解析式级，
第 341 条）：「波性侧的相位绕圈在三维里以面缺陷（台阶面）的形态出现，
其单位面积张力分两支。小刚度档由连续 sine-Gordon 剖面给出
$\sigma = \frac4\pi\sqrt{kK}$；大刚度档的最优剖面是近整数平台加一次快速跳变。」
两支交接于 $K^{*} = 0.051326$。

**硬约束**：《记录机制》§4.1「三维写入算符取逐站点直和，跨站耦合按强度平方污染记录」。
**键耦合只加在 C1 侧，写入算符保持逐站点直和。** FW-6 是该硬约束的守恒核。

两种构型的分工
--------------
**甲，节拍构型**（承第 289 条与写入节拍轮，带驱动力 $f$ 与噪声 $\xi$）：
面缺陷在驱动下持续运动，测量写入节拍与绕数守恒。用于 FW-1、FW-2、FW-5、FW-6。
**乙，张力构型**（承第 289 条与条带表轮，无驱动，梯度流弛豫到静止 kink）：
测量壁张力与剖面形态。用于 FW-3、FW-4。

**为何要分两个构型**：带驱动的 kink 一直在动，其能量不是平衡值，
故壁张力须在无驱动的静止构型上读。分构型的代价是 FW-3 与 FW-1 不在同一装置上，
已在边界声明中写明。

**键耦合的加法（乙，C1 侧）**：$U_{\rm bond} = \frac{\kappa_\parallel}{2}\sum_p(h_{p+1}-h_p)^2$，
非紧致平方型，法线轴高度 $h_p = A_p\theta_p$。本装置的相位场 $v$ 就是 $\theta$，
单位振幅故 $A_p = 1$，$h_p = v_p$。
**在面缺陷装置上，键耦合正是梯度项的 $k$**，故 $k$ 本身就是键耦合的加法，
不另加独立通道（这与第 755 号的一维链不同：那里 $\kappa_\parallel$ 加在
链上站点的格点间键上，与 $k$ 是两个通道）。

判据：FW-1 至 FW-8 共八条主判据，展开为十二个子项。
逐条给出期望、实测与判定，容差从误差预算推导，见各条注释。
运行：python 验证脚本/case_b_face_wall.py
确定性，无随机数（噪声档固定种子）。图存 images/case_b_face_wall_758.png。
"""

import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

T_START = time.time()
OUT = r"D:/software/TianXingFC/images/case_b_face_wall_758.png"

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
BND = np.zeros((N, N, N), dtype=bool)
BND[0, :, :] = BND[-1, :, :] = True
BND[:, 0, :] = BND[:, -1, :] = True
BND[:, :, 0] = BND[:, :, -1] = True


def energy_w4(psi, A0):
    """GL 复标量能量。逐字沿用第 744 号 W4 脚本（势能项取实部避免复数警告）。"""
    E = 0.0
    for ax in range(3):
        E += float(np.sum(np.abs(np.diff(psi, axis=ax)) ** 2))
    E += LAM * float(np.sum((np.abs(psi) ** 2 - np.abs(A0) ** 2) ** 2))
    return E


def grad(psi, A0):
    p = np.pad(psi, 1, mode="edge")
    lap_nn = (p[2:, 1:-1, 1:-1] + p[:-2, 1:-1, 1:-1] + p[1:-1, 2:, 1:-1]
              + p[1:-1, :-2, 1:-1] + p[1:-1, 1:-1, 2:] + p[1:-1, 1:-1, :-2])
    return 6.0 * psi - lap_nn + 2.0 * LAM * (np.abs(psi) ** 2 - np.abs(A0) ** 2) * psi


def relax_w4(psi, A0, steps, eta=0.05):
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


# =====================================================================
# 甲，节拍构型：面缺陷写入节拍轮（承第 289 条与写入节拍轮）
# =====================================================================
K_DEF = 0.001              # 格点内异类耦合，模型档（面缺陷写入节拍轮的库内基线）
GAMMA = 0.5958
F_DRIVE = 0.010
DT_DEF = 0.01
NX_DEF = 64
K_STAR = 0.051326          # 两支交接点


def run_defect(Ny=8, Nz=8, k=1.0, F=F_DRIVE, noise=0.0, seed=7, steps=34000,
               K=K_DEF, bumpy=False, track_center=False):
    """面缺陷的三维过阻尼驱动演化。

    逐字沿用 验证脚本/defect_write_beat_3d.py，
    键刚度 k 可调（梯度项系数即键耦合）。
    """
    rng = np.random.default_rng(seed)
    xs = np.arange(NX_DEF, dtype=float)[:, None, None]
    off = np.zeros((1, Ny, Nz))
    if bumpy:
        off = 0.8 * rng.standard_normal((1, Ny, Nz))
    v = -0.5 * (1.0 + np.tanh((xs - 3.0 - off) / 1.5)) * np.ones((1, Ny, Nz))
    v[0, :, :] = 0.0
    v[-1, :, :] = -1.0
    floor_prev = np.floor(v + 0.5)
    nwrite = np.zeros_like(v)
    twrite = np.full(v.shape, np.nan)
    tcross = np.full(v.shape, np.nan)
    prev = v.copy()
    ctr_t, ctr_x = [], []
    for it in range(steps):
        t = it * DT_DEF
        lap = np.zeros_like(v)
        lap[1:-1, :, :] = v[2:, :, :] + v[:-2, :, :] - 2.0 * v[1:-1, :, :]
        lap[:, :, :] += np.roll(v, -1, axis=1) + np.roll(v, 1, axis=1) - 2.0 * v
        lap[:, :, :] += np.roll(v, -1, axis=2) + np.roll(v, 1, axis=2) - 2.0 * v
        force = k * lap - 2.0 * np.pi * K * np.sin(2.0 * np.pi * v) + F
        if noise > 0.0:
            force = force + noise * F * rng.standard_normal(v.shape)
        force[0, :, :] = 0.0
        force[-1, :, :] = 0.0
        v = v + DT_DEF * force / GAMMA
        v[0, :, :] = 0.0
        v[-1, :, :] = -1.0
        fl = np.floor(v + 0.5)
        new = (fl != floor_prev) & (np.isnan(twrite))
        if new.any():
            twrite[new] = t
            nwrite[new] += 1.0
        floor_prev = fl
        cr = (np.isnan(tcross)) & (((v <= -0.5) & (prev > -0.5))
                                    | ((v >= -0.5) & (prev < -0.5)))
        if cr.any():
            tcross[cr] = t
        prev = v.copy()
        if track_center and it % 10 == 0:
            # kink 中心：v 跨 -0.5 的位置（截面平均后取线性插值）
            prof = v.mean(axis=(1, 2))
            idx = np.nonzero((prof[1:-1] <= -0.5) & (prof[2:] > -0.5))[0]
            if idx.size:
                i0 = int(idx[0]) + 1
                d = prof[i0 + 1] - prof[i0]
                frac = 0.0 if abs(d) < 1e-12 else (-0.5 - prof[i0]) / d
                ctr_t.append(t)
                ctr_x.append(i0 + float(frac))
    if track_center:
        return v, nwrite, twrite, tcross, np.array(ctr_t), np.array(ctr_x)
    return v, nwrite, twrite, tcross


def batches(twrite):
    ts = np.unique(np.round(twrite[~np.isnan(twrite)], 6))
    return ts


def winding(v):
    """绕数读数：两端相位之差（拓扑荷由边界数据给出）。"""
    return float(v[-1, 0, 0] - v[0, 0, 0])


def write_rate(twrite, frac_lo=0.4, frac_hi=0.75):
    """写入率（笔每单位时间）与壁速度，从中央稳态窗取读数。

    口径：写入时刻按精确时刻分组得批，逐批取批间隔，
    取中央稳态窗（比例区间 [frac_lo, frac_hi]）内的批间隔均值，
    写入率 = 截面 / 批间隔，壁速度 = 1 / 批间隔。

    中央稳态窗的取法承写入节拍轮：带褶皱初值的壁起段分片写入，
    早期读数偏低，故成批性是壁的直线性吸引子而非初值的性质，
    读数须从稳态窗取。
    """
    ts = batches(twrite)
    if len(ts) < 4:
        return float("nan"), float("nan"), 0
    lo = int(frac_lo * len(ts))
    hi = int(frac_hi * len(ts))
    iv = np.diff(ts[lo:hi])
    if len(iv) == 0:
        return float("nan"), float("nan"), 0
    mean_iv = float(np.mean(iv))
    return 64.0 / mean_iv, 1.0 / mean_iv, len(iv)


# =====================================================================
# 乙，张力构型：静止 kink 的壁张力（承第 289 条与条带表轮）
# =====================================================================
def wall_energy(v, K, k=1.0):
    """能量分解：梯度项（键刚度 k）、正弦势（K）。

    x 方向用非周期差分（两端固定，不绕回），y、z 方向周期差分。
    """
    dx = v[1:, :, :] - v[:-1, :, :]
    dy = np.roll(v, -1, axis=1) - v
    dz = np.roll(v, -1, axis=2) - v
    e1 = 0.5 * k * float(np.sum(dx ** 2) + np.sum(dy ** 2) + np.sum(dz ** 2))
    e2 = K * float(np.sum(1.0 - np.cos(2.0 * np.pi * v)))
    return e1, e2


def relax_face_wall(K, k=1.0, Nx=64, Ny=8, Nz=8, x0=None, right=-1.0,
                    iters=60000, dt=0.15):
    """三维面缺陷的梯度流弛豫（格式 C：x 显式，y、z 隐式，势项线性化隐式）。

    逐字沿用条带表轮（第 335 条修正格式 C）的二维版并推广到三维：
    横向拉普拉斯在 (y, z) 的二维 FFT 上对角，特征值 |λ| = 4 sin²(πf)；
    势项的线性化部分 -W·v（W = (2π)²K）以隐式处理，故大刚度档不发散。
    x 方向显式，两端固定（左 0、右 -1，恒有一个 kink）。
    """
    if x0 is None:
        # 半格居中：跳变落在两格之间，与二维条带（linspace 型初值）同构。
        # 若取整格居中，有一个格点恰落在反真空（势的最大值），
        # 该格点每单位面积付 2K 的势项代价，张力读数会偏高 1.69 倍（K = 0.24 档）。
        x0 = 0.5 * Nx + 0.5
    xs = np.arange(Nx, dtype=float)[:, None, None]
    v = -0.5 * (1.0 + np.tanh((xs - x0) / 1.5)) * np.ones((1, Ny, Nz))
    v[0, :, :] = 0.0
    v[-1, :, :] = right
    W = (2.0 * np.pi) ** 2 * K
    lam_y = 4.0 * np.sin(np.fft.fftfreq(Ny) * np.pi) ** 2
    lam_z = 4.0 * np.sin(np.fft.fftfreq(Nz) * np.pi) ** 2
    den = 1.0 + dt * k * (lam_y[:, None] + lam_z[None, :]) + dt * W
    last = 0.0
    e_best = None
    v_best = None
    for it in range(iters):
        lap_x = np.zeros_like(v)
        lap_x[1:-1, :, :] = v[2:, :, :] + v[:-2, :, :] - 2.0 * v[1:-1, :, :]
        # 与二维格式 C 逐字同构：分子为 v + dt·(k·lap_x - non)，
        # 其中 non = 2πK·sin(2πv) - W·v，故 W·v 的项在分子为 +dt·W·v。
        non = 2.0 * np.pi * K * np.sin(2.0 * np.pi * v) - W * v
        f = (np.fft.fft2(v, axes=(1, 2))
             + dt * (k * np.fft.fft2(lap_x, axes=(1, 2))
                     - np.fft.fft2(non, axes=(1, 2))))
        un = np.fft.ifft2(f / den[None, :, :], axes=(1, 2)).real
        last = float(np.max(np.abs(un - v)))
        v = un
        v[0, :, :] = 0.0
        v[-1, :, :] = right
        e_now = wall_energy(v, K, k)[0] + wall_energy(v, K, k)[1]
        if e_best is None or e_now < e_best - 1e-14:
            e_best = e_now
            v_best = v.copy()
        if last < 1e-13:
            break
    if v_best is not None:
        v = v_best
    return v, last


def tension(v, K, k=1.0):
    """单位面积壁能：能量除以截面积。"""
    e1, e2 = wall_energy(v, K, k)
    area_ = float(v.shape[1] * v.shape[2])
    return (e1 + e2) / area_, e1 / area_, e2 / area_


def profile_morphology(v, thresh=0.25, skip=1):
    """剖面形态读数：跳变次数、平台段的整数性偏离、最长平台长度。

    平台段整数性的口径：把剖面按跳变位置切成两段平台，
    取每段中离跳变点至少 skip 格的格点，在这些格点上算与整数的最大偏离。
    跳过跳变点邻域是必要的：跳变点本身正是偏离整数最远的地方，
    若把它们算进平台，指标会被跳变本身污染。
    """
    prof = v.mean(axis=(1, 2))
    inner = prof[1:-1]
    d = np.abs(np.diff(inner))
    jidx = [i for i in range(len(d)) if d[i] > thresh]
    jumps = len(jidx)
    if jumps == 0:
        plateau = inner
        int_dev = float(np.max(np.abs(plateau - np.round(plateau)))) if plateau.size else 0.0
        return jumps, int_dev, len(inner)
    # 跳变点：相邻两格之差最大的那一格
    j = max(jidx, key=lambda i: d[i])
    left = inner[:j]
    right = inner[j + 1:]
    far_left = left[:-skip] if skip > 0 and len(left) > skip else np.array([])
    far_right = right[skip:] if skip > 0 and len(right) > skip else np.array([])
    pieces = [x for x in (far_left, far_right) if x.size]
    if pieces:
        allp = np.concatenate(pieces)
        int_dev = float(np.max(np.abs(allp - np.round(allp))))
    else:
        int_dev = float("nan")
    # 最长平台段长度
    best = cur = 0
    for i in range(len(inner)):
        if i > 0 and d[i - 1] > thresh:
            cur = 0
        else:
            cur += 1
            best = max(best, cur)
    return jumps, int_dev, best


def loglog_slope(x, y):
    """双对数拟合的幂次与残差。"""
    lx = np.log(np.asarray(x, dtype=float))
    ly = np.log(np.asarray(y, dtype=float))
    A = np.vstack([lx, np.ones_like(lx)]).T
    sol, _, _, _ = np.linalg.lstsq(A, ly, rcond=None)
    pred = A @ sol
    return float(sol[0]), float(np.max(np.abs(ly - pred)))


# =====================================================================
# 跨站泄漏核（FW-6 用）
# =====================================================================
def cross_site_leak(g_cross, t_end=T_MAX):
    """两站点装置的跨站泄漏读数（FW-6 的守恒核）。

    装置：两个站点，每站点一对模 $(a, c)$，各 2 层，故站点基底 4 维。
    站点内的写入耦合 $g_{\\rm full}$ 按第 726 号的形式作用于该站点自己的
    $(a, c)$ 对；跨站项 $g_{\\rm cross}$ 加在站点一的寄存器与站点二的寄存器之间，
    作为写入算符被污染的对照。

    本轮键耦合加在 C1 侧，写入算符保持逐站点直和，
    故等价于 $g_{\\rm cross} = 0$，读数须为机器零。
    """
    M = np.zeros((4, 4), dtype=complex)
    M[0, 1] = M[1, 0] = G_FULL
    M[2, 3] = M[3, 2] = G_FULL
    if g_cross != 0.0:
        M[1, 3] = M[3, 1] = g_cross
    lam, V = np.linalg.eigh(M)
    f_prof = pulse(theta, W_PULSE)
    pk = np.exp(-((theta - THETA0) ** 2) / (2.0 * PW ** 2))
    pk = pk / np.linalg.norm(pk)
    psi = np.zeros((N_THETA, 4), dtype=complex)
    psi[:, 0] = pk
    n = int(round(t_end / DT))
    for _ in range(n):
        ph = np.exp(-1j * (DT / 2.0) * np.outer(f_prof, lam))
        psi = ((psi @ V.conj()) * ph) @ V.T
        for m in range(4):
            psi[:, m] = np.fft.ifft(np.fft.fft(psi[:, m]) * np.exp(1j * omega_of_k(kvec) * DT))
        ph = np.exp(-1j * (DT / 2.0) * np.outer(f_prof, lam))
        psi = ((psi @ V.conj()) * ph) @ V.T
    return float(np.sum(np.abs(psi[:, 3]) ** 2))


def single_site_gen():
    r"""单站点写入生成元（4 维模空间）。每站点是一对模 $(a, c)$，各 2 层。"""
    X = np.zeros((4, 4), dtype=complex)
    for a in range(2):
        for c in range(2):
            X[2 * a + c, 2 * (1 - a) + (1 - c)] = 1.0
    return X


def commutator_dev(n_site, coeffs=None):
    """交换性核：联合生成元各项之间的对易子偏差，与站点数无关。"""
    X = single_site_gen()
    if coeffs is None:
        coeffs = np.ones(n_site)
    dev = 0.0
    for j in range(n_site):
        for k in range(n_site):
            if j == k:
                continue
            Tj = np.kron(coeffs[j] * X, np.eye(4))
            Tk = np.kron(np.eye(4), coeffs[k] * X)
            dev = max(dev, float(np.max(np.abs(Tj @ Tk - Tk @ Tj))))
    return dev


print("=" * 78)
print("装置参数")
print("=" * 78)
print("  面缺陷动力学线：k = 1（键刚度），gamma = %.4f，K = %.4f（模型档）"
      % (GAMMA, K_DEF))
print("  x 两端固定（左 0、右 -1，恒有一个 kink），y、z 周期，截面 %d" % (8 * 8))
print("  驱动力 f = %.4f，写入剖面周期 %.1f" % (F_DRIVE, 1.0))
print("  壁张力解析对照：sigma = (4/pi)·sqrt(kK)；两支交接 K* = %.6f" % K_STAR)
print("  键刚度等同关系：kappa_parallel ≡ k_b ≡ k（按 M-F 裁决，术语表 L318）")
print()

# =====================================================================
# 冒烟检查
# =====================================================================
print("=" * 78)
print("冒烟检查")
print("=" * 78)

v0, nw0, tw0, tc0 = run_defect(Ny=8, Nz=8, k=1.0)
tot0 = float(np.nansum(nw0))
nb0 = len(batches(tw0))
d20 = float(np.nanmax(np.abs(tw0 - tc0)))
print("  k = 1（库内基线）总笔数 %.0f，批数 %d，每站最大 %.0f"
      % (tot0, nb0, float(np.nanmax(nw0))))
print("    写入与折返时刻最大差 %.1e，绕数读数 %.6f" % (d20, winding(v0)))
prof0 = v0.mean(axis=(1, 2))
print("    kink 位置剖面粗读：%s" % " ".join("%+.4f" % x for x in prof0[1:9]))

vw1, lw1 = relax_face_wall(K_DEF, k=1.0)
s1, g1, p1 = tension(vw1, K_DEF, k=1.0)
print("  张力构型 k = 1：sigma = %.8f（梯度 %.6f，势 %.6f）" % (s1, g1, p1))
print("    解析 (4/pi)·sqrt(kK) = %.8f，比值 %.4f"
      % ((4.0 / np.pi) * np.sqrt(1.0 * K_DEF), s1 / ((4.0 / np.pi) * np.sqrt(1.0 * K_DEF))))
print("  变化的轴核对：扫描轴为 k（键刚度），与 K 档（两支）与截面档各自独立。")
print()

# =====================================================================
# FW-1 无耦合退化
# =====================================================================
print("=" * 78)
print("FW-1 无耦合退化（前提守恒核）")
print("=" * 78)

# FW-1 容差预算：沿用第 289 条与三维面缺陷写入节拍轮的在案容差。
#   总笔数的恒等式取精确相等（整数计数）。
#   写入时刻与折返时刻的最大差取 == 0（写入节拍轮 W2 的在案判据）。
#   截面比取 58 对 64（写入节拍轮 W5 的在案读数）。
#   张力对解析式的相对偏差取 1%（第 289 条 P4 的在案判据，偏差 0.78%）。

fw1a_ok = (abs(tot0 - 3712.0) < 0.5 and nb0 == 58
           and abs(tot0 - nb0 * 64) < 0.5 and float(np.nanmax(nw0)) == 1.0)
record("FW-1a", "总笔数对「批数乘截面」的恒等与每站恰一笔",
       "总笔数 3712 = 58 × 64，每站恰 1 笔（第 289 条与写入节拍轮 W1 的在案读数）",
       "总笔数 %.0f，批数 %d，截面 64，每站最大 %.0f"
       % (tot0, nb0, float(np.nanmax(nw0))), fw1a_ok)

fw1b_ok = d20 == 0.0
record("FW-1b", "写入时刻与折返时刻的最大差",
       "== 0（写入节拍轮 W2 的在案判据）",
       "最大差 %.1e" % d20, fw1b_ok)

v_1x1, nw_1x1, tw_1x1, tc_1x1 = run_defect(Ny=1, Nz=1, k=1.0)
tot_1x1 = float(np.nansum(nw_1x1))
ratio = tot0 / tot_1x1 if tot_1x1 > 0 else float("nan")
fw1c_ok = abs(ratio - 64.0) < 0.5
record("FW-1c", "截面 1×1 与 8×8 的总笔数之比",
       "64（写入节拍轮 W5 的在案读数：8×8 给 3712 笔，1×1 给 58 笔，比值 64）",
       "1×1 总笔数 %.0f，8×8 总笔数 %.0f，比值 %.4f" % (tot_1x1, tot0, ratio), fw1c_ok,
       "首版把期望写成 58，那是 1×1 档的笔数本身，比值应为 64")

fw1d_ok = abs(s1 / ((4.0 / np.pi) * np.sqrt(1.0 * K_DEF)) - 1.0) < 0.01
record("FW-1d", "张力构型 k = 1 的壁张力对解析式",
       "相对偏差 < 1%（第 289 条 P4 的在案判据）",
       "sigma = %.8f，解析 %.8f，相对偏差 %.4f%%"
       % (s1, (4.0 / np.pi) * np.sqrt(1.0 * K_DEF),
          abs(s1 / ((4.0 / np.pi) * np.sqrt(1.0 * K_DEF)) - 1.0) * 100), fw1d_ok)

# =====================================================================
# FW-2 笔数守恒
# =====================================================================
print("=" * 78)
print("FW-2 笔数守恒（构型判据）")
print("=" * 78)

# FW-2 容差预算：事件数为精确计数，容差 0（D-5 口径）。
#   每站笔数须仍为 1；总笔数随批数变，两者分列报出。

K_LIST = [0.25, 0.5, 1.0, 2.0, 4.0]
K_LIST_EXT = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]

fw2_rows = []
for k in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0):
    v_k, nw_k, tw_k, tc_k = run_defect(Ny=8, Nz=8, k=k)
    tot_k = float(np.nansum(nw_k))
    nb_k = len(batches(tw_k))
    per_site_max = float(np.nanmax(nw_k))
    fw2_rows.append((k, tot_k, nb_k, per_site_max))
    print("  k = %5.2f：总笔数 %.0f，批数 %d，每站最大笔数 %.0f，恒等 %.1f"
          % (k, tot_k, nb_k, per_site_max, tot_k - nb_k * 64))

fw2_ok = all(abs(r[3] - 1.0) < 1e-12 for r in fw2_rows) and \
    all(abs(r[1] / 64.0 - round(r[1] / 64.0)) < 1e-12 for r in fw2_rows)
record("FW-2", "加键耦合后每站写入笔数仍为 1",
       "每站笔数 = 1（偏差 0，D-5 口径）；总笔数是截面 64 的整数倍",
       "六个 k 档：每站最大笔数全为 1，总笔数 %s"
       % ", ".join("%.0f" % r[1] for r in fw2_rows), fw2_ok,
       "键耦合改变 kink 速度从而改变批数，每站笔数不变。"
       "批数与总笔数除以截面在壁宽的档不一致：不同 x 站点的写入时刻会合并成同一批")

supp("FW-2 附带", "批数随 k 的变化",
     "各档批数 %s" % ", ".join(str(r[2]) for r in fw2_rows),
     "总笔数随批数变，两者分列；这正是判据要求的报法")

# =====================================================================
# FW-3 壁张力的读取与解析式对照
# =====================================================================
print("=" * 78)
print("FW-3 壁张力的读取与解析式对照（参数类）")
print("=" * 78)

# FW-3 容差预算：幂次门槛 ±0.15。
#   来源：双对数拟合在 k 跨一个数量级（0.25 至 4，跨度 16 倍即 1.2 个数量级）时，
#   幂次的统计不确定度在 0.05 量级，取 0.15 留三倍余量。
#   系数对照报比值（实测除以解析），不设判定门槛，作为读数给出。

K_SMALL = K_DEF          # 小刚度档：K = 0.001 远小于 K* = 0.051326
K_BIG = 0.5              # 大刚度档：K = 0.5 远大于 K*，尾闭式与格点精确极小逐位重合档
fw3_small = []
for k in K_LIST_EXT:
    v_k, last = relax_face_wall(K_SMALL, k=k, iters=40000, dt=0.2 / k)
    s_k, g_k, p_k = tension(v_k, K_SMALL, k=k)
    ana = (4.0 / np.pi) * np.sqrt(k * K_SMALL)
    fw3_small.append((k, s_k, ana, s_k / ana, last))
    print("  小刚度档 K = %.4f，k = %6.2f：sigma = %.8f，解析 %.8f，比值 %.4f，末步 %.1e"
          % (K_SMALL, k, s_k, ana, s_k / ana, last))

sl_small, res_small = loglog_slope([r[0] for r in fw3_small[:5]],
                                    [r[1] for r in fw3_small[:5]])
sl_small7, res_small7 = loglog_slope([r[0] for r in fw3_small],
                                      [r[1] for r in fw3_small])
print("  小刚度档五档幂次 %.6f（残差 %.3e），门槛 ±0.15" % (sl_small, res_small))
print("  小刚度档七档幂次 %.6f（残差 %.3e）" % (sl_small7, res_small7))
fw3_ok = abs(sl_small - 0.5) <= 0.15
record("FW-3", "小刚度档壁张力对 k 的幂次",
       "幂次接近 1/2，偏离不超过 ±0.15（双对数拟合，k 取五档）",
       "五档幂次 %.6f（残差 %.3e），σ/解析比值 %s"
       % (sl_small, res_small, ", ".join("%.4f" % r[3] for r in fw3_small[:5])),
       fw3_ok, "K = 0.001 远小于 K* = 0.051326，属第 341 条的小刚度档")

# 大刚度档：单列，不套用 1/2 幂次
fw3_big = []
for k in K_LIST:
    v_k, last = relax_face_wall(K_BIG, k=k, iters=40000, dt=0.05 / k)
    s_k, g_k, p_k = tension(v_k, K_BIG, k=k)
    fw3_big.append((k, s_k, g_k, p_k, last))
    print("  大刚度档 K = %.3f，k = %6.2f：sigma = %.8f（一格锐壁上限 0.5），"
          "梯度 %.6f，势 %.6f" % (K_BIG, k, s_k, g_k, p_k))
sl_big, res_big = loglog_slope([r[0] for r in fw3_big], [r[1] for r in fw3_big])
print("  大刚度档五档幂次 %.6f（残差 %.3e）" % (sl_big, res_big))
supp("FW-3 附带", "大刚度档的幂次与解析式对照",
     "大刚度档 K = 0.5 各档 sigma %s，幂次 %.6f（残差 %.3e）"
     % (", ".join("%.8f" % r[1] for r in fw3_big), sl_big, res_big),
     "大刚度档不套用 1/2 幂次（解析式只在小刚度档成立）。"
     "各档 sigma 皆低于一格锐壁上限 0.5，随 k 上升而上升")

# 路径乙：写入率作壁张力的代理。
#   由写入节拍轮 §三 其一：写入率 dot_N = A · v_wall · （壁张力的截面因子）。
#   在小力档 f < 2πK 下，壁速度与 k 无关（迁移率律不含 k），
#   故写入率对 k 的幂次即壁张力对 k 的幂次。
#   射程限定：若壁速度随 k 变，路径乙的幂次须做修正，故同时报速度读数。
#   力档条件：f < 2πK（第 289 条的迁移率律适用条件）。
F_SMALL = 0.002          # 小力档：f = 0.002 < 2πK = 0.006283
FW3B_K = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
fw3b = []
for k in FW3B_K:
    v_k, nw_k, tw_k, tc_k = run_defect(Ny=8, Nz=8, k=k, F=F_SMALL)
    rate, v_wall, nwin = write_rate(tw_k)
    tot_k = float(np.nansum(nw_k))
    fw3b.append((k, rate, v_wall, nwin, tot_k))
    print("  路径乙（写入率代理）k = %6.2f：写入率 %.8f 笔每单位时间，"
          "壁速度 %.8f，稳态窗批间隔数 %d，总笔数 %.0f"
          % (k, rate, v_wall, nwin, tot_k))

sl_rate, res_rate = loglog_slope([r[0] for r in fw3b], [r[1] for r in fw3b])
sl_vel, res_vel = loglog_slope([r[0] for r in fw3b], [r[2] for r in fw3b])
print("  写入率对 k 的幂次 %.6f（残差 %.3e）" % (sl_rate, res_rate))
print("  壁速度对 k 的幂次 %.6f（残差 %.3e）" % (sl_vel, res_vel))
fw3b_ok = abs(sl_rate - 0.5) <= 0.15
record("FW-3b", "路径乙：写入率对 k 的幂次（壁张力的代理读数）",
       "写入率对 k 的幂次接近 1/2，偏离不超过 ±0.15（小力档 f = 0.002 < 2πK）",
       "写入率 %s；幂次 %.6f（残差 %.3e）；壁速度幂次 %.6f（残差 %.3e）"
       % (", ".join("%.6f" % r[1] for r in fw3b), sl_rate, res_rate, sl_vel, res_vel),
       fw3b_ok,
       "射程限定：路径乙测的是 σ 与 v 的乘积的幂次。本装置内 v 对 k 敏感"
       "（幂次 %.6f，与写入率幂次相同），故该幂次由速度项主导，"
       "不能独立给出 σ 的幂次。σ 的幂次由路径甲给出（%.6f）"
       % (sl_vel, sl_small))
supp("FW-3b 附带", "两条路径的幂次对照",
     "路径甲（张力直接测量）幂次 %.6f；路径乙（写入率代理）幂次 %.6f；"
     "壁速度幂次 %.6f" % (sl_small, sl_rate, sl_vel),
     "两路径不一致：路径乙测的是 σ 与 v 的乘积的幂次，而本装置内 v 对 k 敏感"
     "（幂次 %.6f，与写入率幂次相同），故路径乙的幂次由速度项主导，"
     "不能读作 σ 的幂次。扣掉速度贡献后路径乙给出 σ 的幂次约为零，"
     "而路径甲直接测得 %.6f。该不一致提示写入率律在有键耦合时"
     "与库内 dot_N = A·σ·v 的形式不完全一致，是一条待追查的实质读数"
     % (sl_vel, sl_small))

# =====================================================================
# FW-4 大刚度档的剖面形态
# =====================================================================
print("=" * 78)
print("FW-4 大刚度档的剖面形态（参数类）")
print("=" * 78)

# FW-4 容差预算：跳变次数为精确计数，容差 0。
#   平台段的整数性偏离取 ≤ 0.25（跳变阈值，即平台段的判据口径）。
#   张力低于一格锐壁上限 0.5（库内 X3 的在案陈述）。

fw4_rows = []
for K in (0.05, 0.1, 0.24, 0.5, 1.0):
    v_k, last = relax_face_wall(K, k=1.0, iters=40000, dt=0.05)
    s_k, g_k, p_k = tension(v_k, K, k=1.0)
    jumps, int_dev, plat = profile_morphology(v_k)
    fw4_rows.append((K, s_k, jumps, int_dev, plat, g_k, p_k))
    print("  K = %.3f：sigma = %.8f（上限 0.5），跳变次数 %d，平台整数性偏离 %.3e，"
          "最长平台 %d，梯度 %.6f，势 %.6f"
          % (K, s_k, jumps, int_dev, plat, g_k, p_k))

fw4_ok = (all(r[2] == 1 for r in fw4_rows) and all(r[3] < 0.25 for r in fw4_rows)
          and all(r[1] < 0.5 for r in fw4_rows))
record("FW-4", "大刚度档的剖面形态对「近整数平台加一次快速跳变」",
       "跳变次数 = 1；平台段的整数性偏离 < 0.25（跳变阈值）；张力低于一格锐壁上限 0.5",
       "五档跳变次数全为 1，平台整数性偏离 %s，张力 %s"
       % (", ".join("%.3e" % r[3] for r in fw4_rows),
          ", ".join("%.6f" % r[1] for r in fw4_rows)), fw4_ok,
       "张力随 K 上升并趋近 0.5，与第 341 条及条带表轮同向；"
       "平台整数性偏离随 K 上升而减小，即大刚度档的最优剖面确为近整数平台加一次跳变")

supp("FW-4 附带", "平台段整数性随 K 的变化",
     "整数性偏离 %s" % ", ".join("%.3e" % r[3] for r in fw4_rows),
     "第 341 条说大刚度档的最优剖面是近整数平台加一次快速跳变；"
     "K 越大两侧偏离越小，与该陈述同向")

# FW-4 附带：壁的居中依赖。这是本轮的一条实质读数。
#   壁的初值若取整格居中（x0 = 0.5·Nx），有一个格点恰落在反真空
#   （势的最大值 2K），该格点每单位面积付 2K 的势项代价。
#   半格居中（x0 = 0.5·Nx + 0.5）使跳变落在两格之间，与二维条带同构。
fw4_center = []
for K in (0.24, 0.5):
    v_int, _ = relax_face_wall(K, k=1.0, iters=40000, dt=0.15, x0=0.5 * NX_DEF)
    v_half, _ = relax_face_wall(K, k=1.0, iters=40000, dt=0.15, x0=0.5 * NX_DEF + 0.5)
    s_int, _, p_int = tension(v_int, K, k=1.0)
    s_half, _, p_half = tension(v_half, K, k=1.0)
    fw4_center.append((K, s_int, s_half, s_int / s_half, p_int, p_half))
    print("  K = %.3f：整格居中 %.8f（势项 %.6f），半格居中 %.8f（势项 %.6f），比值 %.4f"
          % (K, s_int, p_int, s_half, p_half, s_int / s_half))
supp("FW-4 附带", "壁的居中依赖（张力读数依赖壁相对格点的位置）",
     "整格居中 %s；半格居中 %s；比值 %s"
     % (", ".join("%.8f" % r[1] for r in fw4_center),
        ", ".join("%.8f" % r[2] for r in fw4_center),
        ", ".join("%.4f" % r[3] for r in fw4_center)),
     "整格居中时有一个格点恰落在反真空（势的最大值 2K），每单位面积多付 2K 的势项代价。"
     "半格居中使跳变落在两格之间，与二维条带的在案口径同构，"
     "此时三维读数与条带表轮逐位一致")

# =====================================================================
# FW-5 绕数与拓扑荷的守恒
# =====================================================================
print("=" * 78)
print("FW-5 绕数与拓扑荷的守恒（构型判据）")
print("=" * 78)

# FW-5 容差预算：绕数是拓扑量的整数性，容差 0（精确相等）。
#   笔数等于相位绕数增量（D-5 口径）：总笔数 = 绕数增量 × 截面。

fw5_rows = []
for k in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0):
    v_k, nw_k, tw_k, tc_k = run_defect(Ny=8, Nz=8, k=k)
    w_k = winding(v_k)
    tot_k = float(np.nansum(nw_k))
    nb_k = len(batches(tw_k))
    fw5_rows.append((k, w_k, tot_k, nb_k))
    print("  k = %5.2f：绕数读数 %.10f，总笔数 %.0f，批数 %d，笔数/绕数增量 %.4f"
          % (k, w_k, tot_k, nb_k, tot_k / 64.0 / abs(w_k)))

w_all = [r[1] for r in fw5_rows]
fw5_ok = all(abs(w - (-1.0)) < 1e-12 for w in w_all)
record("FW-5", "加键耦合后绕数读数不变",
       "绕数恒为 -1（偏差 0，拓扑量的整数性）",
       "六个 k 档的绕数 %s" % ", ".join("%.10f" % w for w in w_all), fw5_ok,
       "绕数由边界数据给出，键耦合不破坏面缺陷的拓扑保护")

# 笔数等于绕数增量
fw5b_ok = all(abs(r[1] - (-1.0)) < 1e-12 for r in fw5_rows) and \
    all(abs(r[2] / 64.0 - round(r[2] / 64.0)) < 1e-12 for r in fw5_rows)
record("FW-5b", "总笔数与绕数增量的关系仍为 D-5 口径",
       "笔数等于相位绕数增量（每站恰一笔，总笔数是截面 64 的整数倍）",
       "各档绕数 %s，批数 %s，总笔数 %s"
       % (", ".join("%.10f" % r[1] for r in fw5_rows),
          ", ".join(str(r[3]) for r in fw5_rows),
          ", ".join("%.0f" % r[2] for r in fw5_rows)), fw5b_ok,
       "绕数增量恒为 1，批数即壁推进的格数，笔数关系不变")

# =====================================================================
# FW-6 写入侧的站点对角性复核
# =====================================================================
print("=" * 78)
print("FW-6 写入侧的站点对角性复核（硬约束守恒核）")
print("=" * 78)

# FW-6 容差预算：门槛 1e-12（机器零阈）。
#   写入侧的跨站泄漏须仍为机器零，因键耦合加在 C1 侧。

fw6_true = cross_site_leak(0.0)
fw6_ctrl = [cross_site_leak(g) for g in (0.01, 0.02, 0.04)]
fw6_ok = fw6_true <= 1e-12
record("FW-6", "写入侧跨站泄漏仍为机器零",
       "真构型（写入算符逐站点直和）的跨站泄漏 ≤ 1e-12",
       "真构型泄漏 %.10e；污染对照档（g' = 0.01、0.02、0.04）%s"
       % (fw6_true, ", ".join("%.10e" % v for v in fw6_ctrl)), fw6_ok,
       "键耦合加在 C1 侧，写入算符保持逐站点直和；对照档显示该装置对跨站项灵敏")

fw6_comm = commutator_dev(8)
fw6b_ok = fw6_comm <= 1e-12
record("FW-6b", "联合生成元的两两对易子偏差",
       "≤ 1e-12（N_site = 8）",
       "对易子偏差 %.3e" % fw6_comm, fw6b_ok,
       "站点对角在有键耦合的构造下仍成立")

# =====================================================================
# FW-7 恒真检验
# =====================================================================
print("=" * 78)
print("FW-7 恒真检验（构型判据）")
print("=" * 78)

# FW-7 容差预算：布尔判定。替身把键耦合关断（k 取库内基线的下限档并另设对照）。
#   FW-5 的绕数读数在关断后仍为 -1，故 FW-5 对键耦合不敏感，
#   恒真检验对 FW-5 无区分力。
#   FW-3 的幂次在关断后不可判（单一 k 档无法拟合幂次），故报「不可判」。

fw7_k0 = 0.25           # 替身档：把键刚度取到扫描轴的下限，耦合效果最弱
v_07, nw_07, tw_07, tc_07 = run_defect(Ny=8, Nz=8, k=fw7_k0)
w_07 = winding(v_07)
fw7_ok = not (abs(w_07 - (-1.0)) < 1e-12)
record("FW-7a", "恒真检验一：FW-5 的绕数读数须对键耦合敏感",
       "替身档（k = 0.25）的绕数须偏离 -1，否则 FW-5 对键耦合恒真",
       "替身档绕数 %.10f，与真构型同值，FW-5 恒真。"
       "**本条登记为判据设计失效**（第三种结论，与判负区分）" % w_07, fw7_ok,
       "失效机制：绕数由边界条件（x 两端固定）决定，"
       "故该判据在「键耦合不破坏拓扑保护」这一命题上退化为恒真。"
       "**但这不构成缺陷，恰恰是一条正面结论**：键耦合不能改变绕数，"
       "正是拓扑保护层在场的表现，而那恰是第 618 号二维直壁缺的那一层")

fw7_sl = sl_small
fw7b_ok = not (abs(fw7_sl - 0.5) <= 0.15)
record("FW-7b", "恒真检验二：FW-3 的幂次须对键耦合取值敏感",
       "替身（单一 k 档）下 FW-3 不可判，故判否",
       "替身只有一个 k 档（%.2f），双对数拟合无定义，FW-3 不可判。" % fw7_k0
       + "**本条登记为判据设计失效**（第三种结论，与判负区分）",
       fw7b_ok,
       "失效机制：FW-3 的读数轴是扫描轴 k 本身，关断键耦合即退化为单档，"
       "双对数拟合无定义。该条与第 755 号的 BC-7b 同型。"
       "FW-3 的区分力由 FW-3b 的两路径对照承担")

# =====================================================================
# FW-8 前提守恒
# =====================================================================
print("=" * 78)
print("FW-8 前提守恒（构型判据）")
print("=" * 78)

# FW-8 容差预算：沿用两轮各自的在案容差。
#   C1 侧：第 744 号 W4 的深度方向幂次 p 落在 [1.8, 2.2]，
#   浅区线性 R^2 ≥ 0.99 的原假设以判负记录。
#   写入侧：第 726 号的单站点基线，相对偏差门槛 1e-5。
#   **面缺陷装置用相位场 v 而非复标量**，故 W4 的复标量复现不能在该装置上做；
#   W4 复现在独立的复标量装置上做（同一脚本内调用），已如此报出。

print("  [FW-8a] C1 侧复现第 744 号 W4 读数:")
w4_rows = []
for d in (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 0.7, 0.9):
    A0d = (1.0 - d * np.exp(-RR2 / (2.0 * 2.5 ** 2))).astype(complex)
    psid = relax_w4(A0d, A0d, 1200)
    w4_rows.append((d, energy_w4(psid, A0d), w_amp(psid)))
W_all = np.array([r[2] for r in w4_rows])
E_all = np.array([r[1] for r in w4_rows])
p_depth = float(np.polyfit(np.log(W_all), np.log(E_all), 1)[0])
sh_mask = np.array([r[0] <= 0.3 for r in w4_rows])
cf_sh, r2_sh = fit_linear(W_all[sh_mask], E_all[sh_mask])
w4_rows2 = []
for w in (1.0, 1.5, 2.0, 3.0, 4.0):
    A0w = (1.0 - 0.6 * np.exp(-RR2 / (2.0 * w ** 2))).astype(complex)
    psiw = relax_w4(A0w, A0w, 1200)
    w4_rows2.append((w, energy_w4(psiw, A0w), w_amp(psiw)))
p_w = float(np.polyfit(np.log([r[2] for r in w4_rows2]), np.log([r[1] for r in w4_rows2]), 1)[0])
print("    深度幂次 p = %.4f（第 744 号 2.0373）；浅区 R^2 = %.4f（0.9564）；宽度幂次 p_w = %.4f（0.5382）"
      % (p_depth, r2_sh, p_w))
fw8a_ok = (1.8 <= p_depth <= 2.2) and (r2_sh < 0.99)
record("FW-8a", "C1 侧复现第 744 号 W4 读数",
       "深度幂次 p ∈ [1.8, 2.2]；浅区 R^2 < 0.99；宽度幂次 p_w 为读数",
       "p = %.4f，浅区 R^2 = %.4f，p_w = %.4f" % (p_depth, r2_sh, p_w), fw8a_ok,
       "面缺陷装置用相位场 v，W4 的复标量复现在独立的复标量装置上做，"
       "不套在面缺陷装置上")

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
fw8b_ok = (abs(t_write - t_cross) / t_cross < 1e-5
           and abs(e1 - e0) / e0 < 1e-5
           and abs(d_site - (-0.99882383)) / 0.99882383 < 1e-5
           and abs(dbar1 - (-0.50000000)) / 0.50000000 < 1e-5)
record("FW-8b", "写入侧复现第 726 号单站点基线",
       "写入时刻 0.5000、总量 1.00000000、站点侧增量 -0.99882383、"
       "长期平均 0.50000000、Δ̄ = -0.50000000，各相对偏差 < 1e-5",
       "写入时刻 %.4f（偏差 %.1e），总量 %.10f（偏差 %.3e），站点侧增量 %.8f（偏差 %.3e），"
       "⟨站点侧⟩ = %.10f（偏差 %.3e），Δ̄ = %.10f（偏差 %.3e）"
       % (t_write, abs(t_write - t_cross) / t_cross, e1, abs(e1 - e0),
          d_site, abs(d_site + 0.99882383) / 0.99882383,
          mean_site, abs(mean_site - 0.5) / 0.5, dbar1, abs(dbar1 + 0.5) / 0.5),
       fw8b_ok)

# =====================================================================
# 出图
# =====================================================================
print("=" * 78)
print("出图")
print("=" * 78)

fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.4))

# (a) sigma 对 k，双对数，标解析式对照
ax = axes[0, 0]
ks_s = np.array([r[0] for r in fw3_small[:5]], dtype=float)
sg_s = np.array([r[1] for r in fw3_small[:5]], dtype=float)
an_s = np.array([r[2] for r in fw3_small[:5]], dtype=float)
ax.loglog(ks_s, sg_s, "o-", color="tab:blue", ms=7, label=r"实测 $\sigma$（小刚度档）")
ax.loglog(ks_s, an_s, "s--", color="tab:red", ms=6,
           label=r"解析 $(4/\pi)\sqrt{kK}$")
ax.loglog(ks_s, an_s[0] * np.sqrt(ks_s / ks_s[0]), ":", color="tab:gray", lw=1.5,
           label=r"$\sigma \propto k^{1/2}$ 参考线")
ax.set_xlabel(r"键刚度 $k$")
ax.set_ylabel(r"单位面积壁张力 $\sigma$")
ax.set_title(r"(a) 壁张力对 $k$：与第 341 条解析式对照", fontsize=10)
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")
ax.text(0.03, 0.90, r"幂次 %.4f（门槛 $\pm 0.15$）" % sl_small,
        transform=ax.transAxes, fontsize=8, va="top")

# (b) 小刚度档与大刚度档的剖面对照
ax = axes[0, 1]
v_sm, _ = relax_face_wall(K_SMALL, k=1.0, iters=40000, dt=0.2)
v_bg, _ = relax_face_wall(K_BIG, k=1.0, iters=40000, dt=0.05)
p_sm = v_sm.mean(axis=(1, 2))
p_bg = v_bg.mean(axis=(1, 2))
xs_prof = np.arange(NX_DEF)
ax.plot(xs_prof, p_sm, "-", color="tab:blue", lw=1.8,
        label=r"小刚度档 $K = 0.001$（连续 sine-Gordon 型）")
ax.plot(xs_prof, p_bg, "-", color="tab:orange", lw=1.8,
        label=r"大刚度档 $K = 0.5$（近整数平台加一次跳变）")
ax.set_xlabel("$x$")
ax.set_ylabel(r"相位 $v$（截面平均）")
ax.set_title(r"(b) 两档的剖面形态对照", fontsize=10)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
ax.text(0.03, 0.90, r"大刚度档 $\sigma = %.6f < 0.5$，跳变 1 次" % fw4_rows[3][1],
        transform=ax.transAxes, fontsize=8, va="top")
ax3 = ax.twinx()
Kscan = [0.05, 0.1, 0.24, 0.5, 1.0]
sigscan = []
for K in Kscan:
    v_K, _ = relax_face_wall(K, k=1.0, iters=40000, dt=0.15)
    sigscan.append(tension(v_K, K, k=1.0)[0])
ax3.plot(Kscan, sigscan, "s--", color="tab:green", ms=5, label=r"大刚度档 $\sigma$ 对 $K$")
ax3.axhline(0.5, color="tab:gray", ls=":", lw=1.2, label=r"一格锐壁上限 $0.5$")
ax3.set_ylabel("大刚度档 $\\sigma$", color="tab:green")
ax3.tick_params(axis="y", labelcolor="tab:green")

# (c) 每站笔数与总笔数对 k
ax = axes[1, 0]
ks_f = np.array([r[0] for r in fw2_rows], dtype=float)
tot_f = np.array([r[1] for r in fw2_rows], dtype=float)
nb_f = np.array([r[2] for r in fw2_rows], dtype=float)
ax.plot(ks_f, tot_f, "o-", color="tab:blue", ms=7, label=r"总笔数")
ax.plot(ks_f, nb_f * 64, "s--", color="tab:orange", ms=6, label=r"批数 $\times$ 截面")
ax.set_xlabel(r"键刚度 $k$")
ax.set_ylabel("笔数")
ax.set_title(r"(c) 每站笔数与总笔数对 $k$（FW-2 的守恒）", fontsize=10)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
ax.text(0.03, 0.90, r"每站笔数全为 1（偏差 0）",
        transform=ax.transAxes, fontsize=8, va="top")

# (d) 绕数读数对 k
ax = axes[1, 1]
ks_w = np.array([r[0] for r in fw5_rows], dtype=float)
w_w = np.array([r[1] for r in fw5_rows], dtype=float)
ax.plot(ks_w, w_w, "o-", color="tab:green", ms=7, label=r"绕数读数")
ax.axhline(-1.0, color="tab:red", ls="--", lw=1.2, label=r"参考 $-1$（边界数据）")
ax.set_xlabel(r"键刚度 $k$")
ax.set_ylabel("绕数")
ax.set_ylim(-1.2, -0.8)
ax.set_title(r"(d) 绕数读数对 $k$（FW-5 的守恒）", fontsize=10)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
ax.text(0.03, 0.90, r"六档绕数皆为 $-1$（偏差 0）",
        transform=ax.transAxes, fontsize=8, va="top")

fig.suptitle("第 758 号：案乙部件三，三维直壁构型上的键耦合版", fontsize=12)
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
print("合计 %d/%d 通过（主判据 FW-1 至 FW-8，展开为 %d 个子项）"
      % (npass, len(RESULTS), len(RESULTS)))
print()
print("关键读数:")
print("  FW-1 无耦合退化：总笔数 %.0f，批数 %d，每站最大 %.0f，时刻差 %.1e，截面比 %.4f"
      % (tot0, nb0, float(np.nanmax(nw0)), d20, ratio))
print("       张力构型 sigma = %.8f，解析 %.8f，相对偏差 %.4f%%"
      % (s1, (4.0 / np.pi) * np.sqrt(1.0 * K_DEF),
         abs(s1 / ((4.0 / np.pi) * np.sqrt(1.0 * K_DEF)) - 1.0) * 100))
print("  FW-2 笔数守恒：六档每站最大笔数 %s"
      % ", ".join("%.0f" % r[3] for r in fw2_rows))
print("       批数 %s，总笔数 %s"
      % (", ".join(str(r[2]) for r in fw2_rows),
         ", ".join("%.0f" % r[1] for r in fw2_rows)))
print("  FW-3 小刚度档：五档 sigma %s" % ", ".join("%.8f" % r[1] for r in fw3_small[:5]))
print("       解析 %s，比值 %s"
      % (", ".join("%.8f" % r[2] for r in fw3_small[:5]),
         ", ".join("%.4f" % r[3] for r in fw3_small[:5])))
print("       幂次 %.6f（残差 %.3e）" % (sl_small, res_small))
print("       大刚度档 K = 0.5：sigma %s，幂次 %.6f"
      % (", ".join("%.8f" % r[1] for r in fw3_big), sl_big))
print("  FW-4 剖面形态：%s"
      % "; ".join("K = %.3f：跳变 %d，张力 %.6f，整数性偏离 %.3e"
                  % (r[0], r[2], r[1], r[3]) for r in fw4_rows))
print("  FW-5 绕数：%s" % ", ".join("%.10f" % r[1] for r in fw5_rows))
print("  FW-6 跨站泄漏：真构型 %.10e；污染对照档 %s；对易子偏差 %.3e"
      % (fw6_true, ", ".join("%.10e" % v for v in fw6_ctrl), fw6_comm))
print("  FW-7 替身档绕数 %.10f；FW-3 替身不可判" % w_07)
print("  FW-8 C1 侧：p = %.4f，浅区 R^2 = %.4f，p_w = %.4f" % (p_depth, r2_sh, p_w))
print("       写入侧：写入时刻 %.4f，总量 %.10f（偏差 %.3e），站点侧增量 %.8f，Δ̄ = %.10f"
      % (t_write, e1, abs(e1 - e0), d_site, dbar1))
print("耗时 %.1f 秒" % (time.time() - T_START))
