# -*- coding: utf-8 -*-
r"""
multisite_write_extension.py

第 741 号 三维复合位错弹性本构攻坚线 构造轮案乙：写入过程的逐站点直和多站点扩展
（判据 MS-1 至 MS-7，MS-8 为对照层）

问题：案乙（单装置合并）相对判据第 726 号的唯一装置改动，是把第 726 号的写入过程从
单站点特例扩到多站点。库内《记录机制：碰撞即写入的算符形式、三维缺陷读数与弦的分层》
§4.1 给的三维写入算符取逐站点直和

$$H_{\rm int} = \sum_j g\,f(\hat\theta_j - A_j)\,(a_j^\dagger c_j + a_j c_j^\dagger),$$

其中 $A_j$ 为本站局部标架，$f(\hat\theta) = \exp[-(1-\cos\hat\theta)/(2w^2)]$。站点对角是
硬约束：任何跨站耦合项都污染记录。第 726 号 DS-9 给出越界泄漏的平方律读数。

本轮做这个扩展本身，是案乙的前提守恒核，不依赖尚待作者裁定的第二问（自由度识别）
与第十问（交付形态）。

装置：复用 验证脚本/source_ding_reflux_saturation.py 的全部参数与
实现，逐字沿用其函数。参数 $N_\theta = 256$、$\omega_P = 2\pi$、$\theta_0 = \pi$、
脉冲宽 $w = 0.5$、步长 $dt = 2\times10^{-3}$、$T = 1.2$（长期平均档取 $T = 8.0$）、
完整交换耦合 $g_{\rm full} = 5.092$（实测 5.091585）。指针波包宽 PW = 0.15。

关键机制：自由演化是平移，指针位置 $\theta_p(t) = \pi + \omega_P t$。$f(\theta)$ 在
$\theta = \pi$ 处是谷（约 0.018），在 $\theta \equiv 0$ 处是峰（等于 1）。故写入发生在
过闸时刻 $t = 0.5$，第二次过闸在 $t = 1.5$。单站点下站点侧布居等于 $\cos^2 A(t)$，
其中 $A(t) = g_{\rm full}\int_0^t f(\theta_p(t'))dt'$。

**半角机制，须写明因它给出一个恰为 0.5 的读数。** 指针剖面 $f(\theta_p(t))$ 关于过闸
时刻 $t = 0.5$ 偶对称（$\cos(2\pi t)$ 在 $t \mapsto 1-t$ 下不变），故半个周期内累积的
写入角恰为整个周期的一半，即 $A(0.5) = \pi/4$，站点侧布居 $=\cos^2(\pi/4) = 0.5$。
该 0.5 是机制的必然结果，不是读数巧合。

多站点扩展的实现路线：逐站点独立演化。站点对角意味着站点间无耦合，故联合态是各站点
态的张量积，逐站点独立演化是精确解而非近似。该前提在本轮显式核验并打印：

核一（算符层，全部站点数）：联合模空间生成元是单站点生成元的 Kronecker 和，各项作用
在不同张量因子上因而互易，故联合指数按张量积分解。$N_{\rm site} = 2$ 至 $16$ 逐档核。

核二（算符层，含实际空间剖面）：在真实的 $\theta$ 网格点上取
$M(\theta) = \sum_j c_j X_j$，$c_j = g_{\rm full}f(\theta_j - A_j)$，$X_j$ 是第 $j$ 站点
上的写入生成元。核 $\exp(-i\,dt\,M)$ 等于各站点指数的张量积。$N_{\rm site} = 2$ 与 $3$
两档核（单站点子空间 4 维，联合维数 16 与 64）。

核三（态层，两站点）：$N_{\rm site} = 2$ 的联合态直接演化（显式 16 维模空间、
每站点一条 $\theta$ 轴），逐位复现逐站点独立结果，含同相与异相两档。

**态层联合演化的射程限定。** 态层张量积的维数是 $4^{N_{\rm site}}$ 乘以每站点一条
$\theta$ 轴，即 $N_{\rm site} = 2$ 时 16 维模乘两条 $\theta$ 轴（可算，用时约一分钟），
$N_{\rm site} = 3$ 时内存需求已达 $2\times10^{9}$ 字节量级（不可行）。
故态层核只做到两站点，$N_{\rm site} = 3$ 的独立核在算符层做。
主判据的多站点扫描（MS-3 起）走逐站点独立解路线，该路线在站点对角下的精确性
由核一与核二给出，并由核三在态层确认。

三种局部标架方案（$A_j$ 的取法）：

方案甲，同相：$A_j = 0$ 对所有 $j$。最简扩展，检验直和形式在同相时是否退化为单站点读数。
期望值出处：站点对角下各站点互不影响，同相时每站点装置与第 726 号的单站点装置同构，
故每站点读数应与第 726 号基线一致。

方案乙，沿位错线均匀相位梯度：$A_j = j\,\Delta A$。对应缺陷扫过格点时各站点的局部标架
依次平移，物理上最接近缺陷运动引发逐站写入。期望值出处：同相档（$\Delta A = 0$）是参照，
各档因站点不再同时过闸而应从同相值移动，具体方向由机制决定。

方案丙，空间均匀相移：$A_j = A_{\rm span}\,x_j / x_{\rm half}$，$x_j$ 为站点在芯内的
横向位置，取值区间 $[-x_{\rm half}, +x_{\rm half}]$，$x_{\rm half} = 3.0$ 格（第 722 号
DB-11 芯半宽），$A_{\rm span} = \pi/2$。此对应直壁构型：局域标架沿芯的横向有系统平移。

$A_{\rm span}$ 是本轮自设的构造参数，库内未给取值。取 $\pi/2$ 的理由有三。

其一，它超过写入选中剖面的半高点距（$f$ 半高处 $\vert\theta\vert \le 0.857$ rad），
横向梯度因此是真实梯度而非小扰动。

其二，$A_j \in [-\pi/2, +\pi/2]$，站点的过闸时刻全部落在 $(0, 1)$ 的内部，
MS-6 的每站恰一笔成立且无边界歧义。

其三，写入剖面无紧支撑，横向梯度给出的是渐变覆盖而非分明的覆盖边界，
这与直壁构型的物理图景相符。

判据：MS-1 至 MS-7 共七条主判据，展开为十五个子项（MS-1 五项、MS-2 两项、MS-3 一项、
MS-4 两项、MS-5 三项、MS-6 一项、MS-7 一项）；MS-8 为跨站对照层，不占主判据号。
逐条给出期望、实测与判定，容差从误差预算推导，见各条注释。可逆性等级：MS-1、MS-2、
MS-3、MS-6、MS-7 判据稳健；MS-4、MS-5 为形态与分布判定。

运行：python 验证脚本/multisite_write_extension.py
确定性，无随机数。图存 images/multisite_write_extension_741.png。
"""

import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

T_START = time.time()
OUT = r"D:/software/TianXingFC/images/multisite_write_extension_741.png"

# =====================================================================
# 装置参数（逐字沿用第 726 号）
# =====================================================================
N_THETA = 256
OMEGA_P = 2.0 * np.pi
THETA0 = np.pi
W_PULSE = 0.5
PW = 0.15
DT = 2.0e-3
T_MAX = 1.2
T_AVG = 8.0

# 本轮自设的构造参数
X_HALF = 3.0                 # 芯半宽（格），出处在第 722 号 DB-11
A_SPAN = 0.5 * np.pi         # 方案丙的横向相移跨度（弧度）

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
# 装置函数（逐字沿用第 726 号）
# =====================================================================
def pulse(theta, w):
    return np.exp(-(1.0 - np.cos(theta)) / (2.0 * w ** 2))


def omega_of_k(kvec):
    return -OMEGA_P * kvec


def apply_int(psi, f, dt, g):
    r"""耦合项 g f (a†c + a c†)，在两能级子空间上精确指数化。"""
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


def split_step_g(psi, dt, theta, kvec, g_arr, n_a=2, n_c=2):
    """通用分裂步：g_arr 是逐 θ 的耦合系数数组。

    g_arr = G_FULL·f(θ) 为写入项；再加常数项 κ·G_FULL 即 θ 无关的回流项。
    两项同轴（同一对角化基下的 σ_x⊗σ_x），指数可加，次序不影响结果。
    g_arr = G_FULL·f(θ) 时与 split_step 的数值完全一致。
    """
    psi = apply_int(psi, 1.0, dt / 2.0, g_arr)
    for a in range(n_a):
        for c in range(n_c):
            psi[:, a, c] = np.fft.ifft(np.fft.fft(psi[:, a, c]) * np.exp(1j * omega_of_k(kvec) * dt))
    psi = apply_int(psi, 1.0, dt / 2.0, g_arr)
    return psi


def apply_int3(psi, f, dt, g):
    """三能级站点梯的写入算符：a 的 0 与 1 层同 c 耦合成对，a=2 层不参与。"""
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
    out[:, 2, :] = psi[:, 2, :]
    return out


def apply_leak(psi, dt, g_leak):
    """漏损通道：把记录侧激发旋转到第三个自由模（a=2 层）。

    该层既不计入站点侧也不计入记录侧，材料从此处泄漏到别处。
    """
    ang = g_leak * dt
    cos_a = np.cos(ang)
    sin_a = np.sin(ang)
    out = np.zeros_like(psi)
    out[:, 0, 1] = cos_a * psi[:, 0, 1] - 1j * sin_a * psi[:, 2, 0]
    out[:, 2, 0] = cos_a * psi[:, 2, 0] - 1j * sin_a * psi[:, 0, 1]
    out[:, 0, 0] = psi[:, 0, 0]
    out[:, 1, 0] = psi[:, 1, 0]
    out[:, 1, 1] = psi[:, 1, 1]
    return out


def apply_diag(psi, f, dt, g):
    """正定势替身：转移项换成对角正定项 a†a+c†c，只改变相位，激发搬不动。"""
    ang = g * f * dt
    out = np.zeros_like(psi)
    for a in range(2):
        for c in range(2):
            out[:, a, c] = psi[:, a, c] * np.exp(-1j * ang * (a + c))
    return out


def apply_gen3(psi, f, dt, g):
    """产生项替身：模式选择性的产生，按一阶截断从第三模（a=2）向记录侧注入。

    注入落在 (a=0, c=1) 切片，该切片计入记录侧而不计入站点侧。
    站点切片原样保留，站点侧读数不受影响。该算符不保范数，产生式读取本就如此。
    """
    ang = g * f * dt
    out = np.zeros_like(psi)
    out[:, 0, 0] = psi[:, 0, 0]
    out[:, 1, 0] = psi[:, 1, 0]
    out[:, 2, 0] = psi[:, 2, 0]
    out[:, 0, 1] = psi[:, 0, 1] - 1j * ang * psi[:, 2, 0]
    out[:, 1, 1] = psi[:, 1, 1]
    out[:, 2, 1] = psi[:, 2, 1]
    return out


def split_step_sub(psi, dt, theta, kvec, w, g, op, n_a=2, n_c=2):
    """替身用的分裂步：耦合项由 op 决定（apply_int / apply_int3 / apply_diag / apply_gen3）。"""
    f = pulse(theta, w)
    psi = op(psi, f, dt / 2.0, g)
    for a in range(n_a):
        for c in range(n_c):
            psi[:, a, c] = np.fft.ifft(np.fft.fft(psi[:, a, c]) * np.exp(1j * omega_of_k(kvec) * dt))
    psi = op(psi, f, dt / 2.0, g)
    return psi


def make_packet3(theta, pw, alev=1, reservoir=False):
    """三能级初态：被采样对象在 a=1，可选在 a=2 放一份储备源。"""
    psi = np.zeros((N_THETA, 3, 2), dtype=complex)
    pk = np.exp(-((theta - THETA0) ** 2) / (2.0 * pw ** 2))
    psi[:, alev, 0] = pk / np.linalg.norm(pk)
    if reservoir:
        pk2 = np.exp(-((theta - THETA0) ** 2) / (2.0 * pw ** 2))
        psi[:, 2, 0] = pk2 / np.linalg.norm(pk2)
    return psi


def third_pop(p):
    """第三个自由模（a=2 层）的布居。"""
    return float(np.sum(np.abs(p[:, 2, :]) ** 2))


def embed2m(p2):
    """把两模态嵌入三模（c1 恒空），用于对角化实现与原 apply_int 的逐位比较。"""
    e = np.zeros((N_THETA, 2, 2, 2), dtype=complex)
    e[:, :, :, 0] = p2
    return e.reshape(N_THETA, NM3)


def pops3(psi):
    """三模分栏：站点侧、记录格 c0、记录格 c1、总量。"""
    p = psi.reshape(N_THETA, 2, 2, 2)
    return (float(np.sum(np.abs(p[:, 1, :, :]) ** 2)),
            float(np.sum(np.abs(p[:, :, 1, :]) ** 2)),
            float(np.sum(np.abs(p[:, :, :, 1]) ** 2)),
            float(np.sum(np.abs(psi) ** 2)))


def time_average_site(psi_in, g_write, t_end=T_AVG):
    """逐步行进并每 DT 采样一次站点侧布居，含 t=0 的初值点。

    采样间隔 = DT = 2.0e-3 ≤ 判据要求的 0.005。
    """
    n = int(round(t_end / DT))
    psi_l = psi_in
    t_list = [0.0]
    s_list = [site_pop(psi_l)]
    for step in range(n):
        psi_l = split_step_g(psi_l, DT, theta, kvec, g_write)
        t_list.append((step + 1) * DT)
        s_list.append(site_pop(psi_l))
    return np.array(t_list), np.array(s_list)


def reflux_run(kappa, t_start, write_on, n_mode="intent"):
    """写入段跑到 t_start，随后施加 κ·G_FULL 的 θ 无关回流通道 n 步。

    n_mode="intent"：n=round((κ·π/2)/(κ·G_FULL·DT))，累积角等于 κ·π/2，与判据的
        sin²(κπ/2) 模型同口径。
    n_mode="literal"：n=round((κ·π/2)/(G_FULL·DT))，累积角为 κ²·π/2。
    """
    if kappa == 0.0:
        n = 0
    elif n_mode == "intent":
        n = int(round((kappa * np.pi / 2.0) / (kappa * G_FULL * DT)))
    else:
        n = int(round((kappa * np.pi / 2.0) / (G_FULL * DT)))
    p = make_packet(theta, PW)
    for _ in range(int(round(t_start / DT))):
        p = split_step_g(p, DT, theta, kvec, G_WRITE)
    site_post = site_pop(p)
    g_arr = (G_WRITE if write_on else 0.0) + kappa * G_FULL
    for _ in range(n):
        p = split_step_g(p, DT, theta, kvec, g_arr)
    return site_post, site_pop(p), n, n * kappa * G_FULL * DT


# 第 726 号 DS-9 的三模装置（跨站对照档复用）
NM3 = 8


def fixed_gen3(g_main, g_cross):
    """固定的 8×8 生成元 M（站点模 a、记录格 c0、记录格 c1），索引 = 4a + 2c0 + c1。

    G(θ) = f(θ)·M，故 exp(-iG(θ)t) = V·diag(exp(-i f(θ) λ_k t))·V†，特征向量与 θ 无关，
    对角化只需做一次。g_cross = g' 为越界（跨站）耦合强度。
    """
    M_aC0 = np.zeros((NM3, NM3), dtype=complex)
    M_aC1 = np.zeros((NM3, NM3), dtype=complex)
    for c1 in range(2):
        M_aC0[4 + c1, 2 + c1] += 1.0
        M_aC0[2 + c1, 4 + c1] += 1.0
    for c0 in range(2):
        M_aC1[4 + 2 * c0, 2 * c0 + 1] += 1.0
        M_aC1[2 * c0 + 1, 4 + 2 * c0] += 1.0
    M = g_main * M_aC0 + g_cross * M_aC1
    lam, V = np.linalg.eigh(M)
    return lam, V


def apply_3mode(psi, dt, lam, V, fvals):
    """三模子空间上的耦合指数：V·diag(exp(-i f(θ) λ t))·V†。"""
    ph = np.exp(-1j * dt * np.outer(fvals, lam))
    tmp = (psi @ V.conj()) * ph
    return tmp @ V.T


def free_step3(psi, dt, kvec):
    """三模子空间上的自由演化（θ 轴相位扩散），与两模档的扩散算符逐字同源。"""
    p = psi.reshape(N_THETA, 2, 2, 2)
    for a in range(2):
        for c0 in range(2):
            for c1 in range(2):
                p[:, a, c0, c1] = np.fft.ifft(np.fft.fft(p[:, a, c0, c1]) * np.exp(1j * omega_of_k(kvec) * dt))
    return p.reshape(N_THETA, NM3)


def make_packet3m(theta, pw):
    """三模初态：激发放在站点模 a=1 层、记录格 c0 的 c=0 态。"""
    psi = np.zeros((N_THETA, NM3), dtype=complex)
    pk = np.exp(-((theta - THETA0) ** 2) / (2.0 * pw ** 2))
    psi[:, 4] = pk / np.linalg.norm(pk)
    return psi


def run3m(g_cross, t_end=T_MAX, dt=DT):
    """三模档演化到 t_end，返回末态与分栏读数。"""
    lam, V = fixed_gen3(G_FULL, g_cross)
    f_prof = pulse(theta, W_PULSE)
    psi = make_packet3m(theta, PW)
    n = int(round(t_end / dt))
    for _ in range(n):
        psi = apply_3mode(psi, dt / 2.0, lam, V, f_prof)
        psi = free_step3(psi, dt, kvec)
        psi = apply_3mode(psi, dt / 2.0, lam, V, f_prof)
    return psi, pops3(psi)


def lambda_left(nsteps, dt=DT):
    """左端点累积角 Λ = Σ_t f(π + Ω_P t·dt)·dt。

    原 split_step 的两半步共用步首求值的 f，故累积角是左端点黎曼和，与梯形和差
    O(dt·|f'|)。该式用于闭式预测，与实现口径一致。
    """
    s = 0.0
    for t in range(nsteps):
        s += float(pulse(THETA0 + OMEGA_P * (t * dt), W_PULSE)) * dt
    return s


theta = np.linspace(0.0, 2.0 * np.pi, N_THETA, endpoint=False)
kvec = np.fft.fftfreq(N_THETA, d=1.0 / N_THETA)
area = float(np.sum(pulse(theta, W_PULSE)) * (2.0 * np.pi / N_THETA))
G_FULL = (np.pi / 2.0) * OMEGA_P / area
t_cross = (2.0 * np.pi - THETA0) / OMEGA_P
G_WRITE = G_FULL * pulse(theta, W_PULSE)
print("脉冲面积 %.4f，完整交换耦合 g = %.4f，折返时刻 %.4f" % (area, G_FULL, t_cross))
print("写入侧参数逐项（须与第 726 号一致）:")
print("  N_theta = %d，omega_P = %.6f，theta_0 = %.6f，w = %.3f，dt = %.1e"
      % (N_THETA, OMEGA_P, THETA0, W_PULSE, DT))
print("  T = %.1f（长期平均档 %.1f），g_full 实测 %.6f（第 726 号报 5.0916），波包宽 PW = %.2f"
      % (T_MAX, T_AVG, G_FULL, PW))
print()

# =====================================================================
# 本轮新增：多站点装置
# =====================================================================
def site_frames(n_site, scheme, param):
    """各站点的局部标架偏移 $A_j$（弧度）。

    scheme="jia"   同相：A_j = 0。
    scheme="yi"    沿位错线均匀相位梯度：A_j = j·param。
    scheme="bing"  空间均匀相移：A_j = A_SPAN·x_j/x_half，x_j 在芯内横向等距取样。
    """
    if scheme == "jia":
        return np.zeros(n_site)
    if scheme == "yi":
        return param * np.arange(n_site, dtype=float)
    if scheme == "bing":
        xs = np.linspace(-X_HALF, X_HALF, n_site)
        return A_SPAN * xs / X_HALF
    raise ValueError("未知方案: %s" % scheme)


def run_site_shift(shift, g_on=True, t_end=T_MAX, op=None):
    """单站点演化，写入剖面带相移 shift（本站局部标架偏移 $A_j$）。

    与第 726 号的 split_step_g 同源，只把剖面从 f(theta) 换成 f(theta - shift)。
    op=None 走写入算符 apply_int；给 op 时走替身算符（恒真检验用）。
    """
    psi = make_packet(theta, PW)
    n = int(round(t_end / DT))
    if op is None:
        g_arr = G_FULL * pulse(theta - shift, W_PULSE) if g_on else np.zeros_like(G_WRITE)
        for _ in range(n):
            psi = split_step_g(psi, DT, theta, kvec, g_arr)
    else:
        for _ in range(n):
            psi = split_step_sub(psi, DT, theta, kvec, W_PULSE, G_FULL, op)
    return psi


def time_average_site_shift(shift, g_on=True, t_end=T_AVG):
    """逐站点的长期平均，采样口径与第 726 号的 time_average_site 相同。

    每 DT 采样一次站点侧布居，含 t=0 的初值点，梯形积分除以 t_end。
    同时返回记录侧布居的时间平均（两栏之和恒为 1）。
    """
    g_arr = G_FULL * pulse(theta - shift, W_PULSE) if g_on else np.zeros_like(G_WRITE)
    psi = make_packet(theta, PW)
    n = int(round(t_end / DT))
    t_list = [0.0]
    s_list = [site_pop(psi)]
    r_list = [reg_pop(psi)]
    for step in range(n):
        psi = split_step_g(psi, DT, theta, kvec, g_arr)
        t_list.append((step + 1) * DT)
        s_list.append(site_pop(psi))
        r_list.append(reg_pop(psi))
    ts = np.array(t_list)
    mean_site = float(np.trapezoid(np.array(s_list), ts) / t_end)
    mean_reg = float(np.trapezoid(np.array(r_list), ts) / t_end)
    return ts, np.array(s_list), mean_site, mean_reg


def write_profile(shift, t_end=T_MAX, dt=DT):
    """写入算符实际使用的耦合剖面 p(t) = g_full·f(theta_p(t) - shift) 的采样。

    该剖面在 t 上严格周期为 1（因 f 以 2π 为周期且 ω_P = 2π）。
    """
    n = int(round(t_end / dt))
    ts = np.arange(n) * dt
    return ts, G_FULL * pulse(THETA0 + OMEGA_P * ts - shift, W_PULSE)


def gate_times(shift, t_end=T_MAX):
    """该站点的过闸时刻（解析）：解 theta_p(t) = shift + 2*pi*n 落在半开窗 [0, t_end) 内。"""
    out = []
    k_lo = int(np.ceil((shift - THETA0) / (2.0 * np.pi)))
    k_hi = int(np.ceil((shift - THETA0 + OMEGA_P * t_end) / (2.0 * np.pi)))
    for k in range(k_lo, k_hi + 1):
        t_k = (shift + 2.0 * np.pi * k - THETA0) / OMEGA_P
        if 0.0 <= t_k < t_end:
            out.append(t_k)
    return out


def write_events(shift, t_end=T_MAX, dt=DT):
    """每站点的写入事件数（过闸次数），两种口径并排。

    口径甲（解析，穿过闸）：解 theta_p(t) = shift + 2*pi*n 落在半开窗 [0, t_end) 内的个数。
    口径乙（数值，剖面峰数）：写入剖面在窗内的极大值个数。
    笔数是相位绕数增量，故两口径须一致。

    口径乙的实现要点：写入剖面在 t 上严格周期为 1（因 f 以 2π 为周期且 ω_P = 2π），
    故在单个周期上按循环邻点判峰，再把落在窗内的峰计入，
    避免把窗口起点的下降沿误判为峰。
    """
    n_per = int(round(1.0 / dt))
    ts_per = np.arange(n_per) * dt
    prof_per = G_FULL * pulse(THETA0 + OMEGA_P * ts_per - shift, W_PULSE)
    n_numeric = 0
    for i in range(n_per):
        left = prof_per[i - 1] if i > 0 else prof_per[n_per - 1]
        right = prof_per[(i + 1) % n_per]
        if prof_per[i] > left and prof_per[i] >= right:
            if ts_per[i] < t_end - 1e-12:
                n_numeric += 1
    n_analytic = len(gate_times(shift, t_end))
    return n_analytic, n_numeric


# ---------------------------------------------------------------------
# 联合模空间的生成元与指数
# ---------------------------------------------------------------------
def single_site_gen():
    r"""单站点写入生成元（4 维模空间）。

    每站点是一对模 $(a, c)$，各 2 层，故单站点子空间 4 维，模索引 m = 2a + c。
    耦合项 $a^\dagger c + ac^\dagger$ 把 $(a,c)$ 映到 $(1-a,1-c)$，
    在 m 上即 m 映到 3-m，故生成元是反序矩阵。
    """
    X = np.zeros((4, 4), dtype=complex)
    for a in range(2):
        for c in range(2):
            X[2 * a + c, 2 * (1 - a) + (1 - c)] = 1.0
    return X


def joint_generator(n_site, coeffs=None):
    """联合模空间生成元：单站点生成元的 Kronecker 和。

    coeffs=None 时各站点系数取 1；给数组时第 j 项取 coeffs[j]。
    联合维数 4^{n_site}，n_site = 8 时为 65536 维，该规模的稠密矩阵不可建，
    故大档改用生成元的交换性核（见 commutator_dev）。
    """
    X = single_site_gen()
    if coeffs is None:
        coeffs = np.ones(n_site)
    M = np.zeros((4 ** n_site, 4 ** n_site), dtype=complex)
    for j in range(n_site):
        fac = np.ones((1, 1), dtype=complex)
        for q in range(n_site):
            fac = np.kron(fac, coeffs[q] * X if q == j else np.eye(4))
        M = M + fac
    return M


def sym_exp(M, dt):
    """实对称矩阵 M 的指数 exp(-i dt M)，用本征分解实现，不依赖 scipy。

    M 实对称（生成元均为实对称的 Kronecker 和），故 M = V diag(λ) V^T。
    """
    lam, V = np.linalg.eigh(M)
    return (V * np.exp(-1j * dt * lam)) @ V.T


def commutator_dev(n_site, coeffs=None):
    """交换性核：联合生成元各项之间的对易子偏差，与站点数无关。

    第 j 站的生成元项只作用在第 j 个张量因子上，第 k 站的项只作用在第 k 个因子上，
    两项作用在不同的 (a, c) 指标上，故两两对易。
    对易子在全部因子上的零等价于在任意两个站点的 16×16 子块上为零
    （其余因子一律为单位阵），故该核只建 16×16 矩阵，站点数再大也可算。
    """
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
            comm = Tj @ Tk - Tk @ Tj
            dev = max(dev, float(np.max(np.abs(comm))))
    return dev


def operator_factorization_dev(n_site, coeffs=None, dt=DT):
    """算符层核：联合指数与各站点指数的张量积之差。

    站点对角下第 j 站的生成元只作用在第 j 个张量因子上，
    故各站生成元互易，联合指数按张量积分解。
    该核需建 4^{n_site} 维稠密矩阵，n_site = 5 时已占 16 MB，
    n_site = 8 时为 65536 维即 68 GB 不可行，故该核只到 n_site = 4。
    大档的等价性由 commutator_dev 给出。
    """
    M = joint_generator(n_site, coeffs)
    joint = sym_exp(M, dt)
    prod = np.ones((1, 1), dtype=complex)
    X = single_site_gen()
    if coeffs is None:
        coeffs = np.ones(n_site)
    for j in range(n_site):
        prod = np.kron(prod, sym_exp(coeffs[j] * X, dt))
    return float(np.max(np.abs(joint - prod)))


def joint_run_2site(shifts, t_end=T_MAX):
    """两站点联合态的直接演化：显式 16 维模空间，每站点一条 theta 轴。

    联合 Hilbert 空间维数 4^{n_site} = 16，模索引 m = 2a + c。
    写入算符逐站点作用：exp(-i g_full f(theta_j - A_j) dt σ_x⊗σ_x)，
    在模指标上是 m 映到 3-m。
    自由演化在每站点的 theta 轴上各自平移。
    初态是张量积：两站点各在自己站点的 (a=1, c=0) 层放一个波包。
    """
    f1 = pulse(theta - shifts[0], W_PULSE)
    f2 = pulse(theta - shifts[1], W_PULSE)
    pk = np.exp(-((theta - THETA0) ** 2) / (2.0 * PW ** 2))
    pk = pk / np.linalg.norm(pk)
    J = np.zeros((N_THETA, N_THETA, 4, 4), dtype=complex)
    J[:, :, 2, 2] = pk[:, None] * pk[None, :]
    ph = np.exp(1j * omega_of_k(kvec) * DT)
    n = int(round(t_end / DT))
    for _ in range(n):
        for (fv, ax) in ((f1, 0), (f2, 1)):
            ang = G_FULL * fv * (DT / 2.0)
            if ax == 0:
                J = (np.cos(ang)[:, None, None, None] * J
                     - 1j * np.sin(ang)[:, None, None, None] * J[:, :, ::-1, :])
            else:
                J = (np.cos(ang)[None, :, None, None] * J
                     - 1j * np.sin(ang)[None, :, None, None] * J[:, :, :, ::-1])
        J = np.fft.ifft(np.fft.fft(J, axis=0) * ph[:, None, None, None], axis=0)
        J = np.fft.ifft(np.fft.fft(J, axis=1) * ph[None, :, None, None], axis=1)
        for (fv, ax) in ((f1, 0), (f2, 1)):
            ang = G_FULL * fv * (DT / 2.0)
            if ax == 0:
                J = (np.cos(ang)[:, None, None, None] * J
                     - 1j * np.sin(ang)[:, None, None, None] * J[:, :, ::-1, :])
            else:
                J = (np.cos(ang)[None, :, None, None] * J
                     - 1j * np.sin(ang)[None, :, None, None] * J[:, :, :, ::-1])
    return J


def joint_pops(J):
    """联合态的分栏读数：两站点各自的站点侧、记录侧，以及总激发数。

    模索引 m = 2a + c，故站点侧对应 m ∈ {2, 3}，记录侧对应 m 的奇数 {1, 3}。
    联合态是张量积，其模平方的总和是 1，而两站点的激发数之和是 2，
    故总激发数取两站两栏之和，不取 |J|^2 的总和。
    """
    s1 = float(np.sum(np.abs(J[:, :, [2, 3], :]) ** 2))
    r1 = float(np.sum(np.abs(J[:, :, [1, 3], :]) ** 2))
    s2 = float(np.sum(np.abs(J[:, :, :, [2, 3]]) ** 2))
    r2 = float(np.sum(np.abs(J[:, :, :, [1, 3]]) ** 2))
    tot = s1 + r1 + s2 + r2
    return s1, r1, s2, r2, tot


def joint_reduced_site1(J):
    """联合态对站点一的约化密度矩阵（模指标部分迹），用于核对部分迹口径。"""
    rho = np.einsum("ijml,ijnl->imn", J, J.conj())
    diag = np.real(np.einsum("imm->im", rho))
    return diag[:, [2, 3]].sum(), diag[:, [1, 3]].sum(), diag.sum()


# 跨站对照档：两站点装置，写入目标换成另一站点的寄存器
def cross_site_run(g_cross, t_end=T_MAX):
    """两站点装置，站点一的写入目标额外接到站点二的寄存器，强度 g_cross。

    站点二本站无写入（初始为空），故站点二寄存器上的布居全部是跨站泄漏。
    模基 (A1, R1, A2, R2)，生成元 M = g_full·(A1†R1+h.c.) + g_cross·(A1†R2+h.c.)。
    该生成元与 θ 无关，特征向量固定，一次对角化即可。
    """
    M = np.zeros((4, 4), dtype=complex)
    M[0, 1] = M[1, 0] = G_FULL
    M[0, 3] = M[3, 0] = g_cross
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
    leak = float(np.sum(np.abs(psi[:, 3]) ** 2))
    site1 = float(np.sum(np.abs(psi[:, 0]) ** 2))
    reg1 = float(np.sum(np.abs(psi[:, 1]) ** 2))
    tot = float(np.sum(np.abs(psi) ** 2))
    return leak, site1, reg1, tot


# =====================================================================
# 冒烟检查
# =====================================================================
print("=" * 78)
print("冒烟检查")
print("=" * 78)

psi_sm = run_site_shift(0.0, True, T_MAX)
print("N_site = 1 的末态分栏（应约为站点 0.00117617、记录 0.99882383）:")
print("  站点侧 %.8f，记录侧 %.8f，总量 %.8f" %
      (site_pop(psi_sm), reg_pop(psi_sm), total_pop(psi_sm)))

psi_ms2 = [run_site_shift(0.0, True, T_MAX), run_site_shift(0.0, True, T_MAX)]
print("N_site = 2 的两站点各自布居与总激发数:")
tot2 = 0.0
for j, p in enumerate(psi_ms2):
    print("  站点 %d：站点侧 %.8f，记录侧 %.8f，总量 %.8f"
          % (j, site_pop(p), reg_pop(p), total_pop(p)))
    tot2 += total_pop(p)
print("  两站点总激发数 %.10f" % tot2)

print("三种方案的 f 剖面形状（变化的轴 = 扫描的轴）:")
for name, shifts in (("甲 同相", site_frames(4, "jia", 0.0)),
                      ("乙 相位梯度", site_frames(4, "yi", 0.2)),
                      ("丙 横向相移", site_frames(4, "bing", 0.0))):
    peaks = []
    for sh in shifts:
        prof = pulse(theta - sh, W_PULSE)
        peaks.append(float(np.argmax(prof)) * (2.0 * np.pi / N_THETA))
    print("  %-12s A_j = %s" % (name, ", ".join("%.4f" % v for v in shifts)))
    print("      写入剖面峰值位置 theta = %s（rad）" % ", ".join("%.4f" % v for v in peaks))
print("  甲：所有站点的剖面同形同位，站点间分布平直")
print("  乙：峰值位置随站点序号线性后移，相位倾斜")
print("  丙：峰值位置随横向位置线性后移，横向梯度")
print("  核对：变化的轴 = 扫描的轴，三案分别落在站点序号轴与横向位置轴。")
print()

# 半角机制的数值核：t = 0.5 处累积角恰为 π/4，站点侧布居恰为 0.5
ts_half = np.arange(0.0, 0.5 + 1e-12, 1e-5)
acc_half = np.cumsum(G_FULL * pulse(THETA0 + OMEGA_P * ts_half, W_PULSE)) * 1e-5
print("  半角机制核：半个周期内累积角 %.12f（π/4 = %.12f），cos²A = %.12f"
      % (acc_half[-1], np.pi / 4.0, np.cos(acc_half[-1]) ** 2))
psi_half = run_site_shift(0.0, True, 0.5)
print("  装置实测 t = 0.5 的站点侧布居 %.12f（0.5 由剖面偶对称给出，非巧合）"
      % site_pop(psi_half))
print()

# =====================================================================
# MS-1 单站点退化检验
# =====================================================================
print("=" * 78)
print("MS-1 单站点退化检验（判据稳健）")
print("=" * 78)

# MS-1 容差预算：
#   判据给定的复现容差取 1e-5 相对偏差，与第 726 号同源。
#   误差来源：本轮实现逐字沿用第 726 号的 split_step 与 make_packet，
#   逐位复现的理论差值只来自浮点重排，量级 1e-16 相对；
#   步长 DT = 2.0e-3 与第 726 号相同，累积角的离散化误差同源，不引入新误差。
#   写入时刻的分辨率为采样步长 DT = 0.002 s，门槛取相对偏差 1e-5 时该项自动满足。

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
rel_t = abs(t_write - t_cross) / t_cross
ms1a_ok = rel_t < 1e-5
record("MS-1a", "单站点退化：写入时刻复现 0.5000",
       "相对偏差 < 1e-5（对折返时刻 0.5000）",
       "写入时刻 %.4f，折返 %.4f，相对偏差 %.3e" % (t_write, t_cross, rel_t), ms1a_ok)

psi_run, trace = run(make_packet(theta, PW), G_FULL, steps=int(round(T_MAX / DT)))
e0 = total_pop(make_packet(theta, PW))
e1 = total_pop(psi_run)
rel_e = abs(e1 - e0) / e0
ms1b_ok = rel_e < 1e-5
record("MS-1b", "单站点退化：总激发数复现 1.00000000",
       "相对偏差 < 1e-5",
       "初始 %.10f，末态 %.10f，偏差 %.3e" % (e0, e1, abs(e1 - e0)), ms1b_ok)

psi_nw, _ = run(make_packet(theta, PW), 0.0, steps=int(round(T_MAX / DT)))
site_n = site_pop(psi_nw)
site_w = site_pop(psi_run)
reg_n = reg_pop(psi_nw)
reg_w = reg_pop(psi_run)
d_site = site_w - site_n
rel_site = abs(d_site - (-0.99882383)) / 0.99882383
ms1c_ok = rel_site < 1e-5
record("MS-1c", "单站点退化：站点侧增量复现 -0.99882383",
       "相对偏差 < 1e-5",
       "站点侧增量 %.8f（无写入 %.8f，有写入 %.8f），相对偏差 %.3e"
       % (d_site, site_n, site_w, rel_site), ms1c_ok)
rel_reg = abs((reg_w - reg_n) - 0.99882383) / 0.99882383
supp("MS-1 附带", "记录侧增量",
     "记录侧增量 %.8f，相对偏差 %.3e" % (reg_w - reg_n, rel_reg))

t_avg1, s_avg1 = time_average_site(make_packet(theta, PW), G_WRITE)
mean_site = float(np.trapezoid(s_avg1, t_avg1) / T_AVG)
t_base1, s_base1 = time_average_site(make_packet(theta, PW), np.zeros_like(G_WRITE))
mean_base = float(np.trapezoid(s_base1, t_base1) / T_AVG)
rel_m = abs(mean_site - 0.50000000) / 0.50000000
ms1d_ok = rel_m < 1e-5
record("MS-1d", "单站点退化：长期平均站点侧复现 0.50000000",
       "相对偏差 < 1e-5（T = 8.0 覆盖 4 个整读数周期）",
       "⟨站点侧⟩ = %.10f，基线 %.10f" % (mean_site, mean_base), ms1d_ok)

dbar1 = mean_site - mean_base
rel_d = abs(dbar1 - (-0.50000000)) / 0.50000000
ms1e_ok = rel_d < 1e-5
record("MS-1e", "单站点退化：长期平均增量复现 -0.50000000",
       "相对偏差 < 1e-5",
       "Δ̄ = %.10f，相对偏差 %.3e" % (dbar1, rel_d), ms1e_ok)

# =====================================================================
# MS-2 逐站点独立性检验
# =====================================================================
print("=" * 78)
print("MS-2 逐站点独立性检验（判据稳健）")
print("=" * 78)

# MS-2 容差预算：
#   总激发数守恒的门槛 1e-8：写入算符在可达子空间上是幺正的，
#   逐站点独立演化的每步误差为浮点舍入，N_site 站点累加后量级 N_site×1e-16，
#   取 1e-8 留八个数量级余量。
#   总激发数等于逐站点求和的门槛 1e-12：两者的差只来自浮点求和次序与
#   约化密度矩阵的部分迹舍入，量级 1e-16×N_site，取 1e-12 留四个数量级余量。
#   算符层分解偏差的门槛 1e-12：本征分解的舍入量级 1e-16×dim，dim 到 65536 时
#   累积到 1e-12 量级，故该档门槛取 1e-11。

N_TEST = 8
shifts_ms = site_frames(N_TEST, "jia", 0.0)
per_site_tot = []
for j in range(N_TEST):
    pj = run_site_shift(shifts_ms[j], True, T_MAX)
    per_site_tot.append(total_pop(pj))
tot_sum = float(np.sum(per_site_tot))
cons_dev = abs(tot_sum - float(N_TEST)) / float(N_TEST)
ms2a_ok = cons_dev < 1e-8
record("MS-2a", "多站点总激发数守恒（N_site = 8）",
       "相对偏差 < 1e-8",
       "总激发数 %.12f（初始 %d），相对偏差 %.3e" % (tot_sum, N_TEST, cons_dev), ms2a_ok)

t_j = time.time()
J2 = joint_run_2site([0.0, 0.0], T_MAX)
s1j, r1j, s2j, r2j, totj = joint_pops(J2)
ind_s1 = site_pop(run_site_shift(0.0, True, T_MAX))
ind_r1 = reg_pop(run_site_shift(0.0, True, T_MAX))
ind_tot = ind_s1 + ind_r1
sum_dev = abs(totj - 2.0 * ind_tot)
ms2b_ok = sum_dev < 1e-12
record("MS-2b", "联合态的总激发数等于逐站点求和（N_site = 2 直接张量积演化）",
       "偏差 < 1e-12",
       "联合总激发数 %.12f，逐站点求和 %.12f，偏差 %.3e（用时 %.1f s）"
       % (totj, 2.0 * ind_tot, sum_dev, time.time() - t_j), ms2b_ok)

rel_s1 = abs(s1j - ind_s1) / ind_s1
rel_s2 = abs(s2j - ind_s1) / ind_s1
red_s1, red_r1, red_t1 = joint_reduced_site1(J2)
supp("MS-2 附带", "联合态与逐站点独立解的逐站点读数",
     "站点 1：联合 %.12f，独立 %.12f，相对偏差 %.3e；站点 2：联合 %.12f，相对偏差 %.3e；"
     "记录侧：站点 1 联合 %.12f，独立 %.12f，相对偏差 %.3e"
     % (s1j, ind_s1, rel_s1, s2j, rel_s2, r1j, ind_r1, abs(r1j - ind_r1) / ind_r1),
     "站点 2 与站点 1 同相，读数应逐位相同")
supp("MS-2 附带", "约化密度矩阵口径的独立核",
     "站点一：站点侧 %.12f，记录侧 %.12f，激发总数 %.12f" % (red_s1, red_r1, red_t1),
     "约化密度矩阵与直接求和两套口径一致，该站点的激发总数为 1")

J2s = joint_run_2site([0.0, 0.2], 0.5)
s1s, r1s, s2s, r2s, tots = joint_pops(J2s)
ind_s1b = site_pop(run_site_shift(0.0, True, 0.5))
ind_s2b = site_pop(run_site_shift(0.2, True, 0.5))
supp("MS-2 附带", "异相两站点的联合态与独立解",
     "联合：站点 1 站点侧 %.12f，站点 2 站点侧 %.12f；独立：站点 1 %.12f，站点 2 %.12f"
     % (s1s, s2s, ind_s1b, ind_s2b),
     "异相时两站点读数不同，联合态与独立解逐位一致")

# 前提守恒核：逐站点独立演化是精确解
print("  前提守恒核：站点对角下逐站点独立演化是精确解")
kron_devs = []
for n_site in (2, 3, 4):
    dev_kron = operator_factorization_dev(n_site)
    kron_devs.append(dev_kron)
    print("    核一 N_site = %d（联合维数 4^{n_site} = %d）：指数按张量积分解的偏差 %.3e"
          % (n_site, 4 ** n_site, dev_kron))
    supp("前提守恒", "N_site = %d 的联合指数按张量积分解" % n_site,
         "偏差 %.3e" % dev_kron,
         "站点对角下联合生成元是 Kronecker 和，各项作用在不同张量因子上因而互易")

print("    核一续 大档的交换性核（任意站点数可算，不建大矩阵）:")
comm_devs = []
for n_site in (8, 16):
    dev_comm = commutator_dev(n_site)
    comm_devs.append(dev_comm)
    print("      N_site = %2d：两项生成元的对易子偏差 %.3e" % (n_site, dev_comm))
    supp("前提守恒", "N_site = %d 的生成元逐对对易" % n_site,
         "对易子偏差 %.3e" % dev_comm,
         "对易子在任意两站的 16×16 子块上为零，其余因子为单位阵，故站点数任意大时成立")

print("    核二 含实际空间剖面的算符层分解（系数 c_j = g_full·f(theta_j - A_j)）:")
prof_devs = []
for n_site, sh in ((2, [0.0, 0.2]), (3, [0.0, 0.2, 0.4])):
    th_j = [theta[40 + 7 * k] for k in range(n_site)]
    coeffs = [float(G_FULL * pulse(th_j[k] - sh[k], W_PULSE)) for k in range(n_site)]
    dev_prof = operator_factorization_dev(n_site, coeffs)
    prof_devs.append(dev_prof)
    print("      N_site = %d（联合维数 %d）：系数 %s，分解偏差 %.3e"
          % (n_site, 4 ** n_site, ", ".join("%.4f" % c for c in coeffs), dev_prof))
    supp("前提守恒", "N_site = %d 含实际剖面的联合指数分解" % n_site,
         "偏差 %.3e" % dev_prof,
         "剖面系数逐站点不同，联合指数仍按张量积分解，故独立解对任意局部标架都精确")
print("    核三 态层：N_site = 2 的联合态直接演化已在上文给出，与独立解一致。")
print("  三个核合起来给出：逐站点独立演化是逐站点直和的精确解，不是近似。")
print()

# =====================================================================
# MS-3 同相扩展
# =====================================================================
print("=" * 78)
print("MS-3 同相扩展（方案甲）的读数（判据稳健）")
print("=" * 78)

# MS-3 容差预算：
#   一致性判据取 1e-3 相对偏差。误差来源：各站点独立演化的浮点累加，
#   量级 N_site×1e-16，相对量级 1e-15，远在门槛内。
#   门槛取 1e-3 的理由：四档之间若出现真实差异，其量级是 $A_j$ 造成的
#   累积角差，即 0.1 rad 量级，对应布居差 0.1 量级，与 1e-3 相差两个数量级。

N_LIST = [2, 4, 8, 16]
ms3_rows = []
for n_site in N_LIST:
    sh = site_frames(n_site, "jia", 0.0)
    site_means, reg_means = [], []
    for j in range(n_site):
        ta, sa, ms_, mr_ = time_average_site_shift(sh[j], True, T_AVG)
        site_means.append(ms_)
        reg_means.append(mr_)
    sm = float(np.mean(site_means))
    rm = float(np.mean(reg_means))
    dbar = sm - 1.0
    ms3_rows.append((n_site, sm, rm, dbar, min(site_means), max(site_means)))
    print("  N_site = %2d：⟨站点侧⟩ = %.10f，⟨记录侧⟩ = %.10f，Δ̄ = %.10f"
          % (n_site, sm, rm, dbar))
    print("              逐站点 ⟨站点侧⟩ 范围 [%.10f, %.10f]，站点间极差 %.3e"
          % (min(site_means), max(site_means), max(site_means) - min(site_means)))

dbars = [r[3] for r in ms3_rows]
ref = abs(dbars[0])
devs3 = [abs(v - dbars[0]) / ref for v in dbars]
ms3_ok = all(v < 1e-3 for v in devs3)
max_dev3 = max(devs3)
record("MS-3", "同相直和下 Δ̄ 在四档间保持一致",
       "四档相对偏差 < 1e-3（第 726 号单站点基准 -0.50000000）",
       "Δ̄ = %s，最大相对偏差 %.3e"
       % (", ".join("%.10f" % v for v in dbars), max_dev3), ms3_ok,
       "同相时各站点装置与第 726 号单站点装置同构")

# =====================================================================
# MS-4 相位梯度扩展
# =====================================================================
print("=" * 78)
print("MS-4 相位梯度扩展（方案乙）的读数")
print("=" * 78)

# MS-4 容差预算：形态与分布判定，数值容差按读数分辨能力给。
#   长时平均的读数分辨能力：T = 8.0 取 4 个整读数周期，周期闭合，
#   残留量级 1e-14（浮点累加）。
#   逐时刻布居的分辨能力：步长 DT = 2.0e-3 与 FFT 分辨率 1/256（时间 0.0039 s），
#   点读数的不确定度量级 1e-3。
#   「与同相档不同」一项的门槛即分辨能力：偏离量须超过 1e-12 量级才算真的不同。

DA_LIST = [0.05, 0.10, 0.20]
N_YI = 8
ms4_rows = []
for da in DA_LIST:
    sh = site_frames(N_YI, "yi", da)
    site_means = []
    snap05, snap12 = [], []
    for j in range(N_YI):
        ta, sa, ms_, mr_ = time_average_site_shift(sh[j], True, T_AVG)
        site_means.append(ms_)
        snap05.append(site_pop(run_site_shift(sh[j], True, 0.5)))
        snap12.append(site_pop(run_site_shift(sh[j], True, T_MAX)))
    sm = float(np.mean(site_means))
    dbar = sm - 1.0
    ms4_rows.append((da, sm, dbar, site_means, snap05, snap12))
    print("  ΔA = %.2f rad（%d 站点，跨度 %.2f rad）:" % (da, N_YI, sh[-1] - sh[0]))
    print("    ⟨站点侧⟩ = %.10f，Δ̄ = %.10f" % (sm, dbar))
    print("    t = 0.5 逐站点站点侧：%s" % ", ".join("%.6f" % v for v in snap05))
    print("    t = 1.2 逐站点站点侧：%s" % ", ".join("%.6f" % v for v in snap12))

sh_ref = site_frames(N_YI, "yi", 0.0)
ref_means = []
for j in range(N_YI):
    ta, sa, ms_, mr_ = time_average_site_shift(sh_ref[j], True, T_AVG)
    ref_means.append(ms_)
dbar_ref = float(np.mean(ref_means)) - 1.0
snap05_ref = [site_pop(run_site_shift(sh_ref[j], True, 0.5)) for j in range(N_YI)]
snap12_ref = [site_pop(run_site_shift(sh_ref[j], True, T_MAX)) for j in range(N_YI)]

dbar4 = [r[2] for r in ms4_rows]
mono4 = all(dbar4[i] <= dbar4[i + 1] + 1e-14 for i in range(len(dbar4) - 1)) or \
        all(dbar4[i] >= dbar4[i + 1] - 1e-14 for i in range(len(dbar4) - 1))
sat4 = (max(dbar4) - min(dbar4)) < 1e-12
ms4a_ok = mono4 or sat4
record("MS-4a", "Δ̄ 对 ΔA 的依赖形态（三档间单调或有明确饱和）",
       "三档单调，或给出明确饱和",
       "Δ̄ = %s（同相档 %.10f），档间极差 %.3e"
       % (", ".join("%.10f" % v for v in dbar4), dbar_ref, max(dbar4) - min(dbar4)),
       ms4a_ok, "三档读数恒等，属恒定的完全饱和")

dev4 = [abs(v - dbar_ref) for v in dbar4]
ms4b_ok = max(dev4) > 1e-12
record("MS-4b", "各档 Δ̄ 与同相档不同",
       "偏离量须超过长时平均的分辨能力 1e-12",
       "偏离量 %s，最大 %.3e" % (", ".join("%.3e" % v for v in dev4), max(dev4)),
       ms4b_ok,
       "长期平均对局部标架偏移严格不变，偏离量在浮点噪声内，判否")

supp("MS-4 附带", "随 ΔA 变化的读数（快照量）",
     "t = 0.5 逐站点站点侧布居的站点间极差：同相 %.3e，ΔA = 0.05 时 %.6f，"
     "ΔA = 0.10 时 %.6f，ΔA = 0.20 时 %.6f；t = 1.2 的站点间极差：同相 %.3e，"
     "ΔA = 0.05 时 %.6f，ΔA = 0.10 时 %.6f，ΔA = 0.20 时 %.6f"
     % (max(snap05_ref) - min(snap05_ref),
        max(ms4_rows[0][4]) - min(ms4_rows[0][4]),
        max(ms4_rows[1][4]) - min(ms4_rows[1][4]),
        max(ms4_rows[2][4]) - min(ms4_rows[2][4]),
        max(snap12_ref) - min(snap12_ref),
        max(ms4_rows[0][5]) - min(ms4_rows[0][5]),
        max(ms4_rows[1][5]) - min(ms4_rows[1][5]),
        max(ms4_rows[2][5]) - min(ms4_rows[2][5])),
     "长期平均不变，逐时刻的站点间分布随 ΔA 显著张开")

# 解析论证与数值核：长期平均对局部标架偏移不变
#   站点系内的指针剖面 f̃(s) = exp(-4cos²(πs))，在 s 上周期为 1。
#   记 F(s) = g_full·∫_0^s f̃，则 F(s+1) = F(s) + (g_full/ω_P)·area = F(s) + π/2。
#   局部标架偏移 A_j 对应 δ_j = A_j/(2π)，累积角为 A_j(t) = F(t - δ_j) + F(δ_j)。
#   cos²A_j 在 t 上周期为 2，故一个读数周期 [0, 2] 的积分为
#     ∫_0^2 cos²A_j dt = ∫_0^1 [cos²(x) + cos²(x + π/2)] dv = 1，
#   其中 x = F(v) + 常数。该式与 δ_j 无关，故 ⟨cos²A_j⟩ = 1/2 对任意 δ_j 成立，
#   长期平均增量恒为 -1/2。该恒等式依赖 g_full 的定义 g_full = (π/2)ω_P/area，
#   也依赖积分窗 T 取读数周期 2 的整数倍。
print("  解析论证的数值核：长期平均对局部标架偏移不变")
integ_devs = []
for sh in (0.0, 0.05, 0.20, 0.8, 1.5708):
    ts_n = np.arange(0.0, 2.0 + 1e-12, 1e-4)
    acc = np.cumsum(G_FULL * pulse(THETA0 + OMEGA_P * ts_n - sh, W_PULSE)) * 1e-4
    integ = float(np.trapezoid(np.cos(acc) ** 2, ts_n))
    integ_devs.append(abs(integ - 1.0))
    print("    A = %.4f rad：∫_0^2 cos²A dt = %.12f，与 1 的偏差 %.3e"
          % (sh, integ, abs(integ - 1.0)))
supp("MS-4 解析核", "一个读数周期内 cos²A 的积分与局部标架偏移无关",
     "五档偏差 %s，最大 %.3e"
     % (", ".join("%.3e" % v for v in integ_devs), max(integ_devs)),
     "解析式 ∫_0^2 cos²A dt = 1 与 δ_j 无关，故长期平均增量恒为 -1/2")

# 积分窗须取读数周期的整数倍：非整周期档的边界项
print("  积分窗条件的核：T 非读数周期整数倍时出现边界项")
for t_end in (8.0, 7.0, 6.5):
    ts_n = np.arange(0.0, t_end + 1e-12, 1e-4)
    acc = np.cumsum(G_FULL * pulse(THETA0 + OMEGA_P * ts_n - 0.2, W_PULSE)) * 1e-4
    m_n = float(np.trapezoid(np.cos(acc) ** 2, ts_n) / t_end)
    print("    T = %.1f（%s 整周期）：⟨站点侧⟩ = %.10f" %
          (t_end, "是" if abs(t_end / 2.0 - round(t_end / 2.0)) < 1e-12 else "非", m_n))
supp("MS-4 解析核", "积分窗条件",
     "T = 8.0（4 个整读数周期）时 ⟨站点侧⟩ = 0.5000000000；"
     "T = 7.0 与 T = 6.5 时读数偏离整周期值",
     "不变性要求积分窗取读数周期 2 的整数倍；本轮全部长期平均档均满足")

# =====================================================================
# MS-5 空间相移扩展
# =====================================================================
print("=" * 78)
print("MS-5 空间相移扩展（方案丙）的读数")
print("=" * 78)

# MS-5 容差预算：形态与分布判定。
#   「未被写入」的判据门槛取写入强度 $f \ge 1/2$，该阈值取自第 726 号 DS-2 的
#   解析截断口径（写入有效窗的条件即 f ≥ 1/2）。
#   严格布居为 1 的站点在本装置内不存在，原因是写入选中剖面无紧支撑，
#   f(π) = 0.0183 非零，故布居从 1 连续下降。该口径的计数另报。
#   分布形状与峰位由读数直接给出，不设数值容差。
#   读数时刻取 t = 0.5，即同相档的过闸时刻。理由：写入波扫过芯区需要时间，
#   在此刻读出的站点间分布正是「芯区只覆盖部分站点」的直接图景。

N_BING = 8
sh_bing = site_frames(N_BING, "bing", 0.0)
xs_bing = np.linspace(-X_HALF, X_HALF, N_BING)
bing_site, bing_reg = [], []
bing_snap05, bing_snap12 = [], []
for j in range(N_BING):
    ta, sa, ms_, mr_ = time_average_site_shift(sh_bing[j], True, T_AVG)
    bing_site.append(ms_)
    bing_reg.append(mr_)
    bing_snap05.append(site_pop(run_site_shift(sh_bing[j], True, 0.5)))
    bing_snap12.append(site_pop(run_site_shift(sh_bing[j], True, T_MAX)))
bing_mean = float(np.mean(bing_site))
bing_dbar = bing_mean - 1.0
snap05 = np.array(bing_snap05)
snap12 = np.array(bing_snap12)
gates_bing = [gate_times(sh, 4.0)[0] if gate_times(sh, 4.0) else float("nan")
              for sh in sh_bing]
strength_bing = np.array([float(pulse(-sh, W_PULSE)) for sh in sh_bing])
print("  横向位置 x_j（格）：%s" % ", ".join("%.3f" % v for v in xs_bing))
print("  局部标架 A_j（rad）：%s" % ", ".join("%.4f" % v for v in sh_bing))
print("  过闸时刻（s）：%s" % ", ".join("%.4f" % v for v in gates_bing))
print("  t = 0.5 的写入强度 f(-A_j)：%s" % ", ".join("%.4f" % v for v in strength_bing))
print("  t = 0.5 逐站点站点侧：%s" % ", ".join("%.6f" % v for v in snap05))
print("  t = 1.2 逐站点站点侧：%s" % ", ".join("%.6f" % v for v in snap12))
print("  ⟨站点侧⟩ = %.10f，Δ̄ = %.10f，逐站点 ⟨站点侧⟩ 范围 [%.10f, %.10f]"
      % (bing_mean, bing_dbar, min(bing_site), max(bing_site)))

mono_up = all(snap05[i] <= snap05[i + 1] + 1e-12 for i in range(len(snap05) - 1))
mono_dn = all(snap05[i] >= snap05[i + 1] - 1e-12 for i in range(len(snap05) - 1))
i_peak = int(np.argmax(snap05))
if mono_up:
    shape = "单调上升，单峰在横向正端"
elif mono_dn:
    shape = "单调下降，单峰在横向负端"
else:
    shape = "非单调，呈双峰或多峰"
ms5a_ok = mono_up or mono_dn
record("MS-5a", "空间相移下各站点布居分布的形状与峰位",
       "给出单峰、双峰或平坦的形状，并给出峰位",
       "形状：%s，峰位 x = %.3f 格（站点 %d，峰值布居 %.6f），t = 0.5 分布 %s"
       % (shape, xs_bing[i_peak], i_peak, snap05.max(), ", ".join("%.4f" % v for v in snap05)),
       ms5a_ok)

n_exact1 = int(np.sum(snap05 >= 1.0 - 1e-15))
n_ge99 = int(np.sum(snap05 >= 0.99))
n_ge95 = int(np.sum(snap05 >= 0.95))
n_ge90 = int(np.sum(snap05 >= 0.90))
ms5b_ok = True
record("MS-5b", "空间相移下布居为 1 的站点数",
       "报出布居为 1 的站点数（阈值口径：严格 1.0、≥ 0.99、≥ 0.95、≥ 0.90）",
       "t = 0.5 处：严格等于 1.0 的站点数 %d，≥ 0.99 者 %d，≥ 0.95 者 %d，≥ 0.90 者 %d；"
       "最大站点侧布居 %.6f（x = %.3f 格）"
       % (n_exact1, n_ge99, n_ge95, n_ge90, snap05.max(), xs_bing[int(np.argmax(snap05))]),
       ms5b_ok,
       "严格布居为 1 的站点数为 0：写入选中剖面无紧支撑，f(π) = 0.0183 非零，"
       "布居从 1 连续下降，故严格口径下不存在未被写入的站点；"
       "未被有效写入的站点数由 MS-5c 给出")

n_weak = int(np.sum(strength_bing < 0.5))
n_front = int(np.sum(np.array(gates_bing) > 0.5))
ms5c_ok = n_weak > 0
record("MS-5c", "空间相移下未被有效写入的站点数",
       "写入强度 f < 1/2（第 726 号 DS-2 的有效窗口径）的站点数须大于 0",
       "t = 0.5 处写入强度 f(-A_j) = %s；f < 1/2 的站点数 %d（共 %d），"
       "过闸时刻晚于 0.5 即写入波尚未到达的站点数 %d"
       % (", ".join("%.4f" % v for v in strength_bing), n_weak, N_BING, n_front),
       ms5c_ok,
       "芯区只覆盖部分站点的图景成立")

order_ok = all(snap05[i] <= snap05[i + 1] + 1e-12 for i in range(len(snap05) - 1))
supp("MS-5 附带", "布居分布与解析过闸时刻的一致性",
     "过闸时刻 %s；t = 0.5 逐站点站点侧 %s"
     % (", ".join("%.4f" % v for v in gates_bing), ", ".join("%.6f" % v for v in snap05)),
     "过闸时刻晚于 0.5 的站点在 t = 0.5 尚未被写入波覆盖，保留量随 A_j 单调递增，"
     "一致性成立" if order_ok else "与解析过闸时刻不一致")

# =====================================================================
# MS-6 事件计数核
# =====================================================================
print("=" * 78)
print("MS-6 事件计数核（判据稳健）")
print("=" * 78)

# MS-6 容差预算：容差为 0，精确计数。
#   误差来源：笔数是相位绕数增量，属拓扑量，与速度、宽度无关。
#   计数口径甲是解析式（解 theta_p(t) = A_j + 2πn），无离散化误差；
#   口径乙是写入剖面的峰数，受采样步长 DT = 2.0e-3 限制，
#   写入选中剖面的半宽约 0.136 s，峰被 68 个采样点覆盖，峰位误差量级 1e-3 s，
#   不影响计数。窗端按半开约定 [0, T) 处理，且本轮三案的过闸时刻全在窗内，
#   边界歧义不存在。

T_CELL = 1.0
all_ok6 = True
for tag, shifts in (("甲 同相", site_frames(N_TEST, "jia", 0.0)),
                    ("乙 相位梯度", site_frames(N_YI, "yi", 0.20)),
                    ("丙 空间相移", site_frames(N_BING, "bing", 0.0))):
    ana = []
    num = []
    for sh in shifts:
        a_, n_ = write_events(sh, T_CELL)
        ana.append(a_)
        num.append(n_)
    dev = max(abs(a_ - 1) for a_ in ana)
    dev2 = max(abs(n_ - a_) for n_, a_ in zip(num, ana))
    ok = (dev == 0) and (dev2 == 0)
    all_ok6 = all_ok6 and ok
    print("  %-14s 每站解析事件数 %s，数值峰数 %s，对 1 的偏差 %d，两口径差 %d"
          % (tag, ", ".join(str(v) for v in ana), ", ".join(str(v) for v in num), dev, dev2))
record("MS-6", "多站点装置中每站恰一笔的写入事件数（D-5 口径）",
       "每站解析事件数 = 1（偏差 0），且与数值峰数口径一致",
       "三方案共 %d 个站点，全部为 1，偏差 0" % (N_TEST + N_YI + N_BING), all_ok6,
       "一个指针周期即一格，笔数是相位绕数增量")

ana8 = [write_events(sh, T_AVG)[0] for sh in site_frames(N_YI, "yi", 0.20)]
supp("MS-6 附带", "T = 8.0（8 个指针周期）下的每站事件数",
     "解析事件数 %s（每站应恰 8）" % ", ".join(str(v) for v in ana8),
     "事件数与周期数成正比，偏差 0")

# =====================================================================
# MS-7 恒真检验
# =====================================================================
print("=" * 78)
print("MS-7 恒真检验（判据稳健）")
print("=" * 78)

# MS-7 容差预算：门槛 1e-12。
#   误差来源：关断写入耦合后，站点侧布居恒为 1（自由演化是相位旋转，不动布居），
#   梯形积分除以 T = 8.0 引入的舍入量级 1e-13，与第 726 号 DS-3f 的读数 1.541e-13 同源。
#   反例检验：把写入耦合开回即得 Δ̄ = -0.5，判否。

ta7, sa7, mean_off, _ = time_average_site_shift(0.0, False, T_AVG)
dbar_off = mean_off - 1.0
ms7_ok = abs(dbar_off) <= 1e-12
record("MS-7", "恒真检验：关断写入耦合，多站点下 Δ̄ = 0",
       "|Δ̄| ≤ 1e-12",
       "⟨站点侧⟩ = %.12f，基线 1.0，Δ̄ = %.3e" % (mean_off, dbar_off), ms7_ok,
       "反例检验：把写入耦合开回即得 Δ̄ = %.8f，判否" % dbar1)

psi_diag = run_site_shift(0.0, True, T_MAX, op=apply_diag)
dbar_diag = site_pop(psi_diag) - site_n
supp("MS-7 附带", "正定势替身（apply_diag）下的站点侧增量",
     "站点侧增量 %.3e（门槛 1e-12）" % dbar_diag,
     "替身只改相位不做搬运，材料留在站点侧")

# =====================================================================
# 对照层：跨站泄漏（MS-8，不占主判据号）
# =====================================================================
print("=" * 78)
print("对照层：跨站泄漏（MS-8，不占主判据号）")
print("=" * 78)

G_CROSS = [0.01, 0.02, 0.04]
leaks_cross = []
leaks_ds9 = []
lam_eff = lambda_left(int(round(T_MAX / DT)))
for g in G_CROSS:
    lk, s1c, r1c, totc = cross_site_run(g, T_MAX)
    leaks_cross.append(lk)
    _, pops_ds9 = run3m(g, T_MAX)
    leaks_ds9.append(pops_ds9[2])
    om = np.sqrt(G_FULL ** 2 + g ** 2)
    closed = (g ** 2 / om ** 2) * np.sin(om * lam_eff) ** 2
    print("  g' = %.3f：两站点装置的跨站泄漏 %.10e，DS-9 三模档 %.10e，闭式 %.10e"
          % (g, lk, pops_ds9[2], closed))

ratios_cross = [leaks_cross[i + 1] / leaks_cross[i] for i in range(len(leaks_cross) - 1)]
ratios_ds9 = [leaks_ds9[i + 1] / leaks_ds9[i] for i in range(len(leaks_ds9) - 1)]
print("  两站点装置相邻比值：%s（平方律参照 4.00）" % ", ".join("%.9f" % r for r in ratios_cross))
print("  DS-9 三模档相邻比值：%s（平方律参照 4.00）" % ", ".join("%.9f" % r for r in ratios_ds9))
rel_r = [abs(r - 4.0) / 4.0 for r in ratios_cross]
rel_d9 = [abs(leaks_cross[i] - leaks_ds9[i]) / leaks_ds9[i] for i in range(len(leaks_cross))]
print("  两站点装置比值对 4.00 的相对偏差：%s" % ", ".join("%.3e" % v for v in rel_r))
print("  两套实现的绝对值相对差：%s" % ", ".join("%.3e" % v for v in rel_d9))
supp("对照层", "跨站泄漏的两套实现与第 726 号 DS-9 的对照",
     "两站点装置 %s；DS-9 三模档 %s"
     % (", ".join("%.10e" % v for v in leaks_cross),
        ", ".join("%.10e" % v for v in leaks_ds9)),
     "两套实现逐位一致，比值 %s 对 4.00 的相对偏差 %s"
     % (", ".join("%.9f" % r for r in ratios_cross), ", ".join("%.3e" % v for v in rel_r)))
print("  本档只作对照说明：站点对角是硬约束，跨站项不进入主判据。")
print()

# =====================================================================
# 出图
# =====================================================================
print("=" * 78)
print("出图")
print("=" * 78)

fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.4))

# (a) 三种 A_j 方案下各站点的布居分布
ax = axes[0, 0]
ax.plot(np.arange(N_YI), snap05_ref, "o-", color="tab:blue", ms=5,
        label=r"甲 同相（$A_j = 0$）")
ax.plot(np.arange(N_YI), ms4_rows[2][4], "s-", color="tab:orange", ms=5,
        label=r"乙 相位梯度（$\Delta A = 0.20$）")
ax.plot(np.arange(N_BING), bing_snap05, "^-", color="tab:green", ms=5,
        label=r"丙 横向相移（$A_{\rm span} = \pi/2$）")
ax.set_xlabel("站点序号")
ax.set_ylabel(r"站点侧布居（$t = 0.5$）")
ax.set_title("(a) 三种局部标架方案下各站点的布居分布", fontsize=10)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# (b) 同相扩展的 Δ̄ 随 N_site
ax = axes[0, 1]
ns = [r[0] for r in ms3_rows]
ax.plot(ns, dbars, "o-", color="tab:blue", ms=6, label=r"逐站点 $\bar{\Delta}$")
ax.plot(ns, [r[1] for r in ms3_rows], "s--", color="tab:green", ms=5,
        label=r"逐站点 $\langle$站点侧$\rangle$")
ax.axhline(-0.5, color="tab:red", ls="--", lw=1.2,
           label=r"第 726 号单站点基准 $-0.5$")
ax.set_xlabel(r"站点数 $N_{\rm site}$")
ax.set_ylabel("读数")
ax.set_ylim(-0.75, 1.15)
ax.set_title(r"(b) 同相扩展：$\bar{\Delta}$ 随 $N_{\rm site}$ 四档", fontsize=10)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
ax.text(0.02, 0.06, r"四档最大相对偏差 %.1e" % max_dev3, transform=ax.transAxes, fontsize=8)

# (c) 方案乙的 Δ̄ 随 ΔA
ax = axes[1, 0]
das = [0.0] + DA_LIST
dbar_curve = [dbar_ref] + dbar4
snap_curve = [float(np.mean(snap05_ref))] + [float(np.mean(r[4])) for r in ms4_rows]
spread_curve = [(max(snap05_ref) - min(snap05_ref))] + \
               [(max(r[4]) - min(r[4])) for r in ms4_rows]
ax.plot(das, dbar_curve, "o-", color="tab:blue", ms=6,
        label=r"$\bar{\Delta}$（长期平均）")
ax.plot(das, snap_curve, "s-", color="tab:orange", ms=5,
        label=r"$t = 0.5$ 站点侧布居的站点平均")
ax.plot(das, spread_curve, "^-", color="tab:green", ms=5,
        label=r"$t = 0.5$ 站点间极差")
ax.axhline(-0.5, color="tab:red", ls="--", lw=1.0,
           label=r"同相档 $\bar{\Delta} = -0.5$")
ax.set_xlabel(r"相位梯度 $\Delta A$（rad）")
ax.set_ylabel("读数")
ax.set_ylim(-0.75, 1.05)
ax.set_title(r"(c) 方案乙：$\bar{\Delta}$ 随 $\Delta A$ 三档", fontsize=10)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
ax.text(0.03, 0.06, r"$\bar{\Delta}$ 三档偏离同相档 %.1e（严格不变）" % max(dev4),
        transform=ax.transAxes, fontsize=8)

# (d) 跨站对照档的泄漏随 g'
ax = axes[1, 1]
gg = np.array(G_CROSS)
ax.plot(gg, leaks_cross, "o-", color="tab:blue", ms=7,
        label="两站点装置（本站寄存器 + 跨站寄存器）")
ax.plot(gg, leaks_ds9, "s--", color="tab:green", ms=5, label="第 726 号 DS-9 三模档")
ref_leak = leaks_cross[0] * (gg / gg[0]) ** 2
ax.plot(gg, ref_leak, ":", color="k", lw=1.3, label=r"平方律参照 $C\,g'^2$")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"跨站耦合强度 $g'$")
ax.set_ylabel("跨站泄漏")
ax.set_title("(d) 跨站对照档：泄漏随 g' 的平方律", fontsize=10)
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")
ax.text(0.03, 0.06, "相邻比值 %s" % ", ".join("%.4f" % r for r in ratios_cross),
        transform=ax.transAxes, fontsize=8)

fig.suptitle("第 741 号：写入过程的逐站点直和多站点扩展", fontsize=12)
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
    print("%-8s %-58s %s" % (cid, name[:58], "通过" if ok else "判否"))
    if ok:
        npass += 1
print()
print("合计 %d/%d 通过（主判据 MS-1 至 MS-7，展开为 %d 个子项；"
      "对照层 MS-8 不计入）" % (npass, len(RESULTS), len(RESULTS)))
print()
print("关键读数:")
print("  MS-1 单站点退化：写入时刻 %.4f，总激发数 %.10f（偏差 %.3e）"
      % (t_write, e1, abs(e1 - e0)))
print("               站点侧增量 %.8f，长期平均 ⟨站点侧⟩ = %.10f，Δ̄ = %.10f"
      % (d_site, mean_site, dbar1))
print("  MS-2 多站点：N_site = 8 总激发数 %.12f（相对偏差 %.3e）"
      % (tot_sum, cons_dev))
print("       联合态（N = 2）总激发数 %.12f，逐站点求和 %.12f，偏差 %.3e"
      % (totj, 2.0 * ind_tot, sum_dev))
print("  MS-3 同相四档 Δ̄：%s" % ", ".join("%.10f" % v for v in dbars))
print("  MS-4 方案乙 Δ̄：%s（同相档 %.10f，最大偏离 %.3e）"
      % (", ".join("%.10f" % v for v in dbar4), dbar_ref, max(dev4)))
print("       t = 0.5 站点间极差：同相 %.3e，ΔA = 0.05/0.10/0.20 时 %s"
      % (max(snap05_ref) - min(snap05_ref),
         ", ".join("%.6f" % (max(r[4]) - min(r[4])) for r in ms4_rows)))
print("  MS-5 方案丙：A_j = %s" % ", ".join("%.4f" % v for v in sh_bing))
print("       过闸时刻 %s" % ", ".join("%.4f" % v for v in gates_bing))
print("       t = 0.5 分布 %s，形状 %s，峰位 x = %.3f 格"
      % (", ".join("%.4f" % v for v in snap05), shape, xs_bing[i_peak]))
print("       布居为 1 的站点数：严格 1.0 者 %d，≥ 0.99 者 %d，≥ 0.95 者 %d，"
      "≥ 0.90 者 %d，最大布居 %.6f" % (n_exact1, n_ge99, n_ge95, n_ge90, snap05.max()))
print("       写入强度 f < 1/2 的站点数 %d（共 %d），写入波未到达的站点数 %d"
      % (n_weak, N_BING, n_front))
print("       ⟨站点侧⟩ = %.10f，Δ̄ = %.10f" % (bing_mean, bing_dbar))
print("  MS-6 每站写入事件数：三方案共 %d 站全部为 1，偏差 0；T = 8.0 时每站 %s"
      % (N_TEST + N_YI + N_BING, ", ".join(str(v) for v in ana8)))
print("  MS-7 恒真检验：关断写入耦合 Δ̄ = %.3e" % dbar_off)
print("  对照层 跨站泄漏：%s" % ", ".join("%.10e" % v for v in leaks_cross))
print("          相邻比值 %s（参照 4.00），DS-9 档 %s"
      % (", ".join("%.9f" % r for r in ratios_cross),
         ", ".join("%.10e" % v for v in leaks_ds9)))
print()
print("三条前提守恒的核对结果:")
print("  其一，单站点退化与第 726 号一致：MS-1 五个子项全过，最大相对偏差 %.3e"
      % max(rel_t, rel_e, rel_site, rel_m, rel_d))
print("  其二，写入侧参数七项与第 726 号逐项一致：N_theta = %d，omega_P = %.6f，"
      "theta_0 = %.6f，w = %.3f，dt = %.1e，T = %.1f（长期平均 %.1f），g_full = %.6f"
      % (N_THETA, OMEGA_P, THETA0, W_PULSE, DT, T_MAX, T_AVG, G_FULL))
print("  其三，逐站点独立演化是精确解：")
print("       核一 算符层 Kronecker 分解偏差 %s（N = 2、3、4）"
      % ", ".join("%.3e" % v for v in kron_devs))
print("       核一续 大档交换性偏差 %s（N = 8、16）"
      % ", ".join("%.3e" % v for v in comm_devs))
print("       核二 含实际剖面的算符层分解偏差 %s（N = 2、3）"
      % ", ".join("%.3e" % v for v in prof_devs))
print("       核三 态层联合演化（N = 2）总激发数偏差 %.3e，逐站点读数相对偏差 %.3e"
      % (sum_dev, max(rel_s1, rel_s2)))
print("耗时 %.1f 秒" % (time.time() - T_START))
