#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
minimal_verification_vector_sector_parity.py

判据 V：矢量扇区（h_0i，参考系拖曳 / Lense-Thirring）在元相子写入机制中的存在性。

问题
----
线性化度规按自旋分为标量、矢量、张量三块。标量块（h_00 = 2 theta，牛顿极限与史瓦西对齐）
与张量块（判据 D 的 TT 计数 2）均有判据级支撑，矢量块 h_0i（参考系拖曳）未判定。
本脚本判定：现行写入机制能否激发 h_0i。

模型设定
--------
框架侧（写入机制）：有质量粒子每拍写入倾斜幅度 delta_theta = m/m_P，记录场为标量叠加
    theta(r) = sum_a G m_a / (c^2 |r - r_a|)，
度规由两条构造假设给出 g_00 = -(1-2theta)、g_ij = (1+2theta) delta_ij，ansatz 中不含 g_0i。
单位制在结构检验部分取 G = m = c = R = 1，总质量归一为 1（只关心结构与标度，不关心数值大小）。

判据
----
V1 通道判据（同 rho 不同 j）：两种配置在同一时刻有相同的质量密度、不同的质量流，
   框架侧场逐点相等（写规则只是 rho 的泛函，不含 j）。GR 侧的 h_0i 在两种配置下不同。
V2 旋转体判据：旋转刚体的场等于静态场被刚性旋转（逐点机器精度）；非轴对称转动源的时间
   依赖部分是纯 AC（时间平均等于轴对称平均），与拖曳的 DC 结构不同类；时间平均场对 Omega
   为偶（Omega -> -Omega 不变）。
V3 扇区选择判据：在朴解读法（每分量标量弹性）的三维晶格上，纵向（div 型）源只激发无旋
   响应，横向（curl 型）源只激发无散响应。读数：拖曳属横向扇区，标量写规则只给纵向，
   载体本身有能力承载横向（故缺的是源通道）。
V4 观测对照：Gravity Probe B 两个观测量（测地岁差、拖曳岁差）的 GR 预言、框架预言、实测值
   与偏差 sigma。

运行：python 验证脚本/minimal_verification_vector_sector_parity.py
"""

import math

import numpy as np

# ----------------------------------------------------------------------------
# 第一部分：V1 与 V2（环与刚体）
# ----------------------------------------------------------------------------

N_RING = 512
R_RING = 1.0
M_TOT = 1.0


def ring_xy(omega, t, n=N_RING, radius=R_RING, phase0=0.0):
    ang = phase0 + 2.0 * math.pi * np.arange(n) / n + omega * t
    return radius * np.cos(ang), radius * np.sin(ang)


def theta_of_points(xs, ys, px, py, q=1.0):
    """标量叠加：theta(x) = sum_a q / |x - x_a|（单位制 G = m = c = 1）。"""
    dx = xs[:, None] - px[None, :]
    dy = ys[:, None] - py[None, :]
    r = np.sqrt(dx * dx + dy * dy)
    r = np.maximum(r, 1e-9)
    return q * np.sum(1.0 / r, axis=1)


def grid(n=161, half=3.0):
    ax = np.linspace(-half, half, n)
    gx, gy = np.meshgrid(ax, ax, indexing="ij")
    return gx.ravel(), gy.ravel()


def part_v1():
    print("=" * 78)
    print("V1 通道判据：同一时刻同质量密度、不同质量流")
    print("=" * 78)
    gx, gy = grid()
    q = M_TOT / N_RING

    # 配置 A：静止环；配置 B：同一环以 Omega 旋转。t = 0 时刻两者格点位置相同。
    px0, py0 = ring_xy(0.0, 0.0)
    pxB, pyB = ring_xy(0.37, 0.0)
    tA = theta_of_points(gx, gy, px0, py0, q)
    tB = theta_of_points(gx, gy, pxB, pyB, q)
    dev0 = np.max(np.abs(tA - tB)) / np.max(tA)
    print(f"t = 0 时刻两配置的场之差（相对最大值）：{dev0:.3e}")
    print("读数：写规则是质量密度的泛函，同一时刻同密度即同场，与质量流无关。")
    print("      GR 侧同一对配置的 h_0i 为 0 与非零（其符号随 Omega 反号），与框架侧形成对照。")
    print()

    # 时间推进后的对照：A 不动，B 转。检验 A 的场能否由 B 的场经刚性旋转得到。
    om = 0.37
    t_obs = 1.0
    pxT, pyT = ring_xy(om, t_obs)
    fB = theta_of_points(gx, gy, pxT, pyT, q)
    # 把观测点反向旋转 om*t，用静态环求值
    c, s = math.cos(om * t_obs), math.sin(om * t_obs)
    rx = c * gx + s * gy
    ry = -s * gx + c * gy
    pxS, pyS = ring_xy(0.0, 0.0)
    fRot = theta_of_points(rx, ry, pxS, pyS, q)
    dev1 = np.max(np.abs(fB - fRot)) / np.max(np.abs(fB))
    print(f"旋转环的场与静态场刚性旋转的逐点差（相对）：{dev1:.3e}")
    print("读数：旋转体的场是静态解的刚性旋转，场本身不含任何独立于质量密度的内禀角动量内容。")
    return dev0, dev1


def part_v2():
    print()
    print("=" * 78)
    print("V2 旋转体判据：时间平均的偶性与 AC 结构")
    print("=" * 78)
    n_obs = 64
    ang_obs = 2.0 * math.pi * np.arange(n_obs) / n_obs
    gx = 2.0 * np.cos(ang_obs)
    gy = 2.0 * np.sin(ang_obs)
    q = M_TOT / N_RING
    print("观测点：半径 2.0 的圆上 64 点（源的最大半径约 1.15）")

    # 非轴对称源：两个正对的块（m = 2 型图案），绕原点转动。
    n_pat = 64
    ang0 = 2.0 * math.pi * np.arange(n_pat) / n_pat
    pat_r = 1.0 + 0.15 * np.cos(2.0 * ang0)

    def pattern_xy(om, t):
        a = ang0 + om * t
        return pat_r * np.cos(a), pat_r * np.sin(a)

    def field_of(om, t):
        px, py = pattern_xy(om, t)
        return theta_of_points(gx, gy, px, py, q / n_pat * n_pat)

    print(f"{'Omega':>8} {'AC/DC 比':>12} {'(mean(+w)-mean(-w))/DC':>24} {'周期内谐波次数 k':>17}")
    for om in (0.1, 0.2, 0.4, 0.8):
        T = 2.0 * math.pi / om
        ts = np.linspace(0.0, T, 33, endpoint=False)
        fs = np.array([field_of(om, t) for t in ts])
        fmean_p = fs.mean(axis=0)
        fs_neg = np.array([field_of(-om, t) for t in ts])
        fmean_m = fs_neg.mean(axis=0)
        dc = np.max(np.abs(fmean_p))
        ac = np.max(np.abs(fs - fmean_p[None, :]))
        even = np.max(np.abs(fmean_p - fmean_m))
        # 时间依赖部分的主频：对空间最大响应点做 FFT
        dev = np.abs(fs - fmean_p[None, :])
        idx = int(np.unravel_index(np.argmax(dev), dev.shape)[1])
        sig = fs[:, idx] - fmean_p[idx]
        spec = np.abs(np.fft.rfft(sig))[1:]
        k = int(np.argmax(spec)) + 1
        print(f"{om:8.2f} {ac/dc:12.4e} {even/dc:24.3e} {k:17d}")
    print()
    print("读数：时间平均场对 Omega 为偶（第三列为零），说明旋转不引入奇（拖曳型）分量；")
    print("      时间依赖部分是纯 AC，谐波次数 k = 2，即 m = 2 图案每转一周在固定点给出两个周期；")
    print("      AC 幅值与 Omega 无关（刚性图案），说明该时间依赖是运动学转动，不含 Omega 的响应系数。")
    print("      拖曳是 DC 型（自旋的稳态进动漂移），与这里的 AC 附加不同类，不能互相冒充。")

    # 稳态进动观测量（陀螺）在框架侧：拖曳项恒为零
    print()
    print("框架侧的陀螺进动分解（弱场，静止光源不转）：")
    print("  Omega_geo（测地 / de Sitter）：由 g_00 与 g_ij 给出，与 GR 的 gamma = 1 值同阶")
    print("  Omega_LT （拖曳 / Lense-Thirring）：需要 g_0i，框架 ansatz 中不存在，恒为零")
    return


# ----------------------------------------------------------------------------
# 第二部分：V3 扇区选择（三维晶格，朴素读法：每分量标量弹性）
# ----------------------------------------------------------------------------

def kvec(n, dx):
    k = 2.0 * math.pi * np.fft.fftfreq(n, d=dx)
    KX, KY, KZ = np.meshgrid(k, k, k, indexing="ij")
    return KX, KY, KZ


def sector_fractions(ux, uy, uz, KX, KY, KZ):
    """在傅里叶空间做亥姆霍兹分解，返回纵向与横向能量占比。"""
    K2 = KX * KX + KY * KY + KZ * KZ
    K2[0, 0, 0] = 1.0
    UX = np.fft.fftn(ux)
    UY = np.fft.fftn(uy)
    UZ = np.fft.fftn(uz)
    kd = (KX * UX + KY * UY + KZ * UZ) / K2
    Lx, Ly, Lz = kd * KX, kd * KY, kd * KZ
    Tx, Ty, Tz = UX - Lx, UY - Ly, UZ - Lz

    def energy(a, b, c):
        return float(np.sum(np.abs(a) ** 2 + np.abs(b) ** 2 + np.abs(c) ** 2))

    el, et = energy(Lx, Ly, Lz), energy(Tx, Ty, Tz)
    return el / (el + et), et / (el + et)


def radial_profile(ux, uy, uz, dx, radii):
    """沿 +x 轴取 |u|(r)，返回 r^2 |u| 的列表。"""
    n = ux.shape[0]
    i0 = n // 2
    mag = np.sqrt(ux ** 2 + uy ** 2 + uz ** 2)
    return [float(mag[i0 + int(r / dx), i0, i0]) * r * r for r in radii]


def gaussian3(n, dx, width):
    ax = (np.arange(n) - n // 2) * dx
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
    r2 = X ** 2 + Y ** 2 + Z ** 2
    return np.exp(-r2 / (2.0 * width * width))


def long_profile_at(n, r_target, width=2.5):
    """在尺寸为 n 的周期盒上重算纵向响应，返回 r = r_target 处的 r^2 |u|。"""
    dx = 1.0
    KX, KY, KZ = kvec(n, dx)
    K2 = KX * KX + KY * KY + KZ * KZ
    K2[0, 0, 0] = 1.0
    phi = gaussian3(n, dx, width)
    ph_hat = np.fft.fftn(phi)
    ux = np.real(np.fft.ifftn(-1j * KX * ph_hat / K2))
    uy = np.real(np.fft.ifftn(-1j * KY * ph_hat / K2))
    uz = np.real(np.fft.ifftn(-1j * KZ * ph_hat / K2))
    mag = np.sqrt(ux ** 2 + uy ** 2 + uz ** 2)
    i0 = n // 2
    return float(mag[i0 + int(r_target / dx), i0, i0]) * r_target * r_target


def part_v3():
    print()
    print("=" * 78)
    print("V3 扇区选择判据：纵向源只给无旋响应，横向源只给无散响应")
    print("=" * 78)
    n = 48
    dx = 1.0
    KX, KY, KZ = kvec(n, dx)
    K2 = KX * KX + KY * KY + KZ * KZ
    K2[0, 0, 0] = 1.0

    # 纵向源：标量幅度写入（等价的径向倾斜），响应 = -grad Phi，lap Phi = -phi
    phi = gaussian3(n, dx, 2.5)
    ph_hat = np.fft.fftn(phi)
    # lap Phi = -phi  =>  Phi_hat = ph_hat / k^2 ；u = -grad Phi
    uxL = np.real(np.fft.ifftn(-1j * KX * ph_hat / K2))
    uyL = np.real(np.fft.ifftn(-1j * KY * ph_hat / K2))
    uzL = np.real(np.fft.ifftn(-1j * KZ * ph_hat / K2))

    # 横向源：质量流驱动的扭转型源；响应 = curl(psi z)，lap psi = -a
    az = gaussian3(n, dx, 2.5)
    a_hat = np.fft.fftn(az)
    # lap psi = -a  =>  psi_hat = a_hat / k^2 ；u = curl(psi z) = (d_y psi, -d_x psi, 0)
    psi_hat = a_hat / K2
    uxT = np.real(np.fft.ifftn(1j * KY * psi_hat))
    uyT = np.real(np.fft.ifftn(-1j * KX * psi_hat))
    uzT = np.zeros_like(uxT)

    elL, etL = sector_fractions(uxL, uyL, uzL, KX, KY, KZ)
    elT, etT = sector_fractions(uxT, uyT, uzT, KX, KY, KZ)
    print(f"{'源类型':<26}{'纵向能量占比':>14}{'横向能量占比':>14}")
    print(f"{'纵向源（标量幅度写入）':<26}{elL:>14.6f}{etL:>14.3e}")
    print(f"{'横向源（质量流扭转）':<26}{elT:>14.3e}{etT:>14.6f}")
    print()

    radii = (6.0, 8.0, 10.0, 12.0)
    pL = radial_profile(uxL, uyL, uzL, dx, radii)
    pT = radial_profile(uxT, uyT, uzT, dx, radii)
    q_src = float(phi.sum()) * dx ** 3
    asym = q_src / (4.0 * math.pi)
    print("远场径向剖面（r^2 |u|，沿 +x 轴；周期盒尺寸 48，故 r > 12 受镜像污染不计）")
    print(f"{'r':>6}" + "".join(f"{r:>14.0f}" for r in radii))
    print(f"{'纵向':>6}" + "".join(f"{v:>14.6f}" for v in pL))
    print(f"{'横向':>6}" + "".join(f"{v:>14.6f}" for v in pT))
    print(f"{'解析渐近 Q/(4pi)':>6}" + f"{asym:>14.6f}")
    diff = max(abs(a - b) / abs(a) for a, b in zip(pL, pT))
    print(f"两个扇道的径向剖面最大相对差：{diff:.3e}")
    print()
    print("盒尺寸收敛核对（r = 10，检验残余偏离是否来自周期镜像）")
    for nn in (48, 96):
        v = long_profile_at(nn, 10.0)
        print(f"  n = {nn:>3}   r^2 |u| = {v:.6f}   相对渐近 Q/(4pi) = {v/asym:.6f}")
    print()
    print("读数：两个扇道严格分离（各自的对侧占比为机器零），且横向响应存在、与纵向同阶、")
    print("      远场同为 1/r^2 型并逼近解析渐近值 Q/(4pi)；残余偏离随盒尺寸单调减小，来源为周期镜像。")
    print("      算子具备横向扇区，标量写规则只产生纵向源，故横向扇区永不被激发。")
    print("      缺口定位在源通道，不在载体能力。")


# ----------------------------------------------------------------------------
# 第三部分：V4 观测对照（Gravity Probe B）
# ----------------------------------------------------------------------------

G = 6.67430e-11
C_LIGHT = 2.99792458e8
GM_EARTH = 3.986004418e14
R_GPB = 7.018e6
J_EARTH = 5.861e33
ARCSEC_PER_RAD = 206264.806
MAS_PER_RAD = ARCSEC_PER_RAD * 1000.0  # 1 rad = 2.06264806e8 mas
SEC_PER_YEAR = 365.25 * 86400.0


def part_v4():
    print()
    print("=" * 78
          )
    print("V4 观测对照：Gravity Probe B（测地岁差与拖曳岁差）")
    print("=" * 78)
    v_orb = math.sqrt(GM_EARTH / R_GPB)
    omega_geo = 1.5 * GM_EARTH * v_orb / (C_LIGHT ** 2 * R_GPB ** 2)
    geo_mas = omega_geo * SEC_PER_YEAR * MAS_PER_RAD
    print(f"轨道参数：r = {R_GPB/1e3:.0f} km，v = {v_orb:.1f} m/s，"
          f"Omega_LT 参考量 2GJ/(c^2 r^3) = {2*G*J_EARTH/(C_LIGHT**2*R_GPB**3):.3e} /s")
    print(f"首阶测地岁差 (3/2)GMv/(c^2 r^2) = {geo_mas:.1f} mas/yr")
    print()
    print(f"{'观测量':<22}{'GR 预言':>12}{'框架预言':>12}{'实测':>16}{'框架偏差':>12}")
    print(f"{'测地岁差 (mas/yr)':<22}{-6606.1:>12.1f}{-geo_mas:>12.1f}"
          f"{'-6601.8 +- 18.3':>16}{abs(geo_mas-6606.1)/6606.1*100:>11.2f}%")
    print(f"{'拖曳岁差 (mas/yr)':<22}{-39.2:>12.1f}{0.0:>12.1f}"
          f"{'-37.2 +- 7.2':>16}{37.2/7.2:>11.2f} sigma")
    print()
    print("读数：首阶测地岁差与 GPB 的 GR 预言相差 0.3%（高阶与轨道细节），说明弱场度规设定一致；")
    print("      拖曳项框架恒为零，与实测相差 5.2 sigma。这是矢量扇区缺口的直接观测后果。")


def main():
    d0, d1 = part_v1()
    part_v2()
    part_v3()
    part_v4()
    print()
    print("=" * 78)
    print("汇总")
    print("=" * 78)
    print(f"V1 同密度不同流：t = 0 时刻场差 {d0:.3e}；刚性旋转恒等式偏差 {d1:.3e}")
    print("V2 时间平均对 Omega 为偶，时间依赖部分为 AC（谐波次数 k = 2，幅值与 Omega 无关）")
    print("V3 扇区严格分离（对侧占比 ~1e-31），两扇道远场同为 1/r^2 且剖面一致到 1e-15")
    print(f"V4 框架拖曳预言 0，与 GPB 实测 -37.2 +- 7.2 mas/yr 相差 5.2 sigma")
    print()
    print("结论：现行写入机制（幅度 delta_theta = m/m_P，只依赖质量，对时间反演为偶）")
    print("      无法激发法线场的横向（旋度）扇区，故 h_0i 恒为零，参考系拖曳为零。")
    print("      载体具备横向扇区（V3），缺口在源通道；要打开它需要一条时间反演为奇的")
    print("      写入通道（依赖质量流而非质量），该通道在现行分工双场下被分配给实轴取向场。")
    print("      【2026-09-20 改判指针】上面两行的历史判词只适用于只写时间分量的截断版写规则；")
    print("      完整表述（逐拍沿世界线的四维流，判据 Z1-Z4、K1-K5）已带出动量读数项，")
    print("      拖曳张力在方向上解除，现行口径见注记-动量读数项的第一原理。")
    print("      观测后果：框架通过 GPB 的测地岁差检验，不通过拖曳岁差检验。")


if __name__ == "__main__":
    main()
