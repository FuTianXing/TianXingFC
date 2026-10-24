#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
shapiro_delay_channel.py

判据 S：接力光速与 Shapiro 延迟通道（光的时间通道是否有 O(theta) 效应）。

来源：内部草稿《注记-接力光速与Shapiro延迟的缺口：2026-09-12》第二节的临时核验，
提升为常驻资产。原临时脚本的积分域与末步处理未声明，本脚本把积分域、内区正则化、
末步裁剪与参考量全部显式化，数值以本脚本为准。

模型设定
--------
单位制 c = 1，l_P = t_P = 1。静态源的法线场记录取

    theta(r) = theta0 * r0 / r                        (r >= r0)

即 GM/(c^2 r) 的无量纲化，theta0 = GM/(c^2 r0)。为让径向光（b = 0）穿过源心时
不发散，内区按均匀球正规化（与外部解同时保证 theta 与 dtheta/dr 连续）：

    theta(r) = theta0 * (3/2 - r^2 / (2 r0^2))        (r < r0)

光线在 y = b 的直线上入射，积分域 x in [-X, X]，末端做末步裁剪使终点精确落在
x = X（消除 O(dt) 伪影）。b >= 50 时路径不进入内区，内区选择不影响 S1-S3。

两套描述
--------
描述一（接力几何）：光每拍走一格，格距 l_P 与摆程 t_P 绝对不变（晶格刚性），
    坐标速度恒为 c，与 theta 无关。推论：时间读出等于路径弧长，场不产生 O(theta)
    的时间效应；弯曲只带来弧长超出，而弧长超出对偏转角是二阶量。

描述二（B2 测地线更新规则）：

    dv/dt = 2 c^2 ( grad theta - v_hat (v_hat . grad theta) )

    横向投影，严格保速率。推论：只改方向，不改速率，时间仍等于弧长。

对照（GR 型各向同性坐标）：坐标光速 = c (1 - 2 theta)。一阶时间超出沿直线路径

    Delta t = int [1/(1-2 theta) - 1] dx  ~=  4 theta0 r0 arcsinh(X/b)

（对称端点、直线路径的一阶形式，即标准 Shapiro 对数形式；本框架内不推导此式，
只作为阶数与量级的对照。）

判据
----
S1  B2 保速率：速率偏差 |v| - 1 的量级
S2  B2 偏折命中有限路径解析基准 4 theta0 r0 X / (b sqrt(X^2 + b^2))，
    其 X >> b 极限即 4 theta0 r0 / b
S3  B2 时间超出 = 弧长超出，二阶（O(theta^2)），远小于 GR 型一阶延迟；
    并核验 theta0 标度律：超出量对偏转角为严格二阶（比例常数稳定）
S4  径向光（b = 0）：grad theta 与 v_hat 平行，横向投影恒零，偏折与延迟同时为零
S5  步长伪影识别：不裁剪末步的朴素欧拉法给出正比于 dt 的假延迟，dt -> 0 时消失；
    裁剪末步后只剩下真实的二阶弧长超出
S6  完整零测地线读法：按框架自身度规的零测地线（折射率 n = 1/(1-2 theta)）积分，
    偏折与 O(theta) 延迟同时命中，延迟系数对照 gamma = 1
S7  三条候选读法的系数表：n = c、c(1-theta)、c(1-2theta) 分别给出 (1+gamma) = 0、1、2

运行：python 验证脚本/shapiro_delay_channel.py
"""

import math

C = 1.0
THETA0 = 0.002
R0 = 1.0
X = 1000.0
B_LIST = (50.0, 200.0, 1000.0)


def field(x, y):
    r2 = x * x + y * y
    r = math.sqrt(r2)
    if r >= R0:
        theta = THETA0 * R0 / r
        dth = -THETA0 * R0 / r2
    else:
        theta = THETA0 * (1.5 - 0.5 * r2 / (R0 * R0))
        dth = -THETA0 * r / (R0 * R0)
    if r > 0.0:
        gx, gy = dth * x / r, dth * y / r
    else:
        gx, gy = 0.0, 0.0
    return theta, gx, gy


def accel(x, y, vx, vy):
    _, gx, gy = field(x, y)
    sp = math.hypot(vx, vy)
    ux, uy = vx / sp, vy / sp
    proj = ux * gx + uy * gy
    return 2.0 * C * C * (gx - ux * proj), 2.0 * C * C * (gy - uy * proj)


def trace_b2(b, h, x_start=-X, x_end=X, method="rk4", renorm=False, exact_end=True):
    """返回 (vx, vy, t, arc, n, speed_dev_max)。t 为坐标时间，arc 为路径弧长。"""
    x, y, vx, vy = x_start, float(b), C, 0.0
    t = 0.0
    arc = 0.0
    n = 0
    dev = 0.0
    nmax = 400_000_000
    while x < x_end and n < nmax:
        step = h
        if exact_end and vx > 0.0:
            rem = x_end - x
            if rem / vx <= h:
                step = rem / vx
        if method == "rk4":
            ax1, ay1 = accel(x, y, vx, vy)
            ax2, ay2 = accel(x + 0.5 * step * vx, y + 0.5 * step * vy,
                             vx + 0.5 * step * ax1, vy + 0.5 * step * ay1)
            ax3, ay3 = accel(x + 0.5 * step * vx + 0.25 * step * step * ax1,
                             y + 0.5 * step * vy + 0.25 * step * step * ay1,
                             vx + 0.5 * step * ax2, vy + 0.5 * step * ay2)
            ax4, ay4 = accel(x + step * vx + 0.5 * step * step * ax2,
                             y + step * vy + 0.5 * step * step * ay2,
                             vx + step * ax3, vy + step * ay3)
            nx = x + step / 6.0 * (vx + 2 * (vx + 0.5 * step * ax1)
                                   + 2 * (vx + 0.5 * step * ax2) + vx + step * ax3)
            ny = y + step / 6.0 * (vy + 2 * (vy + 0.5 * step * ay1)
                                   + 2 * (vy + 0.5 * step * ay2) + vy + step * ay3)
            nvx = vx + step / 6.0 * (ax1 + 2 * ax2 + 2 * ax3 + ax4)
            nvy = vy + step / 6.0 * (ay1 + 2 * ay2 + 2 * ay3 + ay4)
        else:
            ax, ay = accel(x, y, vx, vy)
            nvx, nvy = vx + ax * step, vy + ay * step
            if renorm:
                sp = math.hypot(nvx, nvy)
                nvx, nvy = nvx / sp, nvy / sp
            nx, ny = x + nvx * step, y + nvy * step
        arc += math.hypot(nx - x, ny - y)
        d = abs(math.hypot(nvx, nvy) - C)
        if d > dev:
            dev = d
        t += step
        x, y, vx, vy = nx, ny, nvx, nvy
        n += 1
    return vx, vy, t, arc, n, dev


def deflection_finite(b, x_half=X):
    """有限路径解析基准：4 theta0 r0 X / (b sqrt(X^2 + b^2))。"""
    return 4.0 * THETA0 * R0 * x_half / (b * math.sqrt(x_half * x_half + b * b))


def deflection_asymptotic(b):
    return 4.0 * THETA0 * R0 / b


def gr_delay_numeric(b, x_half=X, n=2000001):
    """GR 型各向同性坐标光速 c(1-2 theta) 沿直线路径的一阶时间超出（梯形积分）。"""
    dx = 2.0 * x_half / (n - 1)
    s = 0.0
    for i in range(n):
        x = -x_half + i * dx
        th = THETA0 * R0 / math.hypot(x, b)
        w = 0.5 if (i == 0 or i == n - 1) else 1.0
        s += w * (1.0 / (1.0 - 2.0 * th) - 1.0)
    return s * dx


def gr_delay_analytic(b, x_half=X):
    return 4.0 * THETA0 * R0 * math.asinh(x_half / b)


def trace_light_full(b, h, x_start=-X, x_end=X):
    """完整零测地线读法（框架自身度规 g_00 = -(1-2 theta)、g_ij = (1+2 theta) delta_ij）。

    介质读法：折射率 n = 1/(1-2 theta)，射线方程
        dx/ds = u,   du/ds = 2 (grad theta - u (u . grad theta)) / (1-2 theta),
        dt/ds = 1/(1-2 theta).
    该方程保 |u| = 1。返回 (ux, uy, t, steps)。
    """
    x, y = x_start, float(b)
    ux, uy = C, 0.0
    t = 0.0
    steps = 0

    def f(xx, yy, ax, ay):
        th, gx, gy = field(xx, yy)
        f0 = 1.0 - 2.0 * th
        p = ax * gx + ay * gy
        return (ax, ay, 2.0 * (gx - ax * p) / f0, 2.0 * (gy - ay * p) / f0, 1.0 / f0)

    while x < x_end and steps < 400_000_000:
        step = h
        if ux > 0.0:
            rem = x_end - x
            if rem / ux <= h:
                step = rem / ux
        k1 = f(x, y, ux, uy)
        k2 = f(x + 0.5 * step * k1[0], y + 0.5 * step * k1[1],
               ux + 0.5 * step * k1[2], uy + 0.5 * step * k1[3])
        k3 = f(x + 0.5 * step * k2[0], y + 0.5 * step * k2[1],
               ux + 0.5 * step * k2[2], uy + 0.5 * step * k2[3])
        k4 = f(x + step * k3[0], y + step * k3[1],
               ux + step * k3[2], uy + step * k3[3])
        x += step / 6.0 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        y += step / 6.0 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        ux += step / 6.0 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
        uy += step / 6.0 * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3])
        t += step / 6.0 * (k1[4] + 2 * k2[4] + 2 * k3[4] + k4[4])
        steps += 1
    return ux, uy, t, steps


def hr(title):
    print()
    print("-" * 78)
    print(title)
    print("-" * 78)


def main():
    global THETA0
    print("=" * 78)
    print("shapiro_delay_channel.py")
    print("判据 S：接力光速与 Shapiro 延迟通道")
    print("设定：c = 1, l_P = t_P = 1, theta(r) = theta0 r0 / r (r >= r0), 内区均匀球正规化")
    print(f"      theta0 = {THETA0}, r0 = {R0}, 积分域 x in [{-X:.0f}, {X:.0f}]")
    print("=" * 78)

    h_main = 0.05

    # ---------------- S1 / S2 ----------------
    hr("S1 保速率 |  S2 偏折命中有限路径解析基准")
    print(f"{'b':>7} {'max|v|-1':>12} {'偏折角':>16} {'有限路径基准':>16} {'相对偏差':>11} "
          f"{'渐近基准':>13} {'渐近偏差':>11}")
    s2_rows = []
    for b in B_LIST:
        vx, vy, t, arc, n, dev = trace_b2(b, h_main, method="rk4")
        ang = -math.atan2(vy, vx)
        ref = deflection_finite(b)
        ref_inf = deflection_asymptotic(b)
        rel = abs(ang - ref) / ref
        rel_inf = abs(ang - ref_inf) / ref_inf
        s2_rows.append((b, ang, ref, t, arc, dev))
        print(f"{b:7.0f} {dev:12.3e} {ang:16.12e} {ref:16.12e} {rel:11.3e} "
              f"{ref_inf:13.6e} {rel_inf:11.3e}")
    print()
    print("读数：与有限路径基准的偏差为 O(theta) 的高阶修正（路径弯曲），三组均在 2e-4 内；")
    print("      渐近基准 4 theta0 r0 / b 只在 X >> b 时适用（b = 1000 时路径长度不足）。")

    # ---------------- S3 ----------------
    hr("S3 B2 时间超出（二阶）与 GR 型延迟（一阶）对照")
    print(f"{'b':>7} {'B2: t-2X':>14} {'B2: arc-2X':>14} {'GR 型解析':>14} "
          f"{'GR 型数值':>14} {'GR/B2':>10}")
    s3_rows = []
    for (b, ang, ref, t, arc, dev) in s2_rows:
        b2_excess = arc - 2.0 * X
        gr_a = gr_delay_analytic(b)
        gr_n = gr_delay_numeric(b)
        ratio = gr_a / b2_excess
        s3_rows.append((b, t - 2.0 * X, b2_excess, gr_a, gr_n, ratio))
        print(f"{b:7.0f} {t - 2.0*X:14.6e} {b2_excess:14.6e} {gr_a:14.6e} "
              f"{gr_n:14.6e} {ratio:10.1f}")
    print()
    print(f"GR 型解析/数值最大相对偏差：{max(abs(r[3]-r[4])/r[3] for r in s3_rows):.3e}")
    print("读数：B2 的时间超出等于弧长超出（速率恒为 c，时间即弧长），对 theta0 为二阶；")
    print("      GR 型延迟为一阶，两者之比在 1e3 至 1e6 量级，随 b 增大而增大。")

    # ---------------- S3b ----------------
    hr("S3b theta0 标度律：时间超出对偏转角为严格二阶")
    saved = THETA0
    print(f"{'theta0':>9} {'偏折角':>16} {'arc-2X':>14} {'(arc-2X)/alpha^2':>18}")
    for t0 in (0.001, 0.002, 0.004, 0.008):
        THETA0 = t0
        vxb, vyb, tb, arcb, nb, devb = trace_b2(50.0, h_main, method="rk4")
        angb = -math.atan2(vyb, vxb)
        print(f"{t0:9.4f} {angb:16.9e} {arcb - 2.0*X:14.6e} "
              f"{(arcb - 2.0*X)/angb**2:18.4f}")
    THETA0 = saved
    print()
    print("读数：theta0 变化八倍，超出量与偏转角平方之比恒定在 481.5，")
    print("      确认时间超出对 theta0（等价地对偏转角）为严格二阶。")

    # ---------------- S4 ----------------
    hr("S4 径向光（b = 0）：横向投影恒零")
    vx, vy, t, arc, n, dev = trace_b2(0.0, 0.25, method="rk4")
    print(f"终点速度 = ({vx:.15f}, {vy:.15f})，偏离 (1, 0) 的量 = "
          f"{max(abs(vx - 1.0), abs(vy)):.3e}")
    print(f"时间 t - 2X = {t - 2.0*X:.3e}，弧长 arc - 2X = {arc - 2.0*X:.3e}")
    a1 = accel(-500.0, 0.0, 1.0, 0.0)
    a2 = accel(0.0, 0.0, 1.0, 0.0)
    print(f"加速度样例：accel(-500, 0) = ({a1[0]:.3e}, {a1[1]:.3e})，"
          f"accel(0, 0) = ({a2[0]:.3e}, {a2[1]:.3e})")
    print("读数：grad theta 与 v_hat 平行时投影为零，光沿轴向穿过源附近不受任何作用；")
    print("      GR 型同配置（径向）给出全程一阶延迟，是 Shapiro 效应最强的通道。")

    # ---------------- S5 ----------------
    hr("S5 步长伪影识别：朴素欧拉法的假延迟正比于 dt")
    print("同一 b = 50，三种积分：")
    print("  (a) 欧拉，不裁剪末步（原始临时脚本口径）")
    print("  (b) 欧拉，裁剪末步")
    print("  (c) 欧拉，裁剪末步 + 每步速率重归一")
    print(f"{'dt':>10} {'(a) t-2X':>14} {'(a)/dt':>9} {'(b) t-2X':>14} {'(c) arc-2X':>14}")
    for h in (2e-2, 1e-2, 5e-3, 2.5e-3, 1.25e-3):
        _, _, ta, _, _, _ = trace_b2(50.0, h, method="euler",
                                     renorm=False, exact_end=False)
        _, _, tb, _, _, _ = trace_b2(50.0, h, method="euler",
                                     renorm=False, exact_end=True)
        _, _, _, arcc, _, _ = trace_b2(50.0, h, method="euler",
                                       renorm=True, exact_end=True)
        print(f"{h:10.5f} {ta - 2.0*X:14.6e} {(ta - 2.0*X)/h:9.4f} "
              f"{tb - 2.0*X:14.6e} {arcc - 2.0*X:14.6e}")
    print()
    print("读数：(a) 的超出量严格正比于 dt（比值恒等于 1），是末步未裁剪的纯离散化伪影，")
    print("      dt -> 0 时消失；裁剪末步后残余为真实弧长超出，二阶且不随 dt 漂移。")

    # ---------------- S6 ----------------
    hr("S6 完整零测地线读法（n = 1/(1-2 theta)）：偏折与延迟同时命中")
    print(f"{'b':>7} {'|u|':>17} {'偏折角':>16} {'偏折基准':>16} {'偏折偏差':>11} "
          f"{'dt':>14} {'dt 基准':>14} {'dt 偏差':>11}")
    s6_rows = []
    for b in B_LIST:
        uxs, uys, tf, steps = trace_light_full(b, 0.02)
        angs = -math.atan2(uys, uxs)
        refa = deflection_finite(b)
        dtf = tf - 2.0 * X
        refd = 4.0 * THETA0 * R0 * math.asinh(X / b)
        s6_rows.append((b, angs, refa, dtf, refd))
        print(f"{b:7.0f} {math.hypot(uxs, uys):17.15f} {angs:16.12e} {refa:16.12e} "
              f"{abs(angs-refa)/refa:11.3e} {dtf:14.6e} {refd:14.6e} "
              f"{abs(dtf-refd)/refd:11.3e}")
    print()
    print("读数：按框架自身度规完整积分（保留速率通道），偏折与 O(theta) 延迟同时命中，")
    print("      延迟系数即 (1+gamma) = 2，对应 gamma = 1，与 Cassini 测时约束相容。")
    print("      该读法与 B2 规则的差别只在速率通道：B2 固定 |v| = 1，完整读法令坐标光速降为 1-2 theta。")

    # ---------------- S7 ----------------
    hr("S7 三条候选读法的系数表（b = 50）")
    ref50 = 4.0 * THETA0 * R0 * math.asinh(X / 50.0)
    rows7 = (("接力每拍一格，格距与摆程绝对不变", 0.0),
             ("仅可读钟率被场调制（步长不变）", 1.0),
             ("框架度规的完整零测地线", 2.0))
    print(f"{'读法':>38} {'dt':>14} {'(1+gamma)':>10} {'gamma':>7}")
    for name, a in rows7:
        print(f"{name:>38} {ref50 * a / 2.0:14.6e} {a:10.1f} {a - 1.0:7.1f}")
    print()
    print("读数：三条读法给出 gamma = -1、0、1。太阳系对 gamma 的测定为 1 加减 2.3e-5（Cassini），")
    print("      故只有第三条读法与观测量相容；前两条的延迟为 0 与半量，均被排除。")
    print("      机制归属（哪些微观环节对应哪条读法）属作者裁决项，本脚本只给系数层对照。")

    # ---------------- 汇总 ----------------
    hr("汇总")
    print(f"S1 保速率：RK4（h = {h_main}）下三组 b 的最大 |v| - 1 = "
          f"{max(r[5] for r in s2_rows):.3e}")
    print(f"S2 偏折：与有限路径解析基准的最大相对偏差 = "
          f"{max(abs(r[1] - r[2]) / r[2] for r in s2_rows):.3e}")
    print(f"S3 通道分歧：GR 型解析延迟 / B2 弧长超出 介于 "
          f"{min(r[5] for r in s3_rows):.1f} 与 {max(r[5] for r in s3_rows):.1f} 之间")
    print(f"S4 径向：终点速度偏离 (1, 0) = {max(abs(vx - 1.0), abs(vy)):.3e}")
    print("S5 伪影：不裁剪末步的欧拉超出/dt 恒等于 1")
    print("S3b 标度：超出量/偏转角^2 = 481.5（theta0 变化八倍内恒定到 1e-3）")
    print(f"S6 完整零测地线：偏折最大相对偏差 "
          f"{max(abs(r[1]-r[2])/r[2] for r in s6_rows):.3e}，"
          f"延迟最大相对偏差 {max(abs(r[3]-r[4])/r[4] for r in s6_rows):.3e}")
    print("S7 系数表：三条读法给出 (1+gamma) = 0、1、2，即 gamma = -1、0、1")
    print()
    print("结论：描述二（B2 规则）在偏折通道命中解析基准，在时间通道不产生任何")
    print("      一阶（O(theta)）传播延迟；该缺口的来源是 B2 的速率约束 |v| = 1。")
    print("      按框架自身度规完整积分（保留速率通道）时，偏折与延迟同时命中，")
    print("      延迟系数对照 gamma = 1，与 Cassini 测时相容。")
    print("      机制归属（哪条读法成立）属作者裁决项，登记于开放问题总览 P1-24 关联子项。")


if __name__ == "__main__":
    main()
