# -*- coding: utf-8 -*-
r"""
minimal_verification_t1b2_frame_sector.py

[手征线 T1b-2] 标架联络扇区的空间拓扑荷：纯规范平凡性、向错的 SO(3)/SU(2) 二分

定位
----
判据 T1b（内部草稿/注记-判据T1b：四维拓扑数为零及其后果 第五节）把"标架联络扇区
是否有同类超选择结构"登记为 T1b-2，判法与 T1b 同构：先查该扇区的空间拓扑荷是否为
守恒量，再套"四维拓扑数 = 空间拓扑荷的时间变化量"。本脚本把该判定做成可复现读数。

模型（路径 B 规则 1-7 的标架层）
----
规则 2  变形梯度 F(n) 属于 SO(3)（列单位、列正交）
规则 4  链接变量 U_a(n) = F(n)^T F(n + a)
规则 5  格面 holonomy = 转角亏损（Regge）
规则 7  真空平坦：F = 常数 则 U = I，全部格面 holonomy = I
向错构造（Volterra，规则 6）：链接角取极角增量的主值（wrap_pi），再乘 q，Frank 角
即 2*pi*q。主值代表本身携带一条切割链，绕核心一周的链接角和为 2*pi（不是零），
这就是向错的 Frank 角来源。q 为整数时该和是 2*pi 的整数倍，构型在 SO(3) 层与平凡
构型不可分；q 非整数时不可约化，构型是真向错。两种情形走同一套回路读数。

判据（先立判据，再实现；预期写错时修正判据并把教训写入注记）
----
J1 纯规范平坦性（规则 7）：任意单值标架（光滑与逐点随机、幅度 0.05 至 10 rad）的全部
   格面 holonomy = I（小于 1e-12），半径 1..3 的回路 holonomy 亦 = I。
J2 整值绕数在 SO(3) 层不可见：q = n 为整数时，绕核心回路的链接角和 = 2*pi*n（读数），
   而回路 holonomy = I（小于 1e-12），与 n = 1,2,3,-1 及半径 r = 1..4 无关；另核
   不环绕核心的回路 holonomy = I（切割链不是物理通量线）。
J3 对照（判据非空转）：分数向错 q = 1/3、2/3 给绕核心回路 holonomy = rot_z(2*pi*q)，
   非恒等且半径无关。故 J2 的零不是判据恒零。
J4 双覆盖标签：q = n 为整数时提升累积角 = 2*pi*n（小于 1e-12），SU(2) 回路积
   = (-1)^n 乘 I（n = 0,1,2,3）；标架尺寸 L = 24,32,40 给同一读数。
J5 局域冗余精确不变：其一，对单点或全部格点施加任意 SO(3) 重标识（幅度 0.05 至 3.0 rad），
   单值标架构型的全部回路 holonomy 精确不变（小于 1e-13）；其二，对向错构型施加局域
   规范变换 theta_a(n) -> theta_a(n) + [lambda(n+a) - lambda(n)]，全部回路 holonomy 不变。

运行
----
python 验证脚本/minimal_verification_t1b2_frame_sector.py
确定性（随机数种子固定 20260918），运行约 3 秒。
图存 D:/software/TianXingFC/images/t1b2_frame_sector.png。
"""
import time

import numpy as np

T_START = time.time()
CHECKS = []
RNG = np.random.default_rng(20260918)


def check(tag, ok, detail):
    CHECKS.append((tag, bool(ok)))
    print(f"  [{tag}] {'PASS' if ok else 'FAIL'}  {detail}")


# ----------------------------------------------------------------------
# 基本工具
# ----------------------------------------------------------------------
def rot_z(ang):
    """SO(3) 绕 z 轴旋转，支持任意形状角数组 -> (..., 3, 3)"""
    a = np.asarray(ang, dtype=float)
    c, s = np.cos(a), np.sin(a)
    z = np.zeros_like(a)
    o = np.ones_like(a)
    return np.stack([np.stack([c, -s, z], axis=-1),
                     np.stack([s, c, z], axis=-1),
                     np.stack([z, z, o], axis=-1)], axis=-2)


def su2_z(ang):
    """SU(2) 绕 z 轴旋转（双覆盖提升，旋转角一半）-> (..., 2, 2)"""
    a = np.asarray(ang, dtype=float)
    z = np.zeros_like(a, dtype=complex)
    return np.stack([np.stack([np.exp(-0.5j * a), z], axis=-1),
                     np.stack([z, np.exp(0.5j * a)], axis=-1)], axis=-2)


def rot_rand(n, amp):
    """n 个随机 SO(3) 元，绕随机轴、转角均匀于 [-amp, amp]（Rodrigues）"""
    axis = RNG.normal(size=(n, 3))
    axis /= np.linalg.norm(axis, axis=-1, keepdims=True)
    ang = RNG.uniform(-amp, amp, size=n)
    K = np.zeros((n, 3, 3))
    K[:, 0, 1], K[:, 0, 2] = -axis[:, 2], axis[:, 1]
    K[:, 1, 0], K[:, 1, 2] = axis[:, 2], -axis[:, 0]
    K[:, 2, 0], K[:, 2, 1] = -axis[:, 1], axis[:, 0]
    th = ang[:, None, None]
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def wrap_pi(x):
    """折到 (-pi, pi]"""
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def plane_angle(L, core):
    """极角（纯 atan2），以 core 为中心的平面补丁"""
    ii, jj = np.meshgrid(np.arange(L), np.arange(L), indexing='ij')
    return np.arctan2(jj - core[1], ii - core[0])


def volterra_theta(L, core, q):
    """向错构型：链接角取主值（wrap_pi）下的极角增量，再乘 q，Frank 角即 2*pi*q。
    主值代表本身携带一条切割链：绕核心一周的链接角和为 2*pi（单值数组的原始差之和恒为零），乘 q 得
    2*pi*q。整数 q 时该和是 2*pi 的整数倍，故该构型在 SO(3) 层与平凡构型不可分。"""
    a = plane_angle(L, core)
    tx = np.zeros_like(a)
    ty = np.zeros_like(a)
    tx[:-1, :] = wrap_pi(a[1:, :] - a[:-1, :])
    ty[:, :-1] = wrap_pi(a[:, 1:] - a[:, :-1])
    return q * tx, q * ty


def links_from_frame(F):
    """单值标架的链接变量 U_a(n) = F(n)^T F(n + a)（环面回绕）"""
    Ft = np.swapaxes(F, -1, -2)
    return Ft @ np.roll(F, -1, axis=0), Ft @ np.roll(F, -1, axis=1)


def plaquettes(F):
    """格面 holonomy P(n) = Ux(n) Uy(n+ex) Ux(n+ey)^T Uy(n)^T"""
    Ux, Uy = links_from_frame(F)
    UxT, UyT = np.swapaxes(Ux, -1, -2), np.swapaxes(Uy, -1, -2)
    return Ux @ np.roll(Uy, -1, axis=0) @ np.roll(UxT, -1, axis=1) @ UyT


def rect_steps(ci, cj, r):
    """绕 (ci, cj) 的边长 2r 方回路（逆时针），返回链接步骤 (i, j, 方向, 取向)"""
    s = []
    s += [(ci - r + k, cj - r, 'x', +1) for k in range(2 * r)]
    s += [(ci + r, cj - r + k, 'y', +1) for k in range(2 * r)]
    s += [(ci + r - 1 - k, cj + r, 'x', -1) for k in range(2 * r)]
    s += [(ci - r, cj + r - 1 - k, 'y', -1) for k in range(2 * r)]
    return s


def path_holonomy(Ux, Uy, steps):
    """矩阵链接变量：按行进顺序累积（此序下伸缩相消成立）"""
    H = np.eye(3)
    for (i, j, d, o) in steps:
        M = Ux[i, j] if d == 'x' else Uy[i, j]
        H = H @ (M if o > 0 else M.T)
    return H


def path_holonomy_angles(theta_x, theta_y, steps):
    H = np.eye(3)
    for (i, j, d, o) in steps:
        th = theta_x[i, j] if d == 'x' else theta_y[i, j]
        H = H @ rot_z(o * th)
    return H


def path_holonomy_su2(theta_x, theta_y, steps):
    H = np.eye(2, dtype=complex)
    for (i, j, d, o) in steps:
        th = theta_x[i, j] if d == 'x' else theta_y[i, j]
        H = H @ su2_z(o * th)
    return H


def path_angle(theta_x, theta_y, steps):
    return sum(o * (theta_x[i, j] if d == 'x' else theta_y[i, j]) for (i, j, d, o) in steps)


print("=" * 72)
print("最小验证 T1b-2：标架联络扇区的空间拓扑荷")
print("=" * 72)

L, CORE = 24, (12, 12)

# ----------------------------------------------------------------------
# J1 纯规范平坦性（规则 7）
# ----------------------------------------------------------------------
print("\n[J1] 纯规范平坦性（规则 7）：单值标架给格面与回路 holonomy = I")
ii, jj = np.meshgrid(np.arange(L), np.arange(L), indexing='ij')
worst = 0.0
rows = []
for tag, ang in [("光滑 A=0.05", 0.05 * np.sin(2 * np.pi * ii / L) * np.cos(2 * np.pi * jj / L)),
                 ("光滑 A=0.5", 0.5 * np.sin(2 * np.pi * ii / L) * np.cos(2 * np.pi * jj / L)),
                 ("光滑 A=1.5", 1.5 * np.sin(2 * np.pi * ii / L) * np.cos(2 * np.pi * jj / L)),
                 ("光滑 A=3.0", 3.0 * np.sin(2 * np.pi * ii / L) * np.cos(2 * np.pi * jj / L)),
                 ("逐点随机 A=0.5", RNG.uniform(-0.5, 0.5, (L, L))),
                 ("逐点随机 A=3.0", RNG.uniform(-3.0, 3.0, (L, L))),
                 ("逐点随机 A=10.0", RNG.uniform(-10.0, 10.0, (L, L)))]:
    F = rot_z(ang)
    Ux, Uy = links_from_frame(F)
    dev = np.max(np.abs(plaquettes(F) - np.eye(3)))
    dev = max(dev, max(np.max(np.abs(path_holonomy(Ux, Uy, rect_steps(*CORE, r)) - np.eye(3)))
                       for r in (1, 2, 3)))
    worst = max(worst, dev)
    rows.append((tag, dev))
for tag, dev in rows:
    print(f"      {tag:<18s} max|P - I| = {dev:.3e}")
check("J1", worst < 1e-12, f"最大偏差 {worst:.3e}（判据 < 1e-12），七种单值标架构型")

# ----------------------------------------------------------------------
# J2 整值绕数在 SO(3) 层不可见
# ----------------------------------------------------------------------
print("\n[J2] 整值绕数（q = n 为整数）在 SO(3) 层不可见")
j2_worst, j2_angle_ok = 0.0, True
for n in (1, 2, 3, -1):
    tx, ty = volterra_theta(L, CORE, n)
    devs, sums = [], []
    for r in (1, 2, 3, 4):
        s = rect_steps(*CORE, r)
        devs.append(np.max(np.abs(path_holonomy_angles(tx, ty, s) - np.eye(3))))
        sums.append(path_angle(tx, ty, s) / (2 * np.pi))
    j2_worst = max(j2_worst, max(devs))
    j2_angle_ok = j2_angle_ok and abs(abs(sums[0]) - abs(n)) < 1e-12
    print(f"      n = {n:+d}: 链接角和/(2*pi) = " + " ".join(f"{v:+.6f}" for v in sums)
          + "  回路 holonomy 偏差 = " + " ".join(f"{d:.1e}" for d in devs))
tx, ty = volterra_theta(L, CORE, 1)
far = (CORE[0] - 6, CORE[1])
dev_far = np.max(np.abs(path_holonomy_angles(tx, ty, rect_steps(*far, 1)) - np.eye(3)))
print(f"      不环绕核心的回路（圈住一段切割链）偏差 = {dev_far:.1e}")
check("J2", max(j2_worst, dev_far) < 1e-12 and j2_angle_ok,
      f"环绕回路最大偏差 {j2_worst:.3e}、非环绕 {dev_far:.3e}（判据 < 1e-12），角和 = 2*pi*n")

# ----------------------------------------------------------------------
# J3 对照：分数向错给出非平凡 holonomy
# ----------------------------------------------------------------------
print("\n[J3] 对照（判据非空转）：分数向错 q = 1/3、2/3 给 rot_z(2*pi*q) 非恒等")
j3_worst, j3_nontrivial = 0.0, True
for q in (1.0 / 3.0, 2.0 / 3.0):
    tx, ty = volterra_theta(L, CORE, q)
    devs, sums, nontriv = [], [], []
    for r in (1, 2, 3, 4):
        s = rect_steps(*CORE, r)
        tot = path_angle(tx, ty, s)
        H = path_holonomy_angles(tx, ty, s)
        devs.append(np.max(np.abs(H - rot_z(tot))))
        sums.append(tot / (2 * np.pi))
        nontriv.append(np.max(np.abs(H - np.eye(3))))
    j3_worst = max(j3_worst, max(devs))
    j3_nontrivial = j3_nontrivial and abs(abs(sums[0]) - q) < 1e-12 and min(nontriv) > 0.1
    print(f"      q = {q:.4f}: 角和/(2*pi) = " + " ".join(f"{v:+.6f}" for v in sums)
          + "  对 rot_z(角和) 偏差 = " + " ".join(f"{d:.1e}" for d in devs)
          + f"  非恒等度 = {min(nontriv):.3f}")
check("J3", j3_worst < 1e-12 and j3_nontrivial,
      f"偏差 {j3_worst:.3e}（判据 < 1e-12），角和与 q 相符且 holonomy 非恒等")

# ----------------------------------------------------------------------
# J4 双覆盖标签
# ----------------------------------------------------------------------
print("\n[J4] 双覆盖标签：整值绕数的提升累积角 = 2*pi*n，SU(2) 回路积 = (-1)^n 乘 I")
j4_worst, j4_ok = 0.0, True
for Ls in (24, 32, 40):
    core = (Ls // 2, Ls // 2)
    line = []
    for n in (0, 1, 2, 3):
        tx, ty = volterra_theta(Ls, core, n)
        s = rect_steps(*core, 2)
        tot = path_angle(tx, ty, s)
        Hl = path_holonomy_su2(tx, ty, s)
        target = ((-1) ** n) * np.eye(2, dtype=complex)
        dev_tot = abs(abs(tot) - 2 * np.pi * n)
        dev_lift = np.max(np.abs(Hl - target))
        j4_worst = max(j4_worst, dev_tot, dev_lift)
        j4_ok = j4_ok and (dev_tot < 1e-12) and (dev_lift < 1e-12)
        line.append(f"n={n}: {tot / (2 * np.pi):+.6f}")
    print(f"      L = {Ls}: 角和/(2*pi) = " + "  ".join(line))
check("J4", j4_ok, f"最大偏差 {j4_worst:.3e}（判据 < 1e-12），标签 = (-1)^n，三种尺寸一致")

# ----------------------------------------------------------------------
# J5 局域冗余精确不变
# ----------------------------------------------------------------------
print("\n[J5] 局域冗余：标架重标识与局域规范变换下回路 holonomy 精确不变")
j5_worst = 0.0
for amp in (0.05, 0.5, 1.5, 3.0):
    ang = 0.4 * np.sin(2 * np.pi * ii / L) * np.cos(2 * np.pi * jj / L)
    F = rot_z(ang)
    Ux0, Uy0 = links_from_frame(F)
    for mode in ("单点", "全部"):
        G = F.copy()
        if mode == "单点":
            G[CORE[0] + 3, CORE[1] - 2] = rot_rand(1, amp)[0] @ G[CORE[0] + 3, CORE[1] - 2]
        else:
            G = rot_rand(L * L, amp).reshape(L, L, 3, 3) @ G
        Ux1, Uy1 = links_from_frame(G)
        dev = max(np.max(np.abs(path_holonomy(Ux1, Uy1, rect_steps(*CORE, r))
                                   - path_holonomy(Ux0, Uy0, rect_steps(*CORE, r)))) for r in (1, 2, 3))
        j5_worst = max(j5_worst, dev)
        print(f"      幅度 {amp:.2f} rad，{mode}标架重标识：回路偏差 = {dev:.1e}")
    lam = RNG.uniform(-amp, amp, (L, L))
    tx, ty = volterra_theta(L, CORE, 1)
    gx, gy = tx.copy(), ty.copy()
    gx[:-1, :] += lam[1:, :] - lam[:-1, :]
    gy[:, :-1] += lam[:, 1:] - lam[:, :-1]
    devg = max(np.max(np.abs(path_holonomy_angles(gx, gy, rect_steps(*CORE, r))
                                - path_holonomy_angles(tx, ty, rect_steps(*CORE, r)))) for r in (1, 2, 3))
    j5_worst = max(j5_worst, devg)
    print(f"      幅度 {amp:.2f} rad，局域规范变换：回路偏差 = {devg:.1e}")
check("J5", j5_worst < 1e-13, f"最大偏差 {j5_worst:.3e}（判据 < 1e-13），局域冗余是精确的")

# ----------------------------------------------------------------------
# 汇总与出图
# ----------------------------------------------------------------------
print("\n" + "=" * 72)
n_pass = sum(1 for _, ok in CHECKS if ok)
print(f"判据汇总：{n_pass}/{len(CHECKS)} 通过")
for tag, ok in CHECKS:
    print(f"  {tag}: {'PASS' if ok else 'FAIL'}")
print(f"耗时 {time.time() - T_START:.1f} s")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    ax = axes[0, 0]
    tags = [t for t, _ in rows]
    devs = [max(d, 1e-18) for _, d in rows]
    ax.semilogy(range(len(tags)), devs, "o-", color="tab:blue")
    ax.axhline(1e-12, color="tab:red", ls="--", lw=1, label="判据 1e-12")
    ax.set_xticks(range(len(tags)))
    ax.set_xticklabels(tags, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("max |P - I|")
    ax.set_title("J1 单值标架：格面与回路 holonomy = I")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    rr = np.arange(1, 5)
    for n, mk in [(1, "o"), (2, "s"), (3, "^")]:
        tx, ty = volterra_theta(L, CORE, n)
        vals = [np.max(np.abs(path_holonomy_angles(tx, ty, rect_steps(*CORE, r)) - np.eye(3))) for r in rr]
        ax.semilogy(rr, np.maximum(vals, 1e-18), mk + "-", label=f"整值 q = {n}：对 I 偏差")
    for q, mk in [(1.0 / 3.0, "v"), (2.0 / 3.0, "d")]:
        tx, ty = volterra_theta(L, CORE, q)
        vals = []
        for r in rr:
            s = rect_steps(*CORE, r)
            H = path_holonomy_angles(tx, ty, s)
            vals.append(np.max(np.abs(H - rot_z(path_angle(tx, ty, s)))))
        ax.semilogy(rr, np.maximum(vals, 1e-18), mk + "--", label=f"分数 q = {q:.3f}：对 rot_z(角和) 偏差")
    ax.set_xlabel("回路半径 r")
    ax.set_ylabel("回路 holonomy 偏差")
    ax.set_title("J2/J3 整值不可见、分数可见、均与半径无关")
    ax.legend(fontsize=7)

    ax = axes[1, 0]
    ns = np.arange(0, 4)
    angle_ratio = []
    for Ls in (24, 32, 40):
        core = (Ls // 2, Ls // 2)
        vals = []
        for n in ns:
            tx, ty = volterra_theta(Ls, core, n)
            vals.append(np.real(path_holonomy_su2(tx, ty, rect_steps(*core, 2))[0, 0]))
        ax.plot(ns, vals, "o-", label=f"L = {Ls}")
    for n in ns:
        tx, ty = volterra_theta(24, (12, 12), n)
        angle_ratio.append(path_angle(tx, ty, rect_steps(12, 12, 2)) / (2 * np.pi))
    ax.plot(ns, [(-1) ** n for n in ns], "k--", lw=1, label="(-1)^n")
    ax.set_xlabel("绕数 n")
    ax.set_ylabel("SU(2) 回路积的实部")
    ax.set_title("J4 双覆盖标签 = (-1)^n")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    amps = np.array([0.05, 0.5, 1.5, 3.0])
    for mode, mk in [("单点", "o"), ("全部", "s"), ("规范", "d")]:
        vals = []
        for amp in amps:
            if mode == "规范":
                lam = RNG.uniform(-amp, amp, (L, L))
                tx, ty = volterra_theta(L, CORE, 1)
                gx, gy = tx.copy(), ty.copy()
                gx[:-1, :] += lam[1:, :] - lam[:-1, :]
                gy[:, :-1] += lam[:, 1:] - lam[:, :-1]
                dev = max(np.max(np.abs(path_holonomy_angles(gx, gy, rect_steps(*CORE, r))
                                          - path_holonomy_angles(tx, ty, rect_steps(*CORE, r))))
                          for r in (1, 2, 3))
            else:
                ang = 0.4 * np.sin(2 * np.pi * ii / L) * np.cos(2 * np.pi * jj / L)
                F = rot_z(ang)
                G = F.copy()
                if mode == "单点":
                    G[CORE[0] + 3, CORE[1] - 2] = rot_rand(1, amp)[0] @ G[CORE[0] + 3, CORE[1] - 2]
                else:
                    G = rot_rand(L * L, amp).reshape(L, L, 3, 3) @ G
                Ux0, Uy0 = links_from_frame(F)
                Ux1, Uy1 = links_from_frame(G)
                dev = max(np.max(np.abs(path_holonomy(Ux1, Uy1, rect_steps(*CORE, r))
                                        - path_holonomy(Ux0, Uy0, rect_steps(*CORE, r)))) for r in (1, 2, 3))
            vals.append(max(dev, 1e-18))
        ax.semilogy(amps, vals, mk + "-", label=mode)
    ax.axhline(1e-13, color="tab:red", ls="--", lw=1, label="判据 1e-13")
    ax.set_xlabel("幅度 (rad)")
    ax.set_ylabel("回路 holonomy 偏差")
    ax.set_title("J5 局域冗余是精确的")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig("D:/software/TianXingFC/images/t1b2_frame_sector.png", dpi=140)
    print("图存 images/t1b2_frame_sector.png")
    print("J4 角和/(2*pi) 读数：", " ".join(f"n={n}: {v:+.6f}" for n, v in zip(ns, angle_ratio)))
except Exception as exc:
    print(f"绘图跳过：{exc}")