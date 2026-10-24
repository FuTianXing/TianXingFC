# -*- coding: utf-8 -*-
r"""
vortex_init_apparatus.py

C1-0：涡旋初始化与横截面绕数测量装置（纲领 C1 全线公共资产）。

定位
----
纲领 C1（三维复合位错）的第一项任务，计划见
[[三维复合位错攻坚计划：形态、驻留与宽度窗口]] §二、§三。
产物是两类公共资产：目标构型的初始化，以及横截面绕数测量装置。
C1-1 至 C1-6 的每一项读数都依赖本脚本的判据 M1 至 M12。

根因更正与术语更正（2026-09-12）
----
2026-09-12 的中间尝试（`composite_dislocation_3d.py`）
把环芯横截面读数记为零，并据此判定环的涡旋初态有缺陷。判据 M4 与 M5 证明该
判定不成立：所用初态是合法的绕数 1 构型，零读数来自测量平面的选取。法向取成
了环轴方向（顶点法向平面），该平面上环的交叉或落在环外、或为平切，且大窗口会
同时套住两个取向相反的交叉点。环裁决实验 v1 与 v2 已记过同一约定。

术语更正：环芯线缺陷的分类是 $\pi_1(S^1) = \mathbb{Z}$（场取值在 $S^1$），
计划与旧注记中的「$\pi_2$」属于记号误用，判据内容不受影响。

装置约定
----
绕数在横切芯线的平面内测量。法向为 axis 的第 s 层格面上，逐 plaquette 累加
最小像相位差并除以 $2\pi$。回路取向取平面内全局指标升序 $(a, b)$，等价于
法向 $e_a \times e_b$，故读数符号 = 绕数符号 × (芯线切线 · 该法向)。
四条硬性规则（判据 M4、M6 与 M10 逐条实证）：

- 其一，平面须横切芯线。顶点法向平面平切芯线或落在环外，读数恒零。
- 其二，窗口不得同时套住环的两个交叉点。平面 $x = \mathrm{const}$（$|x| < R_c$）
  上环有两个交叉点且取向相反，大窗口的和为零，须用小窗口或 $x > 0$ 半平面。
- 其三，格点须半格偏移。整数格点上芯位落格点时，跨芯键的相位差恰为 $\pm\pi$，
  $\mathrm{mi}(-\pi) = \mathrm{mi}(+\pi) = -\pi$ 破坏反对称，读数退化为
  一侧 $0$、另一侧加倍（判据 M4e）。
- 其四，动力学中窗口须随芯位。局域窗口只对构造态有效；流下芯位会移动，
  读之前须先定位交叉所在 plaquette（判据 M10）。

模型
----
kink3d A 部分模型，开放边界（边界外一圈取真空 $\varphi = 0$）：

$$H = \frac{J_0}{2}\sum_{\text{bonds}}(1 - \cos\Delta\varphi) + \kappa\sum_{\text{sites}}(1 - \cos\varphi)$$

取 $J_0 = 1$、$\kappa = 0.6$、$N = 48$，格距 1。

构造与其边界
----
标准环初态取 $\varphi = \operatorname{arctan2}(z, \rho - R_c)$，零集即芯圆
$\{\rho = R_c, z = 0\}$，相位跳变面为芯圆所张的圆盘，最小像读法下不付能量。
该初态拓扑上干净（每面两个交叉，读数恰为 $\pm 1$），但其相位在无穷远趋于
极角 $\pi/2 - \theta$，边界映射非平凡，故能量随盒增长（判据 M12），
属非局域对象；裸环在梯度流下先缩芯后湮灭（判据 M10）。
遮罩式构造（环裁决实验 v2）作为缺陷对照保留：其 $m \equiv 1$ 区内相位仍是
$\psi = \operatorname{arctan2}(z, \rho - R_{\text{param}})$，故在
$\rho = R_{\text{param}} = R_c + 3.5$ 处另生一个反向涡环（判据 M7），
净通量为零，不可用作单环基准。
本脚本不验证：构型稳定性与驻留、锚核机制、三维动力学，均属 C1-1 之后。

运行
----
`python 验证脚本/vortex_init_apparatus.py`
确定性，随机数种子固定（判据 M11）。图存
`D:/software/TianXingFC/images/vortex_init_apparatus.png`。
"""

import numpy as np

J0, KAPPA = 1.0, 0.6
N = 48
COORD = np.arange(N, dtype=float) - N / 2.0 + 0.5
IC = N // 2
_X, _Y, _Z = np.meshgrid(COORD, COORD, COORD, indexing="ij")
RHO = np.sqrt(_X ** 2 + _Y ** 2)
AXES = (_X, _Y, _Z)
RESULTS = []

NI = 32
ICO = np.arange(NI, dtype=float) - NI / 2.0
_IX, _IY, _IZ = np.meshgrid(ICO, ICO, ICO, indexing="ij")
IRHO = np.sqrt(_IX ** 2 + _IY ** 2)


def mi(d):
    return (d + np.pi) % (2.0 * np.pi) - np.pi


def nearest(v):
    return int(np.argmin(np.abs(COORD - v)))


def plaque_density(P):
    """平面数组逐 plaquette 的绕数密度（单位 2π，整数）。"""
    a00, a10, a11, a01 = P[:-1, :-1], P[1:, :-1], P[1:, 1:], P[:-1, 1:]
    return (mi(a10 - a00) + mi(a11 - a10) + mi(a01 - a11) + mi(a00 - a01)) / (2.0 * np.pi)


def winding(phi, axis, s, center=None, rad=None, half=None):
    """法向 axis 的第 s 层格面上的绕数读数。

    center = 平面内两坐标值（取邻近格点），rad = 窗口半宽（格）；
    half = (全局轴, 符号) 半平面限制。center 与 half 均可单独使用。
    """
    in_axes = [a for a in (0, 1, 2) if a != axis]
    dens = plaque_density(np.take(phi, s, axis=axis))
    mask = np.ones(dens.shape, dtype=bool)
    if center is not None:
        i0, i1 = nearest(center[0]), nearest(center[1])
        r = 1 if rad is None else int(rad)
        u0, u1 = max(i0 - r, 0), min(i0 + r, dens.shape[0])
        v0, v1 = max(i1 - r, 0), min(i1 + r, dens.shape[1])
        win = np.zeros_like(mask)
        win[u0:u1, v0:v1] = True
        mask &= win
    if half is not None:
        haxis, hsign = half
        first = (haxis == in_axes[0])
        n_u = dens.shape[0] if first else dens.shape[1]
        sel = (COORD[:n_u] + 0.5) * hsign > 0.0
        win = np.zeros_like(mask)
        if first:
            win[sel, :] = True
        else:
            win[:, sel] = True
        mask &= win
    return float(dens[mask].sum())


def crossings(phi, axis=1, s=IC):
    """定位法向 axis 平面上的涡量交叉：返回 (柱半径, z 坐标, 读数) 列表。"""
    dens = plaque_density(np.take(phi, s, axis=axis))
    in_axes = [a for a in (0, 1, 2) if a != axis]
    out = []
    for u, v in np.argwhere(np.abs(dens) > 0.5):
        p = [0.0, 0.0, 0.0]
        p[axis] = COORD[s]
        p[in_axes[0]] = COORD[u] + 0.5
        p[in_axes[1]] = COORD[v] + 0.5
        out.append((float(np.hypot(p[0], p[1])), float(p[2]), float(dens[u, v])))
    return out


def straight(axis, core, n=1):
    """沿 axis 的直线涡，core = 垂直平面内两坐标值。"""
    in_axes = [a for a in (0, 1, 2) if a != axis]
    u = AXES[in_axes[0]] - core[0]
    v = AXES[in_axes[1]] - core[1]
    return mi(n * np.arctan2(v, u))


def ring_naive(Rc, n=1):
    """标准环初态：零集 = 芯圆 {rho = Rc, z = 0}。"""
    return mi(n * np.arctan2(_Z, RHO - Rc))


def ring_masked(Rc, n=1, mask=5.0, w=3.0):
    """遮罩式构造（环裁决实验 v2）：另有 rho = Rc + 3.5 处的反向环。"""
    rparam = Rc + mask - w / 2.0
    u = RHO - rparam
    d = np.sqrt(u ** 2 + _Z ** 2)
    m = np.clip((mask - d) / w, 0.0, 1.0)
    psi = np.arctan2(_Z, u)
    return mi(n * np.arctan2(m * np.sin(psi), 1.0 + m * (np.cos(psi) - 1.0)))


def ring_int(Rc, n=1):
    """整数格点上的标准环（用于 ±π 退化读数判据）。"""
    return mi(n * np.arctan2(_IZ, IRHO - Rc))


def block_sum_int(phi, cx, cz, r=1):
    """整数格点 y = 0 平面上以 (cx, cz) 邻近格点为中心的 (2r)^2 plaquette 和。"""
    a = phi[:, NI // 2, :]
    dens = plaque_density(a)
    u0 = int(np.argmin(np.abs(ICO - cx)))
    v0 = int(np.argmin(np.abs(ICO - cz)))
    return float(dens[u0 - r:u0 + r, v0 - r:v0 + r].sum())


def energy(phi):
    e = KAPPA * float(np.sum(1.0 - np.cos(phi)))
    for ax in range(3):
        e += 0.5 * J0 * float(np.sum(1.0 - np.cos(mi(np.diff(phi, axis=ax)))))
    return e


def grad(phi):
    p = np.pad(phi, 1)
    ctr = (slice(1, -1),) * 3
    s = (np.sin(p[2:, 1:-1, 1:-1] - p[ctr]) + np.sin(p[:-2, 1:-1, 1:-1] - p[ctr])
         + np.sin(p[1:-1, 2:, 1:-1] - p[ctr]) + np.sin(p[1:-1, :-2, 1:-1] - p[ctr])
         + np.sin(p[1:-1, 1:-1, 2:] - p[ctr]) + np.sin(p[1:-1, 1:-1, :-2] - p[ctr]))
    return -0.5 * J0 * s + KAPPA * np.sin(phi)


def flow(phi, steps, eta=0.02):
    phi = phi.copy()
    for _ in range(steps):
        phi = phi - eta * grad(phi)
    return phi


def ring_energy_box(n):
    c = np.arange(n, dtype=float) - n / 2.0 + 0.5
    x, y, z = np.meshgrid(c, c, c, indexing="ij")
    phi = mi(np.arctan2(z, np.sqrt(x ** 2 + y ** 2) - 4.0))
    e = KAPPA * float(np.sum(1.0 - np.cos(phi)))
    for ax in range(3):
        e += 0.5 * J0 * float(np.sum(1.0 - np.cos(mi(np.diff(phi, axis=ax)))))
    return e


APEX = [
    ("theta=0", 1, (4.0, 0.0)),
    ("theta=pi/2", 0, (4.0, 0.0)),
    ("theta=pi", 1, (-4.0, 0.0)),
    ("theta=3pi/2", 0, (-4.0, 0.0)),
]


def ring_readings(phi, rad=2):
    out = []
    for tag, axis, ctr in APEX:
        for s in (IC, IC - 1):
            out.append((f"{tag} s={s}", winding(phi, axis, s, center=ctr, rad=rad)))
    return out


def check(tag, got, want, tol=1e-9):
    ok = abs(got - want) <= tol
    RESULTS.append((tag, got, want, ok))
    print(f"  {tag:<44} 实测 {got:+.4f}   期望 {want:+.1f}   {'通过' if ok else '未过'}")
    return ok


def main():
    print("=" * 80)
    print("vortex_init_apparatus.py")
    print("C1-0：涡旋初始化与横截面绕数测量装置")
    print("=" * 80)

    print()
    print("[M1] 装置自检：直线涡的横截面绕数（三方向，每向三处截面）")
    for axis, core in ((0, (0.0, 0.0)), (1, (0.0, 3.0)), (2, (2.0, 0.0))):
        vals = [winding(straight(axis, core), axis, s, center=core, rad=2) for s in (10, IC, 36)]
        print(f"  法向 axis={axis} 芯位 {core}: " + "  ".join(f"{v:+.4f}" for v in vals))
        check(f"M1 直线涡 axis={axis} 三截面", max(abs(v - 1.0) for v in vals), 0.0)

    print()
    print("[M2] 标准环初态（Rc = 4）四方位读数，两个跨芯格面")
    phi_ring = ring_naive(4.0)
    for s in (IC, IC - 1):
        vals = [winding(phi_ring, ax, s, center=ctr, rad=2) for _, ax, ctr in APEX]
        print(f"  平面 s={s}: " + "  ".join(f"{t}={v:+.4f}" for (t, _, _), v in zip(APEX, vals)))
    got = [v for _, v in ring_readings(phi_ring)]
    check("M2 四方位 x 两格面 |W| = 1", float(min(abs(v) for v in got)), 1.0)
    check("M2 读数整数性", float(max(abs(v - round(v)) for v in got)), 0.0, tol=1e-12)
    print("  符号约定：读数 = 绕数 x (平面内升序法向 · 芯线切线)。标准环（n = 1）在 theta=0 读 +1，")
    print("  n = -1 整体反号；同一环在 theta=0 与 theta=pi 两方位反号，因两交叉点的芯线切线反向")

    print()
    print("[M3] 半径扫描：标准环在六种半径下的读数（局域窗口 rad=1）")
    for Rc in (2.0, 2.5, 3.0, 4.0, 5.0, 6.0):
        phi = ring_naive(Rc)
        w1 = winding(phi, 1, IC, center=(Rc, 0.0), rad=1)
        wh = winding(phi, 1, IC, half=(0, +1))
        print(f"  Rc={Rc:.1f}  局域窗口 {w1:+.4f}   x>0 半平面 {wh:+.4f}")
        check(f"M3 Rc={Rc:.1f} 局域窗口 |W| = 1", abs(w1), 1.0)

    print()
    print("[M4] 负控制：五种错误读法")
    for xv in (5.5, 4.5, 3.5):
        v = winding(phi_ring, 0, nearest(xv), center=(0.0, 0.0), rad=3)
        print(f"  (a) 顶点法向平面 x={xv:+.1f} rad=3: {v:+.4f}")
        check(f"M4a 顶点法向平面 x={xv:+.1f}", v, 0.0)
    v = winding(phi_ring, 0, nearest(3.5), center=(0.0, 0.0), rad=2)
    print(f"  (b) 同平面单交叉窗口 x=+3.5 rad=2: {v:+.4f}（规则二：窗口须只套住一个交叉点）")
    check("M4b 单交叉窗口 |W| = 1", abs(v), 1.0)
    check("M4c 整平面求和（两交叉相消）", winding(phi_ring, 1, IC), 0.0)
    check("M4d 环所在平面 z=0", winding(phi_ring, 2, nearest(0.5), center=(0.0, 0.0), rad=6), 0.0)
    phi_i = ring_int(4.0)
    vp = block_sum_int(phi_i, 4.0, 0.0)
    vm = block_sum_int(phi_i, -4.0, 0.0)
    print(f"  (e) 整数格点（芯位落格点）：+x 侧 2x2 块 {vp:+.4f}，-x 侧 2x2 块 {vm:+.4f}")
    check("M4e 整数格点退化读数（+x 侧）", vp, 0.0)
    check("M4e 整数格点退化读数（-x 侧加倍）", vm, -2.0)
    print("      机制：芯位落格点时跨芯键相位差恰为 ±π，mi(-π) = mi(+π) = -π，反对称性破坏")

    print()
    print("[M5] 中间尝试所用初态与旧零读数的复现（更正旧诊断）")
    check("M5 未遮罩环横向读数 |W| = 1", float(min(abs(v) for _, v in ring_readings(phi_ring))), 1.0)
    check("M5 复现旧零读数（顶点法向平面 x=+4.5）",
          winding(phi_ring, 0, nearest(4.5), center=(0.0, 0.0), rad=3), 0.0, tol=1e-12)

    print()
    print("[M6] 窗口几何：横向读数随窗口半宽 rad（theta=0，中心 = 顶点）")
    for r in range(1, 9):
        print(f"      rad={r}: 横向 {winding(phi_ring, 1, IC, center=(4.0, 0.0), rad=r):+.4f}"
              f"   x=+4.5 平面 {winding(phi_ring, 0, nearest(4.5), center=(0.0, 0.0), rad=r):+.4f}"
              f"   x=+3.5 平面 {winding(phi_ring, 0, nearest(3.5), center=(0.0, 0.0), rad=r):+.4f}")
    keep = [abs(abs(winding(phi_ring, 1, IC, center=(4.0, 0.0), rad=r)) - 1.0) for r in range(1, 8)]
    check("M6 rad = 1 至 7 读数稳定", max(keep), 0.0)
    check("M6 rad = 8 触及对侧交叉点", winding(phi_ring, 1, IC, center=(4.0, 0.0), rad=8), 0.0)
    corners = [winding(phi_ring, 1, IC, center=(cx, cz), rad=1)
               for cx in (3.5, 4.5) for cz in (-0.5, 0.5)]
    print(f"  芯位四邻角窗口（rad=1）：" + "  ".join(f"{v:+.4f}" for v in corners))
    check("M6 芯位四邻角窗口仍 |W| = 1", max(abs(abs(v) - 1.0) for v in corners), 0.0)

    print()
    print("[M7] 半平面约定与遮罩构造的缺陷（嵌套反向环）")
    check("M7 标准环 x>0 半平面读数 |W| = 1", abs(winding(phi_ring, 1, IC, half=(0, +1))), 1.0)
    phi_m = ring_masked(4.0)
    real = crossings(phi_m)
    print(f"  遮罩构造真实涡量 plaquette {len(real)} 条（标准环只有 2 条）")
    for rho, z, v in sorted(real):
        print(f"      rho={rho:5.2f} z={z:+5.1f}  dens={v:+.1f}")
    check("M7 遮罩构造 x>0 半平面读数为零（内外环相消）",
          winding(phi_m, 1, IC, half=(0, +1)), 0.0)
    check("M7 遮罩构造内环局域窗口仍读 1",
          abs(winding(phi_m, 1, IC, center=(4.0, 0.0), rad=1)), 1.0)
    check("M7 遮罩构造交叉条数 = 4（内外各一对）", float(len(real)), 4.0)

    print()
    print("[M8] 符号控制：n = 1 与 n = -1 的环、直线涡")
    check("M8 环 n=+1 读数（theta=0 方位）",
          winding(ring_naive(4.0, n=1), 1, IC, center=(4.0, 0.0), rad=2), 1.0)
    check("M8 环 n=-1 读数（theta=0 方位）",
          winding(ring_naive(4.0, n=-1), 1, IC, center=(4.0, 0.0), rad=2), -1.0)
    check("M8 环 n=+1 在 theta=pi 方位反号",
          winding(ring_naive(4.0, n=1), 1, IC, center=(-4.0, 0.0), rad=2), -1.0)
    check("M8 直线涡 n=-1 读数",
          winding(straight(1, (0.0, 0.0), n=-1), 1, IC, center=(0.0, 0.0), rad=2), -1.0)

    print()
    print("[M9] 涡量局域化：标准环每面真实 plaquette 与到芯圆距离")
    real = crossings(phi_ring)
    dmax = max(np.hypot(rho - 4.0, z) for rho, z, _ in real)
    print(f"  非零 plaquette {len(real)} 条，到芯圆最大距离 {dmax:.2f} 格")
    check("M9 涡量条数 = 2（每交叉一条）", float(len(real)), 2.0)
    check("M9 到芯圆距离 <= 1 格", 1.0 if dmax <= 1.0 else 0.0, 1.0)

    print()
    print("[M10] 塌缩认证：梯度流下 (E, 交叉, 读数) 同步（裸环无锚核）")
    print(f"     步数        E          交叉条数   交叉半径    局域读数")
    phi = phi_ring
    samples = []
    blocks = [50, 50, 50, 50, 100, 100, 200, 200, 400]
    total = 0
    prev_e = energy(phi)
    mono = True
    track = True
    for b in blocks:
        phi = flow(phi, b)
        total += b
        e = energy(phi)
        cr = crossings(phi)
        if len(cr):
            loc = max(cr, key=lambda t: abs(t[2]))
            wire = winding(phi, 1, IC, center=(loc[0], loc[1]), rad=1)
            print(f"     {total:>5}  {e:12.4e}  {len(cr):>6}     {loc[0]:6.2f}    {wire:+.4f}")
            if abs(abs(wire) - 1.0) > 1e-9:
                track = False
            samples.append((total, e, loc[0], wire))
        else:
            print(f"     {total:>5}  {e:12.4e}       无交叉       读数 0")
            samples.append((total, e, 0.0, 0.0))
        if e > prev_e + 1e-12:
            mono = False
        prev_e = e
    check("M10 能量单调下降", 1.0 if mono else 0.0, 1.0)
    check("M10 存在交叉时局域读数 |W| = 1", 1.0 if track else 0.0, 1.0)
    check("M10 末态无交叉且能量归零", 1.0 if (len(crossings(phi)) == 0 and prev_e < 1e-6) else 0.0, 1.0)
    print("      读数：裸环先缩芯（4.03 -> 1.12 格）再湮灭（约 200 步），与环裁决实验的塌缩一致")

    print()
    print("[M11] 抗扰动：固定种子光滑扰动（幅度 0.5 rad）后读数保持")
    rng = np.random.default_rng(20260912)
    noise = rng.normal(size=(N, N, N))
    for _ in range(3):
        p = np.pad(noise, 1, mode="edge")
        noise = (p[2:, 1:-1, 1:-1] + p[:-2, 1:-1, 1:-1] + p[1:-1, 2:, 1:-1]
                 + p[1:-1, :-2, 1:-1] + p[1:-1, 1:-1, 2:] + p[1:-1, 1:-1, :-2]) / 6.0
    noise /= np.max(np.abs(noise))
    phi_p = mi(phi_ring + 0.5 * noise)
    check("M11 扰动后四方位 x 两格面 |W| = 1",
          float(min(abs(v) for _, v in ring_readings(phi_p))), 1.0)
    check("M11 扰动 + 30 步流后 |W| = 1",
          float(min(abs(v) for _, v in ring_readings(flow(phi_p, 30)))), 1.0)

    print()
    print("[M12] 标准环初态的能量随盒尺寸（判断是否局域对象）")
    e24 = ring_energy_box(24)
    ratios = []
    for n in (24, 32, 40, 48, 56):
        e = ring_energy_box(n)
        ratios.append(e)
        print(f"      N={n:3d}   E0={e:10.1f}")
    growth = ratios[-1] / ratios[0]
    print(f"      E0(56)/E0(24) = {growth:.2f}（盒增大 2.33 倍）")
    check("M12 能量随盒严格增长且增幅 >= 5 倍", 1.0 if (growth >= 5.0 and all(
        ratios[i] < ratios[i + 1] for i in range(len(ratios) - 1))) else 0.0, 1.0)
    print("      读数：标准环的相位在无穷远趋于极角，边界映射非平凡，故能量随盒增长，")
    print("      属非局域对象；带真空渐近的构造代价是嵌套反环（判据 M7），两者都留给 C1-1 处理")

    print()
    print("[图] 横切平面涡量图与窗口几何")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        dens = plaque_density(np.take(phi_ring, IC, axis=1))
        dm = plaque_density(np.take(phi_m, IC, axis=1))
        fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))
        crop = dens[17:38, 15:32]
        im = ax[0].imshow(crop, origin="lower", cmap="RdBu_r", vmin=-1, vmax=1)
        i0, k0 = nearest(4.0) - 17, nearest(0.0) - 15
        ax[0].add_patch(plt.Rectangle((k0 - 2, i0 - 2), 4, 4, fill=False, ec="k", lw=1.5))
        nzm = np.argwhere(np.abs(dm[17:38, 15:32]) > 0.5)
        ax[0].plot(nzm[:, 1], nzm[:, 0], "kx", ms=6)
        ax[0].set_title("transverse plane y=+0.5: standard ring (o) and masked ring (x)")
        ax[0].set_xlabel("z index")
        ax[0].set_ylabel("x index")
        fig.colorbar(im, ax=ax[0], shrink=0.85)
        rs = np.arange(1, 10)
        ax[1].plot(rs, [winding(phi_ring, 1, IC, center=(4.0, 0.0), rad=int(r)) for r in rs],
                   "o-", label="standard ring")
        ax[1].plot(rs, [winding(phi_m, 1, IC, center=(4.0, 0.0), rad=int(r)) for r in rs],
                   "x-", label="masked ring (nested)")
        ax[1].plot(rs, [winding(phi_ring, 0, nearest(4.5), center=(0.0, 0.0), rad=int(r)) for r in rs],
                   "s-", label="apex-normal plane")
        ax[1].axhline(0, color="k", lw=0.6)
        ax[1].set_xlabel("window half-width (lattice)")
        ax[1].set_ylabel("winding reading")
        ax[1].set_title("reading vs window geometry")
        ax[1].legend(fontsize=8)
        fig.tight_layout()
        out = "D:/software/TianXingFC/images/vortex_init_apparatus.png"
        fig.savefig(out, dpi=140)
        print(f"  已存 {out}")
    except Exception as exc:
        print(f"  绘图跳过：{exc}")

    print()
    print("=" * 80)
    fails = [r for r in RESULTS if not r[3]]
    print(f"判据合计 {len(RESULTS)} 条，通过 {len(RESULTS) - len(fails)} 条，未过 {len(fails)} 条")
    for tag, got, want, _ in fails:
        print(f"  未过：{tag}  实测 {got:+.6f} 期望 {want:+.1f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
