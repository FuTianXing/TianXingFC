# -*- coding: utf-8 -*-
r"""
minimal_verification_amplitude_channel_ring.py

W1 至 W4：振幅通道能否给涡旋环一个稳定半径（复合位错载体层的最小验证）。

定位
----
载体层裁决（见 [[注记-载体层裁决：纯相位的失败边界与振幅通道的实现路径]]）把锚核
定为振幅锁定，并要求判据 W1、W2 与附加读数 W4。本脚本在最小复标量
（Ginzburg-Landau 型）模型上测这些判据，并加一条决定性检验 W3：自由复场弛豫下
环能否驻留。

模型
----
格点复标量 $\psi = A e^{i\theta}$，开放边界（边界外取真空 $\psi = 0$）：

$$H = \sum_{\text{bonds}} |\psi_i - \psi_j|^2 + \lambda \sum_{\text{sites}} (A_i^2 - A_0(x_i)^2)^2$$

$N = 32$，格距 $1$，相干长度 $\xi = 1/\sqrt{4\lambda}$。锚核取管状，把管内偏好振幅
下调：$A_0 = 1 - d\exp(-d_c(R_a)^2/2w^2)$，$d_c$ 为到半径 $R_a$、位于 $z = 0$
平面的圆的距离。

径向扇区（相位锁定）
----
自由弛豫复场会让绕数经相位滑移卸载，测不出径向曲线。先固定相位为环剖面
$\theta = \operatorname{arctan2}(z,\rho - R)$，只弛豫振幅 $A$，则绕数恒为 $1$、芯圆
半径恒为 $R$。该扇区能量是 $A$ 的二次型加四次项，耦合固定为
$J_{ij} = \cos(\theta_i - \theta_j)$，$E(R)$ 即该扇区的最小能量。锁定扇区给出的
$E(R)$ 是同一半径下真实极小值的上界：锁相无稳定半径时，放开相位只会更低，故真实
极小也无稳定半径。反方向不成立，故锁定扇区出现极小后仍须用 W3 复验。

判据
----
W1a 对照（无锚核）：$E(R)$ 严格单调递增，三档 $\lambda$。
W1b 锚核节省 $\Delta E(R) = E_{\text{锚}}(R) - E_{\text{对照}}(R)$：内部极小位置与深度。
W1c 参数网格（$\lambda$、$d$、$w$）：**总能量** $E_{\text{锚}}(R)$ 是否出现内部极小，
    即是否存在稳定半径 $R^*$；给出出现稳定半径的最小 $\lambda$（边界判据）。
W2  锚核半径扫描：$\Delta E(R)$ 的极小位置随 $R_a$ 单调不减，并报位置偏移。
W3  决定性检验：自由复场弛豫（不锁相位）从锁定扇区最优态出发，绕数是否保持 $1$、
    芯半径是否停在有限值。
W4  附加读数：芯处振幅 $A(0)$ 与芯区振幅亏损，及与 $\Delta E$ 的对应。

运行
----
`python 验证脚本/minimal_verification_amplitude_channel_ring.py`
确定性（无随机数），运行约 $2$ 分钟。图存
`D:/software/TianXingFC/images/amplitude_channel_ring.png`。
"""

import numpy as np
from scipy.optimize import minimize

N = 32
N3 = N ** 3
COORD = np.arange(N, dtype=float) - N / 2.0 + 0.5
_X, _Y, _Z = np.meshgrid(COORD, COORD, COORD, indexing="ij")
RHO = np.sqrt(_X ** 2 + _Y ** 2)
SURF = ((RHO - 5.0) ** 2 + _Z ** 2) if True else None
IC = N // 2
UPT = [tuple(slice(2, N + 2) if i == ax else slice(1, N + 1) for i in range(3)) for ax in range(3)]
DNT = [tuple(slice(0, N) if i == ax else slice(1, N + 1) for i in range(3)) for ax in range(3)]
RESULTS = []
RS = [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0]


def ring_phase(R):
    return np.arctan2(_Z, RHO - R)


def cos_cache(theta):
    p = np.pad(theta, 1)
    return [(np.cos(theta - p[UPT[ax]]), np.cos(theta - p[DNT[ax]])) for ax in range(3)]


def make_fun(A0, cache, lam):
    A0sq = (A0 ** 2).reshape(N, N, N)

    def fun(x):
        A = x.reshape(N, N, N)
        A2 = A * A
        E = lam * float(np.sum((A2 - A0sq) ** 2))
        g = np.zeros_like(A)
        for ax in range(3):
            cu, cd = cache[ax]
            p = np.pad(A, 1)
            au, ad = p[UPT[ax]], p[DNT[ax]]
            E += float(np.sum(A2 - 2.0 * A * au * cu + au ** 2))
            E += float(np.sum(A2 - 2.0 * A * ad * cd + ad ** 2))
            g += 2.0 * (A - au * cu) + 2.0 * (A - ad * cd)
        g += 4.0 * lam * (A2 - A0sq) * A
        return E, g.ravel()

    return fun


def anchor(Ra, depth, w):
    if depth <= 0.0:
        return np.ones_like(RHO)
    return 1.0 - depth * np.exp(-(((RHO - Ra) ** 2 + _Z ** 2) / (2.0 * w ** 2)))


def solve_R(R, A0, lam, iters=300):
    cache = cos_cache(ring_phase(R))
    xi = 1.0 / np.sqrt(4.0 * lam)
    A_init = np.minimum(np.tanh(np.sqrt((RHO - R) ** 2 + _Z ** 2) / xi), np.maximum(A0, 0.05))
    res = minimize(make_fun(A0, cache, lam), A_init.ravel(), jac=True, method="L-BFGS-B",
                   bounds=[(0.0, 2.0)] * N3,
                   options=dict(maxiter=iters, ftol=1e-12, gtol=1e-9))
    return float(res.fun), res.x


def curve(A0, lam):
    return [solve_R(R, A0, lam)[0] for R in RS]


def interior_minimum(Rs, vals):
    for i in range(1, len(Rs) - 1):
        if vals[i] < vals[i - 1] and vals[i] < vals[i + 1]:
            return Rs[i], min(vals[i - 1], vals[i + 1]) - vals[i]
    return None


def mean_slope(vals):
    return (vals[-1] - vals[0]) / (RS[-1] - RS[0])


def check(tag, got, want, tol=1e-9):
    ok = abs(got - want) <= tol
    RESULTS.append((tag, got, want, ok))
    print(f"  {tag:<50} 实测 {got:+.4f}  期望 {want:+.1f}  {'通过' if ok else '未过'}")


def circulation_from_A(psi, R, r=3):
    th = np.angle(np.take(psi, IC, axis=1))
    i0 = int(np.argmin(np.abs(COORD - R)))
    k0 = IC
    i0 = min(max(i0, r + 1), N - r - 2)
    idx = [(i0 - r, k0 - r), (i0 + r, k0 - r), (i0 + r, k0 + r), (i0 - r, k0 + r)]
    tot = 0.0
    mif = lambda d: (d + np.pi) % (2 * np.pi) - np.pi
    for a in range(4):
        (u0, v0), (u1, v1) = idx[a], idx[(a + 1) % 4]
        n = max(abs(u1 - u0), abs(v1 - v0))
        for t in range(n):
            ua = u0 + (u1 - u0) * t // n; va = v0 + (v1 - v0) * t // n
            ub = u0 + (u1 - u0) * (t + 1) // n; vb = v0 + (v1 - v0) * (t + 1) // n
            tot += mif(th[ub, vb] - th[ua, va])
    return tot / (2 * np.pi)


def free_relax(R, Ra, d, w, lam, iters=600):
    """自由复场弛豫：从锁定扇区最优态出发，相位不锁。"""
    A0 = anchor(Ra, d, w)
    E, x = solve_R(R, A0, lam)
    A = x.reshape(N, N, N)
    psi0 = A * np.exp(1j * ring_phase(R))

    def fun(xx):
        psi = xx[:N3].reshape(N, N, N) + 1j * xx[N3:].reshape(N, N, N)
        A2 = psi.real ** 2 + psi.imag ** 2
        E = lam * float(np.sum((A2 - A0 ** 2) ** 2))
        g = np.zeros_like(psi)
        p = np.pad(psi, 1)
        for ax in range(3):
            up, dn = p[UPT[ax]], p[DNT[ax]]
            E += float(np.sum(np.abs(psi - up) ** 2)) + float(np.sum(np.abs(psi - dn) ** 2))
            g += (psi - up) + (psi - dn)
        g += 2.0 * lam * (A2 - A0 ** 2) * psi
        return E, np.concatenate([g.real.ravel(), g.imag.ravel()])

    x0 = np.concatenate([psi0.real.ravel(), psi0.imag.ravel()])
    res = minimize(fun, x0, jac=True, method="L-BFGS-B",
                   options=dict(maxiter=iters, maxfun=iters * 2, ftol=1e-11, gtol=1e-8))
    psi = res.x[:N3].reshape(N, N, N) + 1j * res.x[N3:].reshape(N, N, N)
    # 芯位：横截面 y = 0.5 上振幅最小处
    plane = np.abs(np.take(psi, IC, axis=1))
    rr = np.sqrt((COORD + 0.5)[:, None] ** 2 + (COORD + 0.5)[None, :] ** 2)
    band = (rr > 2.0) & (rr < 12.0)
    masked = np.where(band, plane, np.inf)
    i_min, k_min = np.unravel_index(np.argmin(masked), plane.shape)
    rho_core = float(np.hypot(COORD[i_min] + 0.5, 0.5))
    return float(res.fun), psi, rho_core


def main():
    print("=" * 88)
    print("minimal_verification_amplitude_channel_ring.py")
    print("W1 至 W4：振幅通道下的涡旋环径向曲线与自由弛豫驻留检验")
    print("=" * 88)

    print()
    print("[W1a] 对照（无锚核）：E(R) 三档 lambda，R 从 3 到 8 步长 0.5")
    ctrl = {}
    for lam in (0.25, 1.0, 4.0):
        vals = curve(anchor(5.0, 0.0, 1.5), lam)
        ctrl[lam] = vals
        mono = all(vals[i] < vals[i + 1] for i in range(len(vals) - 1))
        print(f"  lambda={lam:<5} E(3)={vals[0]:8.2f}  E(5)={vals[4]:8.2f}  E(8)={vals[-1]:8.2f}"
              f"   线张力斜率={mean_slope(vals):7.1f} 每格")
        check(f"W1a lambda={lam} E(R) 严格单调递增", 1.0 if mono else 0.0, 1.0)

    print()
    print("[W1b] 锚核节省 ΔE(R)（lambda=1，Ra=5，d=0.5，w=1.5）")
    lam = 1.0
    anchor_curve = curve(anchor(5.0, 0.5, 1.5), lam)
    save = [anchor_curve[i] - ctrl[lam][i] for i in range(len(RS))]
    print("  " + "  ".join(f"{R:.1f}:{v:7.1f}" for R, v in zip(RS, save)))
    got = interior_minimum(RS, save)
    print(f"  ΔE 内部极小位置与深度：{got}")
    check("W1b ΔE(R) 存在内部极小（钉扎井成立）", 1.0 if got else 0.0, 1.0)
    if got:
        print(f"        钉扎斜率上界={max(abs(save[i+1]-save[i]) for i in range(len(save)-1)):.1f} 每格"
              f"   对照线张力斜率={mean_slope(ctrl[lam]):.1f} 每格")

    print()
    print("[W1c] 参数网格：总能量 E_锚(R) 是否出现内部极小（稳定半径）")
    print(f"  {'lambda':>7}{'d':>6}{'w':>6}{'总斜率':>10}{'钉扎斜率':>10}{'内部极小':>12}")
    wins = []
    for lam2 in (0.25, 1.0, 4.0, 8.0, 16.0):
        base = ctrl[lam2] if lam2 in ctrl else curve(anchor(5.0, 0.0, 1.5), lam2)
        for d in (0.5, 0.9):
            for w in (1.5, 3.0):
                vals = curve(anchor(5.0, d, w), lam2)
                sv = [vals[i] - base[i] for i in range(len(RS))]
                mn = interior_minimum(RS, vals)
                pin = max(abs(sv[i + 1] - sv[i]) for i in range(len(sv) - 1))
                if mn:
                    wins.append((lam2, d, w, mn))
                if lam2 >= 8.0 and d >= 0.9:
                    steps = [vals[i + 1] - vals[i] for i in range(len(vals) - 1)]
                    print(f"        局部斜率诊断 lambda={lam2} d={d} w={w}: 最小步进斜率={min(steps):+.1f}"
                          f"（负值即出现稳定半径）")
                print(f"  {lam2:>7}{d:>6}{w:>6}{mean_slope(vals):>10.1f}{pin:>10.1f}"
                      f"{(str(mn[0]) + ' / ' + format(mn[1], '.1f')) if mn else '无':>12}")
    print(f"  出现稳定半径的参数组：{[(a, b, c, d2) for a, b, c, d2 in wins]}")
    print(f"  出现稳定半径的最小 lambda：{min([a for a, _, _, _ in wins]) if wins else '无'}")
    check("W1c 网格内存在稳定半径", 1.0 if wins else 0.0, 1.0)
    if wins:
        wlam = min(wins)[0]
        check("W1c 稳定半径只在大 lambda 区出现（lambda >= 4）", 1.0 if wlam >= 4.0 else 0.0, 1.0)

    print()
    print("[W2] 锚核半径扫描：ΔE(R) 极小位置随 Ra（lambda=1，d=0.5，w=1.5）")
    base = ctrl[1.0]
    mins = []
    for Ra in (3.0, 4.0, 5.0, 6.0, 7.0):
        sv = [solve_R(R, anchor(Ra, 0.5, 1.5), 1.0)[0] - base[i] for i, R in enumerate(RS)]
        mn = interior_minimum(RS, sv)
        mins.append(mn[0] if mn else float("nan"))
        print(f"  Ra={Ra:.1f}  极小位置={mins[-1]}  偏移={mins[-1] - Ra:+.1f}")
    ok_num = all(not np.isnan(x) for x in mins)
    nondec = ok_num and all(mins[i] <= mins[i + 1] for i in range(len(mins) - 1))
    check("W2 ΔE 极小位置随 Ra 单调不减", 1.0 if nondec else 0.0, 1.0)

    print()
    print("[W3] 自由复场弛豫：绕数与芯位是否守住（不发散、不卸载）")
    for (lam3, d3, w3) in [(1.0, 0.5, 1.5), (4.0, 0.9, 3.0), (8.0, 0.9, 3.0)]:
        for R in (4.0, 5.0, 6.0):
            E, psi, rho_c = free_relax(R, 5.0, d3, w3, lam3)
            Wval = circulation_from_A(psi, rho_c)
            print(f"  lambda={lam3:<4} d={d3} w={w3}  R_init={R:.0f}  E={E:9.2f}  芯位={rho_c:5.2f}  绕数={Wval:+.4f}")
            if R == 5.0:
                check(f"W3 lambda={lam3} 自由弛豫后绕数保持 |W| = 1", abs(Wval), 1.0)

    print()
    print("[W4] 芯处振幅与芯区振幅亏损（lambda=1，Ra=5，d=0.5，w=1.5）")
    for i, R in enumerate(RS[::2]):
        E, x = solve_R(R, anchor(5.0, 0.5, 1.5), 1.0)
        A = x.reshape(N, N, N)
        ic = int(np.argmin(np.abs(COORD - R))); kc = IC
        acore = float(A[ic, IC, kc])
        mask = np.sqrt((RHO - R) ** 2 + _Z ** 2) < 2.0
        deficit = float(np.sum((1.0 - A ** 2)[mask]))
        print(f"  R={R:.1f}  A(0)={acore:.3f}  芯区振幅亏损={deficit:7.1f}  ΔE={E - ctrl[1.0][2 * i]:7.2f}")
    check("W4 芯处振幅被压低 A(0) < 1", 1.0 if acore < 1.0 else 0.0, 1.0)

    print()
    print("[图] E(R) 与 ΔE(R)")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.3))
        ax[0].plot(RS, ctrl[1.0], "o-", label="control, lambda=1")
        ax[0].plot(RS, anchor_curve, "x-", label="anchor Ra=5, d=0.5, w=1.5")
        if wins:
            lam_w, d_w, w_w, _ = wins[0]
            vw = curve(anchor(5.0, d_w, w_w), lam_w)
            ax[0].plot(RS, vw, "^-", label=f"anchor lambda={lam_w}, d={d_w}, w={w_w}")
        ax[0].set_xlabel("ring radius R")
        ax[0].set_ylabel("E(R), phase-locked sector")
        ax[0].legend(fontsize=8)
        ax[0].set_title("total energy vs R")
        ax[1].plot(RS, save, "s-", label="saving dE(R), lambda=1")
        ax[1].axhline(0, color="k", lw=0.6)
        ax[1].set_xlabel("ring radius R")
        ax[1].set_ylabel("E_anchor - E_control")
        ax[1].set_title("pinning well from the amplitude channel")
        ax[1].legend(fontsize=8)
        fig.tight_layout()
        out = "D:/software/TianXingFC/images/amplitude_channel_ring.png"
        fig.savefig(out, dpi=140)
        print(f"  已存 {out}")
    except Exception as exc:
        print(f"  绘图跳过：{exc}")

    print()
    print("=" * 88)
    fails = [r for r in RESULTS if not r[3]]
    print(f"判据合计 {len(RESULTS)} 条，通过 {len(RESULTS) - len(fails)} 条，未过 {len(fails)} 条")
    for tag, got, want, _ in fails:
        print(f"  未过：{tag}  实测 {got:+.6f} 期望 {want:+.1f}")
    print("=" * 88)


if __name__ == "__main__":
    main()
