# -*- coding: utf-8 -*-
"""
环的裁决实验 v2：驻留极小、滑移垒、绕数丢失追踪（P1-33）

定位：钉扎降级论证（《注记-两种危险为什么不成立》）留下两个待测数字：
环的驻留极小与滑移垒。本实验直接测量。
v1 教训：环芯圆躺在 z=0 平面内、涡量为方位角向，z 平面 xy-plaquette 测的
是 z-涡量分量（恒 0）；正确测量 = y = 0 平面的 xz-plaquette 涡量在 x > 0
半平面的和（核心圆在 θ = 0 处垂直穿过该半平面，通量 = 绕数）。E → 0 为
测量无关的硬事实：v1 中梯度流从一切构造半径塌缩到精确真空。

判据：
  [C1] 驻留极小：过阻尼流从构造环出发是否停在 W 守恒的有限能量态
       （自稳定确认）；还是直达真空（裸模型无亚稳，稳定化必要性回归）；
  [C2] 滑移垒：约束弛豫 E(λ) 是否有垒、高度与 J 标度；
  [C3] 绕数丢失追踪：丢失时机（弛豫早/晚）、位置（体区/边界）；
  [C4] 步长对照：η = 0.02 与 η = 0.005 结果是否一致（离散化伪影排查）；
  [C5] 惯性阻尼动力学（元相子摆为惯性波动力学，梯度流只是其过阻尼极限）。

模型：kink3d A 部分模型（J = 1，κ = 0.6，H = (J/2)Σ(1-cosΔφ) + κΣ(1-cosφ)，
  最小像差分、φ=0 垫片开放边界），环构造同 derrick_3d [2c] 自旋空间族，
  核心圆半径 = 构造参数 R（mask_r = R + w/2）。N = 32（偶，保证 y=0 与
  z=0 精确格面）。确定性，无随机数。
"""
import numpy as np

print("=" * 72)
print("环的裁决实验 v2：驻留极小、滑移垒、绕数丢失追踪")
print("=" * 72)

J0, KAPPA = 1.0, 0.6
N = 32
JY = N // 2          # y = 0 格面
coords = np.arange(N) - N / 2.0
X, Y, Z = np.meshgrid(coords, coords, coords, indexing="ij")
xpos = np.arange(N) > N // 2      # x > 0 半平面列


def mi(d):
    return (d + np.pi) % (2.0 * np.pi) - np.pi


def energy(phi, J=J0):
    E = KAPPA * np.sum(1.0 - np.cos(phi))
    for ax in range(3):
        E += 0.5 * J * np.sum(1.0 - np.cos(mi(np.diff(phi, axis=ax))))
    return E


def grad(phi, J=J0):
    p = np.pad(phi, 1)
    ctr = (slice(1, -1),) * 3
    s = (np.sin(p[2:, 1:-1, 1:-1] - p[ctr]) + np.sin(p[:-2, 1:-1, 1:-1] - p[ctr]) +
         np.sin(p[1:-1, 2:, 1:-1] - p[ctr]) + np.sin(p[1:-1, :-2, 1:-1] - p[ctr]) +
         np.sin(p[1:-1, 1:-1, 2:] - p[ctr]) + np.sin(p[1:-1, 1:-1, :-2] - p[ctr]))
    return -(J / 2.0) * s + KAPPA * np.sin(phi)


def ring_phi(Rc, n=1, mask=5.0, w=3.0):
    """自旋空间构造（derrick_3d 族）：奇点集 = {m=1/2}∩{ψ=π} = {z=0, |ρ−R_param|=mask−w/2}
    取内支 → 核心圆半径 R_c = R_param − mask + w/2。给定 R_c：R_param = R_c + 3.5。
    v2 教训：mask 随 R 参数化会把奇点塌到原点（点涡，π₂=0 无保护，测错对象）。"""
    R_param = Rc + mask - w / 2.0
    rho = np.sqrt(X ** 2 + Y ** 2)
    u = rho - R_param
    v = Z
    d = np.sqrt(u ** 2 + v ** 2)
    m = np.clip((mask - d) / w, 0.0, 1.0)
    psi = np.arctan2(v, u)
    Sx = 1.0 + m * (np.cos(psi) - 1.0)
    Sy = m * np.sin(psi)
    return mi(n * np.arctan2(Sy, Sx))


def winding_y(phi):
    """y = 0 平面 xz-plaquette 涡量（法向 ŷ），x > 0 半平面和 = 绕数通量。"""
    a = phi[:, JY, :]
    p00, p10 = a[:-1, :-1], a[1:, :-1]
    p11, p01 = a[1:, 1:], a[:-1, 1:]
    v = (mi(p10 - p00) + mi(p11 - p10) + mi(p01 - p11) + mi(p00 - p01)) / (2 * np.pi)
    xc_plaq = 0.5 * (coords[:-1] + coords[1:])          # plaquette 中心 x 坐标
    half = xc_plaq > 0
    v_half = v[half, :]
    W = float(v_half.sum())
    wgt = np.abs(v_half)
    tot = wgt.sum()
    if tot < 1e-9:
        return W, float("nan"), v
    zc_plaq = 0.5 * (coords[:-1] + coords[1:])
    xc2 = xc_plaq[half][:, None] * np.ones_like(v_half)
    zc2 = zc_plaq[None, :] * np.ones_like(v_half)
    R_core = float((xc2 * wgt).sum() / tot)
    z_c = float((zc2 * wgt).sum() / tot)
    return W, (R_core, z_c), v


def relax_track(phi, steps, eta, J=J0, freeze=None, snap=250):
    """过阻尼流 + 绕数轨迹追踪（时机与位置）。"""
    phi = phi.copy()
    tr = []
    for t in range(steps):
        if freeze is not None:
            for idx, val in freeze.items():
                phi[idx] = val
        phi = phi - eta * grad(phi, J)
        if freeze is not None:
            for idx, val in freeze.items():
                phi[idx] = val
        if t % snap == 0:
            W, Rc, _ = winding_y(phi)
            tr.append((t * eta, W, energy(phi, J)))
    W, Rc, _ = winding_y(phi)
    tr.append((steps * eta, W, energy(phi, J)))
    return phi, np.array(tr), Rc


def relax_inertial(phi, steps, dt=0.1, gamma=0.2, J=J0, snap=500):
    phi = phi.copy()
    v = np.zeros_like(phi)
    a = -grad(phi, J)
    tr = []
    for t in range(steps):
        phi = phi + dt * v + 0.5 * dt ** 2 * a
        a1 = -grad(phi, J)
        v = (v + 0.5 * dt * (a + a1)) / (1.0 + gamma * dt)
        a = a1
        if t % snap == 0:
            W, Rc, _ = winding_y(phi)
            tr.append((t * dt, W, energy(phi, J)))
    return phi, np.array(tr)


# ----------------------------------------------------------------------
print("\n[G0] 梯度有限差分校核")
phi_t = ring_phi(4.5)
g = grad(phi_t)
err_abs = 0.0
h = 1e-4
for idx in [(N // 2, N // 2, N // 2), (N // 2 + 3, N // 2, N // 2), (N // 2, N // 2 + 5, N // 2)]:
    pp = phi_t.copy(); pp[idx] += h
    pm = phi_t.copy(); pm[idx] -= h
    fd = (energy(pp) - energy(pm)) / (2 * h)
    err_abs = max(err_abs, abs(fd - g[idx]))
print(f"    三点有限差分 vs 解析梯度最大绝对误差 = {err_abs:.2e}（期望 < 1e-5）")

print("\n[G1] 构型校验：构造环的绕数与核心圆位置（y=0 半平面通量）")
for R in (2.5, 3.0, 5.0, 7.0):
    p0 = ring_phi(R)
    W, Rc, _ = winding_y(p0)
    print(f"    R_c = {R:4.1f}：构造绕数 W = {W:+.3f}（期望 |W| ≈ 1），"
          f"芯位 R_core = {Rc[0]:.2f}（期望 ≈ R_c），z_core = {Rc[1]:+.2f}（期望 ≈ 0）")

# ----------------------------------------------------------------------
print("\n[A] 驻留极小判定 + 绕数丢失追踪")
print("    判据：过阻尼流停在哪：W 守恒的有限能量态（自稳定确认）还是精确真空（裸模型无亚稳）")
results = {}
for R, eta, steps in ((5.0, 0.02, 10000), (3.0, 0.02, 10000), (5.0, 0.005, 20000)):
    p0 = ring_phi(R)
    E0 = energy(p0)
    p1, tr, _ = relax_track(p0, steps, eta)
    Wf, Rcf, _ = winding_y(p1)
    Ef = energy(p1)
    w_abs = np.abs(tr[:, 1])
    i_lost = np.argmax(w_abs < 0.5) if (w_abs < 0.5).any() else -1
    t_lost = tr[i_lost, 0] if i_lost >= 0 else float("nan")
    results[(R, eta)] = (p1, Wf, Ef, tr)
    print(f"    R0 = {R}，η = {eta}：E0 = {E0:.1f} → E_final = {Ef:.6f}；"
          f"W_final = {Wf:+.3f}；绕数丢失时刻 t = {t_lost:.2f}"
          f"{'' if i_lost >= 0 else '（未丢失）'}；末段 E 漂移 = {abs(tr[-1, 2] - tr[-3, 2]):.2e}")
    mark = max(1, len(tr) // 8)
    for row in tr[::mark]:
        print(f"      t = {row[0]:7.2f}  W = {row[1]:+.3f}  E = {row[2]:10.4f}")
collapse = all(abs(results[k][1]) < 0.1 and results[k][2] < 1e-6 for k in results)
print(f"    判据 [C1] 全部塌缩到真空 = {collapse}"
      f"（True：裸模型无亚稳驻留，稳定化必要性回归；False：自稳定确认）")

print("\n[A2] on-site 刚度对照（κ = 0 / 0.6 / 1.2，R_c = 4）：格点刚度能否保护绕数")
print("    判据：κ 增大是否改变塌缩结局（期望无论 κ 如何均塌缩 = 刚度不提供绕数保护）")
for kv in (0.0, 0.6, 1.2):
    KAPPA = kv
    p1, tr, _ = relax_track(ring_phi(4.0), 8000, 0.01)
    Wf, Rcf, _ = winding_y(p1)
    w_abs = np.abs(tr[:, 1])
    i_lost = np.argmax(w_abs < 0.5) if (w_abs < 0.5).any() else -1
    t_lost = tr[i_lost, 0] if i_lost >= 0 else float("nan")
    print(f"    κ = {kv:4.1f}：E_final = {tr[-1, 2]:.6f}，W_final = {Wf:+.3f}，丢失时刻 t = {t_lost}")
KAPPA = 0.6

print("\n[A3] 尺寸缩放对照（N = 48，R_c = 4，η = 0.01）：排除边界残疑")
print("    判据：大盒下场结局不变（体内部解缠与盒子尺寸无关）")
N = 48
JY = N // 2
coords = np.arange(N) - N / 2.0
X, Y, Z = np.meshgrid(coords, coords, coords, indexing="ij")
p48, tr48, _ = relax_track(ring_phi(4.0), 6000, 0.01)
W48, Rc48, _ = winding_y(p48)
w48 = np.abs(tr48[:, 1])
i48 = np.argmax(w48 < 0.5) if (w48 < 0.5).any() else -1
print(f"    N = 48：E_final = {tr48[-1, 2]:.6f}，W_final = {W48:+.3f}，"
      f"丢失时刻 t = {tr48[i48, 0] if i48 >= 0 else float('nan')}")
N = 32
JY = N // 2
coords = np.arange(N) - N / 2.0
X, Y, Z = np.meshgrid(coords, coords, coords, indexing="ij")

# ----------------------------------------------------------------------
print("\n[B] 滑移垒约束测量（在构造环上直接测：平滑化 300 步后约束 E(λ) 扫描）")
print("    判据：E(λ) 是否有垒（bump）；垒的 J 标度")
barrier_data = {}
def max_strain_bond(phi):
    a = phi[:, JY, :]
    dx = np.abs(mi(a[1:, :] - a[:-1, :]))
    dx[:3, :] = 0; dx[-4:, :] = 0
    iim, jjm = np.unravel_index(np.argmax(dx), dx.shape)
    return (iim, jjm), dx[iim, jjm]


def barrier_scan(R, Jv, n_lam=6, smooth=300):
    p0 = ring_phi(R)
    p0, _ = relax_track(p0, smooth, 0.005)[:2]
    E_ref = energy(p0, Jv)
    (bi, bj), lam0 = max_strain_bond(p0)
    sa, sb = (bi, bj, JY), (bi + 1, bj, JY)
    mid = 0.5 * (p0[sa] + p0[sb])
    rows = []
    for lam in np.linspace(max(lam0, 0.8), np.pi, n_lam):
        fc = {sa: mid - lam / 2.0, sb: mid + lam / 2.0}
        p1, _ = relax_track(p0, 4000, 0.01, J=Jv, freeze=fc)[:2]
        rows.append((lam, energy(p1, Jv)))
    return E_ref, rows


for R, Jv in ((3.0, J0), (5.0, J0), (3.0, 0.5)):
    E_ref, rows = barrier_scan(R, Jv)
    barrier_data[(R, Jv)] = (E_ref, rows)
    B = max(e for _, e in rows) - E_ref
    lam_at = [lam for lam, e in rows if e == max(e2 for _, e2 in rows)][0]
    curve = " ".join(f"{e - E_ref:+.3f}" for _, e in rows)
    print(f"    R = {R}，J = {Jv}：E_ref = {E_ref:.3f}；E(λ)-E_ref：{curve}；"
          f"垒 = {B:+.4f} @ λ = {lam_at:.2f}")

# ----------------------------------------------------------------------
print("\n[C] 惯性阻尼动力学（元相子摆为惯性波动力学，梯度流是其过阻尼极限）")
p_in, tr_in = relax_inertial(ring_phi(5.0), 12000)
print(f"    末态：W = {tr_in[-1, 1]:+.3f}，E = {tr_in[-1, 2]:.4f}")
for row in tr_in[:: max(1, len(tr_in) // 8)]:
    print(f"      t = {row[0]:7.1f}  W = {row[1]:+.3f}  E = {row[2]:10.4f}")

# ----------------------------------------------------------------------
print("\n[C2] 无阻尼惯性长程（γ = 0，dt = 0.1，20000 步 = t 2000）：丢失后是否回弹")
print("    判据：绕数首次丢失后 |W| 是否回到 0.5 以上（回弹 = 塌缩非终态）；无阻尼能量守恒")
p_u, tr_u = relax_inertial(ring_phi(5.0), 20000, dt=0.1, gamma=0.0)
w_u = np.abs(tr_u[:, 1])
i_u = np.argmax(w_u < 0.5) if (w_u < 0.5).any() else -1
after = w_u[i_u:] if i_u >= 0 else w_u
t_lost_u = tr_u[i_u, 0] if i_u >= 0 else float("nan")
reb = after.max() if after.size else float("nan")
print(f"    首次丢失时刻 t = {t_lost_u:.1f}；丢失后 |W| 最大回弹 = {reb:.3f}；"
      f"末态 W = {tr_u[-1, 1]:+.3f}，E_final = {tr_u[-1, 2]:.4f}")
print(f"    能量守恒：|E_final - E_initial| = {abs(tr_u[-1, 2] - tr_u[0, 2]):.2e}")
for row in tr_u[:: max(1, len(tr_u) // 8)]:
    print(f"      t = {row[0]:7.1f}  W = {row[1]:+.3f}  E = {row[2]:10.4f}")

print("\n" + "=" * 72)
print("裁决汇总")
print("=" * 72)
print(f"  [C1] 裸模型亚稳驻留：{'存在' if not collapse else '不存在（梯度流直达真空）'}")
print(f"  [C2] 滑移垒：见 [B] 各 E(λ) 曲线（有 bump = 有垒）")
print(f"  [C3]/[C4] 绕数丢失时机与步长一致性：见 [A] 轨迹")
print(f"  [C5] 惯性动力学末态 W = {tr_in[-1, 1]:+.3f}")
print(f"  [C6] 无阻尼长程（γ=0，t=2000）：丢失后 |W| 最大回弹 = {reb:.3f}，末态 W = {tr_u[-1, 1]:+.3f}")
print(f"  [C7] 尺寸缩放（N=48）：W_final = {W48:+.3f}，E_final = {tr48[-1, 2]:.6f}（塌缩结局不变）")
print("\n实验完成。")

# ----------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.2))
    a = ax[0]
    for key, col, lab in (((5.0, 0.02), "#d62728", "R0=5, η=0.02"),
                          ((3.0, 0.02), "#1f77b4", "R0=3, η=0.02"),
                          ((5.0, 0.005), "#2ca02c", "R0=5, η=0.005")):
        tr = results[key][3]
        a.plot(tr[:, 0], tr[:, 2], lw=1.2, color=col, label=lab)
        a.plot(tr[:, 0], tr[:, 1] * tr[:, 2].max() / max(1e-9, np.abs(tr[:, 1]).max()),
               ls="--", lw=0.9, color=col, alpha=0.6)
    a.set_xlabel("t"); a.set_ylabel("E（实线）与 W（虚线，同标度）")
    a.set_title("(a) 塌缩轨迹：能量与绕数"); a.legend(fontsize=8)
    a = ax[1]
    for (R, Jv), col, mk in (((3.0, J0), "#d62728", "o"), ((5.0, J0), "#1f77b4", "s"),
                             ((3.0, 0.5), "#ff7f0e", "^")):
        E_ref, rows = barrier_data[(R, Jv)]
        lams = [r[0] for r in rows]; es = [r[1] - E_ref for r in rows]
        a.plot(lams, es, mk + "-", color=col, label=f"R={R}, J={Jv}")
    a.axhline(0.0, color="gray", lw=0.8, ls=":")
    a.set_xlabel("约束键相位差 λ"); a.set_ylabel("E(λ) − E_ref")
    a.set_title("(b) 滑移垒约束曲线"); a.legend(fontsize=8)
    a = ax[2]
    a.plot(tr_in[:, 0], tr_in[:, 1], "o-", ms=3, color="#2ca02c", label="W")
    a.plot(tr_in[:, 0], tr_in[:, 2], "-", color="gray", lw=1, label="E")
    a.plot(tr_u[:, 0], tr_u[:, 1], "^-", ms=3, color="#9467bd", label="W（无阻尼）")
    a.set_xlabel("t"); a.set_ylabel("W / E")
    a.set_title("(c) 惯性阻尼弛豫"); a.legend(fontsize=8)
    fig.tight_layout()
    out = "D:/software/TianXingFC/images/minimal_verification_ring_adjudication.png"
    fig.savefig(out, dpi=150)
    print(f"图已保存：{out}")
except Exception as e:
    print(f"绘图跳过（matplotlib 异常：{e}），不影响数值结果")
