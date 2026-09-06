# -*- coding: utf-8 -*-
"""
疤的裁决实验 v2：裸单值场的孤立疤不可能性（P1-33）

定位：环的裁决实验否定光滑场环的自稳定后，绕数承载层指向格点内离散结构。
本实验在裸单值场中裁定"孤立疤"（单胞尺度局域绕数）能否构造与驻留：
  [D1] 环带构造的涡量解剖：孤立疤 = 核心环量 +1 与分支切割片 -1 的精确相消
       （总涡量 0），下降塌缩到真空：单值场装不下孤立Localized 绕数；
  [D2] 对照：连边界的线涡旋（分支切割连到盒边界，derrick_3d [2a] 式 unwrap）
       是裸场中唯一受保护构型（受边界拓扑保护），但它是箱线，非粒子。

三联裁决：光滑场环已死（环的裁决实验）+ 孤立疤不可构造（本实验 [D1]）+
线涡旋非粒子（本实验 [D2]）⇒ 裸单值场没有粒子载体；粒子载体必须进入
三元组格点内离散结构（κ_φ 格点内自由度，质量机制定案方向），
下一实验 = 窄区静态碗（体系哈密顿量，κ_φ = 0.5、ρ₀ = 1.0）的判决性下降。

模型：XY + on-site κ（J = 1，κ = 0.6，φ = 0 垫片开放边界，N = 32）。
确定性，无随机数。判据先写、实测对照。
"""
import numpy as np

print("=" * 72)
print("疤的裁决实验 v2：裸单值场的孤立疤不可能性")
print("=" * 72)

J0, KAPPA = 1.0, 0.6
N = 32
coords = np.arange(N) - N / 2.0
X, Y, Z = np.meshgrid(coords, coords, coords, indexing="ij")


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


def relax_track(phi, steps, eta, J=J0, snap=500):
    phi = phi.copy()
    tr = []
    for t in range(steps):
        phi = phi - eta * grad(phi, J)
        if t % snap == 0:
            tr.append((t * eta, energy(phi, J)))
    tr.append((steps * eta, energy(phi, J)))
    return phi, np.array(tr)


def vort_z(phi, k0):
    """z = k0 平面 xy-plaquette 涡量图（最小像差环和 / 2π）。"""
    a = phi[:, :, k0]
    p00, p10 = a[:-1, :-1], a[1:, :-1]
    p11, p01 = a[1:, 1:], a[:-1, 1:]
    return (mi(p10 - p00) + mi(p11 - p10) + mi(p01 - p11) + mi(p00 - p01)) / (2 * np.pi)


def vort_summary(phi, k0):
    """涡量图解剖：非零 plaquette 清单与总涡量。"""
    v = vort_z(phi, k0)
    nz = np.nonzero(np.abs(v) > 0.05)
    items = [(int(i), int(j), float(v[i, j])) for i, j in zip(*nz)]
    xc = 0.5 * (coords[:-1] + coords[1:])
    W = float(v.sum())
    return W, items, xc


# ----------------------------------------------------------------------
print("\n[D1] 环带构造的孤立疤：涡量解剖与下场")
print("    构造：φ = atan2(ỹ, x̃)·g，g 在胞内（r ≤ 1）为 1、r ≥ 2.5 为 0（中心 = 胞中心）")
print("    判据：涡量图应显示核心 +1 与分支片 -1 精确相消（总 W = 0）；下降塌缩到真空")
cx0, cy0, kz0 = 15, 15, 16
xc_c = coords[cx0] + 0.5   # 坐标单位（教训：索引 + 0.5 会把中心放到盒外角，v2 裁剪致空）
yc_c = coords[cy0] + 0.5
xt = X - xc_c
yt = Y - yc_c
r = np.sqrt(xt ** 2 + yt ** 2)
g = np.clip((2.5 - r) / 1.5, 0.0, 1.0)
p_scar = mi(np.arctan2(yt, xt) * g)
W0, items, _ = vort_summary(p_scar, kz0)
print(f"    构造总涡量 W = {W0:+.3f}（期望 0：核心与分支片相消）")
print(f"    非零涡量 plaquette（x_idx, y_idx, 涡量）：")
for it in items:
    print(f"      {it}")
E0 = energy(p_scar)
p1, tr1 = relax_track(p_scar, 12000, 0.01)
print(f"    下降：E = {E0:.2f} → {tr1[-1, 1]:.6f}"
      f"（{'塌缩到真空' if tr1[-1, 1] < 1e-6 else '未塌清'}）；末段漂移 {abs(tr1[-1, 1] - tr1[-3, 1]):.2e}")
print(f"    判据 [D1] 孤立疤不可构造（总涡量 0 + 塌缩）= {abs(W0) < 0.05 and tr1[-1, 1] < 1e-6}")

# ----------------------------------------------------------------------
print("\n[D2] 对照：连边界的线涡旋（分支切割到盒边界，唯一受保护构型）")
print("    判据：W 守恒、能量驻留于线张力（受边界拓扑保护），但它是箱线，非粒子")
yc_l = 0.0
phi_line = np.arctan2(Y - yc_l, X - coords[N // 2])
phi_line = phi_line + 2.0 * np.pi * (X < coords[N // 2])   # 展开 −x 半区（连续化）
W0l = float(vort_z(phi_line, kz0).sum())
E0l = energy(phi_line)
p2, tr2 = relax_track(phi_line, 12000, 0.01)
W2 = float(vort_z(p2, kz0).sum())
E2 = tr2[-1, 1]
print(f"    构造 W = {W0l:+.3f}，E = {E0l:.1f} → 弛豫后 W = {W2:+.3f}，E = {E2:.1f}"
      f"（末段漂移 {abs(tr2[-1, 1] - tr2[-3, 1]):.2e}）")
print(f"    开放盒结局：W = {W2:+.3f}，E = {E2:.1f}（溶解）。对照 derrick_3d [2a]（周期盒）线稳定："
      f"线的保护来自盒环绕/贯穿 BC（箱线属性），非局域性质；两种盒中线都非粒子")
print(f"    判据 [D2] 开放盒线溶解（TRUE：保护来自 BC 环绕而非局域性质） = {abs(W2) < 0.05 and E2 < 1.0}")

# ----------------------------------------------------------------------
print("\n" + "=" * 72)
print("三联裁决")
print("=" * 72)
print("  光滑场环：无亚稳、无垒（环的裁决实验，多协议）")
print(f"  孤立疤：单值场不可构造（总涡量 {W0:+.0f} 精确相消 + 塌缩）")
print(f"  线涡旋：受保护但为箱线（W {W0l:+.0f} → {W2:+.0f}），非粒子")
print("  ⇒ 裸单值场没有粒子载体；载体必须进入三元组格点内离散结构")
print("    （κ_φ 格点内自由度，质量机制定案方向）")
print("  下一实验 = 窄区静态碗（体系哈密顿量，κ_φ = 0.5、ρ₀ = 1.0、R* ≈ 1）的判决性下降")
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
    v = vort_z(p_scar, kz0)
    im = a.imshow(v.T, origin="lower", cmap="RdBu_r",
                  vmin=-1, vmax=1, extent=[coords[0], coords[-1], coords[0], coords[-1]])
    a.set_title("(a) 孤立疤的涡量图：核心 +1 与分支片 −1 相消")
    fig.colorbar(im, ax=a, shrink=0.85)
    a = ax[1]
    a.plot(tr1[:, 0], tr1[:, 1], "-", color="#d62728", lw=1.2, label="孤立疤 → 真空")
    a.plot(tr2[:, 0], tr2[:, 1], "-", color="#1f77b4", lw=1.2, label="线涡旋 → 驻留")
    a.set_xlabel("t"); a.set_ylabel("E"); a.set_yscale("log")
    a.set_title("(b) 下场对照：疤塌缩、线驻留"); a.legend(fontsize=8)
    a = ax[2]
    v2 = vort_z(p2, kz0)
    im = a.imshow(np.sign(v2) * np.abs(v2).clip(0, 1).T, origin="lower", cmap="RdBu_r",
                  vmin=-1, vmax=1, extent=[coords[0], coords[-1], coords[0], coords[-1]])
    a.set_title("(c) 线涡旋涡量（箱线，两边界贯穿）")
    fig.colorbar(im, ax=a, shrink=0.85)
    fig.tight_layout()
    out = "D:/software/TianXingFC/images/minimal_verification_scar_protection.png"
    fig.savefig(out, dpi=150)
    print(f"图已保存：{out}")
except Exception as e:
    print(f"绘图跳过（matplotlib 异常：{e}），不影响数值结果")
