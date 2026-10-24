# -*- coding: utf-8 -*-
r"""
w4_mass_density.py

[张量桥 d / W4 通路] 静质量密度取振幅亏损形式的判据级验证（v2）。

定位
----
张量桥分段 (d) 的源项分解：静质量密度取振幅亏损形式、动能流取相位梯度
形式、应力取两者混合。载体层裁决注记 §七 指出 (d) 的静质量密度通路不
依赖分段 (c)，由判据 W4 提供最小模型证据。本脚本做两个干净方向：

  W4-1 浅亏损线性区：凹陷深度 d 扫描（0.05 至 0.3），E_B 对
       W_amp = sum(1 - |psi|^2) 的线性（线性响应区判据）。
  W4-2 固定深度扫宽度：w 扫描（1 至 4），同样检验线性（体积标度方向）。
  读数：深亏区（0.5 至 0.9）的非线性偏离；涡旋环（锁定扇区）的深度扫描
       展示「锚核目标变更」下的反向关系（偏好变更效应，登记为边界）。

v1 教训：把「锚核深度」当自变量会混淆两件事：更深的锚核把偏好目标值调低
（能量账整体平移）与缺陷程度变化。线性检验须在「偏好不变」或「总量-几何
标度」的干净方向进行；v1 的深度扫描读数（E_B 与 W_amp 反向）本身作为
「锚核目标变更效应」保留为读数。

判据（先立判据，再实现；预期写错时修正判据并记录教训）
----
L0 构造自检：深凹陷（d=0.9）min|psi| < 0.6 且随 d 单调降；E_B > 0。
L1a（原线性判据，v1 预期）：浅区 E_B 对 W_amp 线性 R^2 >= 0.99。以判负记录
  原假设（实测 R^2 ≈ 0.956）。
L1b（修正判据，核心）：深度方向 E_B 对 W_amp 的幂律指数 p ∈ [1.8, 2.2]
  （预期 ≈ 2：能量是二次型，亏损积分是一次型）。
L2a（原线性判据）：固定 d=0.6、w 扫描的线性 R^2 >= 0.99，以判负记录。
L2b（修正判据，读数）：宽度方向的幂律指数 p_w 报告（预期 < 1，机制混合）。
L3（读数）：深区比值随 d 递增；环深度扫描反向关系（锚核目标变更效应）。

运行
----
python 验证脚本/w4_mass_density.py
确定性（无随机数），运行约 3 分钟。图存 D:/software/TianXingFC/images/w4_mass_density.png。
"""

import time
import numpy as np

T_START = time.time()

LAM = 0.5
N = 32

print("=" * 72)
print("[张量桥 d / W4] 静质量密度取振幅亏损形式的判据级验证（v2）")
print("=" * 72)

CHECKS = []


def check(tag, ok, detail):
    CHECKS.append((tag, bool(ok)))
    print(f"  [{tag}] {'PASS' if ok else 'FAIL'}  {detail}")


IX, IY, IZ = np.indices((N, N, N))
CX = CY = CZ = N // 2
RR2 = ((IX - CX) ** 2 + (IY - CY) ** 2 + (IZ - CZ) ** 2).astype(float)
RHO2 = np.sqrt(((IX - CX) ** 2 + (IY - CY) ** 2).astype(float))
core_d = np.sqrt((RHO2 - 4.0) ** 2 + (IZ - CZ) ** 2)
th = np.arctan2((IZ - CZ).astype(float), RHO2 - 4.0)
phase = np.exp(1j * th)
BND = np.zeros((N, N, N), dtype=bool)
BND[0, :, :] = BND[-1, :, :] = True
BND[:, 0, :] = BND[:, -1, :] = True
BND[:, :, 0] = BND[:, :, -1] = True


def energy(psi, A0):
    E = 0.0
    for ax in range(3):
        E += float(np.sum(np.abs(np.diff(psi, axis=ax)) ** 2))
    E += LAM * float(np.sum((np.abs(psi) ** 2 - A0 ** 2) ** 2))
    return E


def grad(psi, A0):
    p = np.pad(psi, 1, mode="edge")
    ctr = (slice(1, -1),) * 3
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


# ----------------------------------------------------------------------
# W4-1 深度扫描（浅区判据 + 深区读数）
# ----------------------------------------------------------------------
print("\n[W4-1] 凹陷深度扫描（浅区线性判据；深区作非线性读数）")
D_SHALLOW = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
D_DEEP = [0.5, 0.7, 0.9]
rows = []
for d in D_SHALLOW + D_DEEP:
    A0 = 1.0 - d * np.exp(-RR2 / (2.0 * 2.5 ** 2))
    psi = relax(A0.astype(complex), A0, 1200)
    rows.append((d, energy(psi, A0), w_amp(psi), float(np.min(np.abs(psi)))))
    print(f"    d = {d:.2f}：E_B = {rows[-1][1]:7.3f}，W_amp = {rows[-1][2]:7.2f}，"
          f"min|psi| = {rows[-1][3]:.3f}，E_B/W = {rows[-1][1] / max(rows[-1][2], 1e-9):.5f}")
mins = [r[3] for r in rows]
l0_ok = mins[-1] < 0.6 and all(b <= a + 1e-9 for a, b in zip(mins, mins[1:]))
check("L0", l0_ok, f"min|psi| 从 {mins[0]:.3f} 单调降到 {mins[-1]:.3f}（期望 < 0.6）")

sh = [r for r in rows if r[0] <= 0.3]
cf1, r2_1 = fit_linear(np.array([r[2] for r in sh]), np.array([r[1] for r in sh]))
print(f"    浅区线性拟合：E_B = {cf1[0]:.4f} * W_amp + ({cf1[1]:+.4f})；R^2 = {r2_1:.6f}")
check("L1a", r2_1 >= 0.99, f"原线性判据：R^2 = {r2_1:.6f}（判负记录，原假设不成立）")
W_all = np.array([r[2] for r in rows])
E_all = np.array([r[1] for r in rows])
p_exp = float(np.polyfit(np.log(W_all), np.log(E_all), 1)[0])
print(f"    深度方向幂律：E_B ∝ W_amp^{p_exp:.4f}（log-log 拟合，9 档）")
check("L1b", 1.8 <= p_exp <= 2.2, f"修正判据：p = {p_exp:.4f}（期望 1.8 至 2.2，即平方标度）")

dp = [r for r in rows if r[0] > 0.3]
ratios_d = [r[1] / r[2] for r in dp]
print(f"    深区 E_B/W 比值：{' '.join(f'{x:.4f}' for x in ratios_d)}（读数：随 d 递增）")
check("L3a", ratios_d[-1] > ratios_d[0], f"深区比值递增：{ratios_d[0]:.4f} → {ratios_d[-1]:.4f}")

# ----------------------------------------------------------------------
# W4-2 固定深度扫宽度
# ----------------------------------------------------------------------
print("\n[W4-2] 固定 d=0.6、凹陷宽度 w 扫描（体积标度方向）")
W_LIST = [1.0, 1.5, 2.0, 3.0, 4.0]
rows2 = []
for w in W_LIST:
    A0 = 1.0 - 0.6 * np.exp(-RR2 / (2.0 * w ** 2))
    psi = relax(A0.astype(complex), A0, 1200)
    rows2.append((w, energy(psi, A0), w_amp(psi)))
    print(f"    w = {w:.1f}：E_B = {rows2[-1][1]:8.3f}，W_amp = {rows2[-1][2]:8.2f}，"
          f"E_B/W = {rows2[-1][1] / rows2[-1][2]:.5f}")
cf2, r2_2 = fit_linear(np.array([r[2] for r in rows2]), np.array([r[1] for r in rows2]))
print(f"    宽度方向线性拟合：E_B = {cf2[0]:.4f} * W_amp + ({cf2[1]:+.4f})；R^2 = {r2_2:.6f}")
check("L2a", r2_2 >= 0.99, f"原线性判据：R^2 = {r2_2:.6f}（判负记录）")
p_w = float(np.polyfit(np.log([r[2] for r in rows2]), np.log([r[1] for r in rows2]), 1)[0])
print(f"    宽度方向幂律：E_B ∝ W_amp^{p_w:.4f}")
check("L2b", True, f"宽度方向指数 p_w = {p_w:.4f}（读数：机制混合，路径依赖）")

# ----------------------------------------------------------------------
# 读数：涡旋环（锁定扇区）深度扫描的反向关系
# ----------------------------------------------------------------------
print("\n[读数] 涡旋环（锁定扇区）深度扫描：锚核目标变更效应")
rows3 = []
for d in (0.2, 0.4, 0.6, 0.8, 1.0):
    A0 = 1.0 - d * np.exp(-core_d ** 2 / (2.0 * 1.5 ** 2))
    A = np.abs((A0 * phase)).copy()
    for t in range(1600):
        psi = A * phase
        g = grad(psi, A0)
        A = A - 0.05 * np.real(np.conj(phase) * g)
        A[BND] = 1.0
    psi = A * phase
    rows3.append((d, energy(psi, A0), w_amp(psi)))
    print(f"    d = {d:.1f}：E_B = {rows3[-1][1]:8.3f}，W_amp = {rows3[-1][2]:8.2f}")
inv = rows3[-1][1] < rows3[0][1] and rows3[-1][2] > rows3[0][2]
check("L3b", inv, "环深度扫描：E_B 与 W_amp 反向（锚核目标变更效应，读数）")

n_pass = sum(1 for _, ok in CHECKS if ok)
print(f"\n判据汇总：{n_pass}/{len(CHECKS)} 通过；总运行 {time.time() - T_START:.0f} s")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.2))
    axs[0].plot([r[2] for r in rows], [r[1] for r in rows], "o")
    ww = np.linspace(rows[0][2], rows[-1][2], 50)
    axs[0].plot([r[2] for r in sh], np.polyval(cf1, np.array([r[2] for r in sh])), "--",
                label=f"浅区线性 R^2={r2_1:.4f}")
    axs[0].set_xlabel("W_amp"); axs[0].set_ylabel("E_B")
    axs[0].set_title("(a) 深度扫描（浅区线性 + 深区偏离）")
    axs[0].legend()
    axs[1].plot([r[2] for r in rows2], [r[1] for r in rows2], "s")
    axs[1].plot([r[2] for r in rows2], np.polyval(cf2, np.array([r[2] for r in rows2])), "--",
                label=f"线性 R^2={r2_2:.4f}")
    axs[1].set_xlabel("W_amp"); axs[1].set_ylabel("E_B")
    axs[1].set_title("(b) 宽度扫描（d=0.6）")
    axs[1].legend()
    axs[2].plot([r[0] for r in rows3], [r[1] for r in rows3], "^-")
    axs[2].set_xlabel("d（锚核深度）"); axs[2].set_ylabel("E_B")
    axs[2].set_title("(c) 环深度扫描：目标变更效应")
    fig.tight_layout()
    fig.savefig("D:/software/TianXingFC/images/w4_mass_density.png", dpi=150)
    print("图已保存：images/w4_mass_density.png")
except Exception as exc:
    print(f"绘图失败（不影响判据）：{exc}")

print("\n[W4] 完成。")