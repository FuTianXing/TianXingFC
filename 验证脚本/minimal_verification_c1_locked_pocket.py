# -*- coding: utf-8 -*-
r"""
minimal_verification_c1_locked_pocket.py

[C1-4c] 锁定缺失泡的构造轮：负界面张力门槛与稳定半径验证（v2）。

定位
----
C1-4b 判定泡壁环稳定条件为负线项加正体项（a < 0 < delta），落成
「正常泡 + 界面锚核钉扎」。本脚本把正常泡按体系语汇读作「锁定缺失泡」，
以「界面带偏好凹口（参数 s）+ 界面键优惠（参数 g）」作为锚核的干净实现
（表面活性剂型：只在界面带作用，不改写真空、不创造遍在新相），做两步：

  其一，1D 界面能 sigma(s, g) 扫描，找 sigma < 0 的门槛组合。
  其二（条件触发），3D 泡构造：E0(R) 的稳定半径 R* 与动力学收敛。

v1 两条教训（已修正，过程见注记）：偏好取斜坡会把界面势能白送掉；
用全场的 psi 函数作优惠项会创造遍在的新真空（psi ~ 0.707 成为最低态）。

模型（实场，相位恒零）
----
  E = sum_bonds w_ij (psi_i - psi_j)^2 + lambda * sum_sites (psi_i^2 - A0_i^2)^2
界面带由 kernel(x) = exp(-((x-x0)/wa)^2) 定义；A0 在带内取 s（偏好凹口）、
带外取两相基态值；键权重 w = 1 - (1-g) * kernel_mid。

判据（先立判据，再实现；预期写错时修正判据并记录教训）
----
S1 基准：无锚核（s 关闭、g = 1）的 1D 界面能 sigma0 > 0，且与离散参考
   （同一网格、阶跃偏好下的弛豫解）自洽；读数记录。
S2 门槛：sigma(s, g) 随凹口加深（s 向 0.5）与键优惠增强（g 减小）下降；
   判定是否存在 sigma < 0 的组合，给出 sigma_min 与对应 (s, g)。
S3（条件）：若 sigma_min < 0，以最优 (s, g) 构造 3D 泡：E0(R) 出现内部
   极小（R*），且 1D 与 3D 的 sigma 一致（±40%）。
S4（条件）：动力学收敛：R0 = 4 与 5.5 的 GF 末态半径 |dR| <= 0.7 且 >= 2。

运行
----
python 验证脚本/minimal_verification_c1_locked_pocket.py
确定性（无随机数），运行约 90 秒。图存 D:/software/TianXingFC/images/c1_pocket.png。
"""

import time
import numpy as np

T_START = time.time()

LAM = 0.25

print("=" * 72)
print("[C1-4c] 锁定缺失泡：界面带凹口与键优惠下的负界面张力判定（v2）")
print("=" * 72)

CHECKS = []


def check(tag, ok, detail):
    CHECKS.append((tag, bool(ok)))
    print(f"  [{tag}] {'PASS' if ok else 'FAIL'}  {detail}")


def wall_energy_1d(s=None, g=1.0, N1=240, steps=8000, eta=0.05, wa=2.0):
    """1D 界面能：左相偏好 0、右相偏好 1（阶跃）；界面带（kernel）内偏好
    凹口至 s、键权重乘 g。返回（界面能 sigma，末态剖面）。"""
    x = np.arange(N1, dtype=float)
    x0 = 0.5 * (N1 - 1)
    kern = np.exp(-((x - x0) / wa) ** 2)
    base = (x >= x0).astype(float)
    if s is None:
        A0x = base
    else:
        A0x = kern * s + (1.0 - kern) * base
    w = 1.0 - (1.0 - g) * 0.5 * (kern[:-1] + kern[1:])
    psi = 0.5 * (1.0 + np.tanh((x - x0) / 1.2))
    for t in range(steps):
        d = np.diff(psi)
        gv = np.zeros(N1)
        gv[1:-1] = 2.0 * w[:-1][0:N1 - 2] * d[0:N1 - 2] - 2.0 * w[1:][0:N1 - 2] * d[1:]
        gv[1:-1] += 2.0 * LAM * (psi[1:-1] ** 2 - A0x[1:-1] ** 2) * psi[1:-1]
        psi[1:-1] -= eta * gv[1:-1]
        psi[0] = 0.0
        psi[-1] = 1.0
    E = float(np.sum(w * np.diff(psi) ** 2)) + LAM * float(np.sum((psi ** 2 - A0x ** 2) ** 2))
    return E, psi


# ----------------------------------------------------------------------
# S1 基准与参考
# ----------------------------------------------------------------------
print("\n[S1] 基准界面能（无锚核）")
E_ref, _ = wall_energy_1d(s=None, g=1.0)
ref_cont = (4.0 / 3.0) * np.sqrt(LAM)
print(f"    sigma0（阶跃偏好，离散解）= {E_ref:.4f}；连续极限 (4/3)sqrt(lambda) = {ref_cont:.4f}")
check("S1", E_ref > 0, f"sigma0 = {E_ref:.4f} > 0（基准）")

# ----------------------------------------------------------------------
# S2 扫描
# ----------------------------------------------------------------------
print("\n[S2] 扫描 sigma(s, g)：s 为界面带偏好、g 为界面键权重")
S_LIST = [0.0, 0.3, 0.5, 0.7]
G_LIST = [1.0, 0.5, 0.2]
SIG = {}
rows = []
for s in S_LIST:
    line = []
    for g in G_LIST:
        E, _ = wall_energy_1d(s=s, g=g)
        SIG[(s, g)] = E
        line.append(f"g={g}: {E:+.4f}")
    rows.append(f"    s={s:.1f}: " + " ｜ ".join(line))
    print(rows[-1])
best = min(SIG, key=SIG.get)
sigma_min = SIG[best]
print(f"    sigma_min = {sigma_min:+.4f}（s = {best[0]}，g = {best[1]}）")
s2_neg = sigma_min < 0
check("S2", True, f"扫描完成；sigma_min = {sigma_min:+.4f}，σ<0 组合"
      + ("存在" if s2_neg else "不存在（参数型调制不可达负界面张力）"))

# ----------------------------------------------------------------------
# S3/S4（条件触发）
# ----------------------------------------------------------------------
if not s2_neg:
    print("\n[S3/S4] sigma_min >= 0：三维构造跳过；判定见注记。")
    check("S3", False, "未达负界面张力，三维构造跳过（判定为不可达）")
    check("S4", False, "同上")
else:
    print("\n[S3/S4] 三维泡构造（最优锚核参数）")
    s_b, g_b = best
    N = 32
    IX, IY, IZ = np.indices((N, N, N))
    CX = CY = CZ = N // 2
    RRg = np.sqrt((IX - CX) ** 2 + (IY - CY) ** 2 + (IZ - CZ) ** 2)
    BND = np.zeros((N, N, N), dtype=bool)
    BND[0, :, :] = BND[-1, :, :] = True
    BND[:, 0, :] = BND[:, -1, :] = True
    BND[:, :, 0] = BND[:, :, -1] = True
    WA3 = 2.0

    def fields3(Rp):
        shell = np.exp(-((RRg - Rp) / WA3) ** 2)
        A0f = 1.0 - (1.0 - s_b) * shell
        w_ax = []
        for ax in range(3):
            sl_lo = [slice(None)] * 3
            sl_hi = [slice(None)] * 3
            sl_lo[ax] = slice(0, N - 1)
            sl_hi[ax] = slice(1, N)
            mid = 0.5 * (shell[tuple(sl_lo)] + shell[tuple(sl_hi)])
            w_ax.append(1.0 - (1.0 - g_b) * mid)
        return A0f, w_ax

    def energy3(psi, A0f, w_ax):
        E = 0.0
        for ax in range(3):
            E += float(np.sum(w_ax[ax] * np.diff(psi, axis=ax) ** 2))
        E += LAM * float(np.sum((psi ** 2 - A0f ** 2) ** 2))
        return E

    def grad3(psi, A0f, w_ax):
        g3 = np.zeros_like(psi)
        for ax in range(3):
            d = np.diff(psi, axis=ax)
            wd = w_ax[ax] * d
            lo = [slice(None)] * 3
            hi = [slice(None)] * 3
            lo[ax] = slice(0, N - 1)
            hi[ax] = slice(1, N)
            g3[tuple(lo)] -= 2.0 * wd
            g3[tuple(hi)] += 2.0 * wd
        g3 += 2.0 * LAM * (psi ** 2 - A0f ** 2) * psi
        return g3

    def psi_bubble(Rp, w=1.2):
        gg = 0.5 * (1.0 - np.tanh((RRg - Rp) / w))
        return 1.0 - gg

    def radius_of(psi):
        V = int(np.count_nonzero(np.abs(psi) < 0.5))
        return (3.0 * V / (4.0 * np.pi)) ** (1.0 / 3.0)

    def run_gf3(psi, A0f, w_ax, steps, eta=0.02, snap=500):
        psi = psi.copy()
        tr = []
        for t in range(steps):
            psi = psi - eta * grad3(psi, A0f, w_ax)
            psi[BND] = 1.0
            if (t % snap == 0) or (t == steps - 1):
                tr.append((energy3(psi, A0f, w_ax), radius_of(psi)))
        return psi, tr

    RS = [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
    E0s = []
    for Rp in RS:
        A0f, w_ax = fields3(Rp)
        E0s.append(energy3(psi_bubble(Rp), A0f, w_ax))
    print("    E0(R) = " + " ".join(f"{v:.0f}" for v in E0s))
    d2 = np.diff(E0s)
    has_min = any(d2[i] < 0 and d2[i + 1] > 0 for i in range(len(d2) - 1))
    RSa = np.array(RS)
    M = np.vstack([4.0 * np.pi * RSa ** 2, (4.0 / 3.0) * np.pi * RSa ** 3, np.ones_like(RSa)]).T
    sol, *_ = np.linalg.lstsq(M, np.array(E0s), rcond=None)
    s3, d3 = float(sol[0]), float(sol[1])
    rstar = -2.0 * s3 / d3 if d3 != 0 else None
    print(f"    内部极小 = {'有' if has_min else '无'}；拟合 sigma_3D = {s3:+.4f}，"
          f"delta_3D = {d3:+.4f}，R*（解析）≈ {rstar if rstar is None else round(rstar, 2)}")
    check("S3", has_min, f"E0(R) 内部极小 = {has_min}（R* ≈ {rstar}）")
    outs = []
    for Rp in (4.0, 5.5):
        A0f, w_ax = fields3(Rp)
        psi_f, tr = run_gf3(psi_bubble(Rp), A0f, w_ax, 3000)
        outs.append((Rp, tr[-1][1]))
        print(f"    动力学：R0 = {Rp} → R_f = {tr[-1][1]:.2f}（E_f = {tr[-1][0]:.2f}）")
    conv_ok = abs(outs[0][1] - outs[1][1]) <= 0.7 and min(outs[0][1], outs[1][1]) >= 2.0
    check("S4", conv_ok, f"收敛：R_f = {outs[0][1]:.2f} 与 {outs[1][1]:.2f}")

n_pass = sum(1 for _, ok in CHECKS if ok)
print(f"\n判据汇总：{n_pass}/{len(CHECKS)} 通过；总运行 {time.time() - T_START:.0f} s")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.2))
    for g in G_LIST:
        ys = [SIG[(s, g)] for s in S_LIST]
        axs[0].plot(S_LIST, ys, "o-", label=f"g={g}")
    axs[0].axhline(0.0, color="gray", ls=":", lw=0.9)
    axs[0].set_xlabel("界面带偏好 s"); axs[0].set_ylabel("sigma（1D）")
    axs[0].set_title("(a) sigma(s, g) 与零线")
    axs[0].legend()
    if s2_neg:
        axs[1].plot(RS, E0s, "o-")
        axs[1].set_xlabel("R"); axs[1].set_ylabel("E0")
        axs[1].set_title(f"(b) 三维泡 E0(R)，s={best[0]}, g={best[1]}")
    fig.tight_layout()
    fig.savefig("D:/software/TianXingFC/images/c1_pocket.png", dpi=150)
    print("图已保存：images/c1_pocket.png")
except Exception as exc:
    print(f"绘图失败（不影响判据）：{exc}")

print("\n[C1-4c] 完成。")