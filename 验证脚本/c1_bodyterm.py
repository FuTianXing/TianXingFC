# -*- coding: utf-8 -*-
r"""
c1_bodyterm.py

[C1-4b] 泡壁环的体项可行性判定：稳定条件的精确化与体项扫描。

背景
----
载体层裁决注记 §四其二给出泡壁环机制：E(R) = 2*pi*R*sigma_wall +
pi*R^2*delta_body + 2*pi*R*mu_core，「体项为正时出现稳定 R*」。本脚本
用最小模型把该命题的成立条件精确化并数值核验：

  记 a = sigma_wall + mu_core（线项系数）、delta = delta_body（体项系数），
  E(R) = 2*pi*R*a + pi*R^2*delta。
  稳定 R* 条件（E' = 0 且 E'' > 0）：a < 0 < delta（负线项 + 正体项）。
  纯体项（a > 0 固定）无论 delta 取何号都给不出内部极小（delta > 0 单调增
  至塌缩；delta < 0 为倒钟，内部极值是极大，两侧分别收缩与膨胀）。

模型与扫描
----
u 场模型与 C1-4 同（平方型非紧致弹性、K = 0.25、钉边界），加显式体项
  E_body = epsilon * sum(u - u0)   （h = u - u0 的单位体积代价）
epsilon 五档：-0.10、-0.03、0、+0.03、+0.10。

判据（先立判据，再实现；预期写错时修正判据并记录教训）
----
T1 静态扫描：五档 epsilon 的 E0(R)（R = 3.0 至 6.0）均无内部极小。
T2 形状读数：五档 epsilon 的 E0(R) 在扫描窗（R <= 6）内均单调增。
   修正记录：初版预期 epsilon < 0 档出现内部极大（倒钟）；实测未现，核查
   为体项是 R^3 标度（半球泡体积），倒钟转折点 R_turn ~ A/(|eps|*B) 外推
   ~40 格，远超盒尺度（半径 16），物理内不可达；判据修正为读数报告。
T3 动力学（R = 5，GF 2500 步）：epsilon = +0.10 的末态 ext 小于 epsilon = 0；
   epsilon = -0.10 的末态 ext 大于 epsilon = 0（体项只改变收缩与膨胀方向，
   不制造中间尺寸的驻留）。读数：三档末态 ext 与 E。

运行
----
python 验证脚本/c1_bodyterm.py
确定性（无随机数），运行约 40 秒。
"""

import time
import numpy as np

T_START = time.time()

k = 1.0
N = 32
ETA = 0.01
K_MAIN = 0.25

print("=" * 72)
print("[C1-4b] 泡壁环体项判定：稳定条件精确化与体项扫描")
print("=" * 72)

IX, IY, IZ = np.indices((N, N, N))
u0 = (IX + IY + IZ).astype(float)
CX = CY = CZ = N // 2
XC = IX - CX
YC = IY - CY
ZC = IZ - CZ
RR = np.sqrt(XC * XC + YC * YC + ZC * ZC)
H0 = u0  # 背景

CHECKS = []


def check(tag, ok, detail):
    CHECKS.append((tag, bool(ok)))
    print(f"  [{tag}] {'PASS' if ok else 'FAIL'}  {detail}")


def energy(u, Kp, eps=0.0):
    E = 0.0
    for ax in range(3):
        d = np.diff(u, axis=ax) - 1.0
        E += 0.5 * k * float(np.sum(d ** 2))
    E += Kp * float(np.sum(1.0 - np.cos(2.0 * np.pi * u)))
    E += eps * float(np.sum(u - u0))
    return E


def grad(u, Kp, eps=0.0):
    p = np.pad(u, 1, mode="edge")
    ctr = (slice(1, -1),) * 3
    s = (p[2:, 1:-1, 1:-1] + p[:-2, 1:-1, 1:-1] + p[1:-1, 2:, 1:-1]
         + p[1:-1, :-2, 1:-1] + p[1:-1, 1:-1, 2:] + p[1:-1, 1:-1, :-2])
    return k * (6.0 * u - s) + 2.0 * np.pi * Kp * np.sin(2.0 * np.pi * u) + eps


def pin(u):
    u[0, :, :] = u0[0, :, :]
    u[-1, :, :] = u0[-1, :, :]
    u[:, 0, :] = u0[:, 0, :]
    u[:, -1, :] = u0[:, -1, :]
    u[:, :, 0] = u0[:, :, 0]
    u[:, :, -1] = u0[:, :, -1]


def readout(u):
    h = u - u0
    return float(h.max()), int(np.count_nonzero(h > 0.5))


def run_gf(u_init, Kp, eps, steps, snap=250):
    u = u_init.copy()
    tr = []
    for t in range(steps):
        u = u - ETA * grad(u, Kp, eps)
        pin(u)
        if (t % snap == 0) or (t == steps - 1):
            p, e = readout(u)
            tr.append((energy(u, Kp, eps), p, e))
    return u, tr


def make_ring(Rc, t=1.0, w=2.5):
    g = 0.5 * (1.0 - np.tanh((RR - Rc) / w))
    return 0.5 * (1.0 - np.tanh(ZC / t)) * g


# ----------------------------------------------------------------------
# T1/T2 静态扫描
# ----------------------------------------------------------------------
print("\n[T1/T2] 静态扫描 E0(R)，五档 epsilon")
EPS_LIST = [-0.10, -0.03, 0.0, 0.03, 0.10]
RS = [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
STATIC = {}
t1_ok = True
t2_ok = True
for eps in EPS_LIST:
    row = []
    for Rc in RS:
        row.append(energy(u0 + make_ring(Rc), K_MAIN, eps))
    STATIC[eps] = row
    ds = np.diff(row)
    has_min = any(ds[i] < 0 and ds[i + 1] > 0 for i in range(len(ds) - 1))
    has_max = any(ds[i] > 0 and ds[i + 1] < 0 for i in range(len(ds) - 1))
    mono_inc = bool(np.all(ds > 0))
    t1_ok &= (not has_min)
    t2_ok &= mono_inc
    print(f"    eps={eps:+.2f}: E0(R) = " + " ".join(f"{v:.0f}" for v in row) +
          f" ｜ 内部极小={'有' if has_min else '无'}，内部极大={'有' if has_max else '无'}")
check("T1", t1_ok, "五档 epsilon 的 E0(R) 均无内部极小（体项不造尺寸）")
check("T2", t2_ok, "五档在扫描窗内均单调增（倒钟转折点外推 ~40 格，远超盒尺度）")

# 倒钟转折点外推（读数）：A/R^2 拟合 与 B = dE/deps 的 R^3 标度
A_fit = float(np.polyfit([r * r for r in RS], STATIC[0.0], 1)[0])
B_num = max((STATIC[0.10][-1] - STATIC[0.0][-1]) / 0.10 / (RS[-1] ** 3),
            (STATIC[-0.10][-1] - STATIC[0.0][-1]) / (-0.10) / (RS[-1] ** 3))
R_turn = A_fit / (0.10 * B_num)
print(f"    读数：A（R^2 系数）≈ {A_fit:.2f}，B（体项 R^3 系数）≈ {B_num:.3f}，"
      f"|eps|=0.10 的倒钟转折点外推 R_turn ≈ {R_turn:.0f} 格（盒半径 16）")

# ----------------------------------------------------------------------
# T3 动力学
# ----------------------------------------------------------------------
print("\n[T3] 动力学：R=5，GF 2500 步，三档 epsilon")
E5 = {}
for eps in (-0.10, 0.0, 0.10):
    u_init = u0 + make_ring(5.0)
    peak0, ext0 = readout(u_init)
    uf, tr = run_gf(u_init, K_MAIN, eps, 2500)
    pf, ef = tr[-1][1], tr[-1][2]
    E5[eps] = (ef, tr[-1][0])
    print(f"    eps={eps:+.2f}: ext {ext0} -> {ef}，E_f = {tr[-1][0]:.2f}，peak_f = {pf:.2f}")
t3_ok = (E5[0.10][0] < E5[0.0][0]) and (E5[-0.10][0] > E5[0.0][0])
check("T3", t3_ok,
      f"ext 单调移位：eps=+0.10 → {E5[0.10][0]}；eps=0 → {E5[0.0][0]}；"
      f"eps=-0.10 → {E5[-0.10][0]}（体项只改收缩与膨胀方向）")

# ----------------------------------------------------------------------
# 汇总
# ----------------------------------------------------------------------
n_pass = sum(1 for _, ok in CHECKS if ok)
print(f"\n判据汇总：{n_pass}/{len(CHECKS)} 通过；总运行 {time.time() - T_START:.0f} s")
for tag, ok in CHECKS:
    print(f"  {tag}: {'PASS' if ok else 'FAIL'}")

print("\n理论结论（E(R) = 2*pi*R*a + pi*R^2*delta 的单调性分析）：")
print("  稳定 R* 条件：a < 0 < delta（负线项 + 正体项）；")
print("  纯体项（线项固定为正）无论符号都给不出内部极小，只改变收缩与膨胀方向。")
print("  线项 a = sigma_wall + mu_core 的候选核对：锚核给 mu_core < 0（W1/W2 读数：")
print("  临界缺口约线张力的 10%，即 a 距 0 差 10%），机制上对应球面/柱面界面的")
print("  钉扎能量节省；正体项 delta > 0 的候选见注记的理论判定。")

print("\n[C1-4b] 完成。")