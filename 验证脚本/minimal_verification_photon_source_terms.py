# -*- coding: utf-8 -*-
"""光子源项的四分量构造与离散守恒（第 436 条，P1-24 第二轮）。

定位：光的传播由实轴方向场的波动模式承载（2026-09-26 裁决：曲率与联络的载体是基准取向场，
      波矢、光与静电归实轴方向场，见 [[wiki/基准取向场]]）；本脚本在筛选后的横向场（A_0 = 0、横向 A，
      四分量筛选见 [[光的偏振的元相子几何：四分量筛选、指针与邻居方向的分工]]）上构造标准形式的
      能量-动量四分量，为"无质量粒子源项"接引力侧作准备。

判据（先立判据，再实现）：
  P1 离散平面波精确（1D）：波动方程残差机器零（omega 由 sin(omega/2) = sin(theta/2) 给，a = dt = 1）
  P2 场结构（1D）：格点 E、B 为 sin 结构且 |E| = |B|（光波等幅）
  P3 离散守恒（1D）：res = d_t u + d_z S 机器零（中心差分）
  P4 低阶对照：u 的格点/连续比值 = (sin(theta)/theta)^2，小 theta 时 1 - theta^2/6
  P5 3D 斜波（方向 (1,1,1)）：守恒残差的量级与 theta 标度（格点各向异性的读数）
  P6 应力分量：sigma_ij 按同一口径离散化并输出（源项接口）

口径：格点单位 l_P = t_P = 1；全部中心差分；标准电磁形式 u = (E^2+B^2)/2、S = E x B。
本脚本不做的：进源装配（倾角场/记录场的源）留作下一轮。
"""
import math
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG = os.path.join(ROOT, "images", "photon_source_terms.png")

# ---------- 1D ----------
N = 2048
j = np.arange(N)
theta = 0.05
omega = theta
n0 = 7

def A1(nn, th):
    return np.sin(omega * nn - th * j)

A = A1(n0, theta)
res_wave = (A1(n0 + 1, theta) - 2 * A + A1(n0 - 1, theta)) - (np.roll(A, -1) - 2 * A + np.roll(A, 1))
print("== P1 离散平面波（1D） ==")
print("  波动方程残差 max（内部）= %.3e（期望机器零；边界点为周期卷绕伪影）" % np.max(np.abs(res_wave[2:-2])))

E1 = -(A1(n0 + 1, theta) - A1(n0 - 1, theta)) / 2.0
B1 = (np.roll(A, -1) - np.roll(A, 1)) / 2.0
print()
print("== P2 E、B 场结构（1D） ==")
print("  |E|-|B| max（内部）= %.3e（期望机器零）" % np.max(np.abs(np.abs(E1) - np.abs(B1))[2:-2]))
print("  E 幅度 = %.12f，sin(theta) = %.12f" % (np.max(np.abs(E1)), math.sin(theta)))

def u1(nn, th):
    ph = omega * nn - th * j
    e = -math.sin(omega) * np.cos(ph)
    b = -math.sin(theta) * np.cos(ph)
    return 0.5 * (e ** 2 + b ** 2)

def S1(nn, th):
    ph = omega * nn - th * j
    e = -math.sin(omega) * np.cos(ph)
    b = -math.sin(theta) * np.cos(ph)
    return e * b

res_cons = (u1(n0 + 1, theta) - u1(n0 - 1, theta)) / 2.0 + (np.roll(S1(n0, theta), -1) - np.roll(S1(n0, theta), 1)) / 2.0
print()
print("== P3 离散守恒（1D 单色行波） ==")
print("  |d_t u + d_z S| max（内部）= %.3e；相对 = %.3e（期望机器零）" % (np.max(np.abs(res_cons[2:-2])), np.max(np.abs(res_cons[2:-2])) / np.max(np.abs(u1(n0, theta)))))

print()
print("== P4 低阶对照（u 的格点/连续比值） ==")
for th in (0.02, 0.05, 0.1, 0.2):
    ratio = (math.sin(th) / th) ** 2
    exp = 1.0 - th ** 2 / 3.0
    print("  theta = %.2f：比值 = %.8f；1 - theta^2/3 = %.8f；相对差 = %.2e" % (th, ratio, exp, abs(ratio - exp) / exp))

# ---------- 3D ----------
def droll(f, axis):
    """周期数据上的中心差分：保持 np.roll 卷绕，避免 np.gradient 的单侧边界。"""
    return (np.roll(f, -1, axis=axis) - np.roll(f, 1, axis=axis)) / 2.0
print()
print("== P5 3D 斜波（方向 (1,1,1)，偏振 e1 = (1,-1,0)/sqrt2） ==")
Nd = 64
gr = np.arange(Nd)
X, Y, Z = np.meshgrid(gr, gr, gr, indexing="ij")
e1 = np.array([1.0, -1.0, 0.0]) / math.sqrt(2.0)
n0_3d = 5
th_list = (0.05, 0.1, 0.2, 0.3)
rel_list = []

def fields_at(nn, kap, om):
    ph = om * nn - (kap * X + kap * Y + kap * Z)
    Av = e1[:, None, None, None] * np.sin(ph)[None, :, :, :]
    Ev = -e1[:, None, None, None] * (math.sin(om) * np.cos(ph))[None, :, :, :]
    ax = Av[0]; ay = Av[1]; az = Av[2]
    gax = (droll(ax, 0), droll(ax, 1), droll(ax, 2))
    gay = (droll(ay, 0), droll(ay, 1), droll(ay, 2))
    gaz = (droll(az, 0), droll(az, 1), droll(az, 2))
    bx = gaz[1] - gay[2]
    by = gax[2] - gaz[0]
    bz = gay[0] - gax[1]
    Bv = np.array([bx, by, bz])
    uu = 0.5 * (Ev[0] ** 2 + Ev[1] ** 2 + Ev[2] ** 2 + Bv[0] ** 2 + Bv[1] ** 2 + Bv[2] ** 2)
    Sv = np.array([
        Ev[1] * Bv[2] - Ev[2] * Bv[1],
        Ev[2] * Bv[0] - Ev[0] * Bv[2],
        Ev[0] * Bv[1] - Ev[1] * Bv[0],
    ])
    return uu, Sv, Ev, Bv

sigma_out = None
for th in th_list:
    kap = th / math.sqrt(3.0)
    sw2 = 3.0 * math.sin(kap / 2.0) ** 2
    om = 2.0 * math.asin(math.sqrt(sw2))
    up, Svp, _, _ = fields_at(n0_3d + 1, kap, om)
    um, Svm, _, _ = fields_at(n0_3d - 1, kap, om)
    uu, Sv, Ev, Bv = fields_at(n0_3d, kap, om)
    du = (up - um) / 2.0
    divS = droll(Sv[0], 0) + droll(Sv[1], 1) + droll(Sv[2], 2)
    # 统计只看内部：行波不满足格点周期条件，np.roll 卷绕只污染最外几层
    inner = (slice(3, -3), slice(3, -3), slice(3, -3))
    res3 = du + divS
    rel = np.max(np.abs(res3[inner])) / np.max(np.abs(uu[inner]))
    rel_list.append(rel)
    print("  theta = %.2f：omega = %.6f；残差 max（内部）= %.3e；相对 = %.3e" % (th, om, np.max(np.abs(res3[inner])), rel))
    if sigma_out is None:
        E2 = Ev[0] ** 2 + Ev[1] ** 2 + Ev[2] ** 2
        B2 = Bv[0] ** 2 + Bv[1] ** 2 + Bv[2] ** 2
        sig = np.zeros((3, 3) + uu.shape)
        for a in range(3):
            for b in range(3):
                delta = 1.0 if a == b else 0.0
                sig[a, b] = -(Ev[a] * Ev[b] + Bv[a] * Bv[b]) + 0.5 * delta * (E2 + B2)
        sigma_out = sig
        s11 = float(np.mean(sig[0, 0])); s12 = float(np.mean(sig[0, 1])); s33 = float(np.mean(sig[2, 2]))
        print("  P6 应力分量（theta = 0.05 样本，偏振 (1,-1,0)/sqrt2）：sigma_11 = %.6e，sigma_12 = %.6e，sigma_33 = %.6e（无零断言，作源装配接口输出）"
              % (s11, s12, s33))

print()
print("== P5b 沿轴对照（方向 (0,0,1)）：各向异性来源的隔离 ==")
for th in (0.1, 0.2, 0.3):
    kap = th
    om = th
    def fields_axis(nn):
        ph = om * nn - kap * Z
        Avx = e1[:, None, None, None] * np.sin(ph)[None, :, :, :]
        Evx = -e1[:, None, None, None] * (math.sin(om) * np.cos(ph))[None, :, :, :]
        axv = Avx[0]; ayv = Avx[1]; azv = Avx[2]
        gx = (droll(axv, 0), droll(axv, 1), droll(axv, 2))
        gy = (droll(ayv, 0), droll(ayv, 1), droll(ayv, 2))
        gz = (droll(azv, 0), droll(azv, 1), droll(azv, 2))
        bx = gz[1] - gy[2]; by = gx[2] - gz[0]; bz = gy[0] - gx[1]
        Bvx = np.array([bx, by, bz])
        uux = 0.5 * (Evx[0] ** 2 + Evx[1] ** 2 + Evx[2] ** 2 + Bvx[0] ** 2 + Bvx[1] ** 2 + Bvx[2] ** 2)
        Svx = np.array([
            Evx[1] * Bvx[2] - Evx[2] * Bvx[1],
            Evx[2] * Bvx[0] - Evx[0] * Bvx[2],
            Evx[0] * Bvx[1] - Evx[1] * Bvx[0],
        ])
        return uux, Svx, Evx, Bvx
    up2, Svp2, _, _ = fields_axis(n0_3d + 1)
    um2, Svm2, _, _ = fields_axis(n0_3d - 1)
    uu2, Sv2, Ev2, Bv2 = fields_axis(n0_3d)
    du2 = (up2 - um2) / 2.0
    divS2 = droll(Sv2[0], 0) + droll(Sv2[1], 1) + droll(Sv2[2], 2)
    res2 = du2 + divS2
    inner2 = (slice(3, -3), slice(3, -3), slice(3, -3))
    rel2 = np.max(np.abs(res2[inner2])) / np.max(np.abs(uu2[inner2]))
    E_amp = np.sqrt(np.max(np.abs(Ev2[0]) ** 2 + np.abs(Ev2[1]) ** 2 + np.abs(Ev2[2]) ** 2))
    B_amp = np.sqrt(np.max(np.abs(Bv2[0]) ** 2 + np.abs(Bv2[1]) ** 2 + np.abs(Bv2[2]) ** 2))
    print("  theta = %.2f：残差相对 = %.3e；|E|/|B| - 1 = %.3e" % (th, rel2, abs(E_amp / B_amp - 1.0)))

print()
print("== P5 标度检验（斜向残差对 theta 的幂） ==")
sl = float(np.polyfit(np.log(th_list), np.log(rel_list), 1)[0])
print("  log-log 斜率 = %.3f（期望 3：残差 = 首阶为格点修正的导数，O(theta^3)）" % sl)

print()
print("== 读数小结 ==")
print("  1D：守恒机器零（离散诺特在单色行波上精确）")
print("  3D：(1,1,1) 方向的相对残差 = %s（theta 序列）" % ", ".join("%.2e" % r for r in rel_list))
print("  四分量（源项材料）：u、S_i、sigma_ij 已按标准电磁形式构造并输出")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
ax = axes[0]
ths = np.linspace(0.01, 0.3, 60)
ax.plot(ths, (np.sin(ths) / ths) ** 2, "o", ms=3, label="lattice/continuum u")
ax.plot(ths, 1 - ths ** 2 / 3, "k--", lw=1, label="1 - theta^2/3")
ax.set_xlabel("theta"); ax.set_ylabel("ratio"); ax.set_title("P4: u 的格点/连续比值")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[1]
ax.loglog(th_list, rel_list, "o-")
ax.set_xlabel("theta"); ax.set_ylabel("relative residual")
ax.set_title("P5: 3D (1,1,1) 斜波守恒残差")
ax.grid(alpha=0.3, which="both")

fig.tight_layout()
fig.savefig(IMG, dpi=160)
print("figure saved:", IMG)
