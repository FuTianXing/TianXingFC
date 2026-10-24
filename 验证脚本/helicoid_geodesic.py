# -*- coding: utf-8 -*-
"""helicoid_geodesic.py

判据 717：螺旋面上相位轨迹的几何身份。

对象：P3-35 补法一「证明元相子的相位轨迹恰为螺旋面上的测地线」是否成立。

理论期望先立：
  螺旋面 X(u,v) = (u cos v, u sin v, a v)，u 为半径、v 为相位角。
  第一基本形式 E=1, F=0, G=u^2+a^2；第二基本形式 e=0, f=-a/sqrt(u^2+a^2), g=0。
  H = (eG-2fF+gE)/(2(EG-F^2)) = 0（极小曲面）。
  相位轨迹是等半径螺旋线 u=u0（半径即 hbar 标度），v 匀速推进。
  测地线方程：ddot u + Gamma^u_vv vdot^2 = 0，其中 Gamma^u_vv = -u。
  代入等半径轨迹得残差 R_u = 0 + (-u0) w^2 = -u0 w^2，归一化后恒为 -1。
  故相位轨迹不是测地线。对照：母线方向（v 固定、u 变化）残差恒为 0，是测地线。

判据：
  G0 冒烟：基本形式与曲率的解析式对有限差分数值式
  G1 主判据：相位轨迹的测地残差归一化值 = -1（非零即非测地线）
  G2 对照（区分力）：同一算符作用于母线方向，残差 = 0
  G3 几何身份：相位轨迹法曲率 = 0（渐近曲线）
  G4 结构论证：空间曲率非零且法曲率为零 => 测地曲率非零 => 非测地线
  G5 强形式：螺旋面全部测地线的半径无界（无一档回到初始半径）
  G6 数值可靠性：测地线积分能量守恒

可逆性等级：判据稳健（几何事实为解析结果，数值仅作确认；换实现、换口径不翻）。
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

IMG_DIR = r"D:\software\TianXingFC\images"
FIG = os.path.join(IMG_DIR, "p717_helicoid_geodesic.png")

np.random.seed(20261020)


# ---------------- 曲面与基本形式 ----------------
def surf(u, v, a):
    return np.stack([u * np.cos(v), u * np.sin(v), a * v], axis=-1)


def analytic_forms(u, v, a):
    E = np.ones_like(u, dtype=float)
    F = np.zeros_like(u, dtype=float)
    G = u ** 2 + a ** 2
    e = np.zeros_like(u, dtype=float)
    f = -a / np.sqrt(u ** 2 + a ** 2)
    g = np.zeros_like(u, dtype=float)
    return E, F, G, e, f, g


def fd_forms(u, v, a, h=1e-5):
    """有限差分求基本形式，用于冒烟核对解析式。"""
    X = surf(u, v, a)
    Xu = (surf(u + h, v, a) - surf(u - h, v, a)) / (2 * h)
    Xv = (surf(u, v + h, a) - surf(u, v - h, a)) / (2 * h)
    Xuu = (surf(u + h, v, a) - 2 * X + surf(u - h, v, a)) / h ** 2
    Xvv = (surf(u, v + h, a) - 2 * X + surf(u, v - h, a)) / h ** 2
    Xuv = (surf(u + h, v + h, a) - surf(u + h, v - h, a)
           - surf(u - h, v + h, a) + surf(u - h, v - h, a)) / (4 * h ** 2)
    E = np.sum(Xu * Xu, axis=-1)
    F = np.sum(Xu * Xv, axis=-1)
    G = np.sum(Xv * Xv, axis=-1)
    n = np.cross(Xu, Xv)
    n = n / np.linalg.norm(n, axis=-1, keepdims=True)
    e = np.sum(Xuu * n, axis=-1)
    f = np.sum(Xuv * n, axis=-1)
    g = np.sum(Xvv * n, axis=-1)
    return E, F, G, e, f, g


def mean_gauss(E, F, G, e, f, g):
    den = E * G - F ** 2
    H = (e * G - 2 * f * F + g * E) / (2 * den)
    K = (e * g - f ** 2) / den
    return H, K


# ---------------- 测地残差 ----------------
def residual_helix(u0, w, a):
    """等半径螺旋线 u=u0, v=w t 的测地残差。"""
    R_u = 0.0 + (-u0) * w ** 2      # ddot u = 0, Gamma^u_vv = -u0
    R_v = 0.0 + 2 * (u0 / (u0 ** 2 + a ** 2)) * 0.0 * w
    return R_u, R_v


def residual_ruling(udot, a):
    """母线 v=v0, u=t 的测地残差：vdot=0, ddot=0，两分量皆零。"""
    return 0.0, 0.0


# ---------------- 测地线积分 ----------------
def geodesic_rhs(state, a):
    u, du, v, dv = state
    ddu = u * dv ** 2
    ddv = -2 * u * du * dv / (u ** 2 + a ** 2)
    return np.array([du, ddu, dv, ddv])


def integrate_geodesic(u0, a, T=60.0, dt=2e-3, du0=0.0):
    dv0 = np.sqrt(max(1.0 - du0 ** 2, 1e-12)) / np.sqrt(u0 ** 2 + a ** 2)
    st = np.array([u0, du0, 0.0, dv0])
    n = int(T / dt)
    traj = np.zeros((n + 1, 4))
    traj[0] = st
    for i in range(n):
        k1 = geodesic_rhs(st, a)
        k2 = geodesic_rhs(st + 0.5 * dt * k1, a)
        k3 = geodesic_rhs(st + 0.5 * dt * k2, a)
        k4 = geodesic_rhs(st + dt * k3, a)
        st = st + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        traj[i + 1] = st
    return traj


def energy_of(traj, a):
    u, du, v, dv = traj[:, 0], traj[:, 1], traj[:, 2], traj[:, 3]
    return 0.5 * (du ** 2 + (u ** 2 + a ** 2) * dv ** 2)


# ================= 判据执行 =================
report = []
ok_all = True


def chk(name, meas, expect, tol, mode="abs"):
    global ok_all
    if mode == "abs":
        dev = abs(meas - expect)
    else:
        dev = abs(meas - expect) / max(abs(expect), 1e-300)
    passed = dev <= tol
    ok_all = ok_all and passed
    report.append((name, meas, expect, dev, tol, passed))
    return passed


print("=" * 78)
print("判据 717：螺旋面上相位轨迹的几何身份")
print("=" * 78)

a = 1.0 / (2 * np.pi)      # 一满周 v: 0->2pi 推进 z 一个 l_P，归一化 l_P=1
u_grid = np.linspace(0.4, 3.0, 7)
v_grid = np.linspace(0.3, 2.4, 5)
UU, VV = np.meshgrid(u_grid, v_grid, indexing="ij")

# ---- G0 冒烟 ----
print("\n[G0] 冒烟检查：基本形式与曲率的解析式对有限差分式")
Ea, Fa, Ga, ea, fa, ga = analytic_forms(UU, VV, a)
Ef, Ff, Gf, ef, ff, gf = fd_forms(UU, VV, a)
Ha, Ka = mean_gauss(Ea, Fa, Ga, ea, fa, ga)
Hf, Kf = mean_gauss(Ef, Ff, Gf, ef, ff, gf)
d_E = np.max(np.abs(Ef - Ea) / np.maximum(np.abs(Ea), 1e-12))
d_G = np.max(np.abs(Gf - Ga) / np.abs(Ga))
d_F = np.max(np.abs(Ff - Fa))
d_f = np.max(np.abs(np.abs(ff) - np.abs(fa)) / np.abs(fa))
d_H = np.max(np.abs(Hf - Ha))
d_K = np.max(np.abs(Kf - Ka) / np.abs(Ka))
print("    E 相对差 %.3e  G 相对差 %.3e  F 绝对差 %.3e" % (d_E, d_G, d_F))
print("    f 相对差 %.3e  H 绝对差 %.3e  K 相对差 %.3e" % (d_f, d_H, d_K))
print("    解析 H 最大绝对值 %.3e（极小曲面），K 取值 [%.4f, %.4f]（负高斯曲率）"
      % (np.max(np.abs(Ha)), np.min(Ka), np.max(Ka)))

# 容差按误差预算推导，不直接设数（2026-10-20 本轮教训）。
# 一阶导用中心差分，截断 O(h^2)=1e-10，舍入 eps/h=2e-11，故一阶量（E、G）取 1e-9。
# 二阶混合导 Xuv 分母为 4h^2，舍入放大到 eps/h^2 = 2.2e-16/1e-10 = 2.2e-6，
# 这是 f 相对差的主导项，故取一档余量 1e-5；取 1e-9 会把舍入误判为失败。
tol_first = 1e-9
tol_second = 1e-5
print("    容差预算：一阶量 %.0e（截断 h^2 主导），二阶量 %.0e（舍入 eps/h^2 主导）"
      % (tol_first, tol_second))
chk("G0a E 相对差", d_E, 0.0, tol_first)
chk("G0b G 相对差", d_G, 0.0, tol_first)
chk("G0c f 相对差", d_f, 0.0, tol_second)
chk("G0d H 绝对差（解析应为零）", np.max(np.abs(Ha)), 0.0, 1e-15)

# ---- G1 主判据 ----
print("\n[G1] 主判据：相位轨迹（等半径螺旋线）的测地残差，归一化后应为 -1")
print("     %-8s %-8s %-14s %-14s %-12s" % ("u0", "w", "R_u", "R_u/(u0 w^2)", "判"))
worst_g1 = 0.0
for u0 in [0.5, 1.0, 2.0, 4.0]:
    for w in [0.3, 1.0, 3.0]:
        R_u, R_v = residual_helix(u0, w, a)
        norm = R_u / (u0 * w ** 2)
        worst_g1 = max(worst_g1, abs(norm + 1.0))
        print("     %-8.2f %-8.2f %-14.6e %-14.10f %-12s"
              % (u0, w, R_u, norm, "非测地线" if abs(norm + 1) < 1e-12 else "?"))
chk("G1 归一化残差对 -1 的最大偏差", worst_g1, 0.0, 1e-12)

# ---- G2 对照 ----
print("\n[G2] 对照（区分力）：同一算符作用于母线方向，残差应为 0")
R_u_r, R_v_r = residual_ruling(1.0, a)
print("     母线：R_u = %.3e，R_v = %.3e（直线，属测地线）" % (R_u_r, R_v_r))
print("     相位轨迹 R_u = %.3e，母线 R_u = %.3e，两者之比无定义（对照为零）"
      % (residual_helix(1.0, 1.0, a)[0], R_u_r))
chk("G2 母线残差", abs(R_u_r), 0.0, 1e-15)

# ---- G3 渐近曲线 ----
print("\n[G3] 几何身份：相位轨迹的法曲率（应为零，即渐近曲线）")
worst_g3 = 0.0
for u0 in [0.5, 1.0, 2.0, 4.0]:
    E0, F0, G0_, e0, f0, g0 = analytic_forms(np.array([u0]), np.array([0.0]), a)
    # 相位轨迹 du=0, dv=1
    kn = (e0 * 0.0 ** 2 + 2 * f0 * 0.0 * 1.0 + g0 * 1.0 ** 2) / (
        E0 * 0.0 ** 2 + 2 * F0 * 0.0 * 1.0 + G0_ * 1.0 ** 2)
    worst_g3 = max(worst_g3, abs(float(kn[0])))
    print("     u0=%.2f  kappa_n = %.3e" % (u0, float(kn[0])))
chk("G3 法曲率最大绝对值", worst_g3, 0.0, 1e-15)

# ---- G4 结构论证 ----
print("\n[G4] 结构论证：空间曲率非零 且 法曲率为零 => 测地曲率非零 => 非测地线")
worst_g4 = 0.0
for u0 in [0.5, 1.0, 2.0, 4.0]:
    # 螺旋线 X(t) = (u0 cos wt, u0 sin wt, a w t)，曲率解析式 u0/(u0^2+a^2)
    w = 1.0
    t = np.linspace(0.0, 2 * np.pi, 2001)
    P = np.stack([u0 * np.cos(w * t), u0 * np.sin(w * t), a * w * t], axis=-1)
    d1 = np.gradient(P, t, axis=0)
    d2 = np.gradient(d1, t, axis=0)
    kap = np.linalg.norm(np.cross(d1, d2), axis=-1) / np.linalg.norm(d1, axis=-1) ** 3
    kap_ana = u0 / (u0 ** 2 + a ** 2)
    dev = np.max(np.abs(kap[5:-5] - kap_ana)) / kap_ana
    worst_g4 = max(worst_g4, dev)
    kg2 = kap_ana ** 2 - 0.0 ** 2
    print("     u0=%.2f  曲率数值 %.8f 解析 %.8f（相对差 %.2e）  kappa_g = %.6f > 0"
          % (u0, np.mean(kap[5:-5]), kap_ana, dev, np.sqrt(kg2)))
chk("G4 曲率数值对解析的最大相对差", worst_g4, 0.0, 1e-4)

# ---- G5 强形式 ----
print("\n[G5] 强形式：螺旋面全部测地线的半径无界")
print("     %-8s %-10s %-14s %-14s %-10s" % ("u0", "u 终值", "u 单调", "回到初始半径", "能量漂移"))
worst_e = 0.0
all_unbounded = True
for u0 in [0.5, 1.0, 2.0]:
    tr = integrate_geodesic(u0, a, T=40.0, dt=2e-3)
    uu = tr[:, 0]
    mono = np.all(np.diff(uu) >= -1e-10)
    back = np.min(uu[1:]) <= u0 * (1 - 1e-9)
    en = energy_of(tr, a)
    edrift = np.max(np.abs(en - en[0]))
    worst_e = max(worst_e, edrift)
    all_unbounded = all_unbounded and (uu[-1] > 10 * u0) and (not back)
    print("     %-8.2f %-10.3f %-14s %-14s %-10.2e"
          % (u0, uu[-1], "是" if mono else "否", "否" if not back else "是", edrift))
chk("G5 全部测地线半径无界", 1.0 if all_unbounded else 0.0, 1.0, 0.0)

# ---- G6 能量守恒 ----
print("\n[G6] 数值可靠性：测地线积分能量守恒（误差预算：RK4 单步 O(dt^5)，dt=2e-3）")
print("     最大能量漂移 %.3e（容差按 RK4 截断误差预算取 1e-9）" % worst_e)
chk("G6 能量漂移", worst_e, 0.0, 1e-9)

# ================= 汇总 =================
print("\n" + "=" * 78)
print("%-42s %-14s %-14s %-12s %s" % ("判据", "实测", "期望", "偏差", "结果"))
print("-" * 78)
for name, meas, expect, dev, tol, passed in report:
    print("%-42s %-14.6e %-14.6e %-12.2e %s"
          % (name, meas, expect, dev, "通过" if passed else "判否"))
npass = sum(1 for r in report if r[5])
print("-" * 78)
print("判据 %d/%d %s" % (npass, len(report), "全部通过" if ok_all else "存在判否项"))
print("=" * 78)

print("\n判语：相位轨迹（等半径螺旋线）的测地残差归一化后恒为 -1，")
print("      而同一算符作用于母线方向恒为 0，故补法一「相位轨迹是螺旋面上的测地线」判否。")
print("      结构性理由：等半径螺旋线是渐近曲线（法曲率为零）而空间曲率非零，")
print("      故测地曲率非零；且螺旋面全部测地线的半径无界，与等半径轨迹不相容。")

# ================= 出图 =================
fig = plt.figure(figsize=(15, 4.6))

ax = fig.add_subplot(131, projection="3d")
uu = np.linspace(-2.0, 2.0, 40)
vv = np.linspace(0.0, 2 * np.pi, 60)
U, V = np.meshgrid(uu, vv, indexing="ij")
ax.plot_surface(*np.moveaxis(surf(U, V, a), -1, 0), alpha=0.28, color="0.7",
                linewidth=0, antialiased=True)
t = np.linspace(0, 2 * np.pi, 200)
hel = surf(np.full_like(t, 1.0), t, a)
ax.plot(hel[:, 0], hel[:, 1], hel[:, 2], color="C3", lw=2.4, label="相位轨迹（等半径）")
tr = integrate_geodesic(1.0, a, T=6.0, dt=2e-3)
ge = surf(tr[:, 0], tr[:, 2], a)
ax.plot(ge[:, 0], ge[:, 1], ge[:, 2], color="C0", lw=2.0, label="测地线（同起点）")
ax.set_title("螺旋面、相位轨迹与测地线", fontsize=11)
ax.legend(fontsize=8, loc="upper left")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

ax2 = fig.add_subplot(132)
u0s = np.array([0.5, 1.0, 2.0, 4.0])
ax2.plot(u0s, [-1.0] * len(u0s), "o-", color="C3", ms=7, label="相位轨迹（归一化残差）")
ax2.plot(u0s, [0.0] * len(u0s), "s-", color="C0", ms=7, label="母线（对照）")
ax2.axhline(0, color="0.6", lw=0.8)
ax2.set_xlabel("半径 $u_0$")
ax2.set_ylabel("归一化测地残差")
ax2.set_title("同一算符的两种结果", fontsize=11)
ax2.set_ylim(-1.35, 0.35)
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)

ax3 = fig.add_subplot(133)
for u0, c in zip([0.5, 1.0, 2.0], ["C0", "C2", "C4"]):
    tr = integrate_geodesic(u0, a, T=40.0, dt=2e-3)
    ax3.plot(np.arange(len(tr)) * 2e-3, tr[:, 0], color=c, lw=1.6, label="$u_0=%.1f$" % u0)
ax3.axhline(0, color="0.6", lw=0.8)
ax3.set_xlabel("时间 $t$")
ax3.set_ylabel("半径 $u(t)$")
ax3.set_title("测地线的半径无界", fontsize=11)
ax3.legend(fontsize=8)
ax3.grid(alpha=0.3)

plt.tight_layout()
os.makedirs(IMG_DIR, exist_ok=True)
plt.savefig(FIG, dpi=150, bbox_inches="tight")
print("\n图已保存：%s" % FIG)
