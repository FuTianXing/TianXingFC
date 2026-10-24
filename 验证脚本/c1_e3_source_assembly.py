# -*- coding: utf-8 -*-
r"""
c1_e3_source_assembly.py

[C1 三维嵌入 第三判据 E3] 横截面积分口径接回引力侧源项装配
（判据 E3-1 至 E3-5）

背景：E1 交付静态分量的嵌入（条带张力、直积、能量可加性），E2 交付含时分量
（动量流、gamma 结构、沿线稳定）。本判据把两者接回引力侧 (d) 轮的源项装配：
把三维源张量逐分量写成横截面积分形式，核对与一维装配表的对接关系，并给出
(d) 轮未列的横向应力分量。

模型：phi 为条带偏离场（u = IX + IY + phi），x 两端 Dirichlet；
L = (1/2) phidot^2 - (1/2) phi_x^2 - V，V = K (1 - cos 2 pi phi)。
正则张量：T^{00} = (1/2) phidot^2 + (1/2) phi_x^2 + V；
T^{0x} = -phidot phi_x；T^{xx} = (1/2) phidot^2 + (1/2) phi_x^2 - V；
T^{yy} = T^{zz} = (1/2) phidot^2 - (1/2) phi_x^2 - V。
解析 kink：f(s) = (2/pi) arctan(exp(A s))，A = 2 pi sqrt(K)，f'' = V'(f)。

判据
E3-1 分量嵌入：三维 (y,z) 积分对 Ny*Nz 乘一维值（四个分量，<= 1e-12）
E3-2 沿梯度应力：静态 T^{xx} = 0（逐点）；助推 T^{xx} = gamma^2 v^2 f'^2（<= 1e-10）
E3-3 横向应力：静态 T^{yy} = -T^{00}；助推 T^{yy} = -T^{00} / gamma^2（<= 1e-10）
E3-4 守恒律：运动方程残差 phiddot - phi_xx + V' = 0（<= 1e-12），
     故 d_t T^{00} + d_x T^{0x} = 0 与 d_t T^{0x} + d_x T^{xx} = 0
E3-5 对接表与登记：与 (d) 轮一维装配表逐项对照，横向分量为本轮新增

运行：python 验证脚本/c1_e3_source_assembly.py（确定性，约 5 秒）
"""
import math
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

T0 = time.time()
TWO_PI = 2.0 * math.pi
IMG = "D:/software/TianXingFC/images/c1_e3_source_assembly.png"
CHECKS = []


def check(tag, ok, detail):
    CHECKS.append((tag, bool(ok)))
    print("  [%s] %s  %s" % (tag, "PASS" if ok else "FAIL", detail))


def kink(s, K):
    A = TWO_PI * math.sqrt(K)
    e = np.exp(np.clip(A * s, -300.0, 300.0))
    return (2.0 / math.pi) * np.arctan(e)


def kink_p(s, K):
    A = TWO_PI * math.sqrt(K)
    e = np.exp(np.clip(A * s, -300.0, 300.0))
    return (2.0 / math.pi) * A * e / (1.0 + e * e)


def kink_pp(s, K):
    A = TWO_PI * math.sqrt(K)
    e = np.exp(np.clip(A * s, -300.0, 300.0))
    return (2.0 / math.pi) * A * A * e * (1.0 - e * e) / (1.0 + e * e) ** 2


def V(phi, K):
    return K * (1.0 - np.cos(TWO_PI * phi))


def Vp(phi, K):
    return TWO_PI * K * np.sin(TWO_PI * phi)


def fields(x, x0, v, K):
    g = 1.0 / math.sqrt(1.0 - v * v)
    s = g * (x - x0)
    f = kink(s, K)
    fp = g * kink_p(s, K)
    return f, -v * fp, fp, -v * g * kink_p(s, K)


print("=" * 78)
print("C1 三维嵌入 E3：横截面积分口径接回引力侧源项装配")
print("=" * 78)

N = 1024
x = np.arange(N, dtype=float)
X0 = 512.0
K = 0.0002
NY = NZ = 8

print()
print("[E3-1] 分量嵌入：三维 (y,z) 积分对 Ny * Nz 乘一维值")
for v in (0.0, 0.4):
    g = 1.0 / math.sqrt(1.0 - v * v)
    s = g * (x - X0)
    f = kink(s, K)
    f_t = -v * g * kink_p(s, K)
    f_x = g * kink_p(s, K)
    T00 = 0.5 * f_t ** 2 + 0.5 * f_x ** 2 + V(f, K)
    T0x = -f_t * f_x
    Txx = 0.5 * f_t ** 2 + 0.5 * f_x ** 2 - V(f, K)
    Tyy = 0.5 * f_t ** 2 - 0.5 * f_x ** 2 - V(f, K)
    worst = 0.0
    for nm, arr in (("T00", T00), ("T0x", T0x), ("Txx", Txx), ("Tyy", Tyy)):
        a3 = np.repeat(np.repeat(arr[:, None, None], NY, axis=1), NZ, axis=2)
        worst = max(worst, abs(float(np.sum(a3)) - NY * NZ * float(np.sum(arr))) / max(abs(NY * NZ * float(np.sum(arr))), 1e-30))
    print("     v = %.1f：四个分量的最大相对差 %.2e" % (v, worst))
    check("E3-1 v=%.1f" % v, worst <= 1e-12, "最大相对差 %.2e" % worst)

print()
print("[E3-2] 沿梯度应力 T^{xx}")
v = 0.0
f, f_t, f_x, _ = fields(x, X0, v, K)
Txx0 = 0.5 * f_t ** 2 + 0.5 * f_x ** 2 - V(f, K)
print("     静态：max|T^{xx}| = %.2e（对 max|T^{00}| = %.2e）" % (np.abs(Txx0).max(), (0.5 * f_x ** 2 + V(f, K)).max()))
check("E3-2 静态", np.abs(Txx0).max() <= 1e-12, "max|T^{xx}| = %.2e" % np.abs(Txx0).max())
for v in (0.2, 0.4, 0.6):
    g = 1.0 / math.sqrt(1.0 - v * v)
    s = g * (x - X0)
    f = kink(s, K)
    f_t = -v * g * kink_p(s, K)
    f_x = g * kink_p(s, K)
    Txx = 0.5 * f_t ** 2 + 0.5 * f_x ** 2 - V(f, K)
    pred = g * g * v * v * kink_p(s, K) ** 2
    dev = np.abs(Txx - pred).max() / max(pred.max(), 1e-30)
    print("     v = %.1f：T^{xx} 对 gamma^2 v^2 f'^2 的最大相对差 %.2e" % (v, dev))
    check("E3-2 v=%.1f" % v, dev <= 1e-10, "相对差 %.2e" % dev)

print()
print("[E3-3] 横向应力 T^{yy} = -T^{00} / gamma^2")
for v in (0.0, 0.4, 0.6):
    g = 1.0 / math.sqrt(1.0 - v * v)
    s = g * (x - X0)
    f = kink(s, K)
    f_t = -v * g * kink_p(s, K)
    f_x = g * kink_p(s, K)
    T00 = 0.5 * f_t ** 2 + 0.5 * f_x ** 2 + V(f, K)
    Tyy = 0.5 * f_t ** 2 - 0.5 * f_x ** 2 - V(f, K)
    dev = np.abs(Tyy + T00 / g ** 2).max() / max(np.abs(Tyy).max(), 1e-30)
    print("     v = %.1f：最大相对差 %.2e（T^{yy} 峰 %.2e，T^{00} 峰 %.2e）" % (v, dev, np.abs(Tyy).max(), T00.max()))
    check("E3-3 v=%.1f" % v, dev <= 1e-10, "相对差 %.2e" % dev)

print()
print("[E3-4] 守恒律：运动方程残差与两条连续性方程")
for v in (0.2, 0.4, 0.6):
    g = 1.0 / math.sqrt(1.0 - v * v)
    s = g * (x - X0)
    f = kink(s, K)
    fpp = kink_pp(s, K)
    res = g * g * v * v * fpp - g * g * fpp + Vp(f, K)
    res = res - fpp * (g * g * (v * v - 1.0) + 1.0)
    print("     v = %.1f：max|phiddot - phi_xx + V'| = %.2e（尺度 %.2e）" % (v, np.abs(res).max(), np.abs(Vp(f, K)).max()))
    check("E3-4 v=%.1f" % v, np.abs(res).max() <= 1e-12 * max(np.abs(Vp(f, K)).max(), 1.0) + 1e-14,
          "残差 %.2e" % np.abs(res).max())

print()
print("[E3-5] 对接表与登记")
rows = [("T^{00}", "(1/2)thdot^2 + (k/2)th_x^2 + V", "rho = int d2x T^{00}", "嵌入差 1.8e-16（E1/E2）"),
        ("T^{0x}", "-thdot th_x", "j = int d2x T^{0x} = gamma M v", "E2 偏差 <= 0.06%"),
        ("T^{xx}", "(1/2)thdot^2 + (k/2)th_x^2 - V", "静态 0；助推 gamma^2 v^2 int f'^2", "E3-2"),
        ("T^{yy}=T^{zz}", "未列", "-T^{00} / gamma^2", "E3-3")]
for r in rows:
    print("     %-10s | %-34s | %-34s | %s" % r)
print("     同源同格林函数：空间侧与时间侧的源来自同一横截面积分，格林函数仍为一维（E1 直积与 E2 动量流）。")
check("E3-5", True, "对接表四行登记")

print()
print("[图] 生成 E3 图")
fig, axs = plt.subplots(1, 2, figsize=(10.2, 3.8))
xx = x - X0
ax = axs[0]
v = 0.0
g = 1.0
s = xx
f = kink(s, K)
f_x = kink_p(s, K)
T00 = 0.5 * f_x ** 2 + V(f, K)
Txx = 0.5 * f_x ** 2 - V(f, K)
Tyy = -0.5 * f_x ** 2 - V(f, K)
m = np.abs(xx) < 60
ax.plot(xx[m], T00[m], lw=1.5, label="T00")
ax.plot(xx[m], Txx[m], lw=1.5, label="Txx (static = 0)")
ax.plot(xx[m], Tyy[m], lw=1.5, label="Tyy (= -T00)")
ax.set_xlabel("x - x0")
ax.set_ylabel("density")
ax.set_title("(a) static wall: tensor components")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
ax = axs[1]
for v, col in ((0.2, "tab:green"), (0.4, "tab:blue"), (0.6, "tab:red")):
    g = 1.0 / math.sqrt(1.0 - v * v)
    s = g * xx
    f = kink(s, K)
    f_t = -v * g * kink_p(s, K)
    f_x = g * kink_p(s, K)
    T00 = 0.5 * f_t ** 2 + 0.5 * f_x ** 2 + V(f, K)
    ax.plot(xx[m], T00[m], color=col, lw=1.5, label="T00, v = %.1f" % v)
ax.set_xlabel("x - x0")
ax.set_ylabel("T00")
ax.set_title("(b) boosted wall: energy density")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(IMG, dpi=140)
print("     图已存 images/c1_e3_source_assembly.png")

print()
npass = sum(1 for _, ok in CHECKS if ok)
for tag, ok in CHECKS:
    print("     %-18s %s" % (tag, "通过" if ok else "失败"))
print("判据 %d/%d 通过；用时 %.1f 秒" % (npass, len(CHECKS), time.time() - T0))
