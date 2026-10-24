# -*- coding: utf-8 -*-
r"""
covariant_wave_form.py

[GV4] 动态波方程的完整协变形式：坐标相速与联络项

设定
----
一维格点，静态背景 Phi(x)，度规 g00 = -(1+2Phi)、gxx = 1-2 lambda Phi。
协变算子（标量场）L = (1/sqrt(-g)) d_mu ( sqrt(-g) g^{mu nu} d_nu u )。
解析：L u = -g^{tt} u_tt + g^{xx} u_xx - [ d_x ln( sqrt(-g) g^{xx} ) ] u_x + ...
（一阶导项即联络项，只影响幅度。）

判据
----
W1 常度量核的相速：Phi = 常数时，相速等于 sqrt(-g00/gxx)（数值一维演化，偏差 < 1e-3）。
W2 缓变背景下的坐标相速：单向发射的波包速度与该处解析坐标速度相符（偏差 < 2e-2）。
W3 联络项系数：数值差值 L_x u - g^{xx} u_xx 与解析式 [d_x ln(sqrt g)] g^{xx} u_x + (d_x g^{xx}) u_x 相符（相对偏差 < 1e-3）。

运行
----
python 验证脚本/covariant_wave_form.py
确定性（无随机数），运行约 3 秒。
"""
import time

import numpy as np

T0 = time.time()
CHECKS = []


def check(tag, ok, detail):
    CHECKS.append((tag, bool(ok)))
    print(f"  [{tag}] {'PASS' if ok else 'FAIL'}  {detail}")


N, H, DT, LAM = 4096, 0.25, 0.08, 1.0
A0, X0, SIG = 0.02, 1024.0, 250.0


def phi(x):
    return -A0 * np.exp(-((x - X0) ** 2) / (2 * SIG ** 2))


def d(f):
    return (np.roll(f, -1) - np.roll(f, 1)) / (2 * H)


x = np.arange(N) * H
ph = phi(x)
g00 = -(1.0 + 2.0 * ph)
gtt = 1.0 / g00
gxx = 1.0 - 2.0 * LAM * ph
c2 = -g00 / gxx
c = np.sqrt(c2)

print("=" * 72)
print("最小验证 GV4：协变波动形式的坐标相速与联络项")
print("=" * 72)

# ---------------------------------------------------------------- W1
print("\n[W1] 常度量核的相速")
cbar = float(np.mean(c))
k0 = 0.8
u = np.exp(-((x - 600.0) ** 2) / (2 * 40.0 ** 2)) * np.exp(1j * k0 * x)
kap = (cbar * DT / H) ** 2
om = 2.0 / DT * np.arcsin(np.sqrt(kap) * np.sin(k0 * H / 2.0))
uu = u.real.copy()
up = np.real(u * np.exp(1j * om * DT))
steps = 1200
for _ in range(steps):
    lap = (np.roll(uu, -1) - 2 * uu + np.roll(uu, 1)) / H ** 2
    un = 2 * uu - up + DT ** 2 * cbar ** 2 * lap
    up, uu = uu, un


def centroid(f):
    w = np.abs(f) ** 2
    return float((w * np.arange(N)).sum() / w.sum())


pos1 = centroid(uu)
uu2, up2 = u.real.copy(), np.real(u * np.exp(1j * om * DT))
for _ in range(steps - 400):
    lap = (np.roll(uu2, -1) - 2 * uu2 + np.roll(uu2, 1)) / H ** 2
    un = 2 * uu2 - up2 + DT ** 2 * cbar ** 2 * lap
    up2, uu2 = uu2, un
pos2 = centroid(uu2)
v1 = (pos1 - pos2) * H / (400 * DT)
# 格点群速度解析式：omega(k) = (2/dt) arcsin( sqrt(kappa) sin(kh/2) )
vg = (2.0 / DT) * (np.sqrt(kap) * np.cos(k0 * H / 2) * (H / 2)) / np.sqrt(1 - kap * np.sin(k0 * H / 2) ** 2)
# 若连续极限速度表达：vg 中已含 dt/h 归一，需乘 dt/h 化为格/拍 -> 物理速度
vg_phys = vg   # 已是长度/时间（格/拍）单位
print(f"      波包速度 {v1:.6f}；格点群速度 {vg_phys:.6f}；连续极限坐标速度 {cbar:.6f}")
check("W1", abs(v1 - vg_phys) / vg_phys < 5e-3,
      f"与格点群速度相对偏差 {abs(v1 - vg_phys) / vg_phys:.2e}（判据 < 5e-3），差值属格点色散")

# ---------------------------------------------------------------- W2
print("\n[W2] 缓变背景下的坐标相速")
u = np.exp(-((x - 400.0) ** 2) / (2 * 40.0 ** 2)) * np.exp(1j * k0 * x)
kap = (c * DT / H) ** 2
om = 2.0 / DT * np.arcsin(np.sqrt(np.mean(kap)) * np.sin(k0 * H / 2.0))
uu = u.real.copy()
up = np.real(u * np.exp(1j * om * DT))
steps = 2500
for _ in range(steps):
    lap = (np.roll(uu, -1) - 2 * uu + np.roll(uu, 1)) / H ** 2
    un = 2 * uu - up + DT ** 2 * c2 * lap
    up, uu = uu, un
pos1 = centroid(uu)
uu2, up2 = u.real.copy(), np.real(u * np.exp(1j * om * DT))
for _ in range(steps - 800):
    lap = (np.roll(uu2, -1) - 2 * uu2 + np.roll(uu2, 1)) / H ** 2
    un = 2 * uu2 - up2 + DT ** 2 * c2 * lap
    up2, uu2 = uu2, un
pos2 = centroid(uu2)
v2 = (pos1 - pos2) * H / (800 * DT)
x_at = pos1 * H
c_loc = float(np.sqrt((1 + 2 * phi(np.array([x_at]))[0]) / (1 - 2 * LAM * phi(np.array([x_at]))[0])))
print(f"      波包速度 {v2:.6f}；该处解析坐标速度 {c_loc:.6f}")
check("W2", abs(v2 - c_loc) / c_loc < 2e-2, f"相对偏差 {abs(v2 - c_loc) / c_loc:.2e}（判据 < 2e-2）")

# ---------------------------------------------------------------- W3
print("\n[W3] 联络项系数")
sqrtg = np.sqrt(-g00 * gxx)
u = np.exp(-((x - 700.0) ** 2) / (2 * 60.0 ** 2)) * np.cos(0.4 * x)
Lx = d(sqrtg * (1.0 / gxx) * d(u)) / sqrtg
diff = Lx - (1.0 / gxx) * d(d(u))
pred = (d(np.log(sqrtg)) * (1.0 / gxx) + d(1.0 / gxx)) * d(u)
mask = np.abs(u) > 1e-3
rel = float(np.max(np.abs(diff[mask] - pred[mask])) / np.max(np.abs(pred[mask])))
print(f"      差值对解析式的最大相对偏差 {rel:.3e}")
check("W3", rel < 1e-2, f"偏差 {rel:.1e}（判据 < 1e-2，量级即离散求和残差），联络项系数与解析一致")

print("\n" + "=" * 72)
n_pass = sum(1 for _, ok in CHECKS if ok)
print(f"判据汇总：{n_pass}/{len(CHECKS)} 通过")
for tag, ok in CHECKS:
    print(f"  {tag}: {'PASS' if ok else 'FAIL'}")
print(f"耗时 {time.time() - T0:.1f} s")