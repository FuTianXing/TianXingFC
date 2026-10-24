# -*- coding: utf-8 -*-
r"""
c1_softmode_ring.py

[C1 路线乙第二判] 软体相 + 局域凹口下的三维环驻留与脱钉复核（判据 T-1 至 T-4）

背景：路线乙首判（一维静态）给出窗口互斥可按角色解耦：宽度交软体相位、稳定交
芯处局域偏好凹口，36/36 组同时满足「增益 > 0 且 l* >= 2」。本判据把它搬到
C1-4 的三维环模型上，复核驻留与脱钉阈值。

模型与 C1-2/C1-3/C1-4 同（平方型非紧致弹性，N = 32，背景 u0 = IX+IY+IZ，
边界钉背景，梯度流 eta = 0.01）：
    E = sum_axes (1/2) k (Delta u - 1)^2 + sum K(x) (1 - cos 2 pi u)
锚核 = 偏好凹口（K 在环壁芯处局域压低）：K(x) = K_soft [1 - d exp(-((rho-Rc)^2+z^2)/(2 w_a^2))]
驻留判据沿用 C1-4 的 fate_of（collapse / reside / partial）。

判据
T-1 两端复现：K = 0.02 无凹口应塌缩、K = 0.25 无凹口应驻留（复现 C1-4 的 K_c = 0.25）
T-2 软体相 + 凹口驻留：K_soft = 0.015、0.020（落窗内）配凹口，给出驻留与否
T-3 脱钉阈值：扫描凹口深度 d，给出最小驻留 d_c(K_soft)，与 C1-3 的脱钉阈值 0.15 对照
T-4 宽度核对：驻留态环壁的弹性超量密度二阶矩宽度 l*，核对落窗线 l* >= 2

运行：python 验证脚本/c1_softmode_ring.py（确定性，约 2 分钟）
"""
import time
import numpy as np

T_START = time.time()
k = 1.0
N = 32
ETA = 0.01
STEPS = 2400

IX, IY, IZ = np.indices((N, N, N))
u0 = (IX + IY + IZ).astype(float)
CX = CY = CZ = N // 2
XC, YC, ZC = IX - CX, IY - CY, IZ - CZ
RR = np.sqrt(XC * XC + YC * YC + ZC * ZC)


def energy(u, Karr):
    E = 0.0
    for ax in range(3):
        d = np.diff(u, axis=ax) - 1.0
        E += 0.5 * k * float(np.sum(d ** 2))
    E += float(np.sum(Karr * (1.0 - np.cos(2.0 * np.pi * u))))
    return E


def grad(u, Karr):
    p = np.pad(u, 1, mode="edge")
    s = (p[2:, 1:-1, 1:-1] + p[:-2, 1:-1, 1:-1] + p[1:-1, 2:, 1:-1]
         + p[1:-1, :-2, 1:-1] + p[1:-1, 1:-1, 2:] + p[1:-1, 1:-1, :-2])
    return k * (6.0 * u - s) + 2.0 * np.pi * Karr * np.sin(2.0 * np.pi * u)


def pin(u):
    u[0, :, :] = u0[0, :, :]; u[-1, :, :] = u0[-1, :, :]
    u[:, 0, :] = u0[:, 0, :]; u[:, -1, :] = u0[:, -1, :]
    u[:, :, 0] = u0[:, :, 0]; u[:, :, -1] = u0[:, :, -1]


def readout(u):
    h = u - u0
    return float(h.max()), int(np.count_nonzero(h > 0.5))


def run_gf(u_init, Karr, steps=STEPS):
    u = u_init.copy()
    tr = []
    for t in range(steps):
        u = u - ETA * grad(u, Karr)
        pin(u)
        if (t % 200 == 0) or (t == steps - 1):
            p, e = readout(u)
            tr.append((energy(u, Karr), p, e))
    return u, tr


def fate_of(tr, peak0, ext0):
    peakf, extf = tr[-1][1], tr[-1][2]
    drift = abs(tr[-1][0] - tr[-3][0]) / max(abs(tr[-1][0]), 1e-12)
    if peakf <= 0.15 * peak0 or extf <= 0.15 * ext0:
        return "collapse"
    if peakf >= 0.7 * peak0 and extf > 0.3 * ext0 and drift < 0.01:
        return "reside"
    return "partial"


def make_ring(Rc, t=1.0, w=2.5):
    g = 0.5 * (1.0 - np.tanh((RR - Rc) / w))
    return 0.5 * (1.0 - np.tanh(ZC / t)) * g


def Karr_of(K_soft, d, w_a, Rc):
    if d <= 0:
        return np.full((N, N, N), K_soft)
    core = np.exp(-(((RR - Rc) ** 2 + ZC ** 2) / (2.0 * w_a ** 2)))
    return K_soft * (1.0 - d * core)


def width_wall(u, Rc, half=5.0):
    dens = np.zeros((N, N, N))
    for ax in range(3):
        d = np.diff(u, axis=ax) - 1.0
        dd = np.zeros((N, N, N))
        if ax == 0:
            dd[:-1] = d
        elif ax == 1:
            dd[:, :-1] = d
        else:
            dd[:, :, :-1] = d
        dens += 0.5 * dd ** 2
    w = (np.abs(RR - Rc) <= half) & (np.abs(ZC) <= half)
    wgt = dens[w]
    if wgt.sum() <= 0:
        return float("nan")
    rho = RR[w]
    c0 = (wgt * rho).sum() / wgt.sum()
    return 2.0 * float(np.sqrt((wgt * (rho - c0) ** 2).sum() / wgt.sum()))


print("=" * 78)
print("C1 路线乙第二判：软体相 + 局域凹口下的三维环驻留与脱钉（T-1 至 T-4）")
print("=" * 78)
RC = 4.0
peak0, ext0 = readout(u0 + make_ring(RC))
print(f"环初态读数：peak0 = {peak0:.4f}，ext0 = {ext0}")

# T-1 两端复现
print()
print("[T-1] 两端复现（无凹口）")
for Kp, expect in ((0.02, "collapse"), (0.25, "reside")):
    Karr = np.full((N, N, N), Kp)
    uf, tr = run_gf(u0 + make_ring(RC), Karr)
    f = fate_of(tr, peak0, ext0)
    print(f"     K = {Kp:.2f}：{f:9s}（期望 {expect}）")

# T-2/T-3 软体相 + 凹口
print()
print("[T-2/T-3] 软体相 + 局域凹口：驻留与脱钉阈值")
results = {}
for K_soft in (0.015, 0.020):
    for w_a in (1.5, 2.5):
        print(f"     K_soft = {K_soft:.3f}，w_a = {w_a:.1f}")
        first = None
        for d in (0.0, 0.3, 0.5, 0.7, 0.9):
            Karr = Karr_of(K_soft, d, w_a, RC)
            uf, tr = run_gf(u0 + make_ring(RC), Karr)
            f = fate_of(tr, peak0, ext0)
            lw = width_wall(uf, RC) if f == "reside" else float("nan")
            print(f"       d = {d:.1f}：{f:9s}（E_f = {tr[-1][0]:8.2f}，l* = {lw:.3f}）")
            if f == "reside" and first is None:
                first = d
        results[(K_soft, w_a)] = first
        print(f"       最小驻留深度 d_c = {first}")

# T-3b 宽凹口与近全局凹口对照（把「脚面积」与「机制」分开）
print()
print("[T-3b] 对照：宽凹口与近全局凹口（K_soft = 0.020）")
for w_a, d in ((4.0, 0.9), (6.0, 0.9), (6.0, 0.95)):
    Karr = Karr_of(0.020, d, w_a, RC)
    uf, tr = run_gf(u0 + make_ring(RC), Karr)
    f = fate_of(tr, peak0, ext0)
    lw = width_wall(uf, RC) if f == "reside" else float("nan")
    print(f"     w_a = {w_a:.1f}，d = {d:.2f}：{f:9s}（E_f = {tr[-1][0]:8.2f}，l* = {lw:.3f}）")

# T-3c 中间 K 的过渡（无凹口，钉住驻留阈值）
print()
print("[T-3c] 无凹口的中间 K：驻留阈值周围的过渡")
for Kp in (0.10, 0.20, 0.24, 0.25):
    Karr = np.full((N, N, N), Kp)
    uf, tr = run_gf(u0 + make_ring(RC), Karr)
    f = fate_of(tr, peak0, ext0)
    lw = width_wall(uf, RC) if f == "reside" else float("nan")
    print(f"     K = {Kp:.2f}：{f:9s}（E_f = {tr[-1][0]:8.2f}，l* = {lw:.3f}）")

# T-4 与 C1-3 读数对照
print()
print("[T-4] 对照：C1-3 的脱钉阈值 0.15（锚核幅度口径）")
for key, val in results.items():
    print(f"     (K_soft, w_a) = {key}：d_c = {val}")
print(f"总耗时 {time.time() - T_START:.1f} s")
