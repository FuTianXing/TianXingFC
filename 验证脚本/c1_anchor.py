# -*- coding: utf-8 -*-
r"""
c1_anchor.py

[C1-4] 锚核实现筛选第一轮：格点内锁定（kappa_phi 周期势）的驻留窗口、
宽度核对与绑定能标度。

模型与 C1-2/C1-3 同（平方型非紧致弹性）。锚核实现 = 格点内锁定，即 K 项
（kappa_phi 周期势，把层位钉在整数）。

判据（先立判据，再实现；预期写错时修正判据并记录教训）
----
A1 驻留窗口细化：环 R = 4 与 R = 5，K = 0.20 至 0.28 步 0.02 加密 0.25
   （GF 2400 步）给出最小驻留 K_c(R)；预期 K_c(4) ∈ (0.24, 0.26]（C1-3
   的 0.25 读数与本次加密共同夹定），K_c(5) <= K_c(4)。
A2 宽度核对：三维壁的单界面宽度 l*_3D(K) 与同模型一维 kink 宽度
   l*_1D(K) 一致（准平衡两档 K = 0.02、0.05，偏差 <= 30%）；K = 0.25 档的
   三维驻留锐化态（定态）与一维自由平衡宽度的差值作读数报告。
   v1/v2 教训：宽度窗不得同时套住平板的两个界面；驻留锐化态与自由
   弛豫态是两个归属。
A3 窗口非空性判定：驻留窗口 [K_c, ∞) 与落窗窗口（l* >= 2；一维与三维
   复测给出对应 K 上界）的交集；预期为空。
A4 绑定能标度：驻留态能量 E_res(K)（R = 4，K = 0.24 至 0.30）的读数与
   对 K 的幂次拟合（不做通过条件，供 C1-6 锚核标定）。

运行
----
python 验证脚本/c1_anchor.py
确定性（无随机数），运行约 60 秒。图存
D:/software/TianXingFC/images/c1_anchor.png。
"""

import time
import numpy as np

T_START = time.time()

k = 1.0
N = 32
ETA = 0.01

print("=" * 72)
print("[C1-4] 锚核筛选：格点内锁定的驻留窗口、宽度核对与绑定能标度")
print("=" * 72)

IX, IY, IZ = np.indices((N, N, N))
u0 = (IX + IY + IZ).astype(float)
CX = CY = CZ = N // 2
XC = IX - CX
YC = IY - CY
ZC = IZ - CZ
RR = np.sqrt(XC * XC + YC * YC + ZC * ZC)

CHECKS = []


def check(tag, ok, detail):
    CHECKS.append((tag, bool(ok)))
    print(f"  [{tag}] {'PASS' if ok else 'FAIL'}  {detail}")


def energy(u, Kp):
    E = 0.0
    for ax in range(3):
        d = np.diff(u, axis=ax) - 1.0
        E += 0.5 * k * float(np.sum(d ** 2))
    E += Kp * float(np.sum(1.0 - np.cos(2.0 * np.pi * u)))
    return E


def grad(u, Kp):
    p = np.pad(u, 1, mode="edge")
    ctr = (slice(1, -1),) * 3
    s = (p[2:, 1:-1, 1:-1] + p[:-2, 1:-1, 1:-1] + p[1:-1, 2:, 1:-1]
         + p[1:-1, :-2, 1:-1] + p[1:-1, 1:-1, 2:] + p[1:-1, 1:-1, :-2])
    return k * (6.0 * u - s) + 2.0 * np.pi * Kp * np.sin(2.0 * np.pi * u)


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


def run_gf(u_init, Kp, steps, snap=200):
    u = u_init.copy()
    tr = []
    for t in range(steps):
        u = u - ETA * grad(u, Kp)
        pin(u)
        if (t % snap == 0) or (t == steps - 1):
            p, e = readout(u)
            tr.append((energy(u, Kp), p, e))
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


def make_wall(d, t1=1.2, margin=3.0, mw=1.5):
    z1, z2 = -d / 2.0, d / 2.0
    h = 0.5 * (np.tanh((ZC - z1) / t1) - np.tanh((ZC - z2) / t1))
    dx = np.minimum(IX, N - 1 - IX)
    dy = np.minimum(IY, N - 1 - IY)
    g = 0.5 * (1.0 - np.tanh((margin - np.minimum(dx, dy)) / mw))
    return h * g


# ----------------------------------------------------------------------
# A1 驻留窗口细化
# ----------------------------------------------------------------------
print("\n[A1] 驻留窗口细化：环 R=4、R=5，K = 0.20 至 0.28（加密 0.25）")
K_SCAN = [0.20, 0.22, 0.24, 0.25, 0.26, 0.28]
KC = {}
for Rc in (4.0, 5.0):
    print(f"    R = {Rc}:")
    for Kp in K_SCAN:
        u_init = u0 + make_ring(Rc)
        peak0, ext0 = readout(u_init)
        uf, tr = run_gf(u_init, Kp, 2400)
        f = fate_of(tr, peak0, ext0)
        print(f"      K={Kp:.2f}: {f:9s}（E_f = {tr[-1][0]:8.2f}）")
        if f == "reside" and (Rc not in KC):
            KC[Rc] = Kp
K_C4 = KC.get(4.0, None)
K_C5 = KC.get(5.0, None)
print(f"    最小驻留 K_c：R=4 → {K_C4}；R=5 → {K_C5}")
a1_ok = (K_C4 is not None and 0.24 < K_C4 <= 0.26)
a1_ok &= (K_C5 is None or K_C4 is None or K_C5 <= K_C4)
check("A1", a1_ok, f"K_c(4) = {K_C4}（期望 (0.24, 0.26]）；K_c(5) = {K_C5}（期望 <= K_c(4)）")

# ----------------------------------------------------------------------
# A2 宽度核对（单界面窗 + 一维标定）
# ----------------------------------------------------------------------
print("\n[A2] 宽度核对：三维壁单界面 l*_3D vs 一维 kink l*_1D（同模型同定义）")


def width_from_density(dens, idx, center, half):
    sel = np.abs(idx - center) <= half
    d2 = dens[sel]
    s_ = d2.sum()
    if s_ < 1e-12:
        return 0.0
    i2 = idx[sel]
    c0 = (d2 * i2).sum() / s_
    var = (d2 * (i2 - c0) ** 2).sum() / s_
    return 2.0 * float(np.sqrt(var))


def kink_width_1d(Kp, N1=400, steps=20000, eta=0.02):
    u = np.arange(N1, dtype=float)
    u[N1 // 2:] += 1.0
    u[N1 // 2] = N1 / 2 + 0.5
    for t in range(steps):
        g = np.zeros(N1)
        g[1:-1] = k * (2.0 * u[1:-1] - u[:-2] - u[2:]) + 2.0 * np.pi * Kp * np.sin(2.0 * np.pi * u[1:-1])
        u[1:-1] -= eta * g[1:-1]
        u[0] = 0.0
        u[-1] = float(N1)
    dens = 0.5 * (np.diff(u) - 1.0) ** 2
    idx = np.arange(N1 - 1) + 0.5
    return width_from_density(dens, idx, 0.5 * (N1 - 1), 12)


W_ROWS = []
for Kp in (0.02, 0.05, 0.25):
    steps3 = 2400 if Kp > 0.2 else 600
    u_init = u0 + make_wall(12.0)
    uf, _ = run_gf(u_init, Kp, steps3, snap=600)
    # 第一界面位于 z = -6（d=12），idx 中心 = (N-1)/2 - 6
    zc = 0.5 * (N - 1) - 6.0
    dens = 0.5 * (np.diff(uf[CX, CY, :]) - 1.0) ** 2
    idx = np.arange(N - 1) + 0.5
    w3 = width_from_density(dens, idx, zc, 5)
    w1 = kink_width_1d(Kp)
    dev = abs(w3 - w1) / max(w1, 1e-9)
    W_ROWS.append((Kp, w3, w1, dev))
    print(f"    K={Kp:.2f}: l*_3D = {w3:.3f}，l*_1D = {w1:.3f}，偏差 {dev * 100:.1f}%")
dev_q = max(r[3] for r in W_ROWS if r[0] <= 0.05)
check("A2", dev_q <= 0.30,
      f"准平衡两档最大偏差 {dev_q * 100:.1f}%（期望 <= 30%）；K=0.25 读数：三维驻留锐化 "
      f"{W_ROWS[2][1]:.3f} vs 一维自由平衡 {W_ROWS[2][2]:.3f}（两种归属，作读数）")

# 落窗上界（本例定：l*(K) = l*_1D 同族，扫 K 找 l* = 2）
from numpy import log  # noqa
K_grid = [0.01 * (1.15 ** i) for i in range(30)]
K_win = None
for Kp in K_grid:
    if kink_width_1d(Kp, N1=200, steps=8000) <= 2.0:
        K_win = Kp
        break
print(f"    落窗上界（l*_1D <= 2 的最大 K，粗扫）：K_win ≈ {K_win:.4f}")

# ----------------------------------------------------------------------
# A3 窗口非空性
# ----------------------------------------------------------------------
print("\n[A3] 窗口非空性：驻留窗口与落窗窗口的交集")
print(f"    落窗窗口：K <= {K_win:.4f}")
print(f"    驻留窗口：K >= K_c(4) = {K_C4}")
overlap = (K_C4 is not None) and (K_win is not None) and (K_C4 <= K_win)
print(f"    交集：{'非空' if overlap else '空'}")
check("A3", not overlap, f"驻留窗口与落窗窗口交集为空 = {not overlap}（预期空，负面读数）")

# ----------------------------------------------------------------------
# A4 绑定能标度读数
# ----------------------------------------------------------------------
print("\n[A4] 绑定能标度：驻留态能量 E_res(K)（R=4）")
E_RES = []
for Kp in (0.24, 0.25, 0.26, 0.28, 0.30):
    u_init = u0 + make_ring(4.0)
    uf, tr = run_gf(u_init, Kp, 2400)
    E_RES.append((Kp, tr[-1][0]))
    print(f"    K={Kp:.2f}: E_res = {tr[-1][0]:.3f}")
ks_ = np.array([x[0] for x in E_RES])
es_ = np.array([x[1] for x in E_RES])
m = es_ > 1e-6
if m.sum() >= 3:
    slope = float(np.polyfit(np.log(ks_[m]), np.log(es_[m]), 1)[0])
    print(f"    E_res 对 K 的双对数斜率 = {slope:.3f}（读数，供 C1-6 标定）")
else:
    print("    驻留点不足，斜率未拟合")

# ----------------------------------------------------------------------
# 汇总
# ----------------------------------------------------------------------
n_pass = sum(1 for _, ok in CHECKS if ok)
print(f"\n判据汇总：{n_pass}/{len(CHECKS)} 通过；总运行 {time.time() - T_START:.0f} s")
for tag, ok in CHECKS:
    print(f"  {tag}: {'PASS' if ok else 'FAIL'}")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.2))
    a = axs[0]
    a.plot([x[0] for x in E_RES], [x[1] for x in E_RES], "o-")
    a.set_xlabel("K (kappa_phi)"); a.set_ylabel("E_res")
    a.set_title("(a) 驻留态能量 E_res(K)，R=4")
    b = axs[1]
    kw = [r[0] for r in W_ROWS]
    b.plot(kw, [r[1] for r in W_ROWS], "o-", label="l*_3D（壁单界面）")
    b.plot(kw, [r[2] for r in W_ROWS], "s--", label="l*_1D（kink）")
    b.axhline(2.0, color="gray", ls=":", lw=0.9)
    b.set_xscale("log"); b.set_yscale("log")
    b.set_xlabel("K"); b.set_ylabel("l*")
    b.set_title("(b) 宽度核对与落窗线 l*=2")
    b.legend()
    fig.tight_layout()
    fig.savefig("D:/software/TianXingFC/images/c1_anchor.png", dpi=150)
    print("图已保存：images/c1_anchor.png")
except Exception as exc:
    print(f"绘图失败（不影响判据）：{exc}")

print("\n[C1-4] 完成。")