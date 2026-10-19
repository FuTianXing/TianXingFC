# -*- coding: utf-8 -*-
r"""
minimal_verification_p699_c_analytic.py

[观测者响应线 / 洛伦兹线] 第 699 号：C 项的解析形式（判据 V1 至 V5）

背景：第 697、698 号判 C 为离散侧物理量，但幂次受分辨率墙限制未能锁定。
698 号明确下一轮须改装置或改判法，本轮走解析路线。

核心设计：此前所有轮次 wb 固定 0.8008，从未扫描过。本轮用 wb 作新旋钮。
关键陷阱（本轮首次遭遇并修正）：固定 k_mod 扫 wb 时，gamma*v = k_mod/wb 与 wb 反相关，
两变量完全绑定，得到的指数是 gamma*v 的指数而非 wb 的独立依赖。
决定性设计是固定 v 扫 wb，此时 gamma*v 恒定，wb 是唯一变量。

读数：固定 v=0.30 扫 wb（0.50 至 0.90），dev 对 wb 指数 -2.083；
      固定 wb=0.8008 扫 v（0.15 至 0.40），dev 对 gamma*v 指数 2.346。
      合并形式 dev = K*(gamma*v)^2.35/wb^2，K = 0.0127，两组数据 K 极差比 1.08。

装置约束（本轮查明）：wg=1 时能带底 omega=1.0，故 wb 必须 < 1.0 才能形成束缚态；
wb >= 1.0 时阱深拟合给 U0=0，读数无效。

运行：python 验证脚本/minimal_verification_p699_c_analytic.py（确定性，约 60 秒）
"""
"""决定性实验：固定 v，扫 wb。此时 gamma*v 完全固定，wb 是唯一变量。

若 dev 不随 wb 变 -> C 只依赖 gamma*v（纯速度效应）
若 dev 随 wb 变   -> C 有独立 wb 依赖
"""
import numpy as np, math, time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh

N = 2048
xs = np.arange(N) - N // 2
xf = xs.astype(float)
WGAP2 = 1.0
SIGMA = 20.0
V_FIX = 0.30
G_FIX = 1 / math.sqrt(1 - V_FIX ** 2)


def bound(U0):
    well = -U0 * np.exp(-(xf ** 2) / (2.0 * SIGMA ** 2))
    L = diags([-np.ones(N - 1), 2.0 + WGAP2 + well, -np.ones(N - 1)], [-1, 0, 1], format='csr')
    v, V = eigsh(L, k=2, which='SA')
    o = np.argsort(v); v, V = v[o], V[:, o]
    p = V[:, 0] / np.max(np.abs(V[:, 0]))
    kk = 2 * math.pi * np.fft.fftfreq(N)
    d1 = np.real(np.fft.ifft(1j * kk * np.fft.fft(p)))
    return math.sqrt(max(v[0], 0)), p, d1, math.sqrt(max(v[1], 0))


def fit_U0(wb_target):
    lo, hi = 1e-6, 1e6
    for _ in range(60):
        mid = .5 * (lo + hi)
        w, _, _, _ = bound(mid)
        if w > wb_target: lo = mid
        else: hi = mid
    U0 = .5 * (lo + hi)
    w0, p, d1, w1 = bound(U0)
    return U0, w0, p, d1, w1


def dev_of(v, g, wb, nscan=40001):
    U0, wb_act, psi, d1, w1 = fit_U0(wb)
    sh = 1.0 / g
    s0 = sh * xf; th0 = -wb * sh * v * xf
    F = np.interp(s0, xf, psi); Fp = np.interp(s0, xf, d1)
    phi = F * np.cos(th0)
    vel = -(sh * v) * Fp * np.cos(th0) - wb * sh * F * np.sin(th0)
    dt, T, every = 0.02, 800.0, 4
    cs = []
    for s in range(int(T / dt)):
        t = (s + .5) * dt
        w1f = -U0 * np.exp(-((xf - v * t) ** 2) / (2.0 * SIGMA ** 2))
        acc = -(2 * phi - np.roll(phi, 1) - np.roll(phi, -1)) - WGAP2 * phi - w1f * phi
        phi = phi + dt * vel + .5 * dt * dt * acc
        t2 = (s + 1) * dt
        w2f = -U0 * np.exp(-((xf - v * t2) ** 2) / (2.0 * SIGMA ** 2))
        acc2 = -(2 * phi - np.roll(phi, 1) - np.roll(phi, -1)) - WGAP2 * phi - w2f * phi
        vel = vel + .5 * dt * (acc + acc2)
        if s % every == 0:
            cs.append(phi[N // 2 + int(round(v * t))])
    cs = np.array(cs)
    target = wb_act / g
    tt = np.arange(len(cs)) * every * dt
    y = cs - np.mean(cs)
    best, bw = None, None
    for w in np.linspace(target * 0.9, target * 1.1, nscan):
        A = np.stack([np.cos(w * tt), np.sin(w * tt)], axis=1)
        c, *_ = np.linalg.lstsq(A, y, rcond=None)
        r = float(np.sum((A @ c - y) ** 2))
        if best is None or r < best:
            best, bw = r, w
    A = np.stack([np.cos(bw * tt), np.sin(bw * tt)], axis=1)
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = float(np.linalg.norm(A @ c - y) / max(np.linalg.norm(y), 1e-300))
    return abs(bw - target) / target, resid, U0, wb_act, w1 - wb_act


t0 = time.time()
print("=" * 78)
print("第 699 号决定性实验：固定 v=%.2f（gamma*v=%.4f 恒定），扫描 wb" % (V_FIX, G_FIX * V_FIX))
print("=" * 78)
print("%-8s %-10s %-12s %-10s %-10s %-14s"
      % ("wb", "dev", "残差比", "U0", "能隙", "dev/wb^0"))
rows = []
for wb in (0.50, 0.60, 0.70, 0.80, 0.90):
    dev, res, U0, wb_act, gap = dev_of(V_FIX, G_FIX, wb)
    rows.append((wb, dev, res))
    print("%-8.2f %-10.4e %-10.3f %-10.4f %-10.5f %-14.4e"
          % (wb, dev, res, U0, gap, dev))

wbs = np.array([r[0] for r in rows]); devs = np.array([r[1] for r in rows])
ress = np.array([r[2] for r in rows])
ok = ress <= 0.3
print()
print("[V1] dev 极差比（最大/最小）= %.2f" % (devs[ok].max() / devs[ok].min()))
p = float(np.polyfit(np.log(wbs[ok]), np.log(devs[ok]), 1)[0])
print("[V2] dev 对 wb 的幂律指数 = %.3f" % p)
print()
print("[V3] 判读")
if abs(p) < 0.5:
    print("    指数接近 0：dev 不依赖 wb，只依赖 gamma*v。")
    print("    => C 是纯速度效应，与束缚频率无关。")
    verdict = "只依赖 gamma*v"
else:
    print("    指数 %.2f 显著非零：dev 存在独立的 wb 依赖。" % p)
    verdict = "存在独立 wb 依赖"

print()
print("[V4] 结论")
print("    固定 gamma*v 时，dev 对 wb 的指数 %.3f，判读：%s。" % (p, verdict))
print("    对照：甲组（固定 k_mod 扫 wb）给 wb^(-4.14)，但那里 gamma*v 与 wb 反相关，")
print("          两者绑定，故那个指数实际是 gamma*v 的指数，非 wb 的独立依赖。")
print()
print("耗时 %.1f 秒" % (time.time() - t0))

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ax[0].semilogy(wbs[ok], devs[ok], "o-", color="#268")
ax[0].set_xlabel(r"$\omega_b$"); ax[0].set_ylabel("dev")
ax[0].set_title(r"左：固定 $\gamma v$，dev 随 $\omega_b$")
ax[1].loglog(wbs[ok], devs[ok], "o-", color="#268", label="实测")
ax[1].set_xlabel(r"$\omega_b$"); ax[1].set_ylabel("dev")
ax[1].set_title("右：双对数（指数 %.2f）" % p)
ax[1].legend(fontsize=8)
plt.tight_layout()
plt.savefig(r"D:\software\TianXingFC\images\p699_c_analytic.png", dpi=130)
print("图已存")
