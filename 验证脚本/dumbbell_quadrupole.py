# -*- coding: utf-8 -*-
"""哑铃源的近场四极：桥对照的解析化（第 451 条；443 的近场各向异性解析）。

判据（先立判据，再实现）：
  A1 数值与连续求和对照：格点泊松解 theta 与「站点源 + 连续格林 1/(4pi r) 求和」在
     近场（r = 3 至 12）的「沿 z / 横向」比值一致（相对差小），差值归格点修正
  A2 四极解释：比值的 r 依赖由四极矩 Q_nn 主导（Q_zz 对 Q_xx），公式与读数对照
  A3 缺陷源对照：六链接（八面体）源的 Q 各向同性（沿 z 与沿 x 的四极贡献相同），
     其「沿 z / 横向」比值近 1，443 读到的 0.25% 归离散

器械复用第 443 条：链接源 s（D 分量、N^3）；站点源 f = inc_link(s)/D；fft_solve 解；源外 u_e = theta_mid。
本脚本只做近场的解析化，不改第 443 条的判定。

运行：python 验证脚本/dumbbell_quadrupole.py
确定性，无随机数，运行约数秒。
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
IMG = os.path.join(ROOT, "images", "dumbbell_quadrupole.png")

D = 3
N = 64


def inc_link(u):
    out = np.zeros_like(u[0])
    for mu in range(D):
        out += u[mu] + np.roll(u[mu], 1, axis=mu)
    return out


def fft_solve(f):
    kx = 2 * np.pi * np.fft.fftfreq(N)
    KX, KY, KZ = np.meshgrid(kx, kx, kx, indexing="ij")
    denom = 1.0 - (np.cos(KX) + np.cos(KY) + np.cos(KZ)) / D
    denom.flat[0] = 1.0
    theta = np.real(np.fft.ifftn(np.fft.fftn(f) / denom))
    return theta - theta.mean()


def source_iso():
    # 每方向正负各一条：正链接在原点，负链接在 -e_mu 处（连 -e_mu 与原点）
    s = np.zeros((D, N, N, N))
    for mu in range(D):
        s[mu, 0, 0, 0] += 1.0
        neg = [0, 0, 0]
        neg[mu] = -1
        s[mu, neg[0], neg[1], neg[2]] += 1.0
    return s


def source_long():
    # 真哑铃：沿 z 轴的两条链接（原点与 -z 处），四站点在 z 轴上
    s = np.zeros((D, N, N, N))
    s[2, 0, 0, 0] += 3.0
    s[2, 0, 0, -1] += 3.0
    return s


def site_sources(s):
    f = inc_link(s) / D
    idx = np.nonzero(np.abs(f) > 1e-12)
    pts = np.stack([idx[0], idx[1], idx[2]], axis=1).astype(float)
    # 取最小镜像坐标（把 N-1 视作 -1）
    pts = np.where(pts > N / 2, pts - N, pts)
    w = f[idx]
    return pts, w


def cont_sum(pts, w, rvec):
    d = np.linalg.norm(rvec[None, :] - pts, axis=1)
    return float(np.sum(w / (4 * math.pi * d)))


def quad_tensor(pts, w):
    Q = np.zeros((3, 3))
    for p, wi in zip(pts, w):
        r2 = float(np.dot(p, p))
        Q += wi * (3.0 * np.outer(p, p) - r2 * np.eye(3))
    return Q


def grid_pt(rvec):
    # 映射到网格坐标（源在网格原点）
    return tuple(int(round(c)) % N for c in rvec)


print("== A1/A2/A3 站点源与四极 ==")
for tag, s in (("缺陷（八面体）", source_iso()), ("光（哑铃 z）", source_long())):
    pts, w = site_sources(s)
    Q = quad_tensor(pts, w)
    print("  %s：站点数 = %d，总源 = %.3f，Q = diag(%.3f, %.3f, %.3f)"
          % (tag, len(w), w.sum(), Q[0, 0], Q[1, 1], Q[2, 2]))
    for _p, _wv in zip(pts, w):
        print("    site (%.0f, %.0f, %.0f)  weight %.4f" % (_p[0], _p[1], _p[2], _wv))
    theta = fft_solve(inc_link(s) / D)
    print("  r    数值 theta(z)/theta(x)   连续求和比值   四极截断比值")
    for rr in (3, 4, 6, 8, 12):
        rz = np.array([0.0, 0.0, float(rr)])
        rx = np.array([float(rr), 0.0, 0.0])
        tz = theta[grid_pt(rz)]
        tx = theta[grid_pt(rx)]
        cz = cont_sum(pts, w, rz)
        cx = cont_sum(pts, w, rx)
        Qz = float(rz @ Q @ rz) / rr ** 2
        Qx = float(rx @ Q @ rx) / rr ** 2
        qz = (w.sum() / (4 * math.pi * rr)) + Qz / (2 * (4 * math.pi) * rr ** 3)
        qx = (w.sum() / (4 * math.pi * rr)) + Qx / (2 * (4 * math.pi) * rr ** 3)
        print("  %2d    %.5f                 %.5f         %.5f" % (rr, tz / tx, cz / cx, qz / qx))
print("  注：四极截断取 phi = Q0/(4 pi r) + Q_nn/(8 pi r^3)；常数项已用连续求和校验")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
ax = axes[0]
for tag, s, col in (("iso", source_iso(), "tab:blue"), ("dumbbell", source_long(), "tab:red")):
    theta = fft_solve(inc_link(s) / D)
    rr = np.arange(3, 13)
    ratio = [theta[grid_pt((0, 0, r))] / theta[grid_pt((r, 0, 0))] for r in rr]
    ax.plot(rr, ratio, "o-", color=col, label=tag)
ax.axhline(1.0, color="k", ls="--", lw=1)
ax.set_xlabel("r"); ax.set_ylabel("theta(z)/theta(x)")
ax.set_title("近场各向异性（数值）")
ax.legend(fontsize=8); ax.grid(alpha=0.3)
ax = axes[1]
th_iso = fft_solve(inc_link(source_iso()) / D)
th_dum = fft_solve(inc_link(source_long()) / D)
sl_iso = th_iso[N // 2, N // 2, N // 2 - 12:N // 2 + 12]
sl_dum = th_dum[N // 2, N // 2, N // 2 - 12:N // 2 + 12]
ax.plot(sl_iso, label="iso")
ax.plot(sl_dum, label="dumbbell")
ax.set_title("沿 z 剖面（theta）")
ax.legend(fontsize=8); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(IMG, dpi=160)
print("figure saved:", IMG)
