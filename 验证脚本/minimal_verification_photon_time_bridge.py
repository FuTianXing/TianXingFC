# -*- coding: utf-8 -*-
"""光包含时进桥：平流守恒、瞬时语义与无记忆（第 454 条，原拟 453 与并行线撞号顺延；443 移交"含时源"）。

判据：
  U1 含时离散守恒：光包以 c 沿 khat 平流，检验 d_t u + div S = 0。
     U1a 谱口径（dt -> 0）：相对残差 < 1e-9
     U1b 中心差分口径：注册朴素差分失真，与格点色散预期 1 - sinc(2k/sqrt3) 对照
  U2 含时器械语义：逐帧瞬时解（复用 443 器械 s_e -> f = inc_link(s_e)/D -> FFT 泊松；
     u_e = interp(theta) + s_e；lambda = u_e/theta_mid）：
     U2a 源区代数恒等 (lambda - 1) - s_e/theta_mid 机器零
     U2b 源外（|s_e| < 1e-3 max 且 |theta_mid| > 1）|lambda-1| 的 99 分位 < 5e-3
  U3 无记忆与脉冲结构：固定观测区（原点 8 格内链接）的读数随包通过呈单峰，
     且包远离时读数回基线（< 1e-6）

器械同第 443 条（N 提到 96 以容纳包宽 10）；确定性，运行约十秒。
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
IMG = os.path.join(ROOT, "images", "photon_time_bridge.png")

D = 3
N = 96
a = 10.0
kk = 1.0
khat = np.array([1.0, 1.0, 1.0]) / math.sqrt(3.0)

idx = np.arange(N) - N // 2
X, Y, Z = np.meshgrid(idx, idx, idx, indexing="ij")


def inc_link(u):
    out = np.zeros_like(u[0])
    for mu in range(D):
        out += u[mu] + np.roll(u[mu], 1, axis=mu)
    return out


def interp_site(theta):
    return np.stack([(theta + np.roll(theta, -1, axis=mu)) / 2.0 for mu in range(D)])


def fft_solve(f):
    kx = 2 * np.pi * np.fft.fftfreq(N)
    KX, KY, KZ = np.meshgrid(kx, kx, kx, indexing="ij")
    denom = 1.0 - (np.cos(KX) + np.cos(KY) + np.cos(KZ)) / D
    denom.flat[0] = 1.0
    theta = np.real(np.fft.ifftn(np.fft.fftn(f) / denom))
    return theta - theta.mean()


def solve(s_links):
    f = inc_link(s_links) / D
    theta = fft_solve(f)
    u = interp_site(theta) + s_links
    return theta, u


def packet(dvec):
    rx = X - dvec[0]
    ry = Y - dvec[1]
    rz = Z - dvec[2]
    env = np.exp(-(rx ** 2 + ry ** 2 + rz ** 2) / (2 * a * a))
    ph = kk * (rx + ry + rz) / math.sqrt(3.0)
    u = np.sin(ph) ** 2 * env ** 2
    S = np.stack([u * khat[mu] for mu in range(D)])
    return u, S


def spec_div(S):
    kx = 2 * np.pi * np.fft.fftfreq(N)
    KX, KY, KZ = np.meshgrid(kx, kx, kx, indexing="ij")
    K = [KX, KY, KZ]
    out = np.zeros((N, N, N))
    for mu in range(D):
        out += np.real(np.fft.ifftn(1j * K[mu] * np.fft.fftn(S[mu])))
    return out


def cd_div(S):
    out = np.zeros((N, N, N))
    for mu in range(D):
        out += (np.roll(S[mu], -1, axis=mu) - np.roll(S[mu], 1, axis=mu)) / 2.0
    return out


# 固定观测区：原点的链接中点（站点）8 格内的链接（用环面最近像距离）
TIE = (idx[:, None, None], idx[None, :, None], idx[None, None, :])
ZONE = np.zeros((D, N, N, N), dtype=bool)
for mu in range(D):
    cx, cy, cz = np.meshgrid(idx, idx, idx, indexing="ij")
    if mu == 0:
        cx = cx + 0.5
    elif mu == 1:
        cy = cy + 0.5
    else:
        cz = cz + 0.5
    d2 = cx ** 2 + cy ** 2 + cz ** 2
    ZONE[mu] = d2 <= 8.0 ** 2

print("== U1 含时离散守恒（光包沿 khat 平流，c = 1 格/单位时间） ==")
u0, S0 = packet((0.0, 0.0, 0.0))
dt = 1e-5
up, _ = packet(tuple(dt * khat))
um, _ = packet(tuple(-dt * khat))
dtu = (up - um) / (2 * dt)
Rs = dtu + spec_div(S0)
Rc = dtu + cd_div(S0)
scale = float(np.max(np.abs(dtu)))
m_per = 2.0 * kk / math.sqrt(3.0)
exp_cd = 1.0 - math.sin(m_per) / m_per
print("  谱口径：max|R|/max|d_t u| = %.3e（期望 < 1e-9）" % (float(np.max(np.abs(Rs))) / scale))
print("  中心差分口径：max|R|/max|d_t u| = %.3f（格点色散预期 %.3f）"
      % (float(np.max(np.abs(Rc))) / scale, exp_cd))

print()
print("== U2 含时器械语义（25 帧，包从 -48 移到 +48 格） ==")
shifts = np.arange(-48, 49, 4)
out_p99 = []
alg_err = 0.0
zone_max = []
for sh in shifts:
    dvec = sh * khat
    u_t, _ = packet(tuple(dvec))
    s_links = interp_site(u_t)
    theta, ue = solve(s_links)
    mid = interp_site(theta)
    lam = ue / mid
    dev = np.abs(lam - 1.0)
    thr = 1e-3 * float(np.max(np.abs(s_links)))
    mask = (np.abs(s_links) < thr) & (np.abs(mid) > 1.0)
    vals = dev[mask]
    out_p99.append(float(np.percentile(vals, 99.0)))
    ae = float(np.max(np.abs((lam - 1.0) - s_links / mid)))
    alg_err = max(alg_err, ae)
    zone_max.append(float(np.max(dev[ZONE])))
print("  U2a 代数恒等：max|(lambda-1) - s_e/theta_mid| = %.3e（期望机器零）" % alg_err)
print("  U2b 源外 |lambda-1| 的 99 分位：最大 %.3e（期望 < 5e-3）" % max(out_p99))
print("  U3 固定观测区（原点 8 格）max|lambda-1|：通过帧 %.3e；首帧 %.3e、末帧 %.3e"
      % (max(zone_max), zone_max[0], zone_max[-1]))
idx_mid = int(np.argmax(zone_max))
print("  峰值出现在 shift = %d（期望 |shift| <= 12 的通过区）" % shifts[idx_mid])

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
ax = axes[0]
ax.semilogy(shifts, zone_max, "o-", ms=3)
ax.axvspan(-12, 12, color="0.85", alpha=0.5, label="通过区")
ax.set_xlabel("包心位移（格）"); ax.set_ylabel("原点 8 格内 max|lambda-1|")
ax.set_title("含时源读数波形（固定观测区）"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
ax = axes[1]
ax.semilogy(shifts, out_p99, "o-", ms=3, label="源外 99 分位")
ax.axhline(5e-3, color="k", ls="--", lw=1, label="判据阈 5e-3")
ax.set_xlabel("包心位移（格）"); ax.set_title("源外恒等（全过程）"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(IMG, dpi=160)
print("figure saved:", IMG)
