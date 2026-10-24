# -*- coding: utf-8 -*-
"""光源进桥：纯纵源与等效源的空间度量侧对照（第 443 条，P1-24 第五轮；436/437/439 移交其一）。

判据（先立判据，再实现）：
  B1 源外结构恒等与非敏感性：三种源型（缺陷各向同性、光纯纵、等效质量）下，
     源外同点比值 lambda = u_e / theta_mid(e) 均为 1（机器精度）——「同源同格林函数」与源种类无关
  B2 远场单极一致：光（纯纵）与等效（各向同性）的远场 1/r 系数一致（同总源强约定下偏差小）
  B3 近场各向异性登记：纯纵源的近场沿传播方向与横向的差异（结构读数），各向同性源为对照

器械（复用本体层重建阶段 4）：周期立方格；链接场 u_e；站点记录 theta_i 为入射链接平均；
  稳态 u_e - (theta_i + theta_j)/2 = s_e/kappa；泊松解用 FFT（离散核 1 - (cosKx+cosKy+cosKz)/D）。
  本脚本取 kappa_e = 1（c = 0），只比对源型差异；反馈稳健性已由阶段 4 L2 登记。

源型与强度约定（登记）：缺陷源写在原点六条入射链接各 1（总 6）；光（纯纵）写在沿 z 的
  两条链接各 3（总 6，同总源强）；等效质量源与缺陷源相同（它就是「等效」的参照）。

运行：python 验证脚本/photon_bridge_lambda.py
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
IMG = os.path.join(ROOT, "images", "photon_bridge_lambda.png")

D = 3
N = 64


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


def solve_feedback(s_links, c, iters=60, tol=1e-14):
    """反馈档：kappa_e = 1 + c u_e 的定点迭代（阶段 4 L2 的同形式）。"""
    kap = np.ones((D, N, N, N))
    theta = None
    nu = 0
    for it in range(iters):
        f = inc_link(s_links / kap) / D
        theta = fft_solve(f)
        u = interp_site(theta) + s_links / kap
        kap_new = 1.0 + c * u
        dk = float(np.max(np.abs(kap_new - kap)))
        kap = kap_new
        nu = it + 1
        if dk < tol:
            break
    u = interp_site(theta) + s_links / kap
    return theta, u, nu, dk


def source_iso():
    s = np.zeros((D, N, N, N))
    for mu in range(D):
        s[mu, 0, 0, 0] += 1.0
        s[mu, -1, 0, 0] += 1.0  # 反向入射链接由 roll 语义承担；此处仅取对称强度
    return s


def source_long(z_only=True):
    s = np.zeros((D, N, N, N))
    s[2, 0, 0, 0] += 3.0
    s[2, -1, 0, 0] += 3.0
    return s


def lam_stats(theta, u, rmin=6):
    # 源外同点比值：u_e / theta_mid(e)；源在网格原点，掩码用环面距离（到原点）
    tm = interp_site(theta)
    idx = np.arange(N)
    d = np.minimum(idx, N - idx).astype(float)
    DIST2 = d[:, None, None] ** 2 + d[None, :, None] ** 2 + d[None, None, :] ** 2
    mask_site = DIST2 >= rmin ** 2
    vals = []
    for mu in range(D):
        mm = mask_site & (np.abs(tm[mu]) > 1e-12)
        vals.append((u[mu][mm] / tm[mu][mm]))
    v = np.concatenate(vals)
    return float(np.median(v)), float(np.max(np.abs(v - 1.0)))


print("== B1 源外结构恒等（三源型） ==")
sA = source_iso(); sB = source_long()
thA, uA = solve(sA)
thB, uB = solve(sB)
mA, eA = lam_stats(thA, uA)
mB, eB = lam_stats(thB, uB)
print("  缺陷（各向同性）：lambda 中位 = %.15f；max|lambda-1| = %.3e" % (mA, eA))
print("  光（纯纵）      ：lambda 中位 = %.15f；max|lambda-1| = %.3e" % (mB, eB))
print("  期望：两者均为 1（源外稳态恒等 u_e = theta_mid(e)，机器精度）")

print()
print("== B2 远场单极一致（沿 z 剖面） ==")
idx = np.arange(N) - N // 2
r_axis = idx[N // 2:] if False else None
# 取 +z 轴上剖面
zc = N // 2
profA = thA[zc, zc, N // 2:]
profB = thB[zc, zc, N // 2:]
rr = np.arange(N // 2, N, 1.0) - N // 2 + 1.0
rr = np.arange(len(profA), dtype=float) + 1.0
sel = (rr >= 8) & (rr <= 24)
for (lo, hi) in ((8, 16), (12, 20)):
    sel = (rr >= lo) & (rr <= hi)
    fa = np.polyfit(1.0 / rr[sel], profA[sel], 1)[0]
    fb = np.polyfit(1.0 / rr[sel], profB[sel], 1)[0]
    print("  窗 [%2d, %2d]：缺陷系数 = %.6e；光纯纵 = %.6e；比值 = %.5f" % (lo, hi, fa, fb, fb / fa))
print("  期望：同总源强下单极一致，比值随窗右移趋于 1（多极/离散项衰减）")

print()
print("== B4 反馈档（c = 0.1，定点迭代） ==")
for tag, sl in (("缺陷", sA), ("光纯纵", sB)):
    th, uu, nu, dk = solve_feedback(sl, 0.1)
    mk, ek = lam_stats(th, uu)
    peak = float(np.max(np.abs(th)))
    pk0 = float(np.max(np.abs(thA if tag == "缺陷" else thB)))
    print("  %s：迭代 %d 步（末步残差 %.2e）；lambda 中位 = %.15f；max|lambda-1| = %.3e；峰值 %.4f（无反馈 %.4f）"
          % (tag, nu, dk, mk, ek, peak, pk0))
print("  期望：反馈在场（峰值变化非零）且源外比值仍逐点精确为 1（源外 u_e = theta_mid 与 kappa 无关）")

print()
print("== B3 近场各向异性登记（r = 4 处） ==")
def quad(theta, r):
    vals = {}
    vals["z"] = float(theta[N // 2, N // 2, N // 2 + r])
    vals["x"] = float(theta[N // 2 + r, N // 2, N // 2])
    vals["d"] = float(theta[N // 2 + r // 2, N // 2 + r // 2, N // 2 + r // 2])
    return vals
for tag, th in (("缺陷（各向同性）", thA), ("光（纯纵）", thB)):
    q = quad(th, 4)
    print("  %s：theta(z)/theta(x) = %.5f；theta(d)/theta(x) = %.5f" % (tag, q["z"] / q["x"], q["d"] / q["x"]))
print("  登记：纯纵源的近场沿传播方向与横向的差异即其应力结构的读数")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
ax = axes[0]
sl = thA[N // 2, :, :]
im = ax.imshow(sl, cmap="RdBu_r")
ax.set_title("缺陷源 theta 中心切面")
fig.colorbar(im, ax=ax, fraction=0.046)
ax = axes[1]
sl = thB[N // 2, :, :]
im = ax.imshow(sl, cmap="RdBu_r")
ax.set_title("光（纯纵）源 theta 中心切面")
fig.colorbar(im, ax=ax, fraction=0.046)
fig.tight_layout()
fig.savefig(IMG, dpi=160)
print("figure saved:", IMG)
