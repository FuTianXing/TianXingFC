# -*- coding: utf-8 -*-
"""广义协变性定理化（第 469 条）：坐标重标下的标量性、规范同型与离散比安基。

背景：路线二（[[内部草稿/材料-狭义相对论与广义相对论的公设定理化盘点：四公设加惯性的现状、级别与路线]]）。
[[爱因斯坦场方程的微观根基：从碰撞写入法线场到连续时空几何]] 第四步把"协变推广"作为构造步骤；
本轮把该步骤拆成前提与推论，在弱场线性层把可判定部分落成判据：

  CV1 记录场标量性与重标读数：theta 为标量；物理读数取推前标架分量，重标下不变（机器精度）；
      坐标分量直接当读数会漂移（对照量）；各向同性形式在剪切重标下破缺、在纯旋转下保持（对照量）；
      离散二阶差分的 Hessian 按步长二阶收敛到解析值。
  CV2 无穷小重标与规范同型：重标诱导变化按一阶拆成表述项（d xi + d xi）与输运项（Lie 导数的 h 部分），
      分解恒等式余项按幅度二阶（比值 4）；表述项的爱因斯坦张量为零（机器精度）。
  CV3 离散比安基：乘积格（每轴一个对易差分算子）上 D^mu G_mu nu 为零（机器精度），
      对照关系 D^mu R_mu nu 等于 (1/2) D_nu R；R 单独不无散（判别性对照）；二阶与四阶模板同验。
  CV4 让位链重标的同型核对（模型级）：序保持置换加边缘外扩为纯重标，缺陷间读数不变；
      新缺陷落位为物理事件，读数改变；两类操作在读出层可分离。

确定性；约数秒。图存 images/general_covariance_checks.png
"""
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG = os.path.join(ROOT, "images", "general_covariance_checks.png")
ETA = np.diag([-1.0, 1.0, 1.0, 1.0])
RNG = np.random.default_rng(469)
T0 = time.time()


def check(tag, ok, detail):
    print("  [%s] %s  %s" % (tag, "PASS" if ok else "FAIL", detail))
    return bool(ok)


def maxabs(a):
    return float(np.max(np.abs(a)))


def smooth_field(shape, kmax, rng):
    f = rng.standard_normal(shape)
    ks = [np.fft.fftfreq(n) * n for n in shape]
    k2 = np.zeros(shape)
    for ax, kk in enumerate(ks):
        sh = [1] * len(shape)
        sh[ax] = len(kk)
        k2 = k2 + (kk.reshape(sh) ** 2)
    mask = (k2 <= kmax * kmax).astype(float)
    return np.real(np.fft.ifftn(np.fft.fftn(f) * mask))


def D(f, axis, eps, order=4):
    if order == 2:
        return (np.roll(f, -1, axis=axis) - np.roll(f, 1, axis=axis)) / (2 * eps)
    return (-np.roll(f, -2, axis=axis) + 8 * np.roll(f, -1, axis=axis)
            - 8 * np.roll(f, 1, axis=axis) + np.roll(f, 2, axis=axis)) / (12 * eps)


def box4(f, eps, order=4):
    out = np.zeros_like(f)
    for a in range(4):
        for b in range(4):
            out = out + ETA[a, b] * D(D(f, a, eps, order), b, eps, order)
    return out


def einstein_lin(h, eps, order=4):
    gshape = h.shape[2:]
    A = np.zeros((4,) + gshape)
    for nu in range(4):
        acc = np.zeros(gshape)
        for mu in range(4):
            for a in range(4):
                acc = acc + ETA[mu, a] * D(h[mu, nu], a, eps, order)
        A[nu] = acc
    trh = np.zeros(gshape)
    for a in range(4):
        for b in range(4):
            trh = trh + ETA[a, b] * h[a, b]
    R = np.zeros_like(h)
    for mu in range(4):
        for nu in range(4):
            R[mu, nu] = 0.5 * (D(A[nu], mu, eps, order) + D(A[mu], nu, eps, order)
                               - box4(h[mu, nu], eps, order)
                               - D(D(trh, mu, eps, order), nu, eps, order))
    Rsc = -box4(trh, eps, order)
    for a in range(4):
        for b in range(4):
            Rsc = Rsc + ETA[a, b] * D(A[a], b, eps, order)
    G = R - 0.5 * Rsc[None, None] * ETA.reshape(4, 4, 1, 1, 1, 1)
    return G, R, Rsc


def divergence(T, eps, order=4):
    S = np.zeros((4,) + T.shape[2:])
    for nu in range(4):
        acc = np.zeros(T.shape[2:])
        for mu in range(4):
            for a in range(4):
                acc = acc + ETA[mu, a] * D(T[mu, nu], a, eps, order)
        S[nu] = acc
    return S


print("第 469 条 广义协变性定理化：CV1 至 CV4")

# ---------------------------------------------------------------- CV1
print("CV1 记录场标量性与重标读数（解析层，机器精度）")


def theta_hess(x):
    r2 = np.sum(x ** 2, axis=0)
    t = np.exp(-r2)
    H = (4.0 * np.einsum("in,jn->ijn", x, x) - 2.0 * np.eye(3)[:, :, None]) * t
    return t, H


ang = 0.7
Rrot = np.array([[np.cos(ang), -np.sin(ang), 0.0],
                 [np.sin(ang), np.cos(ang), 0.0],
                 [0.0, 0.0, 1.0]])
nl = 0.35
pts = RNG.uniform(-0.5, 0.5, size=(3, 24))
t_old, H_old = theta_hess(pts)
x0, x1, x2 = pts
qJ = np.stack([
    np.stack([x1, x0, np.zeros_like(x0)], axis=0),
    np.stack([np.zeros_like(x1), x2, x1], axis=0),
    np.stack([x2, np.zeros_like(x2), x0], axis=0)], axis=0)
J = Rrot[:, :, None] + nl * qJ
Minv = np.linalg.inv(np.transpose(J, (2, 0, 1)))
Ht = np.transpose(H_old, (2, 0, 1))
Jt = np.transpose(J, (2, 0, 1))
Eprime = np.einsum("nai,nbj,nab->nij", Minv, Minv, Ht)
read_new = np.einsum("nij,nik,njl->nkl", Eprime, Jt, Jt)
scale_read = maxabs(Ht)
read_dev = maxabs(read_new - Ht) / scale_read
comp_dev = maxabs(Eprime - Ht) / scale_read
check("CV1a", read_dev < 1e-12,
      "推前标架读数不变：相对偏差 %.2e（机器精度）" % read_dev)
check("CV1b", comp_dev > 1e-3,
      "坐标分量直读的漂移（对照量，非零即有判别力）：相对 %.2e" % comp_dev)

Erot = np.einsum("ia,jb,nab->nij", Rrot, Rrot, Ht)
read_rot = np.einsum("nij,ik,jl->nkl", Erot, Rrot, Rrot)
rot_read_dev = maxabs(read_rot - Ht) / scale_read
form_rot = maxabs(np.einsum("ai,aj->ij", Rrot, Rrot) - np.eye(3))
check("CV1c", rot_read_dev < 1e-12 and form_rot < 1e-12,
      "纯旋转重标：推前标架读数偏差 %.2e；各向同性形式偏差 %.2e" % (rot_read_dev, form_rot))

form_shear = maxabs(np.einsum("nai,naj->nij", Minv, Minv) - np.eye(3)[None])
check("CV1d", form_shear > 1e-2,
      "剪切重标下各向同性形式破缺（对照量）：最大破缺 %.2e" % form_shear)


def d2_center(n):
    eps = 2.0 / (n - 1)
    xs = np.linspace(-1.0, 1.0, n)
    X, Y, Z = np.meshgrid(xs, xs, xs, indexing="ij")
    P = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=0)
    tg, Hg = theta_hess(P)
    tg = tg.reshape(n, n, n)
    Hg = Hg.reshape(3, 3, n, n, n)
    c = n // 2
    err = 0.0
    for i in range(3):
        for j in range(3):
            dd = D(D(tg, i, eps, 2), j, eps, 2)
            err = max(err, abs(dd[c, c, c] - Hg[i, j, c, c, c]))
    return err


e1, e2 = d2_center(17), d2_center(33)
ratio_conv = e1 / e2
check("CV1e", 3.5 < ratio_conv < 4.5,
      "离散 Hessian 二阶收敛：e(h)=%.2e，e(h/2)=%.2e，比值 %.3f" % (e1, e2, ratio_conv))

# ---------------------------------------------------------------- CV2
print("CV2 无穷小重标与规范变换（格点层，机器精度）")
N, KM = 16, 2
LEN = 2.0 * np.pi
EPS = LEN / N
grid = (N, N, N, N)
h = np.zeros((4, 4) + grid)
for m in range(4):
    for n in range(m, 4):
        f = smooth_field(grid, KM, RNG)
        h[m, n] = f
        h[n, m] = f
xi = np.stack([smooth_field(grid, KM, RNG) for _ in range(4)])
xi_low = np.einsum("mn,n...->m...", ETA, xi)
s = np.zeros_like(h)
for mu in range(4):
    for nu in range(4):
        s[mu, nu] = D(xi_low[nu], mu, EPS, 4) + D(xi_low[mu], nu, EPS, 4)
G_h, R_h, Rsc_h = einstein_lin(h, EPS, 4)
G_s, _, _ = einstein_lin(s, EPS, 4)
rel_gauge = maxabs(G_s) / maxabs(G_h)
check("CV2a", rel_gauge < 1e-10,
      "表述项（d xi + d xi）的爱因斯坦张量：相对 %.2e（规范不变）" % rel_gauge)


def relabel_decomp(amp):
    rng = np.random.default_rng(4691)
    B = rng.uniform(-0.6, 0.6, size=(4, 4, 4))
    B = 0.5 * (B + np.transpose(B, (1, 0, 2)))
    Emat = rng.uniform(-0.5, 0.5, size=(4, 4)) * amp
    x = rng.uniform(-0.4, 0.4, size=(12, 4))
    xi_v = x @ Emat.T
    h_old = np.einsum("mnl,kl->kmn", B, x)
    h_new = np.einsum("mnl,kl->kmn", B, x + xi_v)
    Jp = np.eye(4) + Emat
    g_new = np.einsum("am,kab,bn->kmn", Jp, ETA[None] + h_new, Jp) - ETA[None] - h_old
    dxi_low = np.einsum("nl,lm->mn", ETA, Emat)
    gauge = dxi_low + dxi_low.T
    trans = (np.einsum("mnl,kl->kmn", B, xi_v)
             + np.einsum("am,kab->kmb", Emat, h_old)
             + np.einsum("kab,bn->kan", h_old, Emat))
    return maxabs(g_new - (gauge[None] + trans))


r_a1 = relabel_decomp(0.40)
r_a2 = relabel_decomp(0.20)
ratio_rel = r_a1 / r_a2
check("CV2b", 3.5 < ratio_rel < 4.5,
      "重标分解余项按幅度二阶：r(A)=%.3e，r(A/2)=%.3e，比值 %.3f" % (r_a1, r_a2, ratio_rel))

# ---------------------------------------------------------------- CV3
print("CV3 离散比安基（乘积格，机器精度）")
cv3 = {}
for order in (2, 4):
    g, r, rsc = einstein_lin(h, EPS, order)
    sg = divergence(g, EPS, order)
    sr = divergence(r, EPS, order)
    dscale = max([maxabs(D(g[mu, nu], a, EPS, order))
                  for mu in range(4) for nu in range(4) for a in range(4)])
    rel_g = maxabs(sg) / dscale
    ctrl = max([maxabs(sr[nu] - 0.5 * D(rsc, nu, EPS, order)) for nu in range(4)])
    rel_ctrl = ctrl / dscale
    rel_sr = maxabs(sr) / dscale
    cv3[order] = (rel_g, rel_ctrl, rel_sr)
    check("CV3-%d" % order,
          rel_g < 1e-10 and rel_ctrl < 1e-10 and rel_sr > 1e-3,
          "D.G 相对 %.2e；D.R-(1/2)dR 相对 %.2e；R 单独散度相对 %.2e（对照）"
          % (rel_g, rel_ctrl, rel_sr))

# ---------------------------------------------------------------- CV4
print("CV4 让位链重标的同型核对（模型级）")
def reading(ds):
    return np.diff(ds)


defects0 = np.array([0, 8])
defects_relabel = defects0 + 1
defects_birth = np.array([0, 1, 8])
defects_move = np.array([0, 9])
r0 = reading(defects0)
r1 = reading(defects_relabel)
r2 = reading(defects_birth)
r3 = reading(defects_move)
inv_ok = np.array_equal(defects_relabel - 1, defects0)
check("CV4a", np.array_equal(r0, r1) and inv_ok,
      "纯重标（序保持置换加边缘外扩）：读数 %s 不变；逆映射复原 %s" % ([int(v) for v in r0], inv_ok))
check("CV4b", (not np.array_equal(r0, r2)) and (not np.array_equal(r0, r3))
      and int(abs(r3[0] - r0[0])) == 1,
      "物理事件：落位读数 %s、位移读数 %s（原 %s），变化 1 格"
      % ([int(v) for v in r2], [int(v) for v in r3], [int(v) for v in r0]))

# ---------------------------------------------------------------- 图
fig, axs = plt.subplots(2, 2, figsize=(11, 8))
ax = axs[0, 0]
labels = ["标架读数\n（重标）", "分量直读\n（剪切重标）", "ansatz形式\n（剪切）", "ansatz形式\n（纯旋转）"]
vals = [max(read_dev, 1e-16), max(comp_dev, 1e-16), max(form_shear, 1e-16), max(form_rot, 1e-16)]
ax.bar(range(4), vals, color=["#2b6cb0", "#c53030", "#c53030", "#2b6cb0"])
ax.set_yscale("log")
ax.set_ylim(1e-16, 10)
ax.axhline(1e-12, color="gray", ls="--", lw=0.8)
ax.set_xticks(range(4))
ax.set_xticklabels(labels, fontsize=8)
ax.set_title("CV1 重标下读数与形式（相对量）", fontsize=10)

ax = axs[0, 1]
amps = np.array([0.4, 0.2, 0.1, 0.05])
ress = np.array([relabel_decomp(a) for a in amps])
ax.loglog(amps, ress, "o-", color="#2b6cb0")
ax.loglog(amps, ress[0] * (amps / amps[0]) ** 2, "--", color="gray", lw=0.9, label="斜率 2 参照")
ax.set_xlabel("重标幅度 A")
ax.set_ylabel("分解余项")
ax.set_title("CV2 重标分解余项按 A^2（规范残差 %.1e）" % rel_gauge, fontsize=10)
ax.legend(fontsize=8)

ax = axs[1, 0]
grp = np.array([[cv3[2][0], cv3[2][1], cv3[2][2]], [cv3[4][0], cv3[4][1], cv3[4][2]]])
w = 0.35
ax.bar(np.arange(3) - w / 2, np.maximum(grp[0], 1e-16), w, label="二阶模板", color="#2b6cb0")
ax.bar(np.arange(3) + w / 2, np.maximum(grp[1], 1e-16), w, label="四阶模板", color="#68a063")
ax.set_yscale("log")
ax.set_ylim(1e-16, 10)
ax.set_xticks(range(3))
ax.set_xticklabels(["D·G", "D·R-(1/2)dR", "D·R（对照）"], fontsize=8)
ax.axhline(1e-12, color="gray", ls="--", lw=0.8)
ax.set_title("CV3 离散比安基（相对量）", fontsize=10)
ax.legend(fontsize=8)

ax = axs[1, 1]
ax.axis("off")
for row, (ds, tag) in enumerate([(defects0, "原状：读数 [8]"),
                                 (defects_relabel, "纯重标：读数 [8]（不变）"),
                                 (defects_birth, "物理事件：读数 [1,7]")]):
    y = 0.78 - 0.27 * row
    for i in range(13):
        ax.add_patch(plt.Rectangle((i * 0.075, y), 0.07, 0.14, fill=False, lw=0.6, ec="#555"))
    for d in ds:
        ax.add_patch(plt.Rectangle((d * 0.075, y), 0.07, 0.14, fc="#2b6cb0", alpha=0.6))
    ax.text(0, y + 0.19, tag, fontsize=9)
ax.set_title("CV4 让位链：重标不变与物理变化分离", fontsize=10)

plt.tight_layout()
plt.savefig(IMG, dpi=150)
print("图已存：%s（用时 %.1f s）" % (IMG, time.time() - T0))
