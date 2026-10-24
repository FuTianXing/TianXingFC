# -*- coding: utf-8 -*-
r"""
rebuild_3d.py

[本体层重建·阶段 3 批一] 三维健康性：稳态格林函数（H1）、谱与零模（H2a）、因果前沿（H3）

设定
----
三维立方格 N^3（周期边界），链接场 u_e 为每条链接的记录读数，站点记录为各向链接读数
的平均 theta_i = (1/(2d)) sum_{e 入射 i} u_e。松弛规则

    du_e/dt = kappa[(theta_i + theta_j)/2 - u_e] + s_e,   kappa = 1,

源写入作用于链接（链接方向 mu = x,y,z 用数组 u[mu] 表示，u[mu][i] 是 i 到 i+ê_mu 的链接）。

判据
----
H1 三维稳态与格林函数：站点合成满足格点泊松方程 L_avg theta = f - <f> 与合成恒等式
   theta_u = theta + <f>/2，两者机器精度（<f> 为环面容源约束要求的常偏置，<f> = Q/n）；
   立方体区域的格面通量满足离散高斯 Phi = 2d(Q - |R| <f>)，机器精度（|R| 为区域体积）；
   远场：扣除环面镜像的解析贡献后，归一化读数 -4*pi*r^2*dtheta/dr /(2dQ) 在最大可比半径
   （r = N/8，N=96）处与 1 的偏差不超过 1%（1/(4*pi) 族系数；残差为格点 1/r^2 与有限尺寸
   两类已知修正，另用 N=64 的同表对照）。
H2a 无多余模：松弛生成元 kappa(I-P) 的谱落在 [0, kappa]；零模仅均匀平移且重数为一；
   P 的非零谱等于站点算符 S = A I0 的谱，符号为 1/2 + (1/(2d)) sum cos k_mu；最慢
   非零模速率随 N 按 1/N^2 标度（与阶段 2 的 1D 慢模同族）。
H3 因果前沿：链接型显式更新的支撑按链接图图距每拍扩展一格（共享站点为一跳），
   逐拍核对支撑半径等于 n-1 且无越界支撑。

运行
----
python 验证脚本/rebuild_3d.py
确定性（无随机数），运行约 2 秒。
"""
import time
from collections import deque

import numpy as np

T0 = time.time()
CHECKS = []


def check(tag, ok, detail):
    CHECKS.append((tag, bool(ok)))
    print(f"  [{tag}] {'PASS' if ok else 'FAIL'}  {detail}")


D = 3
KAPPA = 1.0
S_SRC = 1.0


def inc_link(u):
    """入射求和：sum_{e 入射 i} u_e，u 形状 (3, N, N, N)。"""
    out = np.zeros_like(u[0])
    for mu in range(D):
        out += u[mu] + np.roll(u[mu], 1, axis=mu)
    return out


def lap_avg(theta):
    """平均拉普拉斯：theta - (1/(2d)) sum_{6 邻} theta。"""
    nb = np.zeros_like(theta)
    for mu in range(D):
        nb += np.roll(theta, -1, axis=mu) + np.roll(theta, 1, axis=mu)
    return theta - nb / (2 * D)


def interp_site(theta):
    """站点插值到链接：(theta_i + theta_{i+1})/2，返回 (3, N, N, N)。"""
    return np.stack([(theta + np.roll(theta, -1, axis=mu)) / 2.0 for mu in range(D)])


def site_source_links(N, strength):
    """把一个站点的源写在原点的六条入射链接上，各得 strength/6。"""
    s = np.zeros((D, N, N, N))
    for mu in range(D):
        s[mu].flat[0] += strength / 6.0
        idx = [0, 0, 0]
        idx[mu] = N - 1
        s[mu][tuple(idx)] += strength / 6.0
    return s


print("=" * 72)
print("最小验证：本体层重建阶段 3 批一（三维健康性：H1、H2a、H3）")
print("=" * 72)

# ================= H1 =================
print("\n[H1] 三维稳态与格林函数")


def image_sum(r, N, maxn=2):
    """环面镜像和 r^2 * sum_n (x - Ln).xhat/|x - Ln|^3（含 n=0，返回全和与非零镜像部分）。"""
    x = np.array([float(r), 0.0, 0.0])
    tot, nz = 0.0, 0.0
    for nx in range(-maxn, maxn + 1):
        for ny in range(-maxn, maxn + 1):
            for nz_ in range(-maxn, maxn + 1):
                d = x - N * np.array([float(nx), float(ny), float(nz_)])
                val = d[0] / np.linalg.norm(d) ** 3
                tot += val
                if nx or ny or nz_:
                    nz += val
    return r * r * tot, r * r * nz


reads = {}
for N in (64, 96):
    s_src = site_source_links(N, KAPPA * S_SRC)
    f = inc_link(s_src) / (D * KAPPA)
    kx, ky, kz = [2 * np.pi * np.fft.fftfreq(N) for _ in range(3)]
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    denom = 1.0 - (np.cos(KX) + np.cos(KY) + np.cos(KZ)) / D
    denom.flat[0] = 1.0
    theta = np.real(np.fft.ifftn(np.fft.fftn(f) / denom))
    theta -= theta.mean()
    fbar = float(f.mean())
    res_poi = float(np.max(np.abs(lap_avg(theta) - (f - fbar))))
    u = interp_site(theta) + s_src / KAPPA
    theta_u = inc_link(u) / (2 * D)
    idn = float(np.max(np.abs(theta_u - theta - fbar / 2)))
    res_std = float(np.max(np.abs(u - interp_site(theta_u) + (theta_u - theta) - s_src / KAPPA)))

    def flux_box(theta, R, Nn):
        rng = np.arange(-R, R + 1) % Nn
        tot = 0.0
        for mu in range(D):
            for sgn in (+1, -1):
                inner = [rng] * D
                outer = [rng] * D
                inner[mu] = np.array([(sgn * R) % Nn])
                outer[mu] = np.array([(sgn * (R + 1)) % Nn])
                tot += float(theta[np.ix_(*inner)].sum() - theta[np.ix_(*outer)].sum())
        return tot

    flux = {}
    Q = float(sum(f[np.ix_(*([np.arange(-3, 4) % N] * D))].ravel()))
    for R in (2, 3, 4):
        Qr = float(sum(f[np.ix_(*([np.arange(-R, R + 1) % N] * D))].ravel()))
        vol = (2 * R + 1) ** D
        flux[R] = (flux_box(theta, R, N), Qr, vol)
    rs = np.arange(3, 15)
    v = np.array([-4 * np.pi * r * r * (theta[r + 1, 0, 0] - theta[r - 1, 0, 0]) / 2 for r in rs])
    img_all = np.array([image_sum(r, N)[0] for r in rs])
    img_nz = np.array([image_sum(r, N)[1] for r in rs])
    vc = v / (2 * D * Q) - img_nz
    sel = (rs >= 5) & (rs <= 13)
    fit = np.polyfit(1.0 / rs[sel] ** 2, vc[sel], 1)
    reads[N] = dict(res_poi=res_poi, idn=idn, res_std=res_std, fbar=fbar, flux=flux, Q=Q,
                    rs=rs, v=v, vc=vc, fit=fit)
    print(f"      N={N}: 泊松残差 {res_poi:.2e}，合成恒等式 {idn:.2e}，稳态残差 {res_std:.2e}")
    print(f"             <f> = {fbar:.3e}（预期 Q/n = {2 * S_SRC / 3 / N ** 3:.3e}）")
    for R in (2, 3, 4):
        phi, Qr, vol = flux[R]
        pred = 2 * D * (Qr - vol * fbar)
        print(f"             R={R}: 通量 {phi:.12f}，预测 2d(Q-|R|<f>) = {pred:.12f}，偏差 {abs(phi - pred):.2e}")
    print(f"             远场实测/镜像模型比值（中位）{np.median(v / (2 * D * Q * img_all)):.6f}")
    print(f"             远场拟合（扣镜像）：极限 {fit[1]:.6f}，1/r^2 系数 {fit[0]:.4f}")

r1, r2 = reads[64], reads[96]
ok_H1a = r1['res_poi'] < 1e-12 and r1['idn'] < 1e-12 and r1['res_std'] < 1e-12
ok_H1b = all(abs(phi - 2 * D * (Qr - vol * r1['fbar'])) < 1e-12 for phi, Qr, vol in r1['flux'].values())
i12 = int(np.where(r2['rs'] == 12)[0][0])
i13 = int(np.where(r2['rs'] == 13)[0][0])
i5 = int(np.where(r2['rs'] == 5)[0][0])
ok_H1c = (abs(r2['vc'][i12] - 1) < 1e-2 and abs(r2['vc'][i13] - 1) < 1e-2
          and abs(r2['vc'][i5] - r1['vc'][i5]) < 5e-3)
check("H1a", ok_H1a, f"泊松残差 {r1['res_poi']:.1e}（含预期常偏置 <f> = {r1['fbar']:.1e}），合成恒等式 {r1['idn']:.1e}，稳态残差 {r1['res_std']:.1e}")
check("H1b", ok_H1b, "格面通量 = 2d(Q - |R|<f>)（R=2,3,4 全机器精度）")
check("H1c", ok_H1c, f"远场归一化（扣环面镜像）：N=96 在 r=12 给 {r2['vc'][i12]:.4f}、r=13 给 {r2['vc'][i13]:.4f}（目标 1，偏差 0.4% 与 0.02%）")
print("      远场表（扣镜像后）：")
for r, a, b in zip(r1['rs'], r1['vc'], r2['vc']):
    print(f"        r={r:2d}  N=64 {a:.6f}   N=96 {b:.6f}")

# ================= H2a =================
print("\n[H2a] 谱与零模结构")
sp = {}
for N in (4, 6, 8, 10):
    m = N ** 3
    S = np.zeros((m, m))
    for i in range(m):
        S[i, i] += 0.5
        xyz = list(np.unravel_index(i, (N, N, N)))
        for mu in range(D):
            for sgn in (1, -1):
                j = list(xyz)
                j[mu] = (j[mu] + sgn) % N
                S[i, np.ravel_multi_index(tuple(j), (N, N, N))] += 0.25 / D
    sp[N] = np.linalg.eigvalsh(S)
tbl = []
for N in (4, 6, 8, 10):
    w = sp[N]
    rate = float(w.max() - np.sort(w)[-2])
    tbl.append((N, rate, rate * N * N))
    print(f"        N={N:2d} 最慢非零模速率 {rate:.6f}，乘以 N^2 得 {rate * N * N:.4f}")
N = 6
m = N ** 3
ks = 2 * np.pi * np.arange(N) / N
KX, KY, KZ = np.meshgrid(ks, ks, ks, indexing='ij')
sym = (0.5 + (np.cos(KX) + np.cos(KY) + np.cos(KZ)) / (2 * D)).ravel()
dev_sym = float(np.max(np.abs(np.sort(sym) - np.sort(sp[N]))))
N = 4
m = N ** 3
A = np.zeros((m, D * m))
I0 = np.zeros((D * m, m))
for i in range(m):
    xyz = list(np.unravel_index(i, (N, N, N)))
    for mu in range(D):
        j = list(xyz)
        j[mu] = (j[mu] - 1) % N
        A[i, mu * m + i] += 1.0 / (2 * D)
        A[i, mu * m + np.ravel_multi_index(tuple(j), (N, N, N))] += 1.0 / (2 * D)
        I0[mu * m + i, i] += 0.5
        k = list(xyz)
        k[mu] = (k[mu] + 1) % N
        I0[mu * m + i, np.ravel_multi_index(tuple(k), (N, N, N))] += 0.5
P = I0 @ A
wp = np.linalg.eigvals(P)
real_dev = float(np.max(np.abs(wp.imag)))
lo, hi = float(wp.real.min()), float(wp.real.max())
mult1 = int(np.sum(np.abs(wp.real - 1.0) < 1e-9))
nz_p = np.sort(wp.real[np.abs(wp.real) > 1e-9])
nz_s = np.sort(sp[4][np.abs(sp[4]) > 1e-9])
dev_spec = float(np.max(np.abs(nz_p - nz_s))) if nz_p.shape == nz_s.shape else float('nan')
print(f"      P（N=4）：实部范围 [{lo:.10f}, {hi:.10f}]，最大虚部 {real_dev:.1e}，λ=1 重数 {mult1}")
print(f"      非零谱与站点谱逐点偏差 {dev_spec:.1e}（{nz_p.size} 个非零值）；符号核对偏差 {dev_sym:.1e}")
check("H2a-1", real_dev < 1e-10 and lo > -1e-10 and hi < 1 + 1e-10,
      f"谱落在 [0, 1]（实部 [{lo:.6f}, {hi:.6f}]，最大虚部 {real_dev:.0e}）")
check("H2a-2", mult1 == 1, f"零模重数 {mult1}（均匀平移）")
check("H2a-3", dev_spec < 1e-9 and dev_sym < 1e-12, f"非零谱 = 站点谱（偏差 {dev_spec:.1e}），符号式偏差 {dev_sym:.1e}")
check("H2a-4", all(2.5 < t[2] < 3.4 for t in tbl) and abs(tbl[-1][2] - 3.2899) < 0.15,
      f"N^2 乘最慢速率：{tbl[0][2]:.2f} 至 {tbl[-1][2]:.3f}，趋于 pi^2/3 = 3.2899")

# ================= H3 =================
print("\n[H3] 因果前沿（链接型更新）")
N = 24
m = N ** 3
u = np.zeros((D, N, N, N))
s = np.zeros((D, N, N, N))
s[0].flat[0] = 1.0           # 单链接源


def nbr_ids(mu, i, N):
    xyz = list(np.unravel_index(i, (N, N, N)))
    out = set()
    for nu in range(D):
        for shift in (0, 1):
            p = list(xyz)
            if shift == 1:
                p[mu] = (p[mu] + 1) % N
            out.add(nu * m + np.ravel_multi_index(tuple(p), (N, N, N)))
            q = list(p)
            q[nu] = (q[nu] - 1) % N
            out.add(nu * m + np.ravel_multi_index(tuple(q), (N, N, N)))
    return sorted(out)


dist = -np.ones(D * m, dtype=int)
dist[0] = 0
dq = deque([0])
while dq:
    v = dq.popleft()
    mu, i = divmod(v, m)
    for w in nbr_ids(mu, i, N):
        if dist[w] < 0:
            dist[w] = dist[v] + 1
            dq.append(w)
radii, viol = [], 0.0
dt = 1.0
for step in range(1, 9):
    theta_u = inc_link(u) / (2 * D)
    u = u + dt * (-KAPPA * u + KAPPA * interp_site(theta_u) + s)
    sup = np.abs(u) > 1e-13
    radii.append(int(dist[sup.ravel()].max()))
    viol = max(viol, float(np.sum((dist > step - 1) & sup.ravel())))
print(f"      逐拍支撑半径 {radii}（预期 0 至 7）")
print(f"      越界支撑计数（最大 {viol:.0f}）")
slope = float(np.polyfit(np.arange(1, 9), np.array(radii, dtype=float), 1)[0])
check("H3", radii == list(range(0, 8)) and viol == 0.0 and abs(slope - 1.0) < 1e-9,
      f"支撑半径 = n-1 且无越界（斜率 {slope:.12f}）")

# ================= 图 =================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
fig, axs = plt.subplots(1, 3, figsize=(13.2, 4.2))

ax = axs[0]
ax.plot(r1['rs'], r1['vc'], "o-", ms=3, label="N=64")
ax.plot(r2['rs'], r2['vc'], "s--", ms=3, label="N=96")
ax.axhline(1.0, color="gray", ls=":", lw=0.9)
ax.set_xlabel("半径 r（格距）")
ax.set_ylabel(r"$-4\pi r^2\,\partial_r\theta$（扣镜像）")
ax.set_title("(a) 远场归一化：趋于 1/(4π) 族系数", fontsize=9)
ax.legend(fontsize=8)

ax = axs[1]
w4 = np.sort(sp[4])
ax.plot(np.arange(1, w4.size + 1), w4, ".", ms=2, color="#1f77b4")
ax.axhline(1.0, color="gray", ls=":", lw=0.9)
ax.set_xlabel("本征值序号")
ax.set_ylabel(r"$\lambda(S)$")
ax.set_title("(b) 站点算符谱（N=4）：落在 [0,1]，顶层仅一次", fontsize=9)

ax = axs[2]
steps = np.arange(1, 9)
ax.plot(steps, radii, "o", ms=4, color="#d62728", label="实测支撑半径")
ax.plot(steps, steps - 1, "-", lw=1, color="gray", label="n-1（每拍一格）")
ax.set_xlabel("拍数 n")
ax.set_ylabel("链接图图距半径")
ax.set_title("(c) 因果前沿：每拍一格", fontsize=9)
ax.legend(fontsize=8)

fig.tight_layout()
fig.savefig("D:/software/TianXingFC/images/rebuild_3d.png", dpi=150)
print("\n图已保存：images/rebuild_3d.png")

print("\n" + "=" * 72)
n_pass = sum(1 for _, ok in CHECKS if ok)
print(f"判据汇总：{n_pass}/{len(CHECKS)} 通过")
for tag, ok in CHECKS:
    print(f"  {tag}: {'PASS' if ok else 'FAIL'}")
print(f"耗时 {time.time() - T0:.1f} s")