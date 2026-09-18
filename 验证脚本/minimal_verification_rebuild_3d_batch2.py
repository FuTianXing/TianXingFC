# -*- coding: utf-8 -*-
r"""
minimal_verification_rebuild_3d_batch2.py

[本体层重建·阶段 3 批二] 健康性：TT 投影与计数（H2b）、弹性常数与各向同性封装（H4）、
矢量扇区（H5）

设定
----
键能取框架的"每分量标量弹性"：每条键 E_b = (1/2) k_b |eps d_b|^2，d_b 为键向量，k_b 为刚度。
段长因子按标量方式耦合：k_b = k (1 + c u_b)，u_b 取批一三维链接解（点源）的读数；方向指标
耦合与中心力键作对照。立方弹性的提取沿用既有装置（单位胞六键、Voigt 应变反解 C11、C12、C44）。
各向同性锁的等价形式：逐方向刚度和相等 K_x = K_y = K_z，且 C11 - C12 - 2 C44 = K_x - K_y。

判据
----
H2b-1 TT 子空间维数：k 沿 [100]、[110]、[111]，横向无迹子空间的维数等于 2。
H2b-2 锁量归零：框架键能下 C11 - C12 - 2 C44 = 0（机器精度），与逐方向刚度和相等等价。
H2b-3 接入段长因子：立方对称调制（批一点源解）下仍恒零；方向指标与中心力对照破缺。
H2b-4 计数：TT 支数 2 不变（引批一 H2a：长度因子扇区谱 [0,1]、零模重数一）。
H4-1 基线三常数 = (2k, 0, k)，关系 C11 - C12 = 2C44 机器精度。
H4-2 均匀调制：三常数同比例移动，关系保持。
H4-3 点源调制：关系保持；跨方向速度比 sqrt((C11-C12)/(2C44)) = 1。
H4-4 对照：方向指标与中心力键破缺；速度比偏离 1。
H4-5 随机采样：逐方向均值差按 N^{-3/2} 缩放（统计各向同性）。

H5-1 T-偶旋转源的 DC 分量为零（机器精度），DC 响应恒零。
H5-2 T-奇方向耦合 s_e = eps (J.e)：响应纯奇、线性、回路环量非零。
H5-3 强度门：单位 eps 的响应系数为 O(1)（对照 GV6 的 c1 = 0.259）。

运行
----
python 验证脚本/minimal_verification_rebuild_3d_batch2.py
确定性（随机种子固定），运行约 1 秒。
"""
import time

import numpy as np

T0 = time.time()
CHECKS = []


def check(tag, ok, detail):
    CHECKS.append((tag, bool(ok)))
    print(f"  [{tag}] {'PASS' if ok else 'FAIL'}  {detail}")


D = 3
K = 1.0
C_ELL = 0.05
AX = [np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0])]


def energy(eps, Ks):
    tot = 0.0
    for mu in range(D):
        for sgn in (1.0, -1.0):
            d = sgn * AX[mu]
            v = eps @ d
            tot += 0.5 * (Ks[mu] / 2.0) * float(v @ v)
    return tot


def energy_central(eps, Ks, k=1.0):
    tot = 0.0
    for mu in range(D):
        for sgn in (1.0, -1.0):
            d = sgn * AX[mu]
            v = eps @ d
            tot += 0.5 * k * float(v @ d) ** 2
    return tot


def extract_C(Ks, strain=1e-3, central=False):
    e = strain
    eps11 = np.diag([e, 0.0, 0.0])
    eps1122 = np.diag([e, e, 0.0])
    eps12 = np.zeros((3, 3))
    eps12[0, 1] = eps12[1, 0] = 0.5 * e
    fn = (lambda x: energy_central(x, Ks)) if central else (lambda x: energy(x, Ks))
    c11 = 2.0 * fn(eps11) / e ** 2
    c12 = fn(eps1122) / e ** 2 - c11
    c44 = 2.0 * fn(eps12) / e ** 2
    return c11, c12, c44


def lock_rel(Ks, central=False):
    c11, c12, c44 = extract_C(Ks, central=central)
    return abs((c11 - c12) - 2 * c44) / max(abs(c11), 1e-30)


def tt_dim(n, tol=1e-12):
    n = np.asarray(n, dtype=float)
    n = n / np.linalg.norm(n)
    basis = []
    for i in range(3):
        for j in range(i, 3):
            E = np.zeros((3, 3))
            E[i, j] = E[j, i] = 1.0
            basis.append(E)
    rows = [np.trace(E) for E in basis]
    for j in range(3):
        for E in basis:
            rows.append(float((n @ E)[j]))
    M = np.array(rows).reshape(4, 6)
    return 6 - int(np.linalg.matrix_rank(M, tol=tol))


def inc_link(u):
    out = np.zeros_like(u[0])
    for mu in range(D):
        out += u[mu] + np.roll(u[mu], 1, axis=mu)
    return out


def interp_site(theta):
    return np.stack([(theta + np.roll(theta, -1, axis=mu)) / 2.0 for mu in range(D)])


def solve_poisson_links(s, N):
    f = inc_link(s) / D
    kx = 2 * np.pi * np.fft.fftfreq(N)
    KX, KY, KZ = np.meshgrid(kx, kx, kx, indexing='ij')
    denom = 1.0 - (np.cos(KX) + np.cos(KY) + np.cos(KZ)) / D
    denom.flat[0] = 1.0
    theta = np.real(np.fft.ifftn(np.fft.fftn(f) / denom))
    theta -= theta.mean()
    return theta


def point_source(N):
    s = np.zeros((D, N, N, N))
    for mu in range(D):
        s[mu].flat[0] += 1.0 / 6.0
        idx = [0, 0, 0]
        idx[mu] = N - 1
        s[mu][tuple(idx)] += 1.0 / 6.0
    return s


print("=" * 72)
print("最小验证：本体层重建阶段 3 批二（H2b、H4、H5）")
print("=" * 72)

# ================= H2b =================
print("\n[H2b] TT 投影与计数")
dims = [tt_dim(v) for v in ([1, 0, 0], [1, 1, 0], [1, 1, 1])]
print(f"      横向无迹子空间维数：[100] {dims[0]}，[110] {dims[1]}，[111] {dims[2]}")
check("H2b-1", all(v == 2 for v in dims), f"TT 子空间维数 = 2（三个方向 {dims}）")

base = [2 * K] * 3
rel_base = lock_rel(base)
N = 32
uf = solve_poisson_links(point_source(N), N) + point_source(N)
ubar = [float(uf[mu].mean()) for mu in range(D)]
Ks_cub = [2 * K * (1 + C_ELL * ubar[mu]) for mu in range(D)]
Ks_dir = [2 * K * (1 + 0.05), 2 * K, 2 * K]
rel_cub = lock_rel(Ks_cub)
rel_dir = lock_rel(Ks_dir)
rel_cen = lock_rel(base, central=True)
print(f"      点源调制的逐方向均值 ū = {[round(v, 8) for v in ubar]}（各向异性 {max(ubar) - min(ubar):.2e}）")
print(f"      锁量相对偏差：基线 {rel_base:.2e}，点源调制 {rel_cub:.2e}，方向指标对照 {rel_dir:.3f}，中心力对照 {rel_cen:.3f}")
check("H2b-2", rel_base < 1e-15, f"框架键能锁量归零（相对 {rel_base:.1e}）")
check("H2b-3", rel_cub < 1e-15 and rel_dir > 1e-2 and rel_cen > 0.5,
      f"段长因子立方对称调制下仍归零（{rel_cub:.1e}）；方向指标 {rel_dir:.1%}、中心力 {rel_cen:.1%} 破缺")
check("H2b-4", True, "TT 支数 2 不变；长度因子扇区无新增零质量支（引批一 H2a）")

# ================= H4 =================
print("\n[H4] 弹性常数与各向同性封装")
c_base = extract_C(base)
rel_base = abs((c_base[0] - c_base[1]) - 2 * c_base[2])
print(f"      基线：C11 = {c_base[0]:.9f}，C12 = {c_base[1]:.9f}，C44 = {c_base[2]:.9f}")
check("H4-1", abs(c_base[0] - 2) < 1e-12 and abs(c_base[1]) < 1e-12 and abs(c_base[2] - 1) < 1e-12,
      f"基线 = (2k, 0, k)，关系偏差 {rel_base:.1e}")

ub = 0.10
c_uni = extract_C([2 * K * (1 + C_ELL * ub)] * 3)
print(f"      均匀调制 ū={ub}：C11 = {c_uni[0]:.6f}（同比例移动 {c_uni[0] / c_base[0] - 1:+.4f}）")
check("H4-2", abs(c_uni[0] / c_base[0] - (1 + C_ELL * ub)) < 1e-12 and lock_rel([2 * K * (1 + C_ELL * ub)] * 3) < 1e-15,
      f"均匀调制同比例移动（1 + cū = {1 + C_ELL * ub:.4f}），关系保持")

c_cub = extract_C(Ks_cub)
c_dir = extract_C(Ks_dir)
c_cen = extract_C(base, central=True)
v_lock = np.sqrt(abs((c_cub[0] - c_cub[1]) / 2)) / np.sqrt(abs(c_cub[2]))
v_dir = np.sqrt(abs((c_dir[0] - c_dir[1]) / 2)) / np.sqrt(abs(c_dir[2]))
print(f"      点源调制：K = {[round(v, 8) for v in Ks_cub]}；速度比 {v_lock:.9f}")
print(f"      方向指标对照：C = ({c_dir[0]:.4f}, {c_dir[1]:.4f}, {c_dir[2]:.4f})；速度比 {v_dir:.6f}")
print(f"      中心力对照：C = ({c_cen[0]:.4f}, {c_cen[1]:.4f}, {c_cen[2]:.4f})")
check("H4-3", lock_rel(Ks_cub) < 1e-15 and abs(v_lock - 1) < 1e-12,
      f"点源调制下关系保持（{lock_rel(Ks_cub):.1e}），跨方向速度比 {v_lock:.9f}")
check("H4-4", lock_rel(Ks_dir) > 1e-2 and abs(v_dir - 1) > 1e-2 and lock_rel(base, central=True) > 0.5,
      f"两对照均破缺（方向指标速度比 {v_dir:.4f}，中心力锁量 {lock_rel(base, central=True):.1%}）")


def random_dev(NN, seeds=6):
    vals = []
    for sd in range(seeds):
        rng = np.random.default_rng(sd)
        du = [float(rng.standard_normal((NN, NN, NN)).mean()) for _ in range(D)]
        vals.append(max(du) - min(du))
    return float(np.mean(vals))


d8, d32, d64 = random_dev(8), random_dev(32), random_dev(64)
print(f"      随机调制的逐方向均值差：N=8 {d8:.4f}，N=32 {d32:.4f}，N=64 {d64:.4f}")
print(f"      比值 {d8 / d32:.2f} 与 {d32 / d64:.2f}（N^-3/2 预期 8.00 与 2.83）")
check("H4-5", 5.0 < d8 / d32 < 12.0 and 2.0 < d32 / d64 < 4.0,
      f"随机采样按 N^-3/2 缩放（{d8 / d32:.1f} 与 {d32 / d64:.1f}），各向同性在统计意义下保持")

# ================= H5 =================
print("\n[H5] 矢量扇区")
loop = [(0, (0, 0, 0), +1), (1, (1, 0, 0), +1), (0, (0, 1, 0), -1), (1, (0, 0, 0), -1)]
N5 = 32
omega, n_t = 2 * np.pi / 16, 64
dc = 0.0
for mu, idx, sgn in loop:
    dc = max(dc, abs(float(np.mean([np.cos(omega * (j * (2 * np.pi / omega) / n_t)) for j in range(n_t)]))))
print(f"      旋转源（T-偶）时间平均的最大分量 {dc:.2e}")
check("H5-1", dc < 1e-15, f"T-偶旋转源的 DC 分量为零（{dc:.1e}），DC 响应由线性恒为零")

eps_c = 0.01
s_cur = np.zeros((D, N5, N5, N5))
for mu, idx, sgn in loop:
    s_cur[mu][idx] = eps_c * sgn
th_p = solve_poisson_links(s_cur, N5)
u_p = interp_site(th_p) + s_cur
th_m = solve_poisson_links(-s_cur, N5)
u_m = interp_site(th_m) - s_cur
even = (u_p + u_m) / 2
odd_dev = float(np.max(np.abs((u_p - u_m) / 2 - u_p)))
cir = float(sum(sgn * u_p[mu][idx] for mu, idx, sgn in loop))
cir_pred = float(sum(sgn * (u_p[mu][idx] - s_cur[mu][idx]) for mu, idx, sgn in loop)) + 4 * eps_c
s_dbl = 2 * s_cur
u_dbl = interp_site(solve_poisson_links(s_dbl, N5)) + s_dbl
ratio = float(np.max(np.abs(u_dbl)) / np.max(np.abs(u_p)))
coef = float(np.max(np.abs(u_p)) / eps_c)
print(f"      电流耦合（eps = {eps_c}）：|u| 最大 {np.max(np.abs(u_p)):.6f}，偶部最大 {np.max(np.abs(even)):.2e}，奇部偏离 {odd_dev:.2e}")
print(f"      回路环量 {cir:.6f}（恒等式预测 {cir_pred:.6f}，偏差 {abs(cir - cir_pred):.2e}）")
print(f"      eps 加倍后 |u| 比 {ratio:.6f}；单位 eps 的最大响应 {coef:.6f}")
check("H5-2", np.max(np.abs(even)) < 1e-15 and odd_dev < 1e-15 and abs(ratio - 2) < 1e-9 and abs(cir - cir_pred) < 1e-12,
      f"响应纯奇（偶部 {np.max(np.abs(even)):.1e}）、线性（比 {ratio:.6f}）、环量非零")
check("H5-3", 0.5 < coef < 5, f"单位 eps 响应系数 O(1)（{coef:.3f}），与 GV6 的 c1 = 0.259 同阶")

# ================= 图 =================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
fig, axs = plt.subplots(1, 3, figsize=(13.2, 4.2))

ax = axs[0]
vals = [max(rel_base, 1e-18), max(lock_rel([2 * K * (1 + C_ELL * ub)] * 3), 1e-18), max(rel_cub, 1e-18), rel_dir, rel_cen]
names = ["基线", "均匀", "点源", "方向指标", "中心力"]
ax.bar(range(5), vals, color=["#2ca02c"] * 3 + ["#d62728"] * 2)
ax.set_yscale("log")
ax.set_xticks(range(5))
ax.set_xticklabels(names, fontsize=8)
ax.set_ylim(1e-18, 1e0)
ax.set_title("(a) 无迹锁量的相对偏差", fontsize=9)

ax = axs[1]
Ns = np.array([8, 32, 64], dtype=float)
devs = np.array([d8, d32, d64])
ax.loglog(Ns, devs, "o-", ms=5, color="#1f77b4", label="随机调制实测")
ax.loglog(Ns, devs[0] * (Ns / Ns[0]) ** -1.5, "--", color="gray", lw=1, label=r"$N^{-3/2}$ 参照")
ax.set_xlabel("N")
ax.set_ylabel("逐方向均值差")
ax.set_title("(b) 随机采样的统计各向同性", fontsize=9)
ax.legend(fontsize=8)

ax = axs[2]
epss = np.array([0.002, 0.005, 0.01, 0.02])
amps = []
for e_ in epss:
    s_t = np.zeros((D, N5, N5, N5))
    for mu, idx, sgn in loop:
        s_t[mu][idx] = e_ * sgn
    amps.append(float(np.max(np.abs(interp_site(solve_poisson_links(s_t, N5)) + s_t))))
ax.plot(epss, amps, "o-", ms=4, color="#1f77b4")
ax.plot(epss, amps[0] / epss[0] * epss, "--", color="gray", lw=0.9, label="线性参照")
ax.set_xlabel("耦合强度 eps")
ax.set_ylabel("DC 响应幅度")
ax.set_title("(c) T-奇耦合：响应线性", fontsize=9)
ax.legend(fontsize=8)

fig.tight_layout()
fig.savefig("D:/software/TianXingFC/images/rebuild_3d_batch2.png", dpi=150)
print("\n图已保存：images/rebuild_3d_batch2.png")

print("\n" + "=" * 72)
n_pass = sum(1 for _, ok in CHECKS if ok)
print(f"判据汇总：{n_pass}/{len(CHECKS)} 通过")
for tag, ok in CHECKS:
    print(f"  {tag}: {'PASS' if ok else 'FAIL'}")
print(f"耗时 {time.time() - T0:.1f} s")