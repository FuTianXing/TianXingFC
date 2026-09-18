# -*- coding: utf-8 -*-
r"""
minimal_verification_rebuild_stage4_lambda.py

[本体层重建·阶段 4] 识别式重跑与 lambda 导出的读数（判据 L1 至 L3）

设定
----
复用批一的三维链接装置（周期立方格、链接场 u_e、站点记录 theta = 入射平均、FFT 精确泊松解）。
源写在原点的六条入射链接上，强度取 kappa*S_BOOST（S_BOOST = 6，使记录幅值到 0.5 量级，
反馈可见）。段长因子按标量方式反馈进传导率（与批二 H4 的键能耦合同形式）：

    kappa_e = kappa (1 + c u_e)，  稳态：u_e - (theta_i + theta_j)/2 = s_e / kappa_e。

两支读数按同点配对：钟率 h00 = 2 theta，尺读数 ell = 1 + u（链接自己的读数），
故 lambda = 2 (ell - 1) / h00 = u_e / theta_mid(e)。

判据
----
L1 结构恒等式：推导的泊松源与合成自洽（theta_u = theta，机器精度，非自明核对）；
   源外所有链接上 u_e = (theta_i + theta_j)/2 逐点成立（稳态方程的直接推论），
   即同点比值恒为一，与源强、位置无关。
L2 反馈稳健性：c 取 0、0.05、0.1 时恒等式与比值不漂移（机器精度）；反馈确实改变记录剖面
   （峰值变化非零，证明反馈在场）。
L3 归一化读出：lambda 的导出值在中位意义下为 1，误差为机器精度；剖面变形不进入比值。

运行
----
python 验证脚本/minimal_verification_rebuild_stage4_lambda.py
确定性（无随机数），运行约 12 秒。
"""
import time

import numpy as np

T0 = time.time()
CHECKS = []


def check(tag, ok, detail):
    CHECKS.append((tag, bool(ok)))
    print(f"  [{tag}] {'PASS' if ok else 'FAIL'}  {detail}")


D = 3
KAPPA = 1.0
S_BOOST = 6.0
C_LIST = (0.0, 0.05, 0.1)


def inc_link(u):
    out = np.zeros_like(u[0])
    for mu in range(D):
        out += u[mu] + np.roll(u[mu], 1, axis=mu)
    return out


def interp_site(theta):
    return np.stack([(theta + np.roll(theta, -1, axis=mu)) / 2.0 for mu in range(D)])


def fft_solve(f, N):
    kx = 2 * np.pi * np.fft.fftfreq(N)
    KX, KY, KZ = np.meshgrid(kx, kx, kx, indexing='ij')
    denom = 1.0 - (np.cos(KX) + np.cos(KY) + np.cos(KZ)) / D
    denom.flat[0] = 1.0
    theta = np.real(np.fft.ifftn(np.fft.fftn(f) / denom))
    return theta - theta.mean()


def point_source(N, strength):
    s = np.zeros((D, N, N, N))
    for mu in range(D):
        s[mu].flat[0] += strength / 6.0
        idx = [0, 0, 0]
        idx[mu] = N - 1
        s[mu][tuple(idx)] += strength / 6.0
    return s


def solve(c, N, iters=200):
    s = point_source(N, KAPPA * S_BOOST)
    w = s / KAPPA
    res_hist = []
    for it in range(iters):
        theta = fft_solve(inc_link(w) / D, N)
        u = interp_site(theta) + w
        if c == 0.0:
            return u, theta, s, 0, 0.0, float((inc_link(w) / D).mean())
        kap = 1.0 + c * u
        w_new = s / (KAPPA * kap)
        res = float(np.max(np.abs(w_new - w)))
        res_hist.append(res)
        w = w_new
        if res < 1e-15:
            theta = fft_solve(inc_link(w) / D, N)
            u = interp_site(theta) + w
            return u, theta, s, it + 1, res, float((inc_link(w) / D).mean())
    theta = fft_solve(inc_link(w) / D, N)
    u = interp_site(theta) + w
    return u, theta, s, iters, res_hist[-1], float((inc_link(w) / D).mean())


print("=" * 72)
print("最小验证：本体层重建阶段 4（识别式重跑与 lambda 导出，L1 至 L3）")
print("=" * 72)

# ---------------- L1 / L2：结构恒等式与反馈稳健性 ----------------
print("\n[L1/L2] 源外恒等式与反馈稳健性（N=64）")
N0 = 64
res_tab = {}
ident_tab = {}
for c in C_LIST:
    u, theta, s, iters, res, fbar = solve(c, N0)
    theta_mid = interp_site(theta)
    off = (s == 0.0)
    dev = float(np.max(np.abs(u - theta_mid)[off]))
    idn = float(np.max(np.abs(inc_link(u) / (2 * D) - theta - fbar / 2)))
    lam = float(np.median((u / theta_mid)[off]))
    peak = float(np.max(np.abs(theta)))
    res_tab[c] = (iters, res, peak)
    ident_tab[c] = (dev, idn, lam)
    print(f"      c={c:4.2f}：迭代 {iters}，定点残差 {res:.2e}，源外恒等式偏差 {dev:.2e}，"
          f"合成自洽 {idn:.2e}，比值中位 {lam:.12f}，剖面峰值 {peak:.6f}")

devs = [ident_tab[c][0] for c in C_LIST]
idns = [ident_tab[c][1] for c in C_LIST]
lams = [ident_tab[c][2] for c in C_LIST]
peaks = [res_tab[c][2] for c in C_LIST]
check("L1", max(idns) < 1e-13 and max(devs) < 1e-13 and abs(lams[0] - 1) < 1e-12,
      f"合成 theta_u = theta（含环面常偏置 <f>/2 后偏差 {max(idns):.1e}，即推导的泊松源自洽）；源外比值 = 1（偏差 {max(devs):.1e}）")
peak_shift = abs(peaks[-1] / peaks[0] - 1)
check("L2", max(idns) < 1e-13 and all(abs(l - 1) < 1e-12 for l in lams) and peak_shift > 1e-4,
      f"反馈开启后恒等式与比值不漂移（偏差仍 {max(devs):.1e}），且剖面确被改变（峰值移动 {peak_shift:.2%}）")

# ---------------- L3：归一化读出与格距外推 ----------------
print("\n[L3] lambda 的归一化读出与外推")
out = {}
for N in (64, 96):
    u, theta, s, iters, res, fbar = solve(0.0, N)
    rs = np.arange(2, 11)
    lam_r = np.array([u[0][r, 0, 0] / ((theta[r, 0, 0] + theta[r + 1, 0, 0]) / 2) for r in rs])
    out[N] = (rs, lam_r, float(np.max(np.abs(theta))))
    print(f"      N={N}：沿轴比值 " + " ".join(f"{v:.10f}" for v in lam_r))
fit = np.polyfit(1.0 / out[64][0] ** 2, out[64][1], 1)
fit2 = np.polyfit(1.0 / out[96][0] ** 2, out[96][1], 1)
print(f"      1/r^2 外推：N=64 给 {fit[1]:.12f}，N=96 给 {fit2[1]:.12f}")
check("L3", abs(fit[1] - 1) < 1e-12 and abs(fit2[1] - 1) < 1e-12,
      f"lambda 导出 = 1（外推 {fit[1]:.12f} 与 {fit2[1]:.12f}，机器精度），误差为零偏移")

# ---------------- 图 ----------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
fig, axs = plt.subplots(1, 3, figsize=(13.2, 4.2))

ax = axs[0]
rr = np.arange(1, 13)
for c in C_LIST:
    u, theta, s, _, _, _ = solve(c, N0)
    ax.semilogy(rr, [abs(theta[r, 0, 0]) for r in rr], "o-", ms=3, label=f"c={c}")
ax.set_xlabel("半径 r（格距）")
ax.set_ylabel(r"$|\theta(r)|$")
ax.set_title("(a) 反馈改变记录剖面", fontsize=9)
ax.legend(fontsize=8)

ax = axs[1]
for c in C_LIST:
    u, theta, s, _, _, _ = solve(c, N0)
    lam_r = [u[0][r, 0, 0] / ((theta[r, 0, 0] + theta[r + 1, 0, 0]) / 2) for r in rr]
    ax.plot(rr, lam_r, "o-", ms=3, label=f"c={c}")
ax.axhline(1.0, color="gray", ls=":", lw=0.9)
ax.set_ylim(0.999, 1.001)
ax.set_xlabel("半径 r（格距）")
ax.set_ylabel(r"同点比值 $\lambda$")
ax.set_title("(b) 两支读数比值：三档反馈重合于 1", fontsize=9)
ax.legend(fontsize=8)

ax = axs[2]
cs = np.array([0.0, 0.02, 0.05, 0.08, 0.1])
devs2 = []
for c in cs:
    u, theta, s, _, _, _ = solve(c, N0)
    off = (s == 0.0)
    devs2.append(float(np.max(np.abs(u - interp_site(theta))[off])))
ax.semilogy(cs + 1e-3, np.array(devs2) + 1e-18, "o-", ms=4, color="#1f77b4")
ax.set_xlabel("反馈强度 c")
ax.set_ylabel("源外恒等式偏差")
ax.set_title("(c) 恒等式对反馈强度不敏感（机器精度）", fontsize=9)

fig.tight_layout()
fig.savefig("D:/software/TianXingFC/images/rebuild_stage4_lambda.png", dpi=150)
print("\n图已保存：images/rebuild_stage4_lambda.png")

print("\n" + "=" * 72)
n_pass = sum(1 for _, ok in CHECKS if ok)
print(f"判据汇总：{n_pass}/{len(CHECKS)} 通过")
for tag, ok in CHECKS:
    print(f"  {tag}: {'PASS' if ok else 'FAIL'}")
print(f"耗时 {time.time() - T0:.1f} s")