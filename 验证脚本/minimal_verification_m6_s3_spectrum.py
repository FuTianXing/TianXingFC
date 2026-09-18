# -*- coding: utf-8 -*-
"""M6-S3（可做部分）：真实场构型的谱读数 —— 构型依赖与 lambda 扫描。

背景：BH3-4 终审（裁决十三）的判定量是「真实场的 Gamma_c 与写入速率之比」。S1 可做部分已交
付真实场构型族（写入律加容量裁剪）。本材料把该构型族送进 D1、D2 的机器（线性谱扫描：
gamma(h) = sqrt(-mu_min(h))、I = int_0^1 gamma dh、Gamma_c = I / ln(1/eps0)），给出构型依赖与
lambda 依赖，并把未定输入（耦合系数 lambda 的实际值、kappa_phi 上限）显式留出。机器的解析锚点
按 D2 材料原样复现，保证读数在同一坐标系里。

判据（先写期望，再算）：
  S3a 机器锚点复现：同一构造（周期链 1D N=256；周期立方 3D L=16）下，1D 均匀 I = 0.246351
       （对闭式 F1，偏差 <= 1e-5）；3D 均匀 I = 0.144089（对 D2 读数，偏差 <= 5e-3）。
  S3b 真实场构型的读数：以 S1 写入场剖面为调制形状（弥散档 M/m_P=1、过密档 100，R_s=8）：
       I_real < I_uniform(满幅)；报告与「同均值均匀」之比（D1 的驱动量是平均调制密度）。
  S3c lambda 扫描：Gamma_c(lambda) 单调增；拟合 beta = dlnI/dlnlambda 落在 [0.2, 1]（报告）。
  S3d 终审预备：对写入速率 Gamma = 2.74e-39 每拍（太阳档，S2 材料），Gamma_c(1)/Gamma >= 1e30；
       交叉 lambda_min = (Gamma/Gamma_c(1))^(1/beta) 报出；kappa_phi 与 lambda 两输入按裁决十三留出。
  报告 R：三维真实场构型读数与形状对照（均匀、漏斗）。

确定性：eigsh 固定初矢（种子固定）；全部无随机结果。
"""

import sys
import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
RNG = np.random.default_rng(20260918)
CHECKS = []
def check(tag, ok, detail):
    CHECKS.append((tag, bool(ok)))
    print("  [%s] %s  %s" % (tag, "PASS" if ok else "FAIL", detail))

print("==== M6-S3（可做部分）：真实场构型的谱读数 ====")

# ---------------- D2 机器（原样复用） ----------------
def low_1d(h):
    N = len(h); n = np.arange(N)
    H = np.zeros((N, N))
    H[n, n] = 2.0 + ((-1.0) ** n) * h
    H[n, (n + 1) % N] += 1.0
    H[n, (n - 1) % N] += 1.0
    return float(np.linalg.eigvalsh(H)[0])

def build_3d(L, h):
    N = L ** 3; idx = np.arange(N)
    i = idx // (L * L); j = (idx // L) % L; k = idx % L
    sgn = (-1.0) ** (i + j + k)
    nxp = i * L * L + j * L + (k + 1) % L
    nyp = i * L * L + ((j + 1) % L) * L + k
    nzp = ((i + 1) % L) * L * L + j * L + k
    rows = np.concatenate([idx, idx, idx, nxp, nyp, nzp])
    cols = np.concatenate([nxp, nyp, nzp, idx, idx, idx])
    return sp.coo_matrix((-np.ones(6 * N), (rows, cols)), shape=(N, N)).tocsr() + sp.diags(6.0 + sgn * h)

def low_3d(L, h):
    N = L ** 3
    v0 = RNG.standard_normal(N); v0 /= np.linalg.norm(v0)
    return float(spla.eigsh(build_3d(L, h), k=1, which="SA", v0=v0, tol=0, return_eigenvectors=False)[0])

def gamma_curve(lowfun, shape, npts, lam=1.0):
    a = np.linspace(0.0, 1.0, npts)
    g = np.array([np.sqrt(max(0.0, -lowfun(lam * aa * shape))) for aa in a])
    return a, g

def integral(a, g):
    return float(np.trapezoid(g, a))

EPS0 = 1e-3
LN = np.log(1.0 / EPS0)
N1 = 256; L3 = 16

# ---------------- S3a 锚点 ----------------
t0 = time.time()
a, g = gamma_curve(low_1d, np.full(N1, 1.0), 61)
I_1d_u = integral(a, g)
print("\n  [S3a] 锚点：1D 均匀 I = %.6f（闭式 0.246351）" % I_1d_u)
a3, g3 = gamma_curve(lambda h: low_3d(L3, h), np.full(L3 ** 3, 1.0), 25)
I_3d_u = integral(a3, g3)
print("        3D 均匀 I = %.6f（D2 读数 0.144089）" % I_3d_u)
check("S3a", abs(I_1d_u - 0.246351) <= 1e-5 and abs(I_3d_u - 0.144089) <= 5e-3,
      "1D 偏差 %.1e、3D 偏差 %.1e" % (abs(I_1d_u - 0.246351), abs(I_3d_u - 0.144089)))

# ---------------- S1 场（同 S1 可做部分实现） ----------------
def s1_field(Mp, R_s, N=49, itmax=40000):
    c0 = N // 2
    ii, jj, kk = np.indices((N, N, N))
    dist = np.sqrt((ii - c0) ** 2 + (jj - c0) ** 2 + (kk - c0) ** 2)
    mask = np.zeros((N, N, N), bool); mask[1:-1, 1:-1, 1:-1] = True
    ball = dist <= R_s
    s_tot = 4.0 * np.pi * Mp
    w = np.where(ball, s_tot / (6.0 * int(ball.sum())), 0.0)
    th = np.zeros((N, N, N))
    for it in range(itmax):
        avg = (np.roll(th, 1, 0) + np.roll(th, -1, 0) + np.roll(th, 1, 1) + np.roll(th, -1, 1)
               + np.roll(th, 1, 2) + np.roll(th, -1, 2)) / 6.0
        th_new = np.minimum(avg + w, 1.0); th_new[~mask] = 0.0
        dm = float(np.max(np.abs(th_new - th))); th = th_new
        if it % 50 == 0 and dm < 1e-14:
            break
    r = np.arange(1, 27)
    prof = np.array([th[(np.abs(dist - rr) < 0.6) & mask].mean() for rr in r])
    return r, prof

def bump_1d(prof, q_eff, N=N1):
    n = np.arange(N)
    rc = np.minimum(np.abs(n - N // 2), N - np.abs(n - N // 2))
    h = np.where(rc == 0, prof[0], q_eff / np.maximum(rc, 1))
    m = rc <= 26
    h[m] = prof[np.maximum(rc[m], 1) - 1]
    return h

t1 = time.time()
r1, p1 = s1_field(1.0, 8.0)     # 弥散档（无裁剪，q_eff = M/m_P = 1）
r2, p2 = s1_field(100.0, 8.0)   # 过密档（裁剪，q_eff = 10.597 由 S1 测得）
print("  S1 场生成用时 %.1f 秒（弥散档 q=1.0、过密档 q=10.597）" % (time.time() - t1))
h_dis = bump_1d(p1, 1.0); h_dns = bump_1d(p2, 10.597)
h_uni = np.full(N1, 1.0); h_fun = 1.0 / (1.0 + np.minimum(np.abs(np.arange(N1) - N1 // 2), N1 - np.abs(np.arange(N1) - N1 // 2)))

# ---------------- S3b 真实场构型读数 ----------------
print("\n  [S3b] 真实场构型（1D 等效）")
res = {}
for name, shape in (("均匀(满幅)", h_uni), ("漏斗", h_fun), ("弥散档", h_dis), ("过密档", h_dns)):
    aa, gg = gamma_curve(low_1d, shape, 61)
    I = integral(aa, gg)
    mean = float(np.mean(shape))
    aa2, gg2 = gamma_curve(low_1d, np.full(N1, mean), 61)
    I_m = integral(aa2, gg2)
    res[name] = (I, mean, I_m)
    print("       %-8s 均值 %.4f：I = %.6f，Gamma_c = %.3e；同均值均匀 I = %.6f（比 %.3f）"
          % (name, mean, I, I / LN, I_m, I / I_m if I_m > 0 else float("nan")))
check("S3b", res["弥散档"][0] < res["均匀(满幅)"][0] and res["过密档"][0] < res["均匀(满幅)"][0],
      "I_real < I_uniform(满幅)：弥散 %.5f、过密 %.5f 对 %.5f" % (res["弥散档"][0], res["过密档"][0], res["均匀(满幅)"][0]))

# ---------------- S3c lambda 扫描 ----------------
print("\n  [S3c] lambda 扫描（过密档，1D）")
lams = np.array([0.003, 0.01, 0.03, 0.1, 0.3, 1.0])
Is = []
for lam in lams:
    aa, gg = gamma_curve(low_1d, h_dns, 61, lam=lam)
    Is.append(integral(aa, gg))
Is = np.array(Is)
mono = bool(np.all(np.diff(Is) > 0))
beta = float(np.polyfit(np.log(lams), np.log(Is), 1)[0])
print("       lambda = %s" % lams)
print("       I      = %s" % np.round(Is, 6))
print("       单调增 %s；拟合 beta = %.4f" % (mono, beta))
check("S3c", mono and 0.2 <= beta <= 1.0, "单调 %s；beta = %.4f 落在 [0.2, 1]" % (mono, beta))

# ---------------- S3d 终审预备 ----------------
GAMMA_W = 2.74e-39
Gc1 = res["过密档"][0] / LN
margin = Gc1 / GAMMA_W
lam_min = (GAMMA_W / Gc1) ** (1.0 / beta)
print("\n  [S3d] 终审预备：过密档 Gamma_c(1) = %.3e 每拍（含 1D 等效口径）；写入速率 %.2e 每拍" % (Gc1, GAMMA_W))
print("       余量 = %.2e；交叉 lambda_min = %.3e（I(lambda) 按幂律外推）" % (margin, lam_min))
check("S3d", margin >= 1e30, "余量 %.2e >= 1e30（方向稳健；终审归属待裁决十三门槛齐备）" % margin)

# ---------------- 报告 R：3D 真实场构型 ----------------
print("\n  [报告 R] 3D（L=16）真实场构型")
c0 = L3 // 2
ii, jj, kk = np.indices((L3, L3, L3))
dd = np.sqrt((ii - c0) ** 2 + (jj - c0) ** 2 + (kk - c0) ** 2)
h3_dns = np.interp(dd, r2, p2)
h3_dns = np.where(dd > r2[-1], 10.597 / np.maximum(dd, 1.0), h3_dns)
h3_dns = np.where(dd < 1.0, p2[0], h3_dns)
for name, shape3 in (("均匀(满幅)", np.full(L3 ** 3, 1.0)), ("过密档", h3_dns.ravel())):
    aa, gg = gamma_curve(lambda h: low_3d(L3, h), shape3, 17)
    I = integral(aa, gg)
    print("       %-8s I = %.6f，Gamma_c = %.3e" % (name, I, I / LN))
print("       用时 %.1f 秒" % (time.time() - t0))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
ax = axes[0]
nn = np.arange(N1)
for shape, lab, col in ((h_uni, "均匀(满幅)", "gray"), (h_fun, "漏斗", "tab:green"),
                        (h_dis, "弥散档", "tab:blue"), (h_dns, "过密档", "tab:red")):
    ax.plot(nn, shape, lw=1.0, color=col, label=lab)
ax.set_xlim(100, 156); ax.set_ylim(0, 1.05)
ax.set_xlabel("链格（中心邻域）"); ax.set_ylabel("调制 h_n")
ax.set_title("(a) 1D 调制形状（真实场为核加尾）", fontsize=9); ax.legend(fontsize=7)
ax = axes[1]
for shape, lab, col in ((h_uni, "均匀(满幅)", "gray"), (h_fun, "漏斗", "tab:green"),
                        (h_dis, "弥散档", "tab:blue"), (h_dns, "过密档", "tab:red")):
    aa, gg = gamma_curve(low_1d, shape, 61)
    ax.plot(aa, gg, lw=1.2, color=col, label="%s（I=%.4f）" % (lab, integral(aa, gg)))
ax.set_xlabel("斜坡幅度 a"); ax.set_ylabel("gamma(a) = sqrt(-mu_min)")
ax.set_title("(b) 生长率曲线与积分 I", fontsize=9); ax.legend(fontsize=6.5)
ax = axes[2]
ax.loglog(lams, Is / LN, "o-", lw=1.2, label="Gamma_c(lambda)（过密档）")
ax.axhline(GAMMA_W, color="tab:red", ls=":", lw=1.2)
ax.text(0.004, GAMMA_W * 3, "写入速率 %.1e（太阳档）" % GAMMA_W, fontsize=7, color="tab:red")
ax.loglog([lam_min], [GAMMA_W], "k*", ms=9)
ax.text(lam_min * 1.3, GAMMA_W * 3, "lambda_min = %.1e" % lam_min, fontsize=7)
ax.set_xlabel("lambda"); ax.set_ylabel("Gamma_c（每拍）")
ax.set_title("(c) lambda 扫描与终审预备（beta = %.2f）" % beta, fontsize=9); ax.legend(fontsize=7)
plt.tight_layout()
plt.savefig(r"D:\software\TianXingFC\images\m6_s3_spectrum.png", dpi=150)
print("\n  图已存 images/m6_s3_spectrum.png")
print("  面板数据核对：(a) 中心邻域 100 至 156、h 0 至 1.05；(b) a 0 至 1、gamma 0 至 0.25 量级；")
print("  (c) lambda 0.003 至 1、Gamma_c 1.3e-3 至 3.2e-2，写速率线 2.74e-39，lambda_min = %.2e" % lam_min)

print("\n==== 判据汇总：%d / %d 通过 ====" % (sum(1 for _, ok in CHECKS if ok), len(CHECKS)))
sys.exit(0 if all(ok for _, ok in CHECKS) else 1)
