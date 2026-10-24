# -*- coding: utf-8 -*-
r"""
rebuild_1d.py

[本体层重建·阶段 2] 一维最小模型：链接型写入的稳态、非纯规范与稳定性

设定
----
一维链：站点 0 至 200 共 201 个，链接 0 至 199 共 200 条，链接 e 连接站点 e 与 e+1。
基本自由度在链接（排布二）：每条链接带记录 u_e；两端链接固定为零（Dirichlet）。
站点记录由链接读数合成：theta_s = (u_{s-1} + u_s)/2（内点站点 s = 1 至 199）。
松弛规则取"邻居传导加源写入"，与 GV0 前置清点登记的方向 6（每拍写入幅度 + 邻居传导）同形：

    du_e/dt = (alpha/2)(u_{e-1} - 2 u_e + u_{e+1}) + src_e
            = alpha[(u_{e-1} + u_{e+1})/2 - u_e] + src_e

源写入作用于链接（排布二的要点），单点源置于链接 e = E0 = 100，源强 SRC = 0.5。
稳态方程：alpha u - (alpha/2)(邻) = src，即 -(alpha/2) lap u = src，直接解线性方程组
（迭代法最慢模速率约 1.2e-4，收敛需上万拍，本轮不用）。

判据
----
T1 稳态格林函数保持（D3 第一条）：系统残差机器精度；解与解析帐篷（一维格林函数，
   形状 r^{2-d} 取 d=1，即 |x - x0| 段）逐点一致；通量守恒 q_e - q_{e-1} = src_e
   处处成立，源链接处跳变恰等于源强。
T2 构型不再纯规范（D3 第二条）：交错分量 u_e -> u_e + eta(-1)^e 对站点合成不可见、
   对尺读数 l_e = 1 + u_e 可见；链接到站点的合成映射的核维数为一，核向量即交错模。
T3 小扰动稳定（D3 第三条）：松弛生成元（内点、Dirichlet）本征值全负，最慢模速率对
   解析式 -alpha(1 - cos(pi/(n+1))) 一致；任意初值出发的松弛对稳态偏差单调不增。
T4 附加读数（同源同格林函数）：链接读数与站点读数按同点配对（链接中点取相邻两站点
   记录的平均），源外逐点比值恒为一（机器精度），源链接处偏离恰为 src/(2 alpha)；
   按 l = 1 + lambda h00/2、h00 = 2 theta 的约定读出的 lambda 为一。

运行
----
python 验证脚本/rebuild_1d.py
确定性（无随机数），运行约 2 秒。
"""
import time

import numpy as np

T0 = time.time()
CHECKS = []


def check(tag, ok, detail):
    CHECKS.append((tag, bool(ok)))
    print(f"  [{tag}] {'PASS' if ok else 'FAIL'}  {detail}")


N_SITE = 201
NL = N_SITE - 1
ALPHA = 1.0
SRC = 0.5
E0 = NL // 2

print("=" * 72)
print("最小验证：本体层重建阶段 2 的一维最小模型（链接型写入）")
print("=" * 72)

# ---------- 稳态：直接解线性方程组 alpha u - (alpha/2)(邻) = src ----------
A = np.zeros((NL, NL))
b = np.zeros(NL)
b[E0] = SRC
for e in range(NL):
    if e == 0 or e == NL - 1:
        A[e, e] = 1.0
    else:
        A[e, e] = ALPHA
        A[e, e - 1] = -0.5 * ALPHA
        A[e, e + 1] = -0.5 * ALPHA
u = np.linalg.solve(A, b)

# ---------- T1 稳态格林函数保持 ----------
res = float(np.max(np.abs(A @ u - b)))
P = (SRC / ALPHA) / (1.0 - 0.5 * ((E0 - 1) / E0 + (NL - 2 - E0) / (NL - 1 - E0)))
e_idx = np.arange(NL)
tent = np.where(e_idx <= E0, P * e_idx / E0, P * (NL - 1 - e_idx) / (NL - 1 - E0))
dev = float(np.max(np.abs(u - tent)) / np.max(np.abs(u)))
q = -0.5 * ALPHA * (u[1:] - u[:-1])
dq = q[1:] - q[:-1]
jump_out = float(np.max(np.abs(np.delete(dq, E0 - 1))))
jump_src = float(dq[E0 - 1])
print(f"\n[T1] 稳态格林函数保持")
print(f"      系统残差最大绝对值 {res:.3e}")
print(f"      与解析帐篷的最大相对偏差 {dev:.3e}（数值峰 {np.max(u):.6f}，解析峰 {P:.6f}）")
print(f"      通量守恒：源外最大跳变 {jump_out:.3e}；源链接处跳变 {jump_src:.6f}（源强 {SRC}）")
check("T1", res < 1e-12 and dev < 1e-10 and jump_out < 1e-12 and abs(jump_src - SRC) < 1e-12,
      f"残差 {res:.1e}、帐篷偏差 {dev:.1e}、通量跳变等于源强（{jump_src:.6f}）")

# ---------- T2 构型不再纯规范 ----------
eta = 0.01
u2 = u + eta * (-1.0) ** np.arange(NL)
theta_full = np.zeros(NL + 1)
theta_full[1:NL] = 0.5 * (u[:-1] + u[1:])
theta_full2 = np.zeros(NL + 1)
theta_full2[1:NL] = 0.5 * (u2[:-1] + u2[1:])
d_theta = float(np.max(np.abs(theta_full2[1:NL] - theta_full[1:NL])))
d_ell = float(np.max(np.abs(u2 - u)))
TH = np.zeros((NL - 1, NL))
for s in range(1, NL):
    TH[s - 1, s - 1] = 0.5
    TH[s - 1, s] = 0.5
_, sv, Vt = np.linalg.svd(TH)
alt = (-1.0) ** np.arange(NL)
v_kernel = Vt[-1]
ker_res = float(np.linalg.norm(TH @ v_kernel) / np.linalg.norm(v_kernel))
corr = float(abs(np.dot(v_kernel, alt)) / (np.linalg.norm(v_kernel) * np.linalg.norm(alt)))
print(f"\n[T2] 构型不再纯规范")
print(f"      交错扰动幅度 {eta}：站点合成之差 {d_theta:.3e}，尺读数之差 {d_ell:.3e}")
print(f"      合成映射奇异值：最大 {sv[0]:.6f}，最小 {sv[-1]:.3e}（秩 = 199），核维数 = 200 - 199 = 1")
print(f"      核向量：TH @ v 相对范数 {ker_res:.3e}，与交错模的重合度 {corr:.12f}")
check("T2", d_theta < 1e-12 and abs(d_ell - eta) < 1e-12 and sv[-1] > 1e-6 and ker_res < 1e-13 and corr > 1 - 1e-10,
      f"站点盲（{d_theta:.1e}）、尺可见（{d_ell:.1e}）、核一维且为交错模")

# ---------- T3 小扰动稳定 ----------
n_int = NL - 2
B = np.zeros((n_int, n_int))
for i in range(n_int):
    B[i, i] = -ALPHA
    if i > 0:
        B[i, i - 1] = 0.5 * ALPHA
    if i < n_int - 1:
        B[i, i + 1] = 0.5 * ALPHA
w = np.linalg.eigvalsh(B)
lam_ana = -ALPHA * (1.0 - np.cos(np.pi / (n_int + 1)))
rel = float(abs(np.max(w) - lam_ana) / abs(lam_ana))
dt = 0.2
ws = 1.0 + np.sin(np.arange(NL) * 0.7)
ws[0] = 0.0
ws[-1] = 0.0
prev = float(np.linalg.norm(ws[1:-1] - u[1:-1]))
first = prev
viol = 0.0
for _ in range(400):
    lap = np.zeros_like(ws)
    lap[1:-1] = ws[2:] - 2 * ws[1:-1] + ws[:-2]
    ws[1:-1] = ws[1:-1] + dt * (0.5 * ALPHA * lap[1:-1] + b[1:-1])
    cur = float(np.linalg.norm(ws[1:-1] - u[1:-1]))
    viol = max(viol, cur - prev)
    prev = cur
print(f"\n[T3] 小扰动稳定")
print(f"      生成元本征值：最大 {np.max(w):.6e}（解析最慢模 {lam_ana:.6e}，相对偏差 {rel:.1e}），最小 {np.min(w):.6f}")
print(f"      松弛 400 步：偏差范数 {first:.4f} 降至 {prev:.3e}，单调性最大违反 {viol:.3e}")
check("T3", np.max(w) < 0 and rel < 1e-10 and viol < 1e-12,
      f"本征值全负（最慢模 {np.max(w):.2e}），松弛无增长")

# ---------- T4 附加读数：同源同格林函数 ----------
ee = np.arange(1, NL - 1)
phi_mid = 0.5 * (theta_full[ee] + theta_full[ee + 1])
gam = u[ee] / phi_mid
dev_g = float(np.max(np.abs(gam[ee != E0] - 1.0)))
src_link = float(u[E0] - phi_mid[E0 - 1])
lam_eff = float(np.median(gam))
print(f"\n[T4] 附加读数：同源同格林函数（同点配对）")
print(f"      源外逐点比值对一的最大偏差 {dev_g:.3e}")
print(f"      源链接处偏离 {src_link:.6f}（解析 src/(2 alpha) = {SRC / (2 * ALPHA):.6f}），该点比值 {gam[E0 - 1]:.6f}")
print(f"      有效 lambda 读数（中位）{lam_eff:.6f}")
check("T4", dev_g < 1e-10 and abs(src_link - SRC / (2 * ALPHA)) < 1e-10,
      f"源外逐点同比（{dev_g:.1e}），源链接偏离恰为源强折半，lambda = {lam_eff:.6f}")

# ---------- 图 ----------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
fig, axs = plt.subplots(1, 3, figsize=(13.2, 4.2))

ax = axs[0]
ax.plot(e_idx, tent, "-", lw=1.4, color="#1f77b4", label="解析一维格林函数")
ax.plot(e_idx, u, "o", ms=2.6, color="#d62728", label="链接读数 u_e（数值解）")
ax.plot(np.arange(1, NL) - 0.5, theta_full[1:NL], "s", ms=2.4, color="#2ca02c", label="站点记录（半格对位）")
ax.set_xlabel("链接编号 e")
ax.set_ylabel("记录读数")
ax.set_title("(a) 稳态：链接读数与站点记录同属一条格林函数", fontsize=9)
ax.legend(fontsize=7.5)

ax = axs[1]
ax.plot(ee, gam, "-", lw=0.9, color="#1f77b4")
ax.plot(ee, gam, "o", ms=2.6, color="#d62728")
ax.axhline(1.0, color="gray", lw=0.8, ls=":")
ax.annotate(f"源链接 e={E0}\n比值 {gam[E0 - 1]:.5f}", xy=(E0, gam[E0 - 1]),
            xytext=(E0 + 10, gam[E0 - 1] - 0.0015), fontsize=8,
            arrowprops=dict(arrowstyle="->", lw=0.8))
ax.set_xlabel("链接编号 e")
ax.set_ylabel("同点比值（尺对钟）")
ax.set_title("(b) 同源同格林函数：源外逐点恒为一", fontsize=9)

ax = axs[2]
ax.bar([0, 1], [max(d_theta, 1e-17), d_ell], color=["#d62728", "#2ca02c"], width=0.55)
ax.set_yscale("log")
ax.set_xticks([0, 1])
ax.set_xticklabels(["站点记录差值", "尺读数差值"])
ax.set_ylim(1e-17, 1e-1)
for xi, val in zip([0, 1], [d_theta, d_ell]):
    ax.text(xi, max(val, 1e-17) * 1.8, f"{val:.1e}", ha="center", fontsize=8)
ax.set_title(f"(c) 交错扰动（{eta}）：站点全盲、尺可见", fontsize=9)

fig.tight_layout()
fig.savefig("D:/software/TianXingFC/images/rebuild_1d.png", dpi=150)
print("\n图已保存：images/rebuild_1d.png")

# ---------- 汇总 ----------
print("\n" + "=" * 72)
n_pass = sum(1 for _, ok in CHECKS if ok)
print(f"判据汇总：{n_pass}/{len(CHECKS)} 通过")
for tag, ok in CHECKS:
    print(f"  {tag}: {'PASS' if ok else 'FAIL'}")
print(f"耗时 {time.time() - T0:.1f} s")