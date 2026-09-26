# -*- coding: utf-8 -*-
"""交错通道的双重身份：合成零模与动力学正刚度（第 462 条；局域尺判定指南乙首轮 RD1 至 RD3）。

模型复刻阶段 2 一维链接模型（minimal_verification_rebuild_1d.py）：
链接 0..199，两端 Dirichlet（u_0 = u_199 = 0），内点 1..198；
稳态 alpha u - (alpha/2) 邻 = src，源在 e = 100，src = 0.5；
站点记录 theta_s = (u_{s-1} + u_s)/2；链接中点配对 theta_mid(e) = (u_{e-1} + 2u_e + u_{e+1})/4。

判据：
  RD1 交错方向的体弛豫率 = -2 alpha（解析）；二次型 Q = 2 alpha eta^2 (NL-1)（正定）
  RD2 两个维数分离：稳态算子零空间维数 0（解唯一）；
      合成映射核维数 = 列数 - 秩 = 1（交错通道）
  RD3 通道效应：源外 lambda 恒等（机器零）且源链接偏离 s/(2 alpha)；
      交错扰动对 lambda 的灵敏度线性；同一站点记录下的两个重构解 lambda 读数差
      （合成盲、尺可见的定量）
确定性，约 1 秒。图存 images/staggered_channel.png
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG = os.path.join(ROOT, "images", "staggered_channel.png")

N_SITE = 201
NL = N_SITE - 1
ALPHA = 1.0
SRC = 0.5
E0 = NL // 2


def check(tag, ok, detail):
    print(f"  [{tag}] {'PASS' if ok else 'FAIL'}  {detail}")
    return bool(ok)


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

alt = (-1.0) ** np.arange(NL)
TH = np.zeros((NL - 1, NL))
for s in range(1, NL):
    TH[s - 1, s - 1] = 0.5
    TH[s - 1, s] = 0.5
theta = TH @ u


def theta_mid_of(uu):
    return (uu[:-2] + 2.0 * uu[1:-1] + uu[2:]) / 4.0


print("== RD1 交错方向的动力学刚度 ==")
n_int = NL - 2
B = np.zeros((n_int, n_int))
for i in range(n_int):
    B[i, i] = -ALPHA
    if i > 0:
        B[i, i - 1] = 0.5 * ALPHA
    if i < n_int - 1:
        B[i, i + 1] = 0.5 * ALPHA
v_int = (-1.0) ** np.arange(1, NL - 1)
ratio = (B @ v_int) / v_int
body = ratio[8:-8]
rd1_dev = float(np.max(np.abs(body - (-2.0 * ALPHA))))
print(f"  体区间弛豫率均值 = {float(np.mean(body)):.15f}（解析 -2 alpha = {-2.0*ALPHA:.1f}），最大偏差 {rd1_dev:.3e}")
print(f"  边界两点读数：{ratio[0]:.6f}、{ratio[1]:.6f}、{ratio[-2]:.6f}、{ratio[-1]:.6f}")
eta = 0.01
dalt = np.diff(eta * alt)
Q = 0.5 * ALPHA * float(np.sum(dalt ** 2))
Q_ana = 2.0 * ALPHA * eta ** 2 * (NL - 1)
print(f"  交错扰动的二次型 Q = {Q:.6e}（解析 2 alpha eta^2 (NL-1) = {Q_ana:.6e}）")
rd1_ok = check("RD1", rd1_dev < 1e-12 and abs(Q - Q_ana) < 1e-15 * Q_ana,
               f"体弛豫率 -2 alpha（偏差 {rd1_dev:.1e}）；二次型正定（比值 {Q/Q_ana:.12f}）")

print("== RD2 两个维数的分离 ==")
svA = np.linalg.svd(A, compute_uv=False)
_, svTH, Vt = np.linalg.svd(TH)
rank_A = int(np.sum(svA > 1e-10))
rank_TH = int(np.sum(svTH > 1e-10))
ker_A = NL - rank_A
ker_TH = NL - rank_TH
v_k = Vt[-1]
corr = float(abs(np.dot(v_k, alt)) / (np.linalg.norm(v_k) * np.linalg.norm(alt)))
print(f"  稳态算子：秩 {rank_A}/{NL}，零空间维数 {ker_A}（解唯一）；最小奇异值 {svA[-1]:.6f}")
print(f"  合成映射：秩 {rank_TH}（{NL-1} 行），核维数 = {NL} - {rank_TH} = {ker_TH}；核-交错重合度 {corr:.12f}")
rd2_ok = check("RD2", ker_A == 0 and ker_TH == 1 and corr > 1 - 1e-10,
               f"解唯一（核 0）与合成核一维（交错）并存")

print("== RD3 通道效应 ==")
tm = theta_mid_of(u)
lam = u[1:-1] / tm
off = np.arange(1, NL - 1) != E0
dev_out = float(np.max(np.abs(lam[off] - 1.0)))
src_dev = float(lam[E0 - 1] - 1.0)
src_pred = SRC / (2.0 * ALPHA) / tm[E0 - 1]
print(f"  源外 max|lambda - 1| = {dev_out:.3e}（期望机器零）")
print(f"  源链接 lambda - 1 = {src_dev:.9f}（解析 s/(2 alpha)/theta_mid = {src_pred:.9f}）")

eta1, eta2 = 0.005, 0.01
dl1 = (u + eta1 * alt)[1:-1] / theta_mid_of(u + eta1 * alt) - lam
dl2 = (u + eta2 * alt)[1:-1] / theta_mid_of(u + eta2 * alt) - lam
sel = np.abs(dl1) > 1e-6
lin_dev = float(np.max(np.abs(dl2[sel] / dl1[sel] - 2.0)))
print(f"  交错灵敏度线性核对：加倍比对最大偏差 {lin_dev:.3e}（期望 < 1e-10）")

theta_s = TH @ (u + eta * alt)
d_theta = float(np.max(np.abs(theta_s - theta)))
d_lam_max = float(np.max(np.abs(dl2)))
print(f"  同一站点记录的两个重构解：合成差 {d_theta:.3e}（盲），lambda 读数差 max {d_lam_max:.3e}（可见）")
rd3_ok = check("RD3", dev_out < 1e-12 and abs(src_dev - src_pred) < 1e-12 and lin_dev < 1e-10 and d_theta < 1e-14 and d_lam_max > 1e-4,
               f"源外恒等、源链接偏离解析、灵敏度线性、盲-可见分离")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
ax = axes[0]
e_int = np.arange(1, NL - 1)
ax.plot(e_int, ratio, "o-", ms=2.5)
ax.axhline(-2.0, color="k", ls="--", lw=1, label="解析 -2 alpha")
ax.set_xlabel("内点链接索引"); ax.set_ylabel("(B v)/v")
ax.set_title("交错方向弛豫率（体值与环境修正）"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
ax = axes[1]
ax.plot(e_int, dl2, "o-", ms=2.5, label="lambda 偏移（eta = 0.01）")
ax.set_xlabel("内点链接索引"); ax.set_ylabel("delta lambda")
ax.set_title("交错通道对尺读数的灵敏度（源链接处最大）"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(IMG, dpi=160)
print("figure saved:", IMG)
print("判据：", "全部通过" if (rd1_ok and rd2_ok and rd3_ok) else "存在失败")
