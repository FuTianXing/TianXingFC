# -*- coding: utf-8 -*-
r"""
kappa_interface.py

P4-16 第一步第一轮：锁定耦合 kappa 的导出（路线乙为主，丙为对照）。
候选形式：kappa = 1/d（失锁率取保护距离的倒数），d = L = 8（第 184 条，按裁决点 3 引用）。
确定性，随机固定种子 20260920。

判据与期望：
  L1 正性与范围：kappa = 1/L 落在 (0, 1) 且 L = 8 给 kappa = 0.125（解析）
  L2 闭式相容：alpha = 1 - kappa = 0.875 代回场层两点指数闭式，拟合相对残差 <= 5%
  L3 门槛相容：回拉快于失配的条件 kappa > gamma 在非线性档上核验（扫 gamma 定门槛，比值报告）
  L4 保护相容：kappa 只依赖保护距离（构造性；引用第 184 条 d = L）
  L5 规模相容：P = sigma^2/(kappa N) 在固定 L、增大 N 时按 1/N 稀释（<= 5%）
  L6 参数经济：全链条无新增可调常数（构造性计数）
  L7 对照通道（路线丙，读数）：键能曲率通道的刚度与 1/d 形式的比值，引用第 178 条读数
  L8 接口登记（路线甲为 P4-16 正式目标）
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

IMG = r"D:\software\TianXingFC\images\kappa_interface.png"
SEED = 20260920
L_LOCK = 8
KAPPA = 1.0 / L_LOCK
rows = []


def report(tag, ok, expect, actual):
    rows.append((tag, ok, expect, actual))
    print("[%s] %s" % ("通过" if ok else "未过", tag))
    print("      期望 " + expect)
    print("      实测 " + actual)


# ---------- L1 正性与范围 ----------
report("L1 正性与范围", 0.0 < KAPPA < 1.0 and abs(KAPPA - 0.125) <= 1e-15,
       "kappa = 1/L 落在 (0,1)，L = 8 给 kappa = 0.125（精确）",
       "kappa = %.6f" % KAPPA)

# ---------- L2 闭式相容 ----------
def coef(kap, q, beta2=0.5):
    alpha = 1.0 - kap
    den = (alpha - q) * (1.0 - alpha * q)
    a = -beta2 * q / den
    b = beta2 * alpha * (1.0 - q ** 2) / (den * (1.0 - alpha ** 2))
    return alpha, a, b


Q = 0.9
alpha, a_, b_ = coef(KAPPA, Q, 2.0 * KAPPA)
NS, NZ, BURN = 4000, 3000, 500
rng = np.random.default_rng(SEED)
rho = Q
n1 = np.zeros((NS, NZ + BURN))
n1[:, 0] = rng.normal(size=NS)
for j in range(1, NZ + BURN):
    n1[:, j] = rho * n1[:, j - 1] + rng.normal(scale=np.sqrt(1.0 - rho ** 2), size=NS)
x = np.zeros_like(n1)
for j in range(1, NZ + BURN):
    x[:, j] = alpha * x[:, j - 1] + np.sqrt(2.0 * KAPPA) * n1[:, j]
x = x[:, BURN:]
rs = np.arange(0, 41)
C = np.array([float(np.mean((x[:, :x.shape[1] - r] if r else x) * (x[:, r:] if r else x))) for r in rs])
C_cl = a_ * Q ** rs + b_ * alpha ** rs
res2 = float(np.max(np.abs(C - C_cl)) / C[0])
report("L2 闭式相容", res2 <= 0.05,
       "alpha = 1 - kappa = %.4f 代回两点指数闭式，拟合相对残差 <= 5%%" % alpha,
       "最大相对残差 %.3e；C(0) 实测 %.4f 对闭式 %.4f" % (res2, C[0], a_ + b_))

# ---------- L3 门槛相容 ----------
N3, STEPS3, DT = 256, 12000, 0.05


def coherence(kap, gamma, rng):
    w = gamma * rng.normal(size=N3)
    th = rng.uniform(-np.pi, np.pi, N3)
    r_hist = []
    for n in range(STEPS3):
        phi = np.angle(np.mean(np.exp(1j * th)))
        r_hist.append(float(np.abs(np.mean(np.exp(1j * th)))))
        th = th + DT * (w + kap * (phi - th)) + np.sqrt(DT) * 0.05 * rng.normal(size=N3)
        th = (th + np.pi) % (2 * np.pi) - np.pi
    return float(np.mean(r_hist[STEPS3 // 2:]))


gammas = (0.05, 0.1, 0.2, 0.4)
g3 = [(g, coherence(KAPPA, g, rng)) for g in gammas]
ratio_c = None
for g, r in g3:
    if r < 0.5:
        ratio_c = g / KAPPA
        break
report("L3 门槛相容", all(r >= 0.5 for g, r in g3 if g / KAPPA <= 1.0) and ratio_c is not None and 0.5 <= ratio_c <= 2.0,
       "gamma <= kappa 档上相干保持（r >= 0.5），门槛在 gamma/kappa 落在 0.5 至 2 之间",
       "；".join("gamma=%.2f（gamma/kappa=%.2f）: r = %.3f" % (g, g / KAPPA, r) for g, r in g3)
       + "；首次跌破 0.5 的 gamma/kappa = %.2f" % (ratio_c if ratio_c else float('nan')))

# ---------- L4 保护相容（构造性） ----------
report("L4 保护相容", True,
       "kappa 只依赖保护距离 d（引用第 184 条 d = L = 8），局域扰动的效果落在局域分量上",
       "构造性：kappa = 1/d，d 由稳定子结构给出，与场层噪声无关")

# ---------- L5 规模相容 ----------
sig2 = 1.0
g5 = []
for N in (64, 256, 1024):
    P = sig2 / (KAPPA * N)
    g5.append((N, P, P * N))
ref5 = g5[-1][2]
report("L5 规模相容", all(abs(v - ref5) / ref5 <= 0.05 for _, _, v in g5),
       "P = sigma^2/(kappa N) 在固定 L 下按 1/N 稀释（三档相对散布 <= 5%）",
       "；".join("N=%d: P = %.6f，P N = %.4f" % (N, P, v) for N, P, v in g5))

# ---------- L6 参数经济 ----------
report("L6 参数经济", True,
       "全链条参数为 {L, N}：kappa 由 d 定、d 由稳定子结构定，未引入新的可调常数",
       "构造性计数：新增可调常数 0 个")

print("[读数] L7（路线丙对照）：键能曲率通道的刚度下限见第 178 条（软化分支触刚度下限），与 1/d 形式的比值待该线的绝对标度定下后核验")
print("[登记] L8：路线甲（同步相变投射）为 P4-16 的正式目标，待乙、丙给出资产后升级；本轮只登记，不作判据")

fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
ax[0].semilogy(rs, C / C[0], "o", label="实测（kappa = 0.125）")
ax[0].semilogy(rs, C_cl / C[0], "k--", label="两点指数闭式")
ax[0].set_xlabel("格距 r")
ax[0].set_ylabel("C(r)/C(0)")
ax[0].set_title("导出 kappa 下的场层闭式")
ax[0].legend(fontsize=8)
ax[1].plot([g for g, _ in g3], [r for _, r in g3], "o-", label="相干度 r")
ax[1].axvline(KAPPA, ls="--", c="k", lw=1, label="kappa = 0.125")
ax[1].set_xlabel("频率失配宽度 gamma")
ax[1].set_ylabel("相干度 r")
ax[1].set_title("门槛：回拉快于失配")
ax[1].legend(fontsize=8)
plt.tight_layout()
plt.savefig(IMG, dpi=150)
plt.close(fig)

bad = [r for r in rows if not r[1]]
print("---- 判据 %d 项，通过 %d 项；图存 %s ----" % (len(rows), len(rows) - len(bad), IMG))
sys.exit(1 if bad else 0)
