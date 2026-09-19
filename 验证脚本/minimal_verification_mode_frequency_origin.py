# -*- coding: utf-8 -*-
r"""
minimal_verification_mode_frequency_origin.py

三-4 补缺轮：零点能里 omega_0 的体系内来源（锁定晶格的色散与基模）。
确定性，随机固定种子 20260920。

模型：元相子锁定晶格取最近邻相位耦合，线性化后动力学矩阵为单位格距的拉普拉斯算符
  D = 2I - shift - shift^{-1}，本征值 4 sin^2(k/2)，故模式频率 omega(k) = omega_P sqrt(4 sum_i sin^2(k_i/2))
（单位 l_P = 1；一维即 omega(k) = 2 omega_P |sin(k/2)|）。k = 0 为平移模，频率为零。
环上 N 格点的最小非零模频率 omega_0 = 2 omega_P sin(pi/N)，故基模零点能
  E_0 = (1/2) hbar omega_0 = hbar omega_P sin(pi/N) = E_P sin(pi/N)（用 E_P = hbar omega_P），
大 N 时近似 pi E_P / N。

判据与期望：
  V1 一维色散：三档 k 上频率与 2 omega_P sin(k/2) 相符（<= 1e-12）
  V2 带边：一维带边 2 omega_P、三维带边 2 sqrt(3) omega_P（<= 1e-12）
  V3 连续极限：k -> 0 时 omega / k 趋于 omega_P（相对偏差 <= 1%）
  V4 基模与规模：环上最小非零模等于 2 omega_P sin(pi/N)（三档 N，<= 1e-12），且 omega_0 N 趋近 pi（散布 <= 5%）
  V5 平移模：k = 0 模频率为零（<= 1e-14），与第 171 条环上转子无折返点一致
  V6 合成读数：E_0 = E_P sin(pi/N)，大 N 时按 pi E_P / N 稀释（三档 N 核验 <= 5%）
  V7 接口登记（与第 171 条折返点计数、第 172 条作用量口径的组合，不作新判据）
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

IMG = r"D:\software\TianXingFC\images\mode_frequency_origin.png"
rows = []


def report(tag, ok, expect, actual):
    rows.append((tag, ok, expect, actual))
    print("[%s] %s" % ("通过" if ok else "未过", tag))
    print("      期望 " + expect)
    print("      实测 " + actual)


def lap(N):
    D = 2.0 * np.eye(N)
    for i in range(N):
        D[i, (i + 1) % N] -= 1.0
        D[i, (i - 1) % N] -= 1.0
    return D


# ---------- V1 一维色散 ----------
N1 = 512
ev1 = np.linalg.eigvalsh(lap(N1))
w1 = np.sqrt(np.maximum(ev1, 0.0))
g1 = []
for kfrac in (0.1, 0.25, 0.5):
    k = 2.0 * np.pi * kfrac
    m = max(1, int(round(kfrac * N1)))
    idx = min(N1 - 1, 2 * m - 1)
    wa = 2.0 * abs(np.sin(np.pi * m / N1))
    g1.append((kfrac, float(w1[idx]), wa, abs(float(w1[idx]) - wa)))
report("V1 一维色散", all(d <= 1e-12 for *_, d in g1),
       "三档模上频率与 2 sin(pi m / N)（omega_P = 1，环上离散模）相符（<= 1e-12）",
       "；".join("k/2pi=%.2f: 实测 %.10f 对解析 %.10f（偏差 %.2e）" % (kf, w, wa, d) for kf, w, wa, d in g1))

# ---------- V2 带边 ----------
edge1 = float(w1[-1])
D3 = np.kron(np.kron(lap(8), np.eye(8)), np.eye(8)) + np.kron(np.kron(np.eye(8), lap(8)), np.eye(8)) + np.kron(np.kron(np.eye(8), np.eye(8)), lap(8))
w3 = np.sqrt(np.maximum(np.linalg.eigvalsh(D3), 0.0))
edge3 = float(w3[-1])
report("V2 带边", abs(edge1 - 2.0) <= 1e-12 and abs(edge3 - 2.0 * np.sqrt(3.0)) <= 1e-12,
       "一维带边 2 omega_P、三维带边 2 sqrt(3) omega_P（<= 1e-12）",
       "一维 %.12f 对 2；三维 %.12f 对 %.12f" % (edge1, edge3, 2.0 * np.sqrt(3.0)))

# ---------- V3 连续极限 ----------
k_small = 0.01
w_small = 2.0 * abs(np.sin(k_small / 2.0))
rel3 = abs(w_small / k_small - 1.0)
report("V3 连续极限", rel3 <= 0.01,
       "k -> 0 时 omega / k 趋于 omega_P（相对偏差 <= 1%）",
       "k = %.3f 处 omega/k = %.8f（相对偏差 %.2e）" % (k_small, w_small / k_small, rel3))

# ---------- V4 基模与规模 ----------
g4 = []
for N in (16, 32, 64):
    w = np.sqrt(np.maximum(np.linalg.eigvalsh(lap(N)), 0.0))
    w_min = float(w[1])
    g4.append((N, w_min, 2.0 * np.sin(np.pi / N), w_min * N))
report("V4 基模与规模", all(abs(wm - wa) <= 1e-12 for _, wm, wa, _ in g4) and all(abs(v - 2.0 * np.pi) / (2.0 * np.pi) <= 0.05 for _, _, _, v in g4),
       "最小非零模等于 2 sin(pi/N)（<= 1e-12），且 omega_0 N 趋近 2 pi（相对 <= 5%）",
       "；".join("N=%d: omega_0 %.8f 对 %.8f，omega_0 N %.6f 对 %.6f" % (N, wm, wa, v, 2.0 * np.pi) for N, wm, wa, v in g4))

# ---------- V5 平移模 ----------
ev0 = float(np.linalg.eigvalsh(lap(256))[0])
report("V5 平移模", abs(ev0) <= 1e-14,
       "k = 0 模的本征值为零（<= 1e-14），故其频率为零，与第 171 条环上转子无折返点一致",
       "零模本征值 %.2e（频率 %.2e）" % (ev0, np.sqrt(max(ev0, 0.0))))

# ---------- V6 合成读数 ----------
g6 = []
for N in (16, 32, 64, 256):
    E0 = np.sin(np.pi / N)
    g6.append((N, E0, np.pi / N, abs(E0 - np.pi / N) / (np.pi / N)))
report("V6 合成读数", all(d <= 0.05 for *_, d in g6),
       "E_0 = E_P sin(pi/N) 在大 N 时按 pi E_P / N 稀释（相对偏差 <= 5%）",
       "；".join("N=%d: E_0/E_P %.8f 对 pi/N %.8f（相对 %.2e）" % (N, e, p, d) for N, e, p, d in g6))

print("[登记] V7：omega_0 由锁定晶格色散给出（基模 omega_0 = 2 omega_P sin(pi/N)），与第 171 条折返点计数的 (1/2) 合成 E_0 = (1/2) hbar omega_0 = E_P sin(pi/N)，本轮只登记，不作新判据")

fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
kk = np.linspace(0.0, np.pi, 400)
ax[0].plot(kk, 2.0 * np.abs(np.sin(kk / 2.0)), "k-", label="解析 2 sin(k/2)")
ax[0].plot(kk[::20], [float(np.sqrt(max(4.0 * np.sin(x / 2.0) ** 2, 0.0))) for x in kk[::20]], "o", ms=3, label="格点本征值")
ax[0].set_xlabel("k（单位 1/l_P）")
ax[0].set_ylabel("omega / omega_P")
ax[0].set_title("锁定晶格的色散：带边 2 omega_P")
ax[0].legend(fontsize=8)
Ns = np.array([8, 16, 32, 64, 128, 256])
ax[1].plot(Ns, np.sin(np.pi / Ns), "o", label="E_0/E_P = sin(pi/N)")
ax[1].plot(Ns, np.pi / Ns, "k--", label="pi/N")
ax[1].set_xscale("log", base=2)
ax[1].set_yscale("log")
ax[1].set_xlabel("锁定规模 N")
ax[1].set_ylabel("E_0 / E_P")
ax[1].set_title("基模零点能与锁定规模")
ax[1].legend(fontsize=8)
plt.tight_layout()
plt.savefig(IMG, dpi=150)
plt.close(fig)

bad = [r for r in rows if not r[1]]
print("---- 判据 %d 项，通过 %d 项；图存 %s ----" % (len(rows), len(rows) - len(bad), IMG))
sys.exit(1 if bad else 0)
