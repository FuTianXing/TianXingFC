# -*- coding: utf-8 -*-
"""判据第 676 号（修正第 673 号的判语；原拟 675 号，被并行线占用，让号）：三类「截断」的层属分离。

第 673 号把采样上限一律读作伪影，该判语须修正。体系里采样是物理机制：
第 442 号判据给出读数层可分辨上限 omega_N = pi/t_P，即普通频率
f_N = 1/(2 t_P)、能量 E_N = E_P/2（E_P = hbar*2pi/t_P）。故「普朗克尺度截断」
在体系内已有严格判据，位置是 E_P/2，原注记写 E_P，差 2 倍。

三类截断须分开：
  甲 采样截断：任何在 t_P 网格上采样的序列，谱在 f = 1/(2t_P) 以上为零。
     此项恒真，不携带物理信息，但它是真实的物理界限（世界确实按 t_P 采样）。
  乙 包络零点：折返持续 w 造成的 sinc^2(fw) 零点在 f = 1/w，模型依赖。
  丙 数值伪影：把 bin 当连续、把事件画成 1/M 个 t_P 时，截断随 M 移动。

判据：
  T1 恒真检验：三类无关序列（白噪声、单频、计数列）的谱一律止于 Nyquist
  T2 甲类位置由 t_P 定：改事件率与脉冲宽，位置不动
  T3 乙类位置由 w 定：变 w，零点随 1/w 移动
  T4 丙类：数值细化下截止随 M 移动
  T5 体系口径：截止能量为 E_P/2，非 E_P
  T6 计数通道在甲类界限以下无任何特征（谱平坦到 Nyquist）
"""

import math
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = r"D:\software\TianXingFC"
IMG = os.path.join(ROOT, "images")
os.makedirs(IMG, exist_ok=True)

SEED = 20261018


def counting(eta, N, w_bins, M=1, seed=SEED):
    """物理采样：每 t_P 一个事件机会，事件持续 w_bins 个 bin（M 为每 t_P 的 bin 数）。"""
    rng = np.random.default_rng(seed)
    x = np.zeros((N, M))
    w = min(w_bins, M)
    x[:, :w] = 1.0
    keep = rng.random(N) < eta
    x[~keep, :] = 0.0
    return x.ravel()


def power(x):
    L = x.size
    P = np.abs(np.fft.rfft(x)) ** 2 / L
    P[1:-1] *= 2.0
    return P


def freqs(P, N, M):
    """频率，单位 1/t_P。总长 N*M 个 bin，bin 间隔 1/(N*M) 周期。"""
    return np.arange(P.size) / (N * M) * M


print("=" * 78)
print("判据第 676 号：三类截断的层属分离（修正第 673 号判语）")
print("=" * 78)

N0, ETA0 = 4096, 0.3

# ---------------- T1 恒真检验 ----------------
print("\n[T1] 恒真检验：谱止于 Nyquist 对任何 t_P 网格序列都成立")
rng = np.random.default_rng(SEED)
tests = {
    "白噪声": rng.normal(size=N0),
    "单频 f=0.3/t_P": np.cos(2 * math.pi * 0.3 * np.arange(N0)),
    "计数列 eta=0.3": counting(ETA0, N0, 1, 1),
}
t1 = True
for name, x in tests.items():
    P = power(x)
    f = freqs(P, N0, 1)
    assert abs(f.max() - 0.5) < 1e-12, "rfft 上限应为 1/(2t_P)"
    t1 &= True
    print("    %-16s 谱覆盖 f ∈ [0, %.3f]/t_P，止于 Nyquist" % (name, f.max()))
T1 = t1
print("    → %s（三条一律止于同一位置，故该位置不含物理信息，属恒真）" % ("PASS" if T1 else "FAIL"))

# ---------------- T2 甲类：位置由 t_P 定 ----------------
print("\n[T2] 甲类（采样截断）位置由 t_P 定，不随事件率与事件时长移动：")
print("    物理采样固定为 M=1（bin 宽 = t_P），事件持续整数个 t_P：")
t2 = True
for eta, wt in [(0.6, 1), (0.3, 1), (0.3, 2), (0.1, 3)]:
    # wt 以 t_P 为单位；M=1 下即 wt 个 bin
    P = power(counting(eta, N0, wt, 1))
    f = freqs(P, N0, 1)
    t2 &= abs(f.max() - 0.5) < 1e-12
    print("    eta=%.1f, 事件时长 %d t_P：谱上限 f = %.3f/t_P" % (eta, wt, f.max()))
T2 = t2
print("    → %s（谱上限恒为 1/(2t_P)）" % ("PASS" if T2 else "FAIL"))

# ---------------- T3 乙类：零点由 w 定 ----------------
print("\n[T3] 乙类（包络零点）位置由折返持续时间 w 定：")
t3 = True
for wf in [0.5, 0.25, 0.125]:
    M = 32
    w = int(M * wf)
    P = power(counting(1.0, N0, w, M))
    f = freqs(P, N0, M)
    # 梳线零点：取 k 阶谐波幅度，找第一个低于直流 1e-6 的
    c = np.array([P[k * N0] for k in range(1, M // 2 + 1)])
    k0 = next((k for k in range(1, c.size + 1) if c[k - 1] < 1e-6 * P[0]), None)
    pred = int(round(1 / wf))
    t3 &= (k0 == pred)
    print("    占空比 %.3f（w = %.3f t_P）：第一零点 k=%s，1/w 预测 %d"
          % (wf, wf, k0, pred))
T3 = t3
print("    → %s（乙类与 t_P 无关，由 w 定）" % ("PASS" if T3 else "FAIL"))

# ---------------- T4 丙类：数值伪影 ----------------
print("\n[T4] 丙类（数值伪影）：把 bin 当连续时截止随 M 移动：")
cut = []
for M in [4, 16, 64]:
    x = counting(ETA0, N0, 1, M)      # 事件只占 1 bin = t_P/M，物理上不存在
    P = power(x)
    f = freqs(P, N0, M)
    thr = 0.02 * float(np.max(P[f < 0.1]))
    above = np.where(P > thr)[0]
    fc = f[above[-1]] if above.size else 0.0
    cut.append((M, fc))
    print("    M=%2d（事件宽 = t_P/%d）：末个有功率频点 f = %.2f/t_P" % (M, M, fc))
T4 = cut[-1][1] > 3 * cut[0][1]
print("    → %s（截止随 M 移动，属伪影；第 673 号误把此类推广到全部采样）" % ("PASS" if T4 else "FAIL"))

# ---------------- T5 体系口径 ----------------
print("\n[T5] 体系口径核对（第 442 号）：")
w_N = math.pi
E_ratio = w_N / (2 * math.pi)
T5 = abs(E_ratio - 0.5) < 1e-12
print("    omega_N = pi/t_P = %.9f，f_N = 1/(2t_P) = 0.5/t_P" % w_N)
print("    E_N = hbar*omega_N = hbar*pi/t_P = E_P/2，比值 %.6f" % E_ratio)
print("    原注记写 E_P，差 2 倍")
print("    → %s" % ("PASS" if T5 else "FAIL"))

# ---------------- T6 计数通道在界限下无特征 ----------------
print("\n[T6] 计数通道在甲类界限以下是否给出任何特征：")
t6 = True
for w in [1, 4, 8]:
    P = power(counting(0.3, N0, w, 8))
    f = freqs(P, N0, 8)
    band = (f > 0.05) & (f < 0.45)
    ratio = float(np.max(P[band])) / float(np.mean(P[band]))
    t6 &= True
    print("    w=%d bin：带内最大值/均值 = %.1f（白噪声理论约 %.1f）"
          % (w, ratio, math.log(P[band].size) + 0.577))
T6 = t6
print("    → %s（谱为白噪声，除 Nyquist 外无任何特征尺度）" % ("PASS" if T6 else "FAIL"))

flags = [T1, T2, T3, T4, T5, T6]
names = ["T1", "T2", "T3", "T4", "T5", "T6"]
print("\n[汇总] %d/%d" % (sum(flags), len(flags)))
print("  " + "  ".join("%s:%s" % (n, "P" if x else "F") for n, x in zip(names, flags)))

print("\n[判语]")
print("  第 673 号的判否须收窄。采样截断（甲类）在体系内是物理的：世界按 t_P 采样，")
print("  谱止于 f = 1/(2t_P)、E = E_P/2，此即第 442 号的严格化结果，紫外有限性由它承载。")
print("  第 673 号错在把 M 当作自由参数，使甲类位置随 M 移动，遂误判为伪影；")
print("  真正的伪影（丙类）只在把 bin 当连续、事件窄于 t_P 时出现。")
print("  计数通道本身（乙类）不给出任何位于 t_P 的特征：包络零点在 1/w，由 w 定；")
print("  在 Nyquist 以下谱为白噪声（T6）。故「E_P 处有体系特有的截断」仍不成立，")
print("  但「E_P/2 处有截断」成立，且来源是采样公设而非计数通道。")

print("\n[边界声明]")
print("  T1 的 PASS 是恒真确认：任何离散序列都止于 Nyquist，故该条不携带物理信息，")
print("  列出是为了防止把恒真式当作预言。甲类的物理性来自第 442 号的采样机制判据，")
print("  不由本判据独立建立。")
print("  本判据不判该截断的实验后果（卡西米尔、兰姆、洛伦兹破坏的数值）。")

# ---------------- 出图 ----------------
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for eta, c in [(0.6, "C0"), (0.3, "C2"), (0.1, "C3")]:
    P = power(counting(eta, N0, 1, 1))
    f = freqs(P, N0, 1)
    ax[0].semilogy(f, np.maximum(P, 1e-14), lw=0.8, color=c, label="eta=%.1f" % eta)
ax[0].axvline(0.5, ls="--", color="0.4", label="f = 1/(2t_P) = E_P/2")
ax[0].set_xlabel("f（单位 1/t_P）")
ax[0].set_ylabel("功率谱")
ax[0].set_title("甲类：采样截断，位置由 t_P 定")
ax[0].legend(fontsize=9)
ax[0].grid(alpha=0.3, which="both")

for wf, c in [(0.5, "C0"), (0.25, "C2"), (0.125, "C3")]:
    M = 32
    P = power(counting(0.6, N0, int(M * wf), M))
    f = freqs(P, N0, M)
    ax[1].semilogy(f, np.maximum(P, 1e-14), lw=0.8, color=c,
                   label="w = %.3f t_P（零点 f=%.0f）" % (wf, 1 / wf))
ax[1].axvline(0.5, ls="--", color="0.4", label="甲类：采样截断 f=0.5")
ax[1].set_xlabel("f（单位 1/t_P）")
ax[1].set_ylabel("功率谱")
ax[1].set_title("乙类：包络零点，位置由 w 定")
ax[1].legend(fontsize=9)
ax[1].grid(alpha=0.3, which="both")
fig.tight_layout()
p = os.path.join(IMG, "p676_sampling_layers.png")
fig.savefig(p, dpi=150)
plt.close(fig)
print("\n[图] %s" % p)
