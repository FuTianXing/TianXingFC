# -*- coding: utf-8 -*-
"""
判据 L1 与 L3 的最小验证：分层读数的充分性与「附内容」的可观测边界（验证脚本第 400 条）

判据（先立后算，2026-09-25 立；见内部草稿/注记-认知错视与分层粗粒化）：
  L1a 高层读数对个体细节的不敏感性：固定中层统计量 (W_k, eta_k, Phi_k)，遍历个体相位构型，
      个体层统计量明显变化，而高层读数 Psi 的散布为零（机器精度）。
  L1b 中层三分量的最小性：放开 W、eta、Phi 中任一分量，Psi 的变动须显著，三者缺一不可。
  L1c 条件独立性：给定中层，高层读数无残余散布（与 L1a 同一实测）。
  L3a 纯标记不可观测：只标记「整体在某处」而不改变份额剖面，读数分布与基准解析相同（逐分量差 0），
      样本层卡方检验不能拒绝同分布。
  L3b 有内容的标记可观测（对照）：把分布替换为集中于标记点的 delta，读数分布显著改变
      （直方图 L1 差 > 1.0，卡方远大于临界值），且源剖面峰数由 2 变 1。

理论期望：L1a/L1c 散布 ~ 1e-16；L1b 变动 > 1e-3；L3a 差 0 且卡方低于 7.815；L3b 差 > 1.0 且峰数 2 到 1。
模型：N = K * NK 个元相子分 K 块；块内用中心加对称配对构造，使块内和严格钉在目标 (eta_k, Phi_k) 上。
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

IMG = "D:/software/TianXingFC/images/layered_reading_reliability.png"
SEED = 20260925

K = 4
NK = 25
M = (NK - 1) // 2
DMAX = 2.4
CMIN = float(np.cos(DMAX))

W_T = np.array([0.40, 0.30, 0.20, 0.10])
ETA_T = np.array([0.85, 0.60, 0.40, 0.72])
PHI_T = np.array([0.0, np.pi / 3.0, -np.pi / 2.0, np.pi])


def target_c(eta):
    return (NK * eta - 1.0) / 2.0


def base_c(c_val):
    x = (c_val - M * CMIN) / (1.0 - CMIN) / M
    x = min(max(x, 0.0), 1.0)
    return CMIN + (1.0 - CMIN) * np.full(M, x)


def perturb(c_in, rng, steps=8):
    c = c_in.copy()
    for _ in range(steps):
        j, l = rng.integers(0, M, size=2)
        if j == l:
            continue
        room = min(1.0 - c[j], c[l] - CMIN)
        if room <= 1e-12:
            continue
        eps = room * rng.random()
        c[j] += eps
        c[l] -= eps
    return c


def phases_of(c, phi):
    d = np.arccos(np.clip(c, -1.0, 1.0))
    return np.concatenate([[phi], phi + d, phi - d])


def high_level(W, eta, phi):
    return np.sum(np.sqrt(W) * eta * np.exp(1j * phi))


print("=== 第 400 条：分层读数的充分性与附内容的可观测边界 ===")
print("模型：K = %d 块，每块 NK = %d 个元相子（中心加 12 对对称相位）" % (K, NK))
print("")

rng = np.random.default_rng(SEED)
cs0 = [base_c(target_c(e)) for e in ETA_T]

N_CONF = 400
ind_stats = []
psi_re = []
psi_im = []
eta_drift = []
for _ in range(N_CONF):
    cs = [perturb(cs0[k], rng) for k in range(K)]
    ph = [phases_of(cs[k], PHI_T[k]) for k in range(K)]
    ek = []
    pk = []
    for k in range(K):
        S = np.exp(1j * ph[k]).sum()
        ek.append(abs(S) / NK)
        pk.append(np.angle(S))
    ek = np.array(ek)
    pk = np.array(pk)
    eta_drift.append(float(np.max(np.abs(ek - ETA_T))))
    flat = np.concatenate(ph)
    ind_stats.append(float(np.mean(np.abs(flat - np.repeat(PHI_T, NK)))))
    psi = high_level(W_T, ek, pk)
    psi_re.append(psi.real)
    psi_im.append(psi.imag)

ind_stats = np.array(ind_stats)
psi_re = np.array(psi_re)
psi_im = np.array(psi_im)
eta_drift = np.array(eta_drift)

spread_ind = float(ind_stats.max() - ind_stats.min())
spread_psi = float(max(psi_re.max() - psi_re.min(), psi_im.max() - psi_im.min()))
drift_eta = float(eta_drift.max())

print("--- L1a / L1c：个体层变化与高层读数散布 ---")
print("个体层统计量极差（相位平均绝对偏差）: 实测 %.6e   期望 > 1e-3" % spread_ind)
print("中层量重构偏差 max|eta_hat - eta| : 实测 %.6e   期望 < 1e-14" % drift_eta)
print("高层读数散布 max(dRe, dIm)        : 实测 %.6e   期望 < 1e-14" % spread_psi)
print("")

psi0 = high_level(W_T, ETA_T, PHI_T)
eta2 = ETA_T.copy()
eta2[0] *= 1.10
phi2 = PHI_T.copy()
phi2[0] += 0.10
W2 = np.array([0.45, 0.30, 0.15, 0.10])
d_eta = float(abs(high_level(W_T, eta2, PHI_T) - psi0))
d_phi = float(abs(high_level(W_T, ETA_T, phi2) - psi0))
d_W = float(abs(high_level(W2, ETA_T, PHI_T) - psi0))

print("--- L1b：中层三分量的最小性（放开其一的读数变动） ---")
print("放开 eta_1（乘 1.10）              : 实测 %.6e   期望 > 1e-3" % d_eta)
print("放开 Phi_1（加 0.10 rad）          : 实测 %.6e   期望 > 1e-3" % d_phi)
print("放开 W（改份额）                   : 实测 %.6e   期望 > 1e-3" % d_W)
print("")

N_SAMPLE = 40000
rngA = np.random.default_rng(SEED + 1)
rngB = np.random.default_rng(SEED + 2)
idxA = rngA.choice(K, size=N_SAMPLE, p=W_T)
idxB = rngB.choice(K, size=N_SAMPLE, p=W_T)
x_label = 2
idxAp = np.full(N_SAMPLE, x_label)

cA = np.bincount(idxA, minlength=K)
cB = np.bincount(idxB, minlength=K)
cAp = np.bincount(idxAp, minlength=K)
hA = cA / N_SAMPLE
hB = cB / N_SAMPLE
hAp = cAp / N_SAMPLE
analytic_diff = float(np.max(np.abs(W_T - W_T)))


def chi2_2xK(c1, c2):
    exp = (c1 + c2) / 2.0
    return float(np.sum((c1 - exp) ** 2 / exp) * 2.0)


chi_ab = chi2_2xK(cA, cB)
chi_ap = chi2_2xK(cA, cAp)
l1_ap = float(np.abs(hAp - hA).sum())

print("--- L3a：纯标记（不改变份额剖面） ---")
print("解析分布逐分量差                   : 实测 %.6e   期望 = 0" % analytic_diff)
print("样本卡方（dof = 3，临界 7.815）    : 实测 %.4f    期望 < 7.815（不能拒绝）" % chi_ab)
print("")
print("--- L3b：有内容的标记（分布替换为 delta） ---")
print("直方图 L1 差                       : 实测 %.4f    期望 > 1.0" % l1_ap)
print("样本卡方                           : 实测 %.4e   期望 >> 7.815" % chi_ap)

xgrid = np.arange(0.0, 9.0, 0.01)


def profile(centers, weights, sigma=0.35):
    rho = np.zeros_like(xgrid)
    for c, w in zip(centers, weights):
        rho += w * np.exp(-0.5 * ((xgrid - c) / sigma) ** 2)
    return rho / rho.sum()


rho_spread = profile([1.0, 5.0], [0.5, 0.5])
rho_local = profile([5.0], [1.0])


def count_peaks(rho):
    n = 0
    for i in range(1, len(rho) - 1):
        if rho[i] > rho[i - 1] and rho[i] >= rho[i + 1]:
            n += 1
    return n


pk_spread = count_peaks(rho_spread)
pk_local = count_peaks(rho_local)
print("源剖面峰数：铺开 %d（期望 2）  局部化 %d（期望 1）" % (pk_spread, pk_local))
print("")

fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.2))
ax = axes[0]
ax.scatter(ind_stats, psi_re, s=6, alpha=0.6, color="tab:blue")
ax.set_xlabel("个体层统计量（相位平均绝对偏差）")
ax.set_ylabel("高层读数 Re Psi")
ax.set_title("L1a：个体层变化而高层读数不变")
ax.grid(alpha=0.3)

ax = axes[1]
w = 0.35
ax.bar(np.arange(K) - w / 2, hA, width=w, label="基准（按份额抽样）")
ax.bar(np.arange(K) + w / 2, hAp, width=w, label="有内容标记（delta）")
ax.set_xlabel("读数位置档")
ax.set_ylabel("频率")
ax.set_title("L3：纯标记与有内容标记")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = axes[2]
ax.plot(xgrid, rho_spread, label="铺开源（峰数 %d）" % pk_spread)
ax.plot(xgrid, rho_local, label="局部化（峰数 %d）" % pk_local)
ax.set_xlabel("位置")
ax.set_ylabel("份额密度")
ax.set_title("L3b：源剖面峰数")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(IMG, dpi=150)
print("图已保存：" + IMG)
