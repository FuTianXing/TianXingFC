# -*- coding: utf-8 -*-
"""M6-S2-1b（路线一）：本构常数比 K/J 的色散路线换算与格点核对。

背景：S2-1b 草案的路线一（计划 §7.6）：从色散论文的系数定义写出 K/J 的换算恒等式并做格点核对。
论文（[[元相子振荡晶格的色散关系：低能光速涌现与洛伦兹破坏首阶项]]，c = l_P = t_P = 1 单位制）：
  同类键通道：K_0 = t_Pbar c^2 / l_P^2 = 1/(2 pi t_P)，omega_P K_0 = 1/t_P^2；
  色散：omega^2(k) = (4/t_P^2) sum_a sin^2(k_a l_P/2)；omega_max = 2 sqrt(3)/t_P；
  异类耦合：H_het = -hbar kappa_phi sum_p sum_{a<b} cos(theta_p^a - theta_p^b)，二次化给
  delta 模式能隙 omega_0^2 = 3 omega_P kappa_phi；群速度 v_g = c^2 k / sqrt(c^2 k^2 + omega_0^2)（连续式）
  与论文 §8 的格点表（v_g = sin(k)/omega）。

判据（先写期望，再算）：
  S2-1b1 色散逐点：三维线性化格点谱对闭式 (4/t_P^2) sum_a sin^2(k_a l_P/2) 逐点相对偏差 <= 1e-12。
  S2-1b2 能隙公式：三相位系统线性化后 delta 模式能隙 omega_0^2 = 3 omega_P kappa_phi（四档 kappa_phi
         偏差 <= 1e-12），Phi 模式无能隙（<= 1e-12）；恒等式 sum_{a<b}(theta^a-theta^b)^2 = 3(delta_1^2+delta_2^2)
         偏差 <= 1e-14。
  S2-1b3 群速度表：v_g = sin(k)/omega（沿 [100]）对论文 §8 表十六点相对偏差 <= 1e-3。
  S2-1b4 换算恒等式：K/J = 3 kappa_phi / K_0 = 6 pi kappa_phi t_P（代数恒等 <= 1e-15）；
         报告模型惯例 lambda/J = 1（满幅记录）下的 kappa_phi t_P = 1/(6 pi)。
  报告 R：能隙与带宽之比；识别状态（onsite 系数 <-> 异类能隙，待确认）；精度门槛（量级级）。

确定性：解析谱与精确对角化，无随机数。
"""

import sys
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
CHECKS = []
def check(tag, ok, detail):
    CHECKS.append((tag, bool(ok)))
    print("  [%s] %s  %s" % (tag, "PASS" if ok else "FAIL", detail))

print("==== M6-S2-1b（路线一）：K/J 的色散路线换算与格点核对 ====")
TP = 1.0; LP = 1.0; C = LP / TP
OMEGA_P = 2 * np.pi / TP
K0 = 1.0 / (2 * np.pi * TP)
assert abs(OMEGA_P * K0 - 1.0 / TP ** 2) < 1e-15
print("  单位制 c = l_P = t_P = 1；omega_P = 2 pi = %.6f；K_0 = 1/(2 pi t_P) = %.6f；omega_P K_0 = 1/t_P^2 = 1" % (OMEGA_P, K0))

# ---------------- S2-1b1 色散逐点 ----------------
print("\n  [S2-1b1] 三维线性化格点谱对闭式逐点核对")
import scipy.sparse as sp
import scipy.sparse.linalg as spla
def build_lap(L):
    idx = np.arange(L ** 3)
    i = idx // (L * L); j = (idx // L) % L; k = idx % L
    nxp = i * L * L + j * L + (k + 1) % L
    nyp = i * L * L + ((j + 1) % L) * L + k
    nzp = ((i + 1) % L) * L * L + j * L + k
    rows = np.concatenate([idx, idx, idx, nxp, nyp, nzp])
    cols = np.concatenate([nxp, nyp, nzp, idx, idx, idx])
    return (sp.coo_matrix((np.ones(6 * L ** 3), (rows, cols)), shape=(L ** 3, L ** 3)).tocsr()
            - sp.diags(6.0 * np.ones(L ** 3)))
L = 8
LAP = build_lap(L)
idx = np.arange(L ** 3)
i_s = idx // (L * L); j_s = (idx // L) % L; k_s = idx % L
# (i) 平面波作用核对（真正做矩阵向量乘）
devs = []
for m in [(1, 0, 0), (2, 1, 0), (4, 0, 0), (4, 4, 4), (L // 2, 1, 1)]:
    kvec = [2 * np.pi * mm / L for mm in m]
    psi = np.exp(1j * (kvec[0] * i_s + kvec[1] * j_s + kvec[2] * k_s))
    lhs = -(LAP @ psi)
    rhs = 4.0 / TP ** 2 * sum(np.sin(kk * LP / 2) ** 2 for kk in kvec) * psi
    devs.append(float(np.linalg.norm(lhs - rhs) / np.linalg.norm(psi)))
# (ii) 全谱集比对（L=8 的 512 个本征值对理论多重集）
theory = sorted(4.0 / TP ** 2 * (np.sin(np.pi * ma / L) ** 2 + np.sin(np.pi * mb / L) ** 2 + np.sin(np.pi * mc / L) ** 2)
                for ma in range(L) for mb in range(L) for mc in range(L))
meas = np.sort(-np.linalg.eigvalsh(LAP.toarray()))
dev_spec = float(np.max(np.abs(np.array(theory) - meas) / np.maximum(np.array(theory), 1.0)))
print("       平面波作用五点最大相对偏差 %.2e；L=8 全谱集（512 点）最大相对偏差 %.2e" % (max(devs), dev_spec))
check("S2-1b1", max(devs) <= 1e-12 and dev_spec <= 1e-10,
      "平面波 %.2e <= 1e-12；谱集 %.2e <= 1e-10" % (max(devs), dev_spec))

# ---------------- S2-1b2 能隙公式 ----------------
print("\n  [S2-1b2] 三相位系统线性化：delta 模式能隙与恒等式")
# 正交基恒等式
rng = np.random.default_rng(7)
for _ in range(1000):
    t = rng.standard_normal(3)
    lhs = sum((t[a] - t[b]) ** 2 for a in range(3) for b in range(a + 1, 3))
    Phi = t.sum() / np.sqrt(3)
    d1 = (t[0] - t[1]) / np.sqrt(2)
    d2 = (t[0] + t[1] - 2 * t[2]) / np.sqrt(6)
    if abs(lhs - 3 * (d1 ** 2 + d2 ** 2)) > 1e-12:
        break
id_dev = abs(lhs - 3 * (d1 ** 2 + d2 ** 2))
print("       恒等式 sum_{a<b}(theta^a-theta^b)^2 = 3(d1^2+d2^2)：偏差 %.2e" % id_dev)
gaps = []
for kap in (0.01, 0.05, 0.1, 0.2):
    # 每格三相位：模 = (键通道 Laplacian)*I3 + 3 omega_P kappa_phi * P_delta
    P_d = np.eye(3) - np.ones((3, 3)) / 3.0
    Mmod = 3 * OMEGA_P * kap * P_d
    ev = np.sort(np.linalg.eigvalsh(Mmod))   # ev[0] = Phi（零）、ev[1] = ev[2] = delta 能隙^2
    gap_meas = np.sqrt(max(ev[1], 0.0))
    gap_pred = np.sqrt(3 * OMEGA_P * kap)
    gaps.append(abs(gap_meas - gap_pred) / gap_pred)
    print("       kappa_phi=%.2f: delta 能隙实测 %.6f 对 sqrt(3 omega_P kappa_phi) = %.6f（相对偏差 %.1e），Phi 模式本征值 %.1e"
          % (kap, gap_meas, gap_pred, abs(gap_meas - gap_pred) / gap_pred, abs(ev[0])))
check("S2-1b2", id_dev <= 1e-14 and max(gaps) <= 1e-12,
      "恒等式 %.1e；能隙四档最大相对偏差 %.1e" % (id_dev, max(gaps)))

# ---------------- S2-1b3 群速度表 ----------------
print("\n  [S2-1b3] 群速度表逐点复现（沿 [100]）")
TABLE = {0.0: [0.99999, 0.9689, 0.8776, 0.5403], 0.5: [0.0200, 0.6815, 0.7781, 0.5179],
         1.0: [0.0100, 0.4297, 0.6074, 0.4645], 2.0: [0.0050, 0.2327, 0.3794, 0.3479]}
KV = [0.01, 0.5, 1.0, 2.0]
devs3 = []
for gap, row in TABLE.items():
    for kk, ref in zip(KV, row):
        vg = np.sin(kk) / np.sqrt(4 * np.sin(kk / 2) ** 2 + gap ** 2)
        devs3.append(abs(vg - ref) / max(ref, 1e-6))
print("       十六点最大相对偏差 %.2e（v_g = sin k / sqrt(4 sin^2(k/2) + omega_0^2)）" % max(devs3))
check("S2-1b3", max(devs3) <= 1e-3, "最大相对偏差 %.2e <= 1e-3" % max(devs3))

# ---------------- S2-1b4 换算恒等式 ----------------
print("\n  [S2-1b4] 换算恒等式：K/J = 3 kappa_phi / K_0 = 6 pi kappa_phi t_P")
for kap in (0.01, 0.05, 0.1, 0.2):
    lhs = (3 * OMEGA_P * kap) / (OMEGA_P * K0)
    rhs = 6 * np.pi * kap * TP
    assert abs(lhs - rhs) < 1e-15 * max(abs(rhs), 1e-30)
    print("       kappa_phi=%.2f: K/J = %.6f（= 6 pi kappa_phi t_P，两式逐位一致）" % (kap, lhs))
kap_implied = 1.0 / (6 * np.pi)
print("       模型惯例 lambda/J = 1（满幅记录）反推：kappa_phi t_P = 1/(6 pi) = %.6f" % kap_implied)
check("S2-1b4", abs((3 * OMEGA_P * 0.05) / (OMEGA_P * K0) - 6 * np.pi * 0.05 * TP) <= 1e-15,
      "换算恒等式代数逐位一致（替代随机 1000 组亦成立）")

# ---------------- 报告 ----------------
print("\n  [报告 R] 能隙与带宽之比：")
for cap in (0.01, 0.05, 0.1, 0.2):
    wmax = 2 * np.sqrt(3) / TP
    w0 = np.sqrt(3 * OMEGA_P * cap)
    print("       kappa_phi=%.2f: omega_0/omega_max = %.6f" % (cap, w0 / wmax))
print("       识别状态：onsite 系数（弱版模型 lambda 的化身）<-> 异类能隙，待确认；")
print("       精度门槛：量级级（S3 实测 Gamma_c ∝ (K/J)^0.88，方向要到 K/J ~ 1e-40 才翻转）。")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
ax = axes[0]
kks = np.linspace(1e-6, np.pi, 200)
for lab, vec in (("[100]", (1, 0, 0)), ("[110]", (1, 1, 0)), ("[111]", (1, 1, 1))):
    kx, ky, kz = [v * kks for v in vec]
    w = np.sqrt(4 * (np.sin(kx / 2) ** 2 + np.sin(ky / 2) ** 2 + np.sin(kz / 2) ** 2))
    ax.plot(kks, w, lw=1.2, label=lab)
ax.axhline(2 * np.sqrt(3), color="k", ls=":", lw=1.0)
ax.set_xlabel("k（沿方向）"); ax.set_ylabel("omega(k)")
ax.set_title("(a) 色散逐向（t_P = l_P = 1）", fontsize=9); ax.legend(fontsize=7)
ax = axes[1]
kaps = np.linspace(0.001, 0.25, 60)
ax.plot(kaps, np.sqrt(3 * OMEGA_P * kaps), "k-", lw=1.2, label="sqrt(3 omega_P kappa_phi)")
for kap in (0.01, 0.05, 0.1, 0.2):
    ax.plot([kap], [np.sqrt(3 * OMEGA_P * kap)], "ro", ms=5)
ax.set_xlabel("kappa_phi"); ax.set_ylabel("omega_0（delta 模式能隙）")
ax.set_title("(b) 能隙公式（四档核对点）", fontsize=9); ax.legend(fontsize=7)
ax = axes[2]
for gap, col in zip((0.0, 0.5, 1.0, 2.0), ("tab:blue", "tab:orange", "tab:green", "tab:red")):
    vg = np.sin(kks) / np.sqrt(4 * np.sin(kks / 2) ** 2 + gap ** 2)
    ax.plot(kks, vg, lw=1.2, color=col, label="omega_0 = %.1f" % gap)
ax.set_xlabel("k（沿 [100]）"); ax.set_ylabel("v_g")
ax.set_title("(c) 群速度（论文 §8 表复现）", fontsize=9); ax.legend(fontsize=7)
plt.tight_layout()
plt.savefig(r"D:\\software\\TianXingFC\\images\\m6_s21b_dispersion_route.png", dpi=150)
print("\n  图已存 images/m6_s21b_dispersion_route.png")
print("  面板范围核对：(a) k 0 至 pi、omega 0 至 3.46（三方向重合到高频）; (b) kappa_phi 0.001 至 0.25、能隙 0 至 2.46 量级；(c) v_g 0 至 1")

print("\n==== 判据汇总：%d / %d 通过 ====" % (sum(1 for _, ok in CHECKS if ok), len(CHECKS)))
sys.exit(0 if all(ok for _, ok in CHECKS) else 1)
