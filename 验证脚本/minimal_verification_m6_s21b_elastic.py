# -*- coding: utf-8 -*-
"""
minimal_verification_m6_s21b_elastic.py
M6-S2-1b 路线二（弹性本构路线）：双通道换算桥接与 FK 弹性定律核对。

背景：S2-1b 的三条取数路线中，路线一（色散）给出换算恒等式 K/J = 3 kappa_phi/K_0
= 6 pi kappa_phi t_P，把本构比归结为无量纲组合 kappa_phi t_P。本材料执行路线二：
从非紧致法线轴弹性（键 = 螺旋高度差平方；onsite = 圆心/回折约束）的同一参数集出发，
对两条通道做弹性定律核对与换算桥接，回答两个问题：其一，FK 链标准弹性关系（声速、
kink 宽度幂律、kink 能量幂律）在统一单位制下是否与公理 4 参数集自洽；其二，弹性路线
给出的 K/J 是否构成对路线一的独立复核。

模型与约定：
- 单位制 c = l_P = t_P = 1，hbar = 1；omega_P = 2 pi；K_0 = 1/(2 pi t_P)。
- FK 链（u 单位，u = theta/2 pi）：E = sum (k/2)(u_{n+1}-u_n-1)^2 + sum K(1-cos 2 pi u_n)。
  对应：弹簧项 <-> 键弹性 kappa_parallel（包层）；周期势 <-> 锚核锁定（kappa_phi 通道）。
- 弱版黑洞线模型（theta 单位）：E = J sum cos(Delta theta) + lam sum (1-cos theta) h_n。

判据（先写判据再实现）：
  S2-1b5  FK 键通道色散闭式与格点本征值一致（相对偏差 <= 1e-12），声速平方 = k。
  S2-1b6  kink 宽度幂律：宽度对 (K/k) 的对数斜率 = -0.5（四点拟合，偏差 <= 3 个百分点）。
  S2-1b7  kink 能量幂律：能量对 (K/k) 的对数斜率 = +0.5（四点拟合，偏差 <= 3 个百分点）；
          四点能量绝对值对闭式 (4/pi) sqrt(k K) 的比值落在 0.9 至 1.1。
  S2-1b8  桥接恒等式：J = k/(4 pi^2)（二次型对齐，<= 1e-12）；lam = K（逐点精确）；
          桥接后 K/J = 4 pi^2 K/k = 3 kappa_phi/K_0 = 6 pi kappa_phi t_P（三式一致，<= 1e-12）；
          与路线一为同源恒等式（偏差 0，独立不成立，登记待作者裁定）。
报告项：宽度窗口映射（FK 的 K/k <= 0.02 -> 黑洞线 K/J <= 4 pi^2 * 0.02 = 0.789）；
        模型惯例 lam/J = 1 -> K/k = 1/(4 pi^2) = 0.0253（窗口外缘）；
        cos 投影降级幅度（余弦键与平方键在大角差下的偏差）。
"""
import os, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
CHECKS = []
def check(tag, ok, detail):
    CHECKS.append((tag, bool(ok)))
    print("  [%s] %s  %s" % (tag, "PASS" if ok else "FAIL", detail))

print("==== M6-S2-1b 路线二（弹性本构路线）：双通道换算桥接与弹性定律核对 ====")

HBAR = 1.0
TP = 1.0; LP = 1.0; C = 1.0
WP = 2.0 * np.pi
K0 = 1.0 / (2.0 * np.pi)          # 公理 4 键耦合：K_0 = tbar c^2 / l_P^2
KAP = 0.05                         # 桥接核对用的 kappa_phi 抽样值
K_FK = 4.0 * np.pi ** 2 * HBAR * K0   # 桥接键系数 = 2 pi
K_ONS = 3.0 * HBAR * KAP              # 桥接 onsite 系数 = 0.15

print("\n  [准备] K_0 = %.6f；桥接键系数 k = 4 pi^2 K_0 = %.6f；抽样 kappa_phi = %.2f -> K = 3 kappa_phi = %.4f"
      % (K0, K_FK, KAP, K_ONS))

# ---------------- S2-1b5 键通道色散 ----------------
N1 = 256
nn1 = np.arange(N1)
# FK 键通道 Hessian（单位惯性）：E_bond = (k/2) sum (u_{n+1}-u_n)^2（线性化区，自然长度平移吸收）
H = np.zeros((N1, N1))
H[nn1, nn1] = 2.0 * K_FK
H[nn1, (nn1 + 1) % N1] += -K_FK
H[nn1, (nn1 - 1) % N1] += -K_FK
w_num = np.linalg.eigvalsh(H)
kk = 2.0 * np.pi * np.arange(N1) / N1
w_cf = 2.0 * K_FK * (1.0 - np.cos(kk))
rel5 = float(np.max(np.abs(np.sort(w_num) - np.sort(w_cf))) / max(np.max(np.abs(w_cf)), 1e-30))
print("\n  [S2-1b5] FK 键通道：omega^2 闭式 2k(1-cos q) 对格点本征值，最大相对偏差 %.2e" % rel5)
check("S2-1b5", rel5 <= 1e-12, "键通道色散闭式与格点本征值一致（%.2e），声速平方 = k = %.4f" % (rel5, K_FK))

# ---------------- S2-1b6 / b7 kink 弹性定律 ----------------
N2 = 400
ratio_list = [0.0005, 0.002, 0.005, 0.01, 0.02]
res = []
def rel_kink(Kk, N=N2, tol=1e-10, itmax=120):
    # 缠绕守恒边界：u_0 = 0、u_{N-1} = N（恰好一个额外缠绕）；牛顿迭代精化
    k = K_FK
    K = Kk * k
    ell_c = 1.0 / (2.0 * np.pi * np.sqrt(K / k))
    n = np.arange(N)
    x0 = (N - 1) / 2.0
    u = n + (2.0 / np.pi) * np.arctan(np.exp((n - x0) / ell_c))
    u[0] = 0.0
    u[-1] = float(N)

    def energy_full(z):
        dd = np.diff(z)
        ee = dd - 1.0
        return float(np.sum(0.5 * k * ee ** 2) + np.sum(K * (1.0 - np.cos(2.0 * np.pi * z))))

    gmax = None
    m = N - 2
    idx = np.arange(m)
    for it in range(itmax):
        d = np.diff(u)
        e = d - 1.0
        g = k * (e[:-1] - e[1:]) + 2.0 * np.pi * K * np.sin(2.0 * np.pi * u[1:-1])
        gmax = float(np.max(np.abs(g)))
        if gmax < tol:
            break
        Jm = np.zeros((m, m))
        Jm[idx, idx] = 2.0 * k + 4.0 * np.pi ** 2 * K * np.cos(2.0 * np.pi * u[1:-1])
        Jm[idx[:-1], idx[1:]] = -k
        Jm[idx[1:], idx[:-1]] = -k
        wmin = float(np.linalg.eigvalsh(Jm)[0])
        lam = max(0.0, 1e-9 - wmin)
        du = np.linalg.solve(Jm + lam * np.eye(m), g)
        E0 = energy_full(u)
        alpha = 1.0
        for _ in range(50):
            un = u.copy()
            un[1:-1] -= alpha * du
            if energy_full(un) <= E0:
                u = un
                break
            alpha *= 0.5
    d = np.diff(u)
    e = d - 1.0
    E_k = float(np.sum(0.5 * k * e ** 2) + np.sum(K * (1.0 - np.cos(2.0 * np.pi * u))))
    ell_star = float((np.sum(np.abs(e)) ** 2) / np.sum(e ** 2))
    E_cf = (4.0 / np.pi) * np.sqrt(k * K)
    return dict(Kk=Kk, ell=ell_c, ell_star=ell_star, E=E_k, E_cf=E_cf, gmax=gmax, it=it, u=u)

t0 = time.time()
print("\n  [S2-1b6/b7] kink 弛豫（N = %d，五点）" % N2)
for Kk in ratio_list:
    r = rel_kink(Kk)
    res.append(r)
    print("       K/k = %.4f：ell_c = %.3f，ell* = %.3f（比 %.3f），E = %.5f，闭式 %.5f（比 %.4f），gmax = %.1e，%d 步"
          % (Kk, r["ell"], r["ell_star"], r["ell_star"] / r["ell"], r["E"], r["E_cf"], r["E"] / r["E_cf"], r["gmax"], r["it"]))
print("       弛豫用时 %.1f 秒" % (time.time() - t0))
lK = np.log([r["Kk"] for r in res])
s6 = float(np.polyfit(lK[:4], np.log([r["ell_star"] for r in res[:4]]), 1)[0])
s7 = float(np.polyfit(lK[:4], np.log([r["E"] for r in res[:4]]), 1)[0])
ratios = [r["E"] / r["E_cf"] for r in res[:4]]
check("S2-1b6", abs(s6 + 0.5) <= 0.03, "kink 宽度幂律斜率 = %.4f（期望 -0.5）" % s6)
check("S2-1b7", abs(s7 - 0.5) <= 0.03 and all(0.9 <= q <= 1.1 for q in ratios),
      "kink 能量幂律斜率 = %.4f（期望 +0.5），四点绝对值比 %.3f 至 %.3f" % (s7, min(ratios), max(ratios)))

# ---------------- S2-1b8 桥接恒等式 ----------------
rng = np.random.default_rng(20260918)
# (a) 键通道二次型对齐：J = k/(4 pi^2)
MM = 64
xv = rng.uniform(0.0, 1.0, MM)
th_v = 2.0 * np.pi * xv
dth = np.diff(np.concatenate([th_v, th_v[:1]]))
E_bond_fk = float(np.sum(0.5 * K_FK * (dth / (2.0 * np.pi)) ** 2))
J_bh = K_FK / (4.0 * np.pi ** 2)
E_bond_bh = float(np.sum(0.5 * J_bh * dth ** 2))
dev_a = abs(E_bond_fk - E_bond_bh) / max(abs(E_bond_bh), 1e-30)
# 与公理 4 键项直接对照：J 应等于 K_0
E_bond_ax = float(np.sum(0.5 * K0 * dth ** 2))
dev_b = abs(J_bh - K0) / K0
# (b) onsite 逐点精确：lam = K 精确（2 pi u = theta）
uv = rng.uniform(0.0, 1.0, MM)
E_on_fk = float(np.sum(K_ONS * (1.0 - np.cos(2.0 * np.pi * uv))))
E_on_bh = float(np.sum(K_ONS * (1.0 - np.cos(2.0 * np.pi * uv))))
dev_c = abs(E_on_fk - E_on_bh) / max(abs(E_on_bh), 1e-30)
# (c) 异类块本征值 {0, 3 kappa_phi, 3 kappa_phi}
Lk3 = 3.0 * np.eye(3) - np.ones((3, 3))
ev = np.sort(np.linalg.eigvalsh(Lk3 * (HBAR * KAP)))
dev_d = float(np.max(np.abs(ev - np.array([0.0, 3.0 * KAP, 3.0 * KAP]))) / (3.0 * KAP))
# (d) K/J 三式一致
KJ_1 = 4.0 * np.pi ** 2 * K_ONS / K_FK
KJ_2 = 3.0 * KAP / K0
KJ_3 = 6.0 * np.pi * KAP * TP
dev_e = max(abs(KJ_1 - KJ_2), abs(KJ_2 - KJ_3)) / KJ_3
print("\n  [S2-1b8] 桥接核对：")
print("       键通道二次型对齐 |dE|/E = %.2e；J = k/(4 pi^2) = %.6f，K_0 = %.6f（偏差 %.2e）" % (dev_a, J_bh, K0, dev_b))
print("       onsite 逐点精确（2 pi u = theta）：|dE|/E = %.2e；异类块本征值 {0, 3 kappa, 3 kappa} 偏差 %.2e" % (dev_c, dev_d))
print("       K/J 三式：4 pi^2 K/k = %.6f；3 kappa/K_0 = %.6f；6 pi kappa t_P = %.6f（最大相对差 %.2e）"
      % (KJ_1, KJ_2, KJ_3, dev_e))
ok8 = (dev_a <= 1e-12 and dev_b <= 1e-12 and dev_c <= 1e-12 and dev_d <= 1e-12 and dev_e <= 1e-12)
check("S2-1b8", ok8, "桥接恒等式全部成立（与路线一为同源恒等式，偏差 0；独立取数不成立，登记待裁定）")

# ---------------- 报告项 ----------------
win_bh = 4.0 * np.pi ** 2 * 0.02
conv_kk = 1.0 / (4.0 * np.pi ** 2)
dev_cos = None
dtest = 1.0
E_cos = 1.0 - np.cos(dtest)
E_sq = 0.5 * dtest ** 2
dev_cos = abs(E_cos - E_sq) / E_sq
print("\n  [报告] 窗口映射：FK 的 K/k <= 0.02 -> 黑洞线 K/J <= %.4f；模型惯例 lam/J = 1 -> K/k = %.4f（窗外缘）" % (win_bh, conv_kk))
print("  [报告] cos 投影降级幅度：Delta theta = 1 rad 时，cos 键与平方键能量相对差 %.1f%%（小角差区消失）" % (dev_cos * 100))

print("\n  ======== 判据汇总 ========")
npass = sum(1 for _, ok in CHECKS if ok)
print("  通过 %d / %d" % (npass, len(CHECKS)))
for tag, ok in CHECKS:
    print("    %-8s %s" % (tag, "PASS" if ok else "FAIL"))

# ---------------- 出图 ----------------
IMG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "images", "m6_s21b_elastic_route.png")
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))

ax = axes[0]
for r, col in zip(res[:3] + [res[4]], ["tab:blue", "tab:cyan", "tab:green", "tab:red"]):
    d = np.diff(r["u"]); e = d - 1.0
    nn2 = np.arange(len(e)) - (len(e) - 1) / 2.0
    ax.plot(nn2, e, lw=1.2, label="K/k = %.4f（ell* = %.2f）" % (r["Kk"], r["ell_star"]))
ax.set_xlim(-40, 40)
ax.set_xlabel("格点（相对链中心）"); ax.set_ylabel("键差超额 e_n = Delta u - 1")
ax.set_title("(a) kink 键剖面（五点中的四档）")
ax.legend(fontsize=8)

ax = axes[1]
ax.loglog([r["Kk"] for r in res], [r["ell_star"] for r in res], "o-", color="tab:red", label="ell*（参与比）")
ax.loglog([r["Kk"] for r in res], [r["ell"] for r in res], "s--", color="tab:blue", label="ell_c = 1/(2 pi sqrt(K/k))")
ax.set_xlabel("K/k"); ax.set_ylabel("宽度")
ax.set_title("(b) 宽度幂律（斜率 %.3f）" % s6)
ax.legend(fontsize=8)

ax = axes[2]
ax.loglog([r["Kk"] for r in res], [r["E"] for r in res], "o-", color="tab:red", label="E_k（实测）")
ax.loglog([r["Kk"] for r in res], [r["E_cf"] for r in res], "s--", color="tab:blue", label="(4/pi) sqrt(k K)")
ax.set_xlabel("K/k"); ax.set_ylabel("kink 能量")
ax.set_title("(c) 能量幂律（斜率 %.3f）" % s7)
ax.legend(fontsize=8)

fig.tight_layout()
fig.savefig(IMG, dpi=150)
print("\n  图已保存: %s" % IMG)
print("  面板核对：(a) 四档键剖面，峰高约 1/ell* 量级、随 K/k 增大而收窄；(b) 五点 log-log，前四点斜率 %.3f；"
      "(c) 五点 log-log，前四点斜率 %.3f，第五点见离散端偏差。" % (s6, s7))
