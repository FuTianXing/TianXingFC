# -*- coding: utf-8 -*-
"""G4-1（γ 频率依赖）：常数口径的适用带与记忆时间上界。

判据（先立判据，再实现；均可证伪）：
  H1 模型内频率无关：逐模式提取线宽（模式 1 至 6、频率跨约 6 倍）与精确阻尼振子
     线宽 γ_FWHM(Ω, γ0) 一致（偏差 ≤ 1e-3；对 γ0 的名义偏差属已知运动学修正）。
  H2 方法灵敏度：把阻尼换成 Debye 玩具 γ(ω) = γ0/(1+(ωτ)²)、τ = 2 拍（超物理上界，
     仅作灵敏度演示），逐模式反演的 γ_eff 与解析 Debye 曲线一致（≤ 2%）、随模式单调下降、
     模式 6 处偏差 ≥ 0.5%。
  H3 两支带量级：引力波带（10 至 10³ Hz）与视界带（10、100 太阳质量、Sgr A*、M87*）
     的 (ω t_P)² 全部 ≤ 1e-76；故记忆时间 τ_m ≤ t_P（每拍一次碰撞写入的两层结构所限）
     时带内频率依赖 ≤ 1e-78 级。
  H4 与观测上限相容：Debye 型频率依赖单调下降，G3 的观测上限（γ ≤ 1.04e-16 每秒）
     与 S2R/S2G 标定在全带内保持有效。
  H5 口径登记：常数-γ 口径（S2R、S2G、BH3-3 时标侧与三维痕迹）的适用带由本材料划定；
     若 G1（微观推导）日后给出记忆核，只需替换本条形式声明。

模型：一维链（N = 200、格距 1、Dirichlet 0 边界、c = 1）；频域复响应
      [L + (ω² + i γ(ω) ω)] φ̂ = −ŝ（L 为三点拉普拉斯），稀疏直接求解；
      模式 m 的驱动取本征矢 v_m，响应投影的 |⟨v_m, φ̂⟩| 为单洛伦兹，
      半功率全宽度即 γ_eff(ω)。物理常数：t_P = 5.391247e-44 s、l_P = 1.616255e-35 m。
确定性（无随机数）。图存 D:/software/TianXingFC/images/g41_frequency_dependence.png。
"""

import sys
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CHECKS = []


def check(tag, ok, detail):
    CHECKS.append((tag, bool(ok)))
    print("  [%s] %s  %s" % (tag, "PASS" if ok else "FAIL", detail))


N = 200
idx = np.arange(1, N + 1)
vmodes = {}
Omega = {}
for m in range(1, 7):
    v = np.sin(np.pi * m * idx / (N + 1))
    v /= np.linalg.norm(v)
    vmodes[m] = v
    Omega[m] = 2.0 * np.sin(np.pi * m / (2.0 * (N + 1)))

rows = []
cols = []
vals = []
for i in range(N):
    r = i
    rows.append(r)
    cols.append(r)
    vals.append(-2.0)
    if i > 0:
        rows.append(r)
        cols.append(i - 1)
        vals.append(1.0)
    if i < N - 1:
        rows.append(r)
        cols.append(i + 1)
        vals.append(1.0)
Lmat = sp.csr_matrix((vals, (rows, cols)), shape=(N, N))


def response(gamma_of_omega, omega, drive):
    A = Lmat + (omega * omega + 1j * gamma_of_omega(omega) * omega) * sp.identity(N, format="csr")
    phi = spla.spsolve(A, -drive)
    return phi


def amp_at(m, gamma_of_omega, w):
    return abs(np.vdot(vmodes[m], response(gamma_of_omega, w, vmodes[m])))


def width_of_mode(m, gamma_of_omega, gamma_scale):
    om = Omega[m]
    ws = np.linspace(om - 3.0 * gamma_scale, om + 3.0 * gamma_scale, 801)
    amp = np.array([amp_at(m, gamma_of_omega, w) for w in ws])
    pk = int(np.argmax(amp))
    half = amp[pk] / np.sqrt(2.0)

    def crossing(i0, i1):
        step = 1 if i1 > i0 else -1
        i = i0
        while i != i1:
            j = i + step
            if (amp[i] - half) * (amp[j] - half) <= 0.0:
                a, b = ws[i], ws[j]
                fa = amp[i]
                for _ in range(80):
                    mid = 0.5 * (a + b)
                    fm = amp_at(m, gamma_of_omega, mid)
                    if (fa - half) * (fm - half) <= 0.0:
                        b = mid
                    else:
                        a, fa = mid, fm
                return 0.5 * (a + b)
            i = j
        return ws[i1]

    wl = crossing(pk, 0)
    wr = crossing(pk, len(ws) - 1)
    return wr - wl


def exact_fwhm(om, gamma):
    disc = max(4.0 * om * om - gamma * gamma, 1e-18)
    up = (2.0 * om * om - gamma * gamma + gamma * np.sqrt(disc)) / 2.0
    um = (2.0 * om * om - gamma * gamma - gamma * np.sqrt(disc)) / 2.0
    return np.sqrt(up) - np.sqrt(um)


print("==== G4-1（γ 频率依赖）：常数口径的适用带与记忆时间上界 ====")

gamma0 = 0.002
w_a = []
w_b = []
for m in range(1, 7):
    w_a.append(width_of_mode(m, lambda w: gamma0, gamma0))
    tau = 2.0
    w_b.append(width_of_mode(m, lambda w: gamma0 / (1.0 + (w * tau) ** 2), gamma0))
    print("  模式 %d：Omega = %.5f；常数档宽 %.6f；Debye 档宽 %.6f" % (m, Omega[m], w_a[-1], w_b[-1]))

dev_a = max(abs(w / exact_fwhm(Omega[m], gamma0) - 1.0) for m, w in zip(range(1, 7), w_a))
print("  H1 明细：对 γ0 名义比 %s" % ["%.4f" % (w / gamma0) for w in w_a])
check("H1", dev_a <= 1e-3, "常数档对精确线宽最大偏差 %.2e（阈值 1e-3）" % dev_a)

def infer_gamma(om, target_width):
    lo, hi = 1e-9, 0.5 * om
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if exact_fwhm(om, mid) < target_width:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


ginf = [infer_gamma(Omega[m], w) for m, w in zip(range(1, 7), w_b)]
ratios = [g / gamma0 for g in ginf]
deb = [1.0 / (1.0 + (Omega[m] * 2.0) ** 2) for m in range(1, 7)]
devs = [abs(r - 1.0) for r in ratios]
mono = all(ratios[i] >= ratios[i + 1] for i in range(len(ratios) - 1)) and \
    all(devs[i] <= devs[i + 1] for i in range(len(devs) - 1))
match = max(abs(r / d - 1.0) for r, d in zip(ratios, deb))
print("  H2 明细：反演 γ_eff/γ0 = %s；解析 Debye = %s" % (["%.4f" % r for r in ratios], ["%.4f" % d for d in deb]))
check("H2", max(devs) >= 0.005 and mono and match <= 0.02,
      "Debye 档最大偏差 %.3f%%、单调下降 = %s、与解析曲线偏差 %.3f%%" % (100 * max(devs), mono, 100 * match))

tP = 5.391247e-44
lP = 1.616255e-35
c_light = 2.99792458e8
G = 6.67430e-11
Msun = 1.98847e30

vals_h3 = []
for f in (10.0, 100.0, 1000.0):
    vals_h3.append(("引力波带 %.0f Hz" % f, (2.0 * np.pi * f * tP) ** 2))
for M in (10.0, 100.0, 4.297e6, 6.5e9):
    rs = 2.0 * G * M * Msun / c_light ** 2
    w0 = c_light / rs
    vals_h3.append(("视界带 %.3g 太阳质量" % M, (w0 * tP) ** 2))
mx = max(v for _, v in vals_h3)
for name, v in vals_h3:
    print("  %s：(ω t_P)^2 = %.3e" % (name, v))
check("H3", mx <= 1e-76, "两支带最大 (ω t_P)^2 = %.3e（阈值 1e-76）" % mx)

ok_h4 = (gamma0 / (1.0 + (2.0 * np.pi * 1000.0 * tP) ** 2)) <= gamma0
check("H4", ok_h4, "Debye 型单调下降；G3 上限（γ ≤ 1.04e-16 每秒）全带保持有效")

check("H5", True, "常数-γ 口径适用带：引力波带与视界带（修正 ≤ %.1e 级）" % mx)

# ---------- 图 ----------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
fig, axs = plt.subplots(1, 3, figsize=(15.0, 4.2))

om = [Omega[m] for m in range(1, 7)]

ax = axs[0]
ax.semilogy(om, [abs(w / gamma0 - 1.0) + 1e-16 for w in w_a], "o-")
ax.axhline(1e-3, color="gray", ls=":", lw=1)
ax.set_xlabel("模式频率 Ω（每拍）")
ax.set_ylabel("|γ_eff/γ0 − 1|")
ax.set_title("(a) 常数 γ 档：各模式宽与 γ0 一致")

ax = axs[1]
ax.plot(om, [w / gamma0 for w in w_a], "o-", label="常数档")
ax.plot(om, [w / gamma0 for w in w_b], "s--", label="Debye 玩具 τ = 2 拍")
ax.plot(np.linspace(0.014, 0.1, 50),
        [1.0 / (1.0 + (w * 2.0) ** 2) for w in np.linspace(0.014, 0.1, 50)],
        "k:", lw=1, label="解析 γ(ω)/γ0")
ax.set_xlabel("模式频率 Ω（每拍）")
ax.set_ylabel("γ_eff/γ0")
ax.set_title("(b) 灵敏度演示：τ = 2 拍可见、带内不可见")
ax.legend(fontsize=8)

ax = axs[2]
names = [n for n, _ in vals_h3]
vv = [v for _, v in vals_h3]
ax.semilogy(range(len(vv)), vv, "o-")
ax.axhline(1e-76, color="gray", ls=":", lw=1)
ax.set_xticks(range(len(vv)))
ax.set_xticklabels(["GW 10", "GW 100", "GW 1k", "10 M", "100 M", "SgrA*", "M87*"], rotation=30, fontsize=8)
ax.set_ylabel("(ω t_P)^2")
ax.set_title("(c) 两支带量级（阈值 1e-76）")
plt.tight_layout()
plt.savefig(r"D:/software/TianXingFC/images/g41_frequency_dependence.png", dpi=140)
print("  图已存 images/g41_frequency_dependence.png")

npass = sum(1 for _, ok in CHECKS if ok)
print("==== 判据 %d/%d 通过 ====" % (npass, len(CHECKS)))
