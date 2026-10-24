# -*- coding: utf-8 -*-
"""系统因子 1.365 的剖分：现行有限段 eikonal、宽段积分、乘载波 k0 的动量变体与时间步细化。
问题：格点波包实测偏折角与 eikonal 期望之比稳定在 1.364 至 1.366（与 lambda 无关）。
本脚本在同一次模拟下并排四种口径，定位该因子的来源（截断、k0 因子、kx 归一、离散色散）。
输出：images/system_factor_1365.png（绝对路径写入，工作区移动需同步改）。
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

N, H, EPS = 256, 1.0, 2.0
A0, B0, K0, SIGMA = 0.02, 20.0, 1.2, 8.0

def potential(A):
    x = np.arange(N) - N / 2
    X, Y = np.meshgrid(x, x, indexing='ij')
    return -A / np.sqrt(X ** 2 + Y ** 2 + EPS ** 2)

def cfield(phi, lam):
    h00 = 2.0 * phi
    return np.sqrt(np.clip(1.0 + h00, 1e-6, None) / np.clip(1.0 - lam * h00, 1e-6, None))

def lap(u):
    out = -4.0 * u
    out[1:, :] += u[:-1, :]
    out[:-1, :] += u[1:, :]
    out[:, 1:] += u[:, :-1]
    out[:, :-1] += u[:, 1:]
    return out

def run(lam, dt, steps, A=A0, b=B0, k0=K0, sigma=SIGMA):
    c = cfield(potential(A), lam)
    x0 = N * 0.25
    x = np.arange(N)
    X, Y = np.meshgrid(x, x, indexing='ij')
    yb = N / 2 + b
    psi = np.exp(-((X - x0) ** 2 + (Y - yb) ** 2) / (2 * sigma ** 2)) * np.exp(1j * k0 * X)
    k2 = (c * dt / H) ** 2
    kap = float(np.mean(k2))
    omega = 2.0 / dt * np.arcsin(np.sqrt(kap) * np.sin(k0 * H / 2.0))
    u_prev = np.real(psi * np.exp(1j * omega * dt))
    u = np.real(psi)
    for _ in range(steps):
        u_next = 2.0 * u - u_prev + k2 * lap(u)
        u_prev, u = u, u_next
    return u

def measure(u, k0=K0):
    U = np.fft.fft2(u)
    kx = np.fft.fftfreq(N) * 2 * np.pi / H
    KX, KY = np.meshgrid(kx, kx, indexing='ij')
    mask = (np.abs(KX - k0) < 0.6) & (np.abs(KY) < 1.0)
    w = (np.abs(U) ** 2) * mask
    ky = float((w * KY).sum() / w.sum())
    kxm = float((w * KX).sum() / w.sum())
    x = np.arange(N)
    X, Y = np.meshgrid(x, x, indexing='ij')
    ww = (u ** 2) * np.exp(-((Y - (N / 2 + B0)) ** 2) / (2 * (3 * SIGMA) ** 2))
    xc = float((ww * X).sum() / ww.sum())
    return ky, kxm, xc

def eikonal_int(lam, x1, x2, delta=0.5, nq=6000, A=A0, b=B0):
    xs = np.linspace(x1, x2, nq)
    def nloc(y):
        phi = -A / np.sqrt(xs ** 2 + y ** 2 + EPS ** 2)
        h00 = 2.0 * phi
        return np.sqrt(np.clip(1.0 - lam * h00, 1e-6, None) / np.clip(1.0 + h00, 1e-6, None))
    dndy = (nloc(b + delta) - nloc(b - delta)) / (2 * delta)
    return float(np.trapezoid(dndy, xs))

def eikonal_wide(lam, A=A0, b=B0, span=40):
    return eikonal_int(lam, -span * b, span * b, A=A, b=b)

print("=" * 72)
print("剖分：系统因子 1.365 的来源")
print("=" * 72)

rows = []
for lam in [0.0, 1.0]:
    u = run(lam, dt=0.4, steps=300)
    ky, kxm, xc = measure(u)
    x1 = N * 0.25 - N / 2
    x2 = xc - N / 2
    E1 = eikonal_int(lam, x1, x2)
    E2 = eikonal_wide(lam)
    meas = ky / kxm
    r_old = meas / E1
    print(f"lambda = {lam:.1f}: ky = {ky:+.6e}, kx = {kxm:.6f}, xc = {xc:.2f}（x2 = {x2:.2f}）")
    print(f"    E1（现行段 [{x1:.0f}, {x2:.0f}]）= {E1:+.6e}；E2（宽段）= {E2:+.6e}；E1/E2 = {E1/E2:.4f}")
    print(f"    r_old = meas/E1 = {r_old:.4f}（复现 1.364 至 1.366）")
    print(f"    ky/(k0*E2) = {ky/(K0*E2):.4f}；ky/E2 = {ky/E2:.4f}；kx/k0 = {kxm/K0:.4f}")
    rows.append((lam, ky, kxm, xc, x2, E1, E2, meas, r_old))

print("-" * 72)
print("时间步细化（lambda = 1，dt = 0.2, steps = 600，同距离）：")
u = run(1.0, dt=0.2, steps=600)
ky2, kx2, xc2 = measure(u)
E1b = eikonal_int(1.0, N * 0.25 - N / 2, xc2 - N / 2)
E2b = eikonal_wide(1.0)
print(f"  ky = {ky2:+.6e}, kx = {kx2:.6f}, xc = {xc2:.2f}")
print(f"  r_old = {(ky2/kx2)/E1b:.4f}；ky/(k0*E2) = {ky2/(K0*E2b):.4f}")

print("-" * 72)
print("读数：r_old 的分解 r_old = (k0/kx) * (E2_true/E1) * [ky/(k0 E2_true)]")
for (lam, ky, kxm, xc, x2, E1, E2, meas, r_old) in rows:
    f1 = K0 / kxm
    f2 = E2 / E1
    f3 = ky / (K0 * E2)
    print(f"  lambda = {lam:.1f}: 因子 (k0/kx) = {f1:.4f}，截断 (E2/E1) = {f2:.4f}，残余 ky/(k0 E2) = {f3:.4f}，乘积 = {f1*f2*f3:.4f}")


print("-" * 72)
print("离散色散 eikonal（格点自洽版）：kick = -∫ (dOmega/dy)/v_g dx")
DT_R, STEPS_R = 0.4, 300

def omega_of(lam, A=A0):
    phi = potential(A)
    h00 = 2.0 * phi
    c = np.sqrt(np.clip(1.0 + h00, 1e-6, None) / np.clip(1.0 - lam * h00, 1e-6, None))
    k2 = (c * DT_R / H) ** 2
    return 2.0 / DT_R * np.arcsin(np.sqrt(float(np.mean(k2))) * np.sin(K0 * H / 2.0))

def kick_discrete(lam, x1, x2, delta=0.5, nq=6000, A=A0, b=B0):
    xs = np.linspace(x1, x2, nq)
    om = omega_of(lam)
    ks = np.sin(K0 * H / 2.0)

    def kappa(y):
        phi = -A / np.sqrt(xs ** 2 + y ** 2 + EPS ** 2)
        h00 = 2.0 * phi
        c = np.sqrt(np.clip(1.0 + h00, 1e-6, None) / np.clip(1.0 - lam * h00, 1e-6, None))
        return (c * DT_R / H) ** 2

    def Om(y):
        kap = kappa(y)
        return 2.0 / DT_R * np.arcsin(np.sqrt(kap) * ks)

    def vg(y):
        kap = kappa(y)
        return (H / DT_R) * np.sqrt(kap) * np.cos(K0 * H / 2.0) / np.sqrt(1.0 - kap * ks ** 2)

    dOdy = (Om(b + delta) - Om(b - delta)) / (2 * delta)
    return -float(np.trapezoid(dOdy / vg(b), xs)), om

for (lam, ky, kxm, xc, x2, E1, E2, meas, r_old) in rows:
    kick, om = kick_discrete(lam, -40 * B0, 40 * B0)
    kick_fin, _ = kick_discrete(lam, N * 0.25 - N / 2, x2)
    print(f"  lambda = {lam:.1f}: omega = {om:.6f}, 离散 kick（宽段）= {kick:+.6e}，实测 ky = {ky:+.6e}，比 = {ky/kick:.4f}")
    print(f"      离散 kick（现行段）= {kick_fin:+.6e}，实测/它 = {ky/kick_fin:.4f}")
print("  读数：若 ky/离散kick ≈ 1，则 1.365 整解释为口径（截断加连续极限近似），无剩余物理因子。")


print("-" * 72)
print("包宽扫描（lambda = 1）：残余是否随 sigma 按 sigma^2 收缩（横向平均假说的判定）")
for sg in [4.0, 8.0, 16.0]:
    u = run(1.0, dt=0.4, steps=300, sigma=sg)
    ky_s, kx_s, xc_s = measure(u)
    kick_w, _ = kick_discrete(1.0, -40 * B0, 40 * B0)
    E2_s = eikonal_wide(1.0)
    print(f"  sigma = {sg:5.1f}: ky = {ky_s:+.6e}, ky/kick(宽带) = {ky_s/kick_w:.4f}, ky/(k0 E2) = {ky_s/(K0*E2_s):.4f}, xc = {xc_s:.1f}")
print("  期望：若残余来自横向平均，则 sigma 减半时 (比值 - 1) 约降为四分之一。")

fig, axs = plt.subplots(1, 2, figsize=(11.0, 3.8))
lams = [r[0] for r in rows]
x = np.arange(len(lams))
ax = axs[0]
ax.bar(x - 0.18, [abs(r[5]) for r in rows], width=0.36, color='#9aa4ae', label='E1 现行段')
ax.bar(x + 0.18, [abs(r[6]) for r in rows], width=0.36, color='#14508c', label='E2 宽段')
for i, r in enumerate(rows):
    ax.text(i - 0.18, abs(r[5]), f'{abs(r[5]):.2e}', ha='center', va='bottom', fontsize=9)
    ax.text(i + 0.18, abs(r[6]), f'{abs(r[6]):.2e}', ha='center', va='bottom', fontsize=9)
ax.set_xticks(x); ax.set_xticklabels([f'lambda = {l:.0f}' for l in lams])
ax.set_ylabel('|eikonal 积分|', fontsize=10); ax.set_title('截断：E1 对 E2', fontsize=11, color='#14508c'); ax.legend(fontsize=9)
ax = axs[1]
parts = [[K0/r[2], r[6]/r[5], r[1]/(K0*r[6])] for r in rows]
labels = ['k0/kx', 'E2/E1', 'ky/(k0 E2)']
colors = ['#2ca02c', '#ff7f0e', '#b3261e']
bottom = np.zeros(len(lams))
for j in range(3):
    vals = np.array([p[j] for p in parts])
    ax.bar(x, vals, bottom=bottom, width=0.5, color=colors[j], label=labels[j])
    for i, v in enumerate(vals):
        ax.text(i, bottom[i] + v/2, f'{v:.3f}', ha='center', va='center', fontsize=9, color='white')
    bottom += vals
ax.axhline(1.365, color='#555555', ls='--', lw=1)
ax.text(len(lams)-0.5, 1.372, 'r_old = 1.365', fontsize=9, color='#555555', ha='right')
ax.set_xticks(x); ax.set_xticklabels([f'lambda = {l:.0f}' for l in lams])
ax.set_title('r_old 的三因子分解', fontsize=11, color='#14508c'); ax.legend(fontsize=9, loc='upper right'); ax.set_ylim(0, 1.6)
ax.text(0.02, 0.02, '残余（离散 eikonal 见终端读数）', transform=ax.transAxes, fontsize=8.5, color='#777777')
fig.subplots_adjust(left=0.07, right=0.98, top=0.87, bottom=0.14, wspace=0.22)
plt.savefig("D:/software/TianXingFC/images/system_factor_1365.png", dpi=150)
print("图已存：images/system_factor_1365.png")
