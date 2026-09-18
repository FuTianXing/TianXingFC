# -*- coding: utf-8 -*-
r"""
minimal_verification_lambda_light_deflection.py

[张量桥分段 d] 空间系数 lambda 的光偏折锚：偏折角随 lambda 的线性读数

定位
----
P1-5 分段 (d) 要交付的是"源到空间度量"那一环，其中的未知量是空间系数 lambda（即 PPN
gamma，工作区已登记其唯一独立经验锚为光偏折系数）。本脚本把 lambda 做成偏折读数里的
可测参数：在弱场中取 h00 = 2 Phi，局域坐标速度按各向同性形式
  c(x) = sqrt( (1 + h00) / (1 - lambda h00) )，
则折射率 n = 1/c 一阶为 1 - (1+lambda) Phi，偏折角应为 (1+lambda) 乘 2GM/b。
判据核对这条线性关系，并给出 lambda = 1 与观测值 4GM/b 的对应。

判据
----
M1 一阶结构：lambda = 0 给半值（2A/b），lambda = 1 给全值（4A/b），比值 2（偏差 < 10%）。
M2 线性律：lambda 取 0、0.5、1、1.5 四点，偏折角对 (1+lambda) 的线性拟合斜率与截距
   （截距应为 0）符合（斜率相对偏差 < 10%，截距小于斜率的 2%）。
M3 与 eikonal 的逐点一致：每个 lambda 的实测偏折角与该 lambda 下的 eikonal 部分路径积分
   之比一致（比值散布 < 5%）。

运行
----
python 验证脚本/minimal_verification_lambda_light_deflection.py
确定性（无随机数），运行约 8 秒。
图存 D:/software/TianXingFC/images/lambda_light_deflection.png。
"""
import time

import numpy as np

T0 = time.time()
CHECKS = []


def check(tag, ok, detail):
    CHECKS.append((tag, bool(ok)))
    print(f"  [{tag}] {'PASS' if ok else 'FAIL'}  {detail}")


N, H, DT, STEPS, EPS = 256, 1.0, 0.4, 300, 2.0


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


def run(lam, A, b, k0=1.2, sigma=8.0, steps=STEPS):
    c = cfield(potential(A), lam)
    x0 = N * 0.25
    x = np.arange(N)
    X, Y = np.meshgrid(x, x, indexing='ij')
    yb = N / 2 + b
    psi = np.exp(-((X - x0) ** 2 + (Y - yb) ** 2) / (2 * sigma ** 2)) * np.exp(1j * k0 * X)
    k2 = (c * DT / H) ** 2
    kap = float(np.mean(k2))
    omega = 2.0 / DT * np.arcsin(np.sqrt(kap) * np.sin(k0 * H / 2.0))
    u_prev = np.real(psi * np.exp(1j * omega * DT))
    u = np.real(psi)
    for _ in range(steps):
        u_next = 2.0 * u - u_prev + k2 * lap(u)
        u_prev, u = u, u_next
    return u


def angle(u, k0=1.2):
    U = np.fft.fft2(u)
    kx = np.fft.fftfreq(N) * 2 * np.pi / H
    KX, KY = np.meshgrid(kx, kx, indexing='ij')
    mask = (np.abs(KX - k0) < 0.6) & (np.abs(KY) < 1.0)
    w = (np.abs(U) ** 2) * mask
    return float((w * KY).sum() / w.sum()), float((w * KX).sum() / w.sum())


def centroid_x(u, yb, sigma=8.0):
    x = np.arange(N)
    X, Y = np.meshgrid(x, x, indexing='ij')
    w = (u ** 2) * np.exp(-((Y - yb) ** 2) / (2 * (3 * sigma) ** 2))
    return float((w * X).sum() / w.sum())


def eikonal(lam, A, b, x1, x2, delta=0.5, nq=1500):
    xs = np.linspace(x1, x2, nq)

    def nloc(y):
        phi = -A / np.sqrt(xs ** 2 + y ** 2 + EPS ** 2)
        h00 = 2.0 * phi
        return np.sqrt(np.clip(1.0 - lam * h00, 1e-6, None) / np.clip(1.0 + h00, 1e-6, None))

    dndy = (nloc(b + delta) - nloc(b - delta)) / (2 * delta)
    return float(np.trapezoid(dndy, xs))


print("=" * 72)
print("最小验证：空间系数 lambda 的光偏折锚（偏折角随 lambda）")
print("=" * 72)

A0, B0 = 0.02, 20.0
X1 = N * 0.25 - N / 2
lams = [0.0, 0.5, 1.0, 1.5]
meas, exp = {}, {}
for lam in lams:
    u = run(lam, A0, B0)
    ky, kx = angle(u)
    xc = centroid_x(u, N / 2 + B0)
    meas[lam] = ky / kx
    exp[lam] = eikonal(lam, A0, B0, X1, xc - N / 2)
    print(f"      lambda = {lam:.1f}：偏折角 {meas[lam]:+.4e}，eikonal 期望 {exp[lam]:+.4e}")

print("\n[M1] 一阶结构：lambda = 0 给半值、lambda = 1 给全值")
r01 = meas[1.0] / meas[0.0]
print(f"      偏折角之比（lambda = 1 对 0）= {r01:.3f}（期望 2）")
check("M1", abs(r01 - 2.0) < 0.2, f"比值 {r01:.3f}（判据 2 ± 10%）")

print("\n[M2] 线性律：偏折角正比于 (1 + lambda)")
xs = np.array([1.0 + l for l in lams])
ys = np.array([meas[l] for l in lams])
fit = np.polyfit(xs, ys, 1)
resid = float(np.max(np.abs(np.polyval(fit, xs) - ys)))
print(f"      拟合斜率 {fit[0]:+.4e}，截距 {fit[1]:+.2e}，最大残差 {resid:.1e}")
check("M2", abs(fit[1]) < 0.02 * abs(fit[0]) and resid < 0.05 * abs(fit[0]),
      f"截距 {fit[1]:+.1e}（小于斜率 2%），残差 {resid:.1e}（小于斜率 5%）")

print("\n[M3] 与 eikonal 逐点一致")
rats = [meas[l] / exp[l] for l in lams]
print(f"      四个 lambda 的比值：{', '.join(f'{r:.3f}' for r in rats)}")
spread = (max(rats) - min(rats)) / np.mean(rats)
check("M3", spread < 0.05, f"比值散布 {spread:.1%}（判据 < 5%）")

print("\n" + "=" * 72)
n_pass = sum(1 for _, ok in CHECKS if ok)
print(f"判据汇总：{n_pass}/{len(CHECKS)} 通过")
for tag, ok in CHECKS:
    print(f"  {tag}: {'PASS' if ok else 'FAIL'}")
print(f"耗时 {time.time() - T0:.1f} s")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(6, 4.2))
    lgg = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    vv = []
    for lam in lgg:
        u = run(lam, A0, B0)
        kyy, kxx = angle(u)
        vv.append(abs(kyy / kxx))
    ax.plot(1 + lgg, vv, "o-", label="实测")
    ax.plot(1 + lgg, abs(fit[0]) * (1 + lgg), "k--", lw=1, label="线性律 (1+lambda)")
    ax.axvline(2.0, color="tab:red", ls=":", lw=1, label="lambda = 1（广义相对论）")
    ax.set_xlabel("1 + lambda")
    ax.set_ylabel("偏折角大小 (rad)")
    ax.set_title("偏折角随空间系数的线性律")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("D:/software/TianXingFC/images/lambda_light_deflection.png", dpi=140)
    print("图存 images/lambda_light_deflection.png")
except Exception as exc:
    print(f"绘图跳过：{exc}")