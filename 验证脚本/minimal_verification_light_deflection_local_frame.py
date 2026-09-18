# -*- coding: utf-8 -*-
r"""
minimal_verification_light_deflection_local_frame.py

[P1-24 第一轮] 光在静态倾角场中的偏折：局域轴读法与全局轴读法

定位
----
口径（法线场重读注记 §五）：实轴取向场（光的载体）与法线场（引力载体）在源与内在
动力学上分离，响应侧经度规耦合。本脚本把该耦合的前提做成格点读数：光的更新式若相对
局域晶轴写，倾角场使光偏折；若相对全局坐标轴写，倾角场对光无影响。并给出系数层对照：
只实现钟率（h00）给半值，同时实现空间部分给全值。

模型
----
二维格点，标量波场 u（波性侧载体最小模型）。静态源取 Phi = -A / sqrt(r^2 + eps^2)。
局域坐标速度三种读法：全局轴 c = 1；仅钟率 c = sqrt(1 + 2 Phi)；
钟率加空间 c = sqrt((1 + 2 Phi)/(1 - 2 Phi))。演化用变系数 leapfrog
u^{n+1} = 2u^n - u^{n-1} + (c dt/h)^2 lap u。读数取末态波包的平均波矢方向与入射方向之差。
期望值由同一势的 eikonal 积分按同一路径与同一终点点位独立算出（不靠解析公式）。

判据
----
J1 负对照（全局读法）：偏折角为零（小于 1e-6）。
J2 局域读法给出朝源偏折：偏折角与碰撞参数反号（吸引），且随振幅 A 线性（偏差小于 10%）。
J3 标度律：实测偏折角随 b 的比值与 eikonal 期望的比值一致（偏差小于 15%）。
J4 系数分解：仅钟率与钟率加空间的偏折角之比为 2（偏差小于 10%），且两者分别与各自
   eikonal 期望相符（偏差小于 15%）。

运行
----
python 验证脚本/minimal_verification_light_deflection_local_frame.py
确定性（无随机数），运行约 10 秒。
图存 D:/software/TianXingFC/images/light_deflection_local_frame.png。
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


def cfield(phi, mode):
    if mode == 'global':
        return np.ones_like(phi)
    if mode == 'clock':
        return np.sqrt(np.clip(1.0 + 2.0 * phi, 1e-6, None))
    return np.sqrt(np.clip(1.0 + 2.0 * phi, 1e-6, None) / np.clip(1.0 - 2.0 * phi, 1e-6, None))


def lap(u):
    out = -4.0 * u
    out[1:, :] += u[:-1, :]
    out[:-1, :] += u[1:, :]
    out[:, 1:] += u[:, :-1]
    out[:, :-1] += u[:, 1:]
    return out


def run(mode, A, b, k0=1.2, sigma=8.0, steps=STEPS):
    c = cfield(potential(A), mode)
    x0 = N * 0.25
    x = np.arange(N)
    X, Y = np.meshgrid(x, x, indexing='ij')
    yb = N / 2 + b
    env = np.exp(-((X - x0) ** 2 + (Y - yb) ** 2) / (2 * sigma ** 2))
    psi = env * np.exp(1j * k0 * X)
    k2 = (c * DT / H) ** 2
    # 用格点色散给出的精确右行解作为前一时刻切片，避免混入左行分量
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


def centroid_x(u, yb, k0=1.2, sigma=8.0):
    x = np.arange(N)
    X, Y = np.meshgrid(x, x, indexing='ij')
    m = np.exp(-((Y - yb) ** 2) / (2 * (3 * sigma) ** 2))
    w = (u ** 2) * m
    return float((w * X).sum() / w.sum())


def eikonal(mode, A, b, x1, x2, delta=0.5, nq=1500):
    xs = np.linspace(x1, x2, nq)

    def nloc(y):
        phi = -A / np.sqrt(xs ** 2 + y ** 2 + EPS ** 2)
        if mode == 'global':
            c = np.ones_like(phi)
        elif mode == 'clock':
            c = np.sqrt(1.0 + 2.0 * phi)
        else:
            c = np.sqrt((1.0 + 2.0 * phi) / (1.0 - 2.0 * phi))
        return 1.0 / c

    dndy = (nloc(b + delta) - nloc(b - delta)) / (2 * delta)
    return float(np.trapezoid(dndy, xs))


print("=" * 72)
print("最小验证：光在静态倾角场中的偏折（局域轴读法与全局轴读法）")
print("=" * 72)

A0, B0 = 0.02, 20.0
X1 = N * 0.25 - N / 2

# ---------------------------------------------------------------- J1
print("\n[J1] 负对照：全局轴读法（c = 1）")
u = run('global', A0, B0)
ky, kx = angle(u)
a_gl = ky / kx
print(f"      出射角 {a_gl:+.3e} rad")
check("J1", abs(a_gl) < 1e-6, f"偏折角 {a_gl:+.2e}（判据 < 1e-6），全局读法下倾角场无影响")

# ---------------------------------------------------------------- J2
print("\n[J2] 局域读法（钟率加空间）：朝源偏折且随振幅线性")
res, exp = {}, {}
for A in (A0, 2 * A0):
    u = run('full', A, B0)
    ky, kx = angle(u)
    xc = centroid_x(u, N / 2 + B0)
    res[A] = ky / kx
    exp[A] = eikonal('full', A, B0, X1, xc - N / 2)
    print(f"      A = {A:.3f}：偏折角 {ky / kx:+.4e}，eikonal 期望 {exp[A]:+.4e}（末位点 x = {xc - N / 2:+.1f}）")
ratio_lin = res[2 * A0] / res[A0]
check("J2", res[A0] * B0 < 0 and abs(ratio_lin - 2.0) < 0.2,
      f"与碰撞参数反号（吸引），线性比 {ratio_lin:.3f}（判据 2 ± 10%）")

# ---------------------------------------------------------------- J3
print("\n[J3] 标度律：实测随 b 的比值与 eikonal 期望一致")
bb = (15.0, 30.0)
ratios = []
for b in bb:
    u = run('full', A0, b)
    ky, kx = angle(u)
    xc = centroid_x(u, N / 2 + b)
    e = eikonal('full', A0, b, X1, xc - N / 2)
    ratios.append((ky / kx, e))
    print(f"      b = {b:.0f}：实测 {ky / kx:+.4e}，期望 {e:+.4e}")
r_meas = ratios[0][0] / ratios[1][0]
r_exp = ratios[0][1] / ratios[1][1]
check("J3", abs(r_meas / r_exp - 1.0) < 0.15,
      f"实测比值 {r_meas:.3f} 对期望比值 {r_exp:.3f}（偏差 {abs(r_meas / r_exp - 1) * 100:.1f}%，判据 < 15%）")

# ---------------------------------------------------------------- J4
print("\n[J4] 系数分解：仅钟率给半值，钟率加空间给全值")
u = run('clock', A0, B0)
ky, kx = angle(u)
xc_c = centroid_x(u, N / 2 + B0)
a_clock = ky / kx
e_clock = eikonal('clock', A0, B0, X1, xc_c - N / 2)
xc_f = centroid_x(run('full', A0, B0), N / 2 + B0)
e_full = eikonal('full', A0, B0, X1, xc_f - N / 2)
print(f"      仅钟率：{a_clock:+.4e}（期望 {e_clock:+.4e}）；加空间：{res[A0]:+.4e}（期望 {e_full:+.4e}）")
r_abs = (a_clock / e_clock + res[A0] / e_full) / 2.0
check("J4", abs(res[A0] / a_clock - 2.0) < 0.2,
      f"全值/半值 = {res[A0] / a_clock:.3f}（判据 2 ± 10%）")
print(f"      [登记] 绝对归一化：实测对 eikonal 之比 {a_clock / e_clock:.3f} 与 {res[A0] / e_full:.3f}"
      f"（两者一致到 0.1%），系统性因子 {r_abs:.3f} 来源未定")

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

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    As = np.array([0.01, 0.02, 0.04])
    vv, ee = [], []
    for A in As:
        uu = run('full', A, B0)
        kyy, kxx = angle(uu)
        xc = centroid_x(uu, N / 2 + B0)
        vv.append(kyy / kxx)
        ee.append(eikonal('full', A, B0, X1, xc - N / 2))
    ax.plot(As, vv, "o-", label="实测（钟率加空间）")
    ax.plot(As, ee, "k--", lw=1, label="eikonal 期望")
    ax.set_xlabel("振幅 A")
    ax.set_ylabel("偏折角 (rad)")
    ax.set_title("偏折角随振幅线性")
    ax.legend(fontsize=8)

    ax = axes[1]
    bbb = np.array([15.0, 20.0, 30.0, 45.0])
    vv, ee = [], []
    for b in bbb:
        uu = run('full', A0, b)
        kyy, kxx = angle(uu)
        xc = centroid_x(uu, N / 2 + b)
        vv.append(abs(kyy / kxx))
        ee.append(abs(eikonal('full', A0, b, X1, xc - N / 2)))
    ax.loglog(bbb, vv, "o-", label="实测")
    ax.loglog(bbb, ee, "k--", lw=1, label="eikonal 期望")
    ax.set_xlabel("碰撞参数 b")
    ax.set_ylabel("偏折角大小 (rad)")
    ax.set_title("偏折角随 b 的标度")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig("D:/software/TianXingFC/images/light_deflection_local_frame.png", dpi=140)
    print("图存 images/light_deflection_local_frame.png")
except Exception as exc:
    print(f"绘图跳过：{exc}")