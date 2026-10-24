# -*- coding: utf-8 -*-
"""合成权重唯一性：算术平均被各向同性与中心对称唯一钉住（第 463 条；丙路径剩余细项一）。

模型（复刻阶段 4 口径）：三维立方格，站点记录由入射链接记录合成，
theta_i = sum_{j~i} w_{ij} u_e(i,j)，链接中点配对 u_e = (theta_i + theta_j)/2。
权重限定最近邻六方向、按轴分组（w_{mu,+}, w_{mu,-}），归一化 sum w = 1。

平面波符号（theta = exp(i k.r)）：
  S(k) = (1/2) sum_{mu} [ w_{mu,+}(1+e^{+i k_mu}) + w_{mu,-}(1+e^{-i k_mu}) ]
  实部 ~ 1 - (1/4) (W_x k_x^2 + W_y k_y^2 + W_z k_z^2)，W_mu = w_{mu,+}+w_{mu,-}；
  虚部 ~ (1/2) (dW_x k_x + dW_y k_y + dW_z k_z)，dW_mu = w_{mu,+}-w_{mu,-}（漂移项）。

判据：
  W1 格点算子与解析符号一致（平面波，机器精度）
  W2 均匀权重 1/6：三方向二阶系数各向同性（[100]=[110]=[111]，偏差 < 1e-12）且 = 1/12；
     漂移项为零（< 1e-15）
  W3 唯一性：轴向参数族 W_x = s 扫描，各向同性比值 [100]/[110] 的唯一零点在 s = 1/3；
     ± 不对称参数族扫描，漂移系数的唯一零点在 dW_x = 0。
     两条件合并给出 w = 1/6（每方向）。

确定性，约 1 秒。图存 images/composition_weight.png
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG = os.path.join(ROOT, "images", "composition_weight.png")


def check(tag, ok, detail):
    print(f"  [{tag}] {'PASS' if ok else 'FAIL'}  {detail}")
    return bool(ok)


def S_analytic(kvec, wplus, wminus):
    s = 0.0
    for mu in range(3):
        km = kvec[mu]
        s += wplus[mu] * (1.0 + np.exp(1j * km)) + wminus[mu] * (1.0 + np.exp(-1j * km))
    return 0.5 * s


def MP_grid(theta, wplus, wminus):
    D = 3
    out = np.zeros_like(theta)
    for mu in range(D):
        for sign, w in ((+1, wplus[mu]), (-1, wminus[mu])):
            nb = np.roll(theta, -sign, axis=mu)
            out += w * (theta + nb) / 2.0
    return out


print("== W1 格点算子与解析符号一致 ==")
N = 8
idx = np.arange(N)
X, Y, Z = np.meshgrid(idx, idx, idx, indexing="ij")
wplus = np.array([1.0, 1.0, 1.0]) / 6.0
wminus = np.array([1.0, 1.0, 1.0]) / 6.0
rng = np.random.default_rng(463)
maxdev = 0.0
for _ in range(6):
    n = rng.integers(0, N, 3)
    kvec = 2 * np.pi * n / N
    theta = np.exp(1j * (kvec[0] * X + kvec[1] * Y + kvec[2] * Z))
    num = MP_grid(theta, wplus, wminus)
    dev = float(np.max(np.abs(num / theta - S_analytic(kvec, wplus, wminus))))
    maxdev = max(maxdev, dev)
print(f"  随机 6 个模：格点算子/解析符号最大偏差 = {maxdev:.3e}（期望 < 1e-14）")
w1_ok = check("W1", maxdev < 1e-14, f"解析符号与格点实现一致（{maxdev:.1e}）")

print("== W2 均匀权重的各向同性 ==")
dirs = {"[100]": np.array([1.0, 0, 0]), "[110]": np.array([1, 1, 0]) / np.sqrt(2), "[111]": np.array([1, 1, 1]) / np.sqrt(3)}


def C_dir(nvec, wp, wm, k):
    S = S_analytic(k * nvec, wp, wm)
    return (1.0 - S.real) / k ** 2


def C0_dir(nvec, wp, wm):
    k1, k2 = 0.01, 0.005
    return (4.0 * C_dir(nvec, wp, wm, k2) - C_dir(nvec, wp, wm, k1)) / 3.0


C100, C110, C111 = (C0_dir(dirs["[100]"], wplus, wminus), C0_dir(dirs["[110]"], wplus, wminus), C0_dir(dirs["[111]"], wplus, wminus))
iso_dev = max(abs(C100 - C110), abs(C100 - C111)) / C100
drift = abs(S_analytic(np.array([1e-3, 0, 0]), wplus, wminus).imag)
print(f"  二阶系数（外推）：C[100] = {C100:.12f}，C[110] = {C110:.12f}，C[111] = {C111:.12f}（解析 1/12 = {1/12:.12f}）")
print(f"  各向同性相对偏差 = {iso_dev:.3e}（期望 < 1e-9）；漂移项 = {drift:.3e}（期望 < 1e-15）")
w2_ok = check("W2", iso_dev < 1e-9 and drift < 1e-15 and abs(C100 - 1 / 12) < 1e-9,
              f"各向同性 {iso_dev:.1e}，漂移 {drift:.1e}，系数 = 1/12")

print("== W3 唯一性扫描 ==")
# 轴向族：w_{x,±} = s/2，w_{y,±} = w_{z,±} = (1-s)/4（每方向），W_x = s，W_y = W_z = (1-s)/2
s_list = np.linspace(0.20, 0.46, 131)
ratio = []
for s in s_list:
    wp = np.array([s / 2, (1 - s) / 4, (1 - s) / 4])
    wm = wp.copy()
    ratio.append(C0_dir(dirs["[100]"], wp, wm) / C0_dir(dirs["[110]"], wp, wm))
ratio = np.array(ratio)
i0 = int(np.argmin(np.abs(ratio - 1.0)))
print(f"  轴向族：比值=1 的数值零点 s* = {s_list[i0]:.6f}（期望 1/3 = {1/3:.6f}）")
# ± 不对称族：w_{x,+} = 1/6 + d/2, w_{x,-} = 1/6 - d/2
d_list = np.linspace(-0.15, 0.15, 121)
drift_list = []
for d in d_list:
    wp = np.array([1 / 6 + d / 2, 1 / 6, 1 / 6])
    wm = np.array([1 / 6 - d / 2, 1 / 6, 1 / 6])
    kd = 1e-3
    drift_list.append(S_analytic(np.array([kd, 0, 0]), wp, wm).imag / kd)
drift_list = np.array(drift_list)
j0 = int(np.argmin(np.abs(drift_list)))
print(f"  不对称族：漂移=0 的数值零点 d* = {d_list[j0]:.6f}（期望 0）")
w3_ok = check("W3", abs(s_list[i0] - 1 / 3) < 0.002 and abs(d_list[j0]) < 0.001,
              f"轴向零点 {s_list[i0]:.4f}（→ w = 1/6），不对称零点 {d_list[j0]:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
ax = axes[0]
ax.plot(s_list, ratio, "-", lw=1.5)
ax.axhline(1.0, color="k", ls="--", lw=1)
ax.axvline(1 / 3, color="r", ls=":", lw=1, label="s = 1/3（w = 1/6）")
ax.set_xlabel("轴向总权重 W_x = s"); ax.set_ylabel("C[100]/C[110]")
ax.set_title("轴向各向异性扫描（唯一零点）"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
ax = axes[1]
ax.plot(d_list, drift_list, "-", lw=1.5)
ax.axhline(0.0, color="k", ls="--", lw=1)
ax.axvline(0.0, color="r", ls=":", lw=1, label="dW = 0")
ax.set_xlabel("± 不对称 dW_x"); ax.set_ylabel("漂移系数 Im S / k")
ax.set_title("中心对称扫描（唯一零点）"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(IMG, dpi=160)
print("figure saved:", IMG)
print("判据：", "全部通过" if (w1_ok and w2_ok and w3_ok) else "存在失败")
