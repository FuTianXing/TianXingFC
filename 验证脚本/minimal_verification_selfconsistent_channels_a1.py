# -*- coding: utf-8 -*-
r"""
minimal_verification_selfconsistent_channels_a1.py

[识别式阶段 2 第一轮] 自撑阱的四通道显式权重与系数核对（判据 A1-1 至 A1-4）

定位：阶段 2 的任务 A1——把"约束势由包层自撑"从判据 P 的同齐次度推进到逐通道读数。
既有材料（识别式乘数验证）只算了**外加阱**的两组权重（谐振与四次）；本轮补**自撑阱**
（phi^4 的 BPS kink）的权重表，并按均匀标度族核对 kappa = 1。

模型（与判据 P 同族）：V(phi) = (lam/4)(phi^2 - 1)^2，kink phi = tanh(kx)，k^2 = lam/2；
涨落算符 A = -d^2/dx^2 + V''，自洽恒等式 V'' = 4k^2 - 6k^2 sech^2(kx)（常数项 = 能隙，
定域项 = 阱 U = -6k^2 sech^2(kx)）。时钟取形状模（唯一非零束缚态，omega_b^2 = 3k^2）；
权重 w_k = <-d^2>/omega_b^2、w_g = 4k^2/omega_b^2、w_U = <U>/omega_b^2。

判据
A1-1 自洽恒等式点态成立（机器精度）。
A1-2 束缚态谱：两个束缚态本征值 {0, 3k^2}，且随 k 按 k^2 标度（k = 0.5/1/2）。
A1-3 权重表：三通道权重求和恒为 1（构造恒等），给出自撑阱的显式数值。
A1-4 系数核对：均匀标度 (a1,a2,a4) = a、a3 = a+2 下 kappa = 1，且与 a 及权重无关
     （a = 0/1/2 三点核对）。

运行：python 验证脚本/minimal_verification_selfconsistent_channels_a1.py
确定性、无图，运行约 2 秒。
"""
import numpy as np
from scipy.linalg import eigh_tridiagonal

L, NPTS = 20.0, 4001

def build(k):
    x = np.linspace(-L, L, NPTS)
    h = x[1] - x[0]
    phi = np.tanh(k * x)
    # V'' 由势定义独立算：V = (lam/4)(phi^2-1)^2、lam = 2k^2 -> V'' = 2k^2(3 phi^2 - 1)
    vpp = 2.0 * k ** 2 * (3.0 * phi ** 2 - 1.0)
    diag = 2.0 / h ** 2 + vpp
    off = np.full(NPTS - 1, -1.0 / h ** 2)
    w, v = eigh_tridiagonal(diag, off)
    return x, h, phi, vpp, w, v

print("=" * 72)
print("识别式阶段 2 第一轮：自撑阱四通道显式权重与系数核对（A1-1 至 A1-4）")
print("=" * 72)
res = {}
for k in (0.5, 1.0, 2.0):
    x, h, phi, vpp, w, v = build(k)
    ident = float(np.max(np.abs(vpp - (4.0 * k ** 2 - 6.0 * k ** 2 / np.cosh(k * x) ** 2))))
    print(f"[A1-1] k = {k:.1f}：自洽恒等式最大点态偏差 {ident:.2e}")
    bound = w[w < 4.0 * k ** 2 - 1e-6]
    print(f"[A1-2] k = {k:.1f}：束缚态本征值 {np.round(bound, 6)}（解析 0 与 {3*k*k:.4f}）")
    idx = int(np.argmin(np.abs(w - 3.0 * k ** 2)))
    psi = v[:, idx]
    psi = psi / np.linalg.norm(psi)
    kin = float(((psi[1:] - psi[:-1]) ** 2).sum() / h ** 2)
    U = -6.0 * k ** 2 / np.cosh(k * x) ** 2
    wk = kin / w[idx]
    wg = 4.0 * k ** 2 / w[idx]
    wU = float(psi @ (U * psi)) / w[idx]
    res[k] = (wk, wg, wU, float(w[idx]))
    print(f"[A1-3] k = {k:.1f}：omega_b^2 = {w[idx]:.6f}（解析 {3*k*k:.6f}）")
    wa = (7.0 / 15.0, 4.0 / 3.0, -4.0 / 5.0)
    print(f"        w_k = {wk:.6f}，w_g = {wg:.6f}，w_U = {wU:.6f}，求和 = {wk+wg+wU:.12f}（恒为 1）")
    print(f"        解析对照 (7/15, 4/3, -4/5) = ({wa[0]:.6f}, {wa[1]:.6f}, {wa[2]:.6f})；最大偏差 {max(abs(wk-wa[0]), abs(wg-wa[1]), abs(wU-wa[2])):.2e}")

print()
print("[A1-4] 系数核对（用 k = 1 的权重）")
wk, wg, wU, om = res[1.0]
for a in (0.0, 1.0, 2.0):
    a1 = a2 = a4 = a
    a3 = a + 2.0
    kappa = 0.5 * (a3 - a1 * wk - a2 * wg - a4 * wU)
    print(f"   a = {a:.1f}：(a1,a2,a3,a4) = ({a1:.0f},{a2:.0f},{a3:.0f},{a4:.0f}) → kappa = {kappa:.12f}")
print("   读数：系数族为 (a, a, a+2, a)，a 只决定算符侧与本征侧的分摊；kappa = 1 对 a 与权重恒成立。")
print("[A1-2 附] k 标度：omega_b^2/k^2 = " + ", ".join(f"{res[k][3]/k**2:.6f}" for k in (0.5, 1.0, 2.0)) + "（期望 3）")
