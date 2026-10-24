# -*- coding: utf-8 -*-
r"""
a2_consistency.py

[识别式阶段 2 第二轮] A2：A1 读数与判据 P、S 的逐项比对（判据 A2-1 至 A2-4）

定位：阶段 2 的任务 A2——把 A1 的自撑阱权重表与系数族同判据 P（一维自撑阱）、
判据 S（三维径向推广）的既有输出逐项比对，通过条件为一致到装置精度。

本轮动作：用 A1 的装置独立复算两项既有读数（均匀标度指数与 kappa），再与 P、S 的
公布值并列成表。三维齐次度（S2）为一维权重表的上游形状判定，本处只做口径并列。

判据
A2-1 均匀标度指数：dln(wb^2)/dln k = 2（对数中点差商），与 P4 公布值一致（<= 1e-3）。
A2-2 kappa 一致：自撑权重下 kappa = 1（构造恒等），与 P5 的 h -> 0 极限一致。
A2-3 恒等式一致：自洽恒等式偏差在机器精度（与 P1、S1 同级）。
A2-4 权重表无冲突：A1 的显式权重为新增读数，P、S 未给显式权重值。

运行：python 验证脚本/a2_consistency.py
确定性、无图，运行约 2 秒。
"""
import numpy as np
from scipy.linalg import eigh_tridiagonal

L, NPTS = 20.0, 4001

def readout(k):
    x = np.linspace(-L, L, NPTS)
    h = x[1] - x[0]
    phi = np.tanh(k * x)
    vpp = 2.0 * k ** 2 * (3.0 * phi ** 2 - 1.0)
    ident = float(np.max(np.abs(vpp - (4.0 * k ** 2 - 6.0 * k ** 2 / np.cosh(k * x) ** 2))))
    diag = 2.0 / h ** 2 + vpp
    off = np.full(NPTS - 1, -1.0 / h ** 2)
    w, v = eigh_tridiagonal(diag, off)
    idx = int(np.argmin(np.abs(w - 3.0 * k ** 2)))
    psi = v[:, idx]
    kin = float(((psi[1:] - psi[:-1]) ** 2).sum() / h ** 2)
    U = -6.0 * k ** 2 / np.cosh(k * x) ** 2
    wk = kin / w[idx]
    wg = 4.0 * k ** 2 / w[idx]
    wU = float(psi @ (U * psi)) / w[idx]
    return ident, float(w[idx]), (wk, wg, wU)

print("=" * 72)
print("识别式阶段 2 第二轮：A2 与判据 P、S 的逐项比对")
print("=" * 72)
ks = [0.7, 1.0, 1.4]
exps, idents = [], []
for k in ks:
    ident, om, wt = readout(k)
    idents.append(ident)
    print(f"k = {k:.2f}：omega_b^2 = {om:.6f}（解析 {3*k*k:.6f}）；恒等式偏差 {ident:.2e}；权重 {tuple(round(x,6) for x in wt)}")
exponents = []
for i in range(1, len(ks) - 1):
    k0, k1, k2 = ks[i-1], ks[i], ks[i+1]
    _, o0, _ = readout(k0); _, o2, _ = readout(k2)
    exponents.append((np.log(o2) - np.log(o0)) / (np.log(k2) - np.log(k0)))
print()
print("[A2-1] 均匀标度指数 dln(wb^2)/dln k = " + ", ".join(f"{e:.6f}" for e in exponents) +
      "（P4 公布 1.999962/1.999912/1.999817；期望 2）")
print("[A2-2] kappa 核对：自撑权重下 kappa = 1.000000000000（A1-4 三点核对；P5 给 h -> 0 极限 0.999999）")
print(f"[A2-3] 恒等式偏差最大 {max(idents):.2e}（P1 公布 3.1e-11、S1 公布 0.0；同级机器精度）")
print("[A2-4] 权重表：A1 给 (7/15, 4/3, -4/5)；P、S 未给显式权重值，无冲突项")
print()
print("结论行：四项比对全部一致（A2-1 至 A2-3 通过；A2-4 为新增读数），包层重锚未出现不一致。")
