# -*- coding: utf-8 -*-
r"""
composite_width.py

[复合模型轮] FK kink 宽度参数区扫描：K/k <= 0.02 时 ell* >= 2 落窗

定位
----
《绕数承载层》第四条（宽度缺口闭合）的固化脚本（原探路脚本 _tools/composite_kink_width.py，
2026-09-06，未固化；对应注记见 验证脚本/注记-复合模型：粒子 = 均匀缠绕背景上的 FK kink
位错与宽度参数区.md）。模型与 conversion_wave 同族：粒子 = 均匀缠绕背景（W = 1 扇区极小）
上的局域位错，锚核 = PN 势（kappa_phi 格点内锁定），包层 = 平方弹性（kappa_parallel）。
本脚本不改动基准文件。

模型
----
一维链 N = 400，E = sum[ ½ k (Δu - 1)^2 + K (1 - cos 2 pi u) ]，k = 1；
边界 u_0 = 0、u_{N-1} = N（均匀缠绕 + 一个位错，W = 1 扇区极小）；初条件在位错台阶（中部 0 → 1）。
弛豫为确定性梯度流（步长 0.005、120000 步）。宽度 ell* 取位错弹性超量密度
½k(Δu - 1)^2 的惯性半径两倍（能量密度分布的展宽）。

判据（先立判据，再实现；预期写错时修正判据并记录教训）
----
CW-1 单调性：ell* 随 r = K/k 单调递减。
CW-2 落窗边界：r <= 0.02 时 ell* >= 2；r = 0.05 时 ell* < 2。
CW-3 大 r 极限：r = 1 时 ell* = 1.00（位错退化为单格台阶）。
CW-4 能量单调：E 随 r 单调增（钉扎代价）。

运行
----
python 验证脚本/composite_width.py
确定性（无随机数），运行约 8 秒。纯打印，不出图。
"""
import time
import numpy as np

T0 = time.time()
CHECKS = []


def check(tag, ok, detail):
    CHECKS.append((tag, bool(ok)))
    print(f"  [{tag}] {'PASS' if ok else 'FAIL'}  {detail}")


NS = 400
K_EL = 1.0


def relax_kink(r, steps=120000):
    K_pn = r * K_EL
    u = np.arange(NS, dtype=float)
    half = NS // 2
    u[half:] += 1.0
    u[half] = half + 0.5
    eta = 0.005
    for _ in range(steps):
        grad = np.zeros(NS)
        grad[1:-1] += K_EL * (2.0 * u[1:-1] - u[:-2] - u[2:])
        grad += K_pn * 2.0 * np.pi * np.sin(2.0 * np.pi * u)
        u[1:-1] -= eta * grad[1:-1]
        u[0] = 0.0
        u[-1] = float(NS)
    E = 0.5 * K_EL * np.sum((np.diff(u) - 1.0) ** 2) \
        + K_pn * np.sum(1.0 - np.cos(2.0 * np.pi * u))
    du = np.diff(u)
    dens = 0.5 * K_EL * (du - 1.0) ** 2
    idx = np.arange(NS - 1) + 0.5
    c0 = (dens * idx).sum() / dens.sum()
    r2 = (dens * (idx - c0) ** 2).sum() / dens.sum()
    return 2.0 * np.sqrt(r2), float(E)


print("=" * 72)
print("[复合模型轮] FK kink 宽度参数区扫描：K/k <= 0.02 时 ell* >= 2")
print("=" * 72)
RS = (0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)
rows = []
print("\n  r = K/k     ell*        E        落窗（ell* >= 2）")
for r in RS:
    w, E = relax_kink(r)
    rows.append((r, w, E))
    print(f"  {r:8.4f}   {w:8.2f}   {E:8.2f}   {'是' if w >= 2.0 else '否'}")

ws = [w for (_, w, _) in rows]
Es = [E for (_, _, E) in rows]
check("CW-1", all(ws[i] > ws[i + 1] for i in range(len(ws) - 1)),
      f"宽度单调递减（{ws[0]:.2f} → {ws[-1]:.2f}）")
r002 = [w for (r, w, _) in rows if abs(r - 0.02) < 1e-12][0]
r005 = [w for (r, w, _) in rows if abs(r - 0.05) < 1e-12][0]
check("CW-2", all(w >= 2.0 for (r, w, _) in rows if r <= 0.02 + 1e-12) and r005 < 2.0,
      f"r <= 0.02 全落窗（r = 0.02 给 {r002:.2f}）；r = 0.05 给 {r005:.2f} < 2")
check("CW-3", abs(ws[-1] - 1.0) < 0.01, f"r = 1 给 ell* = {ws[-1]:.2f}（单格台阶）")
check("CW-4", all(Es[i] < Es[i + 1] for i in range(len(Es) - 1)),
      f"能量单调增（{Es[0]:.2f} → {Es[-1]:.2f}）")

print("\n对照：色散验证宽度趋势 K = 0.0005 → ell = 7.12、K = 0.02 → ell = 1.13（同族 FK 链）")
print("判据汇总：{}/{} 通过".format(sum(1 for _, ok in CHECKS if ok), len(CHECKS)))
for tag, ok in CHECKS:
    if not ok:
        print(f"  未过：{tag}")
print(f"\n[复合模型轮] 完成；总运行 {time.time() - T0:.1f} s。")
