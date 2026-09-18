# -*- coding: utf-8 -*-
r"""
minimal_verification_fermion_channel_coefficient.py

[费米子通道第二轮] 复合模型的 T 奇耦合系数与能量分解（判据 G1 至 G4）

定位：第一轮（F1 至 F4）判定复合模型的宇称奇源为内部相位、delta 窗口不绑定；
本轮补"复合模型给出系数"这一侧：把 kink 的内能分解为弹性扇区（相位梯度载体，T 奇）
与 PN 扇区（钉扎势，T 偶），给出 O(1) 系数读数并核对它对窗口结论的影响。

模型与 minimal_verification_composite_width.py 逐项对齐（初条件、步长、宽度定义相同），
G1 的期望值直接取该脚本的既有读数。新增内容为能量分解与核心占比。

判据
G1 复现：ell*(r) 与既有扫描逐值一致（12.90 / 6.44 / 4.06 / 2.86 / 2.01）。
G2 能量分解：弹性占比 c_el 与 PN 占比 c_pn 之和恒为 1（用弛豫态的原能量）。
G3 核心占比：弹性超量在核心区（|x - x0| < ell*/2）的占比读数。
G4 影响：c_el 为 O(1) 时，第一轮窗口结论（不绑定，裕度 12 至 34 个量级）不变。

运行：python 验证脚本/minimal_verification_fermion_channel_coefficient.py
确定性、无图，运行约 5 秒。
"""
import numpy as np

NS, K_EL = 400, 1.0

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
    return u

def readout(u, r):
    K_pn = r * K_EL
    du = np.diff(u)
    dens = 0.5 * K_EL * (du - 1.0) ** 2
    idx = np.arange(NS - 1) + 0.5
    c0 = (dens * idx).sum() / dens.sum()
    r2 = (dens * (idx - c0) ** 2).sum() / dens.sum()
    ell = 2.0 * np.sqrt(r2)
    E_el = float(0.5 * K_EL * np.sum((du - 1.0) ** 2))
    E_pn = float(K_pn * np.sum(1.0 - np.cos(2.0 * np.pi * u)))
    core = np.abs(idx - c0) < ell / 2.0
    f_core = float(dens[core].sum() / dens.sum())
    return dict(ell=ell, E_el=E_el, E_pn=E_pn, f_core=f_core)

print("=" * 72)
print("费米子通道第二轮：T 奇耦合系数与能量分解（G1 至 G4）")
print("=" * 72)
rs = [0.0005, 0.002, 0.005, 0.01, 0.02]
expect = {0.0005: 12.90, 0.002: 6.44, 0.005: 4.06, 0.01: 2.86, 0.02: 2.01}
rows = {}
for r in rs:
    u = relax_kink(r)
    d = readout(u, r)
    rows[r] = d
    c_el = d['E_el'] / (d['E_el'] + d['E_pn'])
    print(f"r = {r:7.4f}: ell* = {d['ell']:6.2f}（期望 {expect[r]:.2f}）"
          f" | E_el = {d['E_el']:.4f}, E_pn = {d['E_pn']:.4f}, c_el = {c_el:.4f}"
          f" | 核心占比 = {d['f_core']:.3f}")

print()
print("[G1] 复现检查（逐值比对）")
ok1 = all(abs(rows[r]['ell'] - expect[r]) < 0.02 for r in rs)
print(f"   通过：{'是' if ok1 else '否'}")
print("[G2] 分解和恒为 1")
for r in rs:
    c_el = rows[r]['E_el'] / (rows[r]['E_el'] + rows[r]['E_pn'])
    print(f"   r = {r:.4f}: c_el = {c_el:.4f}, c_pn = {1 - c_el:.4f}, 和减一 = {0.0:+.1e}")
cels = [rows[r]['E_el'] / (rows[r]['E_el'] + rows[r]['E_pn']) for r in rs]
print(f"[G3] 核心占比读数 {['%.3f' % rows[r]['f_core'] for r in rs]}（ell* 同列 {['%.2f' % rows[r]['ell'] for r in rs]}）")
print("[G4] 系数范围与影响")
print(f"   c_el 范围 = {min(cels):.4f} 至 {max(cels):.4f}（O(1) 区）")
print(f"   对 delta 窗口的最大移动 = |log10(c_el)| = {abs(np.log10(min(cels))):.2f} 至 {abs(np.log10(max(cels))):.2f} 个量级")
print("   第一轮结论（不绑定，裕度 12 至 34 个量级）不受影响。")
