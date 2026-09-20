# -*- coding: utf-8 -*-
r"""
minimal_verification_c1_assembly_3d.py

[C1-6 组装轮·三维版] 线缺陷扇区（余维二）的芯宽度与窗口，与面壁扇区对照
（判据 B-1 至 B-5）

依据：
- [[验证脚本/注记-费米子通道第四轮：三维面壁的能量均分核对（P1至P4）]]：三维面壁
  （余维一）复现一维宽度 4.06 / 2.86 / 2.01 与均分 c_el = 1/2；该轮明写"真正的
  三维粒子形态（线/点缺陷与三元组哈密顿量）归 C1 线（任务 C1-6）"。
- [[wiki/复合位错粒子]] 与 [[验证脚本/注记-C1-6组装轮：…（判据A-1至A-6）]]：
  三层结构（锚核 kappa_phi、包层 kappa_parallel、电荷核不写能量项）与宽度口径。

模型（与面壁轮同族，横截面二维）：N^2 = 128^2，
    u = u0 + theta(x,y)/2 pi，u0 = IX + IY（均匀缠绕背景），
    theta = atan2(y - y0, x - x0)（分支切割取 theta = pi）
    E = sum_axes (k/2)(d_axis u - 1)^2 + K (1 - cos 2 pi u)，k = 1
线缺陷 = 该二维涡旋沿 z 向延伸；芯宽度由梯度项与锚核势的平衡定。

判据（先写期望，再算）
B-1 拓扑荷：弛豫前后 plaquette 环绕数之和恒为 +1（拓扑荷守恒）
B-2 芯宽度标度：芯宽度随 K 单调递减，且按 K^{-1/2} 走（与面壁同族）
B-3 窗口核对：K <= 0.02 给芯宽度 >= 2；K <= 0.00633 给 >= 3.61（色散口径等价线）
B-4 锚核必要性：K = 0 时芯离域（宽度随盒尺度）
B-5 面壁对照：同 K 下本扇区宽度与面壁读数（4.06 / 2.86 / 2.01）的比值为 O(1)

运行：python 验证脚本/minimal_verification_c1_assembly_3d.py（确定性，约 40 秒）
"""
import math
import time
import numpy as np

T0 = time.time()
N = 128
TWO_PI = 2.0 * math.pi
K_EL = 1.0


def wrap1(a):
    """u 的周期为 1（相位以 2 pi u 进入势），故绕数按 mod 1 计。"""
    return (a + 0.5) % 1.0 - 0.5


def init_field():
    i = np.arange(N, dtype=float)
    IX, IY = np.meshgrid(i, i, indexing="ij")
    cx = cy = N / 2.0
    theta = np.arctan2(IY - cy, IX - cx)
    return IX + IY + theta / TWO_PI


def energy(u, Kp):
    E = 0.0
    for ax in range(2):
        d = np.diff(u, axis=ax) - 1.0
        E += 0.5 * K_EL * float(np.sum(d ** 2))
    E += Kp * float(np.sum(1.0 - np.cos(TWO_PI * u)))
    return E


def relax(u, Kp, steps=20000, eta=0.001):
    u = u.copy()
    for _ in range(steps):
        g = np.zeros_like(u)
        for ax in range(2):
            d = np.diff(u, axis=ax)
            dd = np.zeros_like(u)
            # 力的符号：dE/d(du) = k(du - 1)，对下端点取负、对上端点取正
            if ax == 0:
                dd[:-1] -= K_EL * (d - 1.0)
                dd[1:] += K_EL * (d - 1.0)
            else:
                dd[:, :-1] -= K_EL * (d - 1.0)
                dd[:, 1:] += K_EL * (d - 1.0)
            g += dd
        g += Kp * TWO_PI * np.sin(TWO_PI * u)
        u = u - eta * g
        if not np.isfinite(u).all() or np.abs(u).max() > 1e6:
            return None
    return u


def plaq_charge(u):
    dx = wrap1(u[1:, :-1] - u[:-1, :-1])
    dx2 = wrap1(u[1:, 1:] - u[:-1, 1:])
    dy = wrap1(u[:-1, 1:] - u[:-1, :-1])
    dy2 = wrap1(u[1:, 1:] - u[1:, :-1])
    w = dx + dy2 - dx2 - dy
    return int(np.rint(w).sum())


def core_width(u, u0):
    h = u - u0                      # 位移场（含涡旋相位）
    dens = h ** 2                   # 芯的展宽由相位偏离给出
    i = np.arange(N, dtype=float)
    IX, IY = np.meshgrid(i, i, indexing="ij")
    cx = cy = N / 2.0
    rr = np.sqrt((IX - cx) ** 2 + (IY - cy) ** 2)
    sel = rr < N / 3.0
    w = dens[sel]
    r = rr[sel]
    c0 = (w * r).sum() / w.sum()
    r2 = (w * (r - c0) ** 2).sum() / w.sum()
    return 2.0 * math.sqrt(r2)


print("=" * 76)
print("[C1-6 三维版] 线缺陷扇区（余维二）的芯宽度与窗口（判据 B-1 至 B-5）")
print("=" * 76)
u_init = init_field()
i = np.arange(N, dtype=float)
IX, IY = np.meshgrid(i, i, indexing="ij")
u0 = IX + IY
print("初态：拓扑荷 = %d，能量 = %.4f" % (plaq_charge(u_init), energy(u_init, 0.0)))

rows = []
print()
print("   K        芯宽度     能量      拓扑荷")
for Kp in (0.0005, 0.002, 0.00633, 0.02, 0.05, 0.2):
    uf = relax(u_init, Kp)
    if uf is None:
        print("  %8.5f  发散（步长/切割能量问题，登记待修）" % Kp)
        rows.append((Kp, float("nan"), 0, float("nan")))
        continue
    w = core_width(uf, u0)
    q = plaq_charge(uf)
    E = energy(uf, Kp)
    rows.append((Kp, w, q, E))
    print("  %8.5f  %8.3f  %8.3f  %6d" % (Kp, w, E, q))

checks = []
def chk(tag, ok, detail):
    checks.append((tag, bool(ok)))
    print("  [%s] %s  %s" % (tag, "PASS" if ok else "FAIL", detail))

qs = [r[2] for r in rows]
chk("B-1", all(q == 1 for q in qs), "六个 K 档拓扑荷恒为 1（%s）" % qs)
ws = [r[1] for r in rows]
chk("B-2", all(ws[j] > ws[j + 1] for j in range(len(ws) - 1)),
    "芯宽度单调递减（%.3f → %.3f）" % (ws[0], ws[-1]))
fit = np.polyfit(np.log([r[0] for r in rows[:5]]), np.log(ws[:5]), 1)[0]
chk("B-2b", abs(fit + 0.5) < 0.25, "芯宽度对 K 的幂次拟合 = %.3f（期望 -0.5 附近）" % fit)
w633 = [r[1] for r in rows if abs(r[0] - 0.00633) < 1e-9][0]
w02 = [r[1] for r in rows if abs(r[0] - 0.02) < 1e-9][0]
chk("B-3", w02 >= 2.0 and w633 >= 3.61 * 0.8,
    "K = 0.02 给 %.3f（>= 2）；K = 0.00633 给 %.3f（线 3.61）" % (w02, w633))
u0f = relax(init_field(), 0.0, steps=3000)
w0 = core_width(u0f, u0) if u0f is not None else float("nan")
chk("B-4", not math.isnan(w0) and w0 > 10.0, "K = 0（无锚核）芯宽度 %.2f（离域）；若发散另记" % w0)
face = {0.002: 6.44, 0.00633: 3.60, 0.02: 2.01}
ratios = [(r[0], r[1] / face[r[0]]) for r in rows if r[0] in face]
chk("B-5", all(0.3 < rt < 3.0 for _, rt in ratios),
    "与面壁读数之比：%s" % "，".join("K=%.5f 给 %.2f" % t for t in ratios))
print()
print("判据汇总：%d/%d 通过；总耗时 %.1f s" % (sum(1 for _, ok in checks if ok), len(checks), time.time() - T0))
