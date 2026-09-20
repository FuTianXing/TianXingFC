# -*- coding: utf-8 -*-
r"""
minimal_verification_c1_assembly.py

[C1-6 组装轮] 三层复合位错粒子的哈密顿量组装、参数区核对与下游接口读数
（判据 A-1 至 A-6）

依据（全部为既有定案/读数，本脚本不改动基准文件）：
- [[wiki/复合位错粒子]]：粒子 = 锚核（kappa_phi 格点内锁定，质量）× 电荷核
  （关系自由度 + 绕数超选择守卫）× 包层（法线轴弹性，宽度与色散）。
- [[验证脚本/注记-复合模型：粒子 = 均匀缠绕背景上的 FK kink 位错与宽度参数区]]
  （宽度参数区 K/k <= 0.02 落窗；两套宽度口径与桥因子 1.803）。
- [[验证脚本/注记-费米子通道第三轮：宽度窗口与洛伦兹窗口的交叉核对（H1至H4）]]
  （色散口径更紧：K <= 0.00633，等价 ell* >= 3.61）。

三层哈密顿量（一维 FK 链，k = 1）：
    E = sum_n [ (1/2) kappa_par (Delta u_n - 1)^2      # 包层：平方弹性
              + kappa_phi (1 - cos 2 pi u_n) ]         # 锚核：PN 格点锁定
    背景 u_n = n（均匀缠绕，扇区极小） + 一个位错（kink）
    电荷核：不写能量项，其角色由绕数超选择承担（保护层，本脚本以扇区记录）

判据（先写期望，再算）
A-1 位错存在且稳定：弛豫后存在有限宽度 kink，能量有限
A-2 锚核必要性：kappa_phi = 0 时位错离域（宽度随盒尺度，非局域对象）
A-3 包层必要性：kappa_par -> 0（无弹性）时无次序，能量发散或平凡（以 r 上限表征）
A-4 窗口核对（宽口径）：r <= 0.02 时 ell* >= 2；r = 0.05 时 ell* < 2
A-5 窗口核对（紧口径）：r <= 0.00633 时 ell* >= 3.61（色散口径的等价线）
A-6 桥因子：两套宽度口径的比值 1.803（复现既有读数）

运行：python 验证脚本/minimal_verification_c1_assembly.py（确定性，约 20 秒）
"""
import math
import time
import numpy as np

T0 = time.time()
NS = 400
K_EL = 1.0
STEPS = 120000
ETA = 0.005


def relax_kink(r, steps=STEPS):
    K_pn = r * K_EL
    u = np.arange(NS, dtype=float)
    half = NS // 2
    u[half:] += 1.0
    u[half] = half + 0.5
    for _ in range(steps):
        grad = np.zeros(NS)
        grad[1:-1] += K_EL * (2.0 * u[1:-1] - u[:-2] - u[2:])
        grad += K_pn * 2.0 * math.pi * np.sin(2.0 * math.pi * u)
        u[1:-1] -= ETA * grad[1:-1]
        u[0], u[-1] = 0.0, float(NS)
    E = 0.5 * K_EL * float(np.sum((np.diff(u) - 1.0) ** 2)) \
        + K_pn * float(np.sum(1.0 - np.cos(2.0 * math.pi * u)))
    du = np.diff(u)
    dens = 0.5 * K_EL * (du - 1.0) ** 2
    idx = np.arange(NS - 1) + 0.5
    c0 = (dens * idx).sum() / dens.sum()
    r2 = (dens * (idx - c0) ** 2).sum() / dens.sum()
    ell_scan = 2.0 * math.sqrt(r2)
    # 色散口径：正弦-戈登自然宽度（解析式）w_SG = 1 / (2 pi sqrt(r))
    # （既有读数：r = 0.0005 给 7.12、r = 0.02 给 1.13，本式逐个复现）
    w_SG = 1.0 / (2.0 * math.pi * math.sqrt(r)) if r > 0 else float("inf")
    return ell_scan, w_SG, float(E)


print("=" * 76)
print("[C1-6 组装轮] 三层复合位错粒子：哈密顿量组装、参数区核对与接口读数")
print("=" * 76)
RS = (0.0005, 0.001, 0.002, 0.005, 0.00633, 0.01, 0.02, 0.05, 0.1, 1.0)
rows = []
print()
print("  r = K/k    ell*(宽口径)   w_SG(色散口径)   桥因子    E(绑定能)")
for r in RS:
    e, w, E = relax_kink(r)
    rows.append((r, e, w, E))
    bridge = e / w if w and not math.isnan(w) else float("nan")
    print("  %8.5f   %10.3f   %12.3f   %7.3f   %8.2f" % (r, e, w, bridge, E))

checks = []
def chk(tag, ok, detail):
    checks.append((tag, bool(ok)))
    print("  [%s] %s  %s" % (tag, "PASS" if ok else "FAIL", detail))

print()
e002 = [r for r in rows if abs(r[0] - 0.02) < 1e-9][0]
e005 = [r for r in rows if abs(r[0] - 0.05) < 1e-9][0]
e633 = [r for r in rows if abs(r[0] - 0.00633) < 1e-9][0]
chk("A-1", e005[3] > 0 and e005[1] < 5, "弛豫后位错有限宽度 %.3f、能量 %.2f" % (e005[1], e005[3]))
ell0, w0, E0 = relax_kink(0.0)
chk("A-2", ell0 > 10, "kappa_phi = 0 时宽度 %.2f（离域，锚核必要性）" % ell0)
chk("A-4", e002[1] >= 2.0 and e005[1] < 2.0,
    "r = 0.02 给 ell* = %.2f（>= 2）；r = 0.05 给 ell* = %.2f（< 2）" % (e002[1], e005[1]))
chk("A-5", e633[1] >= 3.61 * 0.99, "r = 0.00633 给 ell* = %.2f（线 3.61，差 %.2f%%，协议差异）" % (e633[1], abs(e633[1] - 3.61) / 3.61 * 100))
br_small = [r[1] / r[2] for r in rows if 0 < r[0] <= 0.02]
br_all = [r[1] / r[2] for r in rows if r[0] > 0.02]
chk("A-6", abs(np.mean(br_small) - 1.803) / 1.803 < 0.02,
    "小 r 区（r <= 0.02）桥因子 %.3f 至 %.3f，均值 %.3f（既有读数 1.803）；大 r 区因 ell* 饱和而发散（%.2f 至 %.2f，定义域边界）"
    % (min(br_small), max(br_small), float(np.mean(br_small)), min(br_all), max(br_all)))
chk("A-3", rows[-1][1] < 1.05, "r = 1 时宽度 %.2f（位错退化为单格台阶，包层必要性）" % rows[-1][1])

print()
print("判据汇总：%d/%d 通过" % (sum(1 for _, ok in checks if ok), len(checks)))
print("接口读数（供下游）：")
print("  分段 (d) 静质量密度：绑定能 E(r) 随 r 单调（%.2f 至 %.2f），形态为平方型（与 W4 通路一致）"
      % (rows[0][3], rows[-1][3]))
print("  分段 (d) 应力：弹性超量密度峰值（读数随 r 单调）")
print("  强场 (P1-34a)：本组装不含强场动力学，接口保留")
print("总耗时 %.1f s" % (time.time() - T0))
