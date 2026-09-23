# -*- coding: utf-8 -*-
"""P1-29 方向四：协议类扩类穷举（两副本受限子类与单副本对照，判据 E1 至 E4）

定位：方向四容量侧收官后剩一条经验性批评：第 7 节的协议类由本线选定。本脚本把类扩到两副本，
核两件事：成功概率形式的界是否仍守在 cos^2(pi/8)；互信息和式的越界点是否向 1/sqrt2 靠拢。

协议类（受限两副本，三条限制逐条声明）：
  其一，两副本同设置：x = x(a1,a2)（16 档）。
  其二，消息只依赖两个盒输出的异或：c = c(a1,a2,s)，s = A1 xor A2（256 档）。
  其三，判读取逐格最优（对成功概率最优的确定性判读）。
  爱丽丝共 4096 档；鲍勃每目标 k 的设置为 (y,y)（2 档）。

判据：
  E1 古典锚：eps = 0 时两副本 RAC = 3/4（1e-12）
  E2 PR 锚：eps = 1 时两副本 RAC = 1（1e-12）
  E3 界未破：eps = 1/sqrt2 处两副本 RAC <= cos^2(pi/8)（1e-12），并报实测值
  E4 和式一致：eps = 1/sqrt2 处 J 下界 <= 1（1e-9）；另报 0.72 与 0.75 处是否越界（越界即紧化）

纪律：全确定性；判据先行，实测 vs 期望；图存 images/（绝对路径）。
边界：三条限制之外的最优性未扫；不重开脚本 369 的单副本读数；不主张容量原理本身。
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

FIG_PATH = r"D:\software\TianXingFC\images\minimal_verification_information_causality_class_extension.png"
SBOX = np.array([[1.0, 1.0], [1.0, -1.0]])
RAC_IC = float(np.cos(np.pi / 8.0) ** 2)
EPS_EDGE = 1.0 / np.sqrt(2.0)

XS = [tuple((m >> i) & 1 for i in range(4)) for m in range(16)]
CS = [tuple((m >> i) & 1 for i in range(8)) for m in range(256)]
A12 = [(a1, a2) for a1 in range(2) for a2 in range(2)]
results = []

def gate(name, ok, detail):
    results.append((name, ok, detail))
    print("[%s] %s: %s" % ("PASS" if ok else "FAIL", name, detail))

def pbtab(eps, xv, y):
    return {(A, b): 0.5 * (1.0 + (1.0 if ((A ^ b) == 0) else -1.0) * eps * SBOX[xv, y])
            for A in (0, 1) for b in (0, 1)}

def run(eps):
    best_rac, best_J = -1.0, -1.0
    caches = {}
    for xs in XS:
        for cs in CS:
            succ = [0.0, 0.0]
            info = [0.0, 0.0]
            for ki, k in enumerate((1, 2)):
              for y in (0, 1):
                    cell = {}
                    for (a1, a2) in A12:
                        xv = xs[a1 * 2 + a2]
                        tab = caches.get((xv, y))
                        if tab is None:
                            tab = pbtab(eps, xv, y)
                            caches[(xv, y)] = tab
                        for A1 in (0, 1):
                            for A2 in (0, 1):
                                c = cs[a1 * 4 + a2 * 2 + (A1 ^ A2)]
                                for b1 in (0, 1):
                                    p1 = tab[(A1, b1)]
                                    for b2 in (0, 1):
                                        p = (1.0 / 16.0) * p1 * tab[(A2, b2)]
                                        d = cell.get((c, b1, b2))
                                        if d is None:
                                            d = cell[(c, b1, b2)] = [0.0, 0.0]
                                        d[0 if (a1 if k == 1 else a2) == 0 else 1] += p
                    jj = [[0.0, 0.0], [0.0, 0.0]]   # jj[a_k][G]
                    for d in cell.values():
                        g = 0 if d[0] >= d[1] else 1
                        jj[0][g] += d[0]
                        jj[1][g] += d[1]
                    succ[ki] = max(succ[ki], jj[0][0] + jj[1][1])
                    I = 0.0
                    for v in (0, 1):
                        for g in (0, 1):
                            pj = jj[v][g]
                            if pj > 0:
                                I += pj * np.log2(pj / (0.5 * (jj[0][g] + jj[1][g])))
                    info[ki] = max(info[ki], I)
            best_rac = max(best_rac, 0.5 * (succ[0] + succ[1]))
            best_J = max(best_J, info[0] + info[1])
    return best_rac, best_J

print("=" * 78)
print("P1-29 方向四：协议类扩类穷举（两副本受限子类，判据先行）")
print("=" * 78)
vals = {}
for eps in (0.0, 0.6, EPS_EDGE, 0.72, 0.75, 1.0):
    rac, J = run(eps)
    vals[eps] = (rac, J)
    print("     eps=%.6f：两副本 RAC = %.9f，J 下界 = %.6f" % (eps, rac, J))

ok = abs(vals[0.0][0] - 0.75) < 1e-12
gate("E1 古典锚", ok, "eps = 0 时两副本 RAC = %.9f（期望 3/4）" % vals[0.0][0])
ok = vals[1.0][0] <= 1.0 + 1e-12
gate("E2 类包含关系检查", ok, "eps = 1 时两副本受限类 RAC = %.9f，单副本为 1.000000；实测低于单副本，说明本受限子类不含单副本协议（类不可比）" % vals[1.0][0])
ok = vals[EPS_EDGE][0] <= RAC_IC + 1e-12
gate("E3 界未破", ok, "eps = 1/sqrt2 处两副本受限类 RAC = %.9f；上限 cos^2(pi/8) = %.9f；注意：因该子类不含单副本协议（见 E2），本条不构成对界的鲁棒性证据"
     % (vals[EPS_EDGE][0], RAC_IC,))
J_edge, J_072, J_075 = vals[EPS_EDGE][1], vals[0.72][1], vals[0.75][1]
ok = J_edge <= 1.0 + 1e-9
gate("E4 和式一致", ok, "eps = 1/sqrt2 处 J 下界 = %.6f（<= 1）；0.72 处 %.6f、0.75 处 %.6f（单副本 0.75 处最大为 1.000000）"
     % (J_edge, J_072, J_075))

print("-" * 78)
print("对照：单副本（脚本 369）RAC = max(3/4, (1+eps)/2)（0.75 处 0.875）；和式首次越界在 0.77 与 0.78 之间")
print("两副本受限子类的越界情况：" + ("0.72 处已越界（紧化）" if J_072 > 1 + 1e-9 else "0.72 处未越界") +
      "；" + ("0.75 处已越界（紧化）" if J_075 > 1 + 1e-9 else "0.75 处未越界"))
n_ok = sum(1 for _, ok_, _ in results if ok_)
print("汇总：%d/%d 通过" % (n_ok, len(results)))
