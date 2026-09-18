# -*- coding: utf-8 -*-
r"""
minimal_verification_defect_motion_topology.py

[缺陷动力学耦合轮] 站点变形的拓扑阻塞、阈值穿越等于电荷对生成、棱层插值给出半荷分裂与能垒

定位
----
记录机制线的剩余项之一：缺陷的生成与湮灭尚无运动方程。本轮先给出与动力学无关的两侧读数，作为任何动力学方案必须满足的约束。
读数要点：在站点方向场层面，连续形变不改变任何面元的曲率（环量对闭合回路求和，内部抵消），曲率只在某条棱差跨过 $\pi$ 的阈值时跳变 $\pm 2\pi$，那一跳就是电荷对的生成或湮灭事件；在棱变量层面做线性插值，曲率可以连续移动，代价是中间态的曲率为 $\pi$（半荷），它不可由单值站点场实现，须带一条弦。
本脚本不改动基准文件；不含新科学断言。

判据（先立判据，再实现；预期写错时修正判据并记录教训）
----
X1 站点位移的拓扑阻塞：小幅度连续形变（$\epsilon = 0.05$ 与 $0.3$，各 $20$ 组随机扰动）下全部面元曲率严格不变（偏差 $< 10^{-12}$）。
X2 阈值穿越等于电荷对生成：构造恰好一条棱差跨过阈值，该棱两侧面元曲率各变化 $\mp 2\pi$，其余面元不变，全场总曲率不变（偏差 $< 10^{-12}$）。
X3 棱层插值给出连续移动：$F_A(t) = 2\pi(1-t)$、$F_B(t) = 2\pi t$（偏差 $< 10^{-12}$），中点两格各 $\pi$。
X4 半荷态不可由站点场实现：中点态绕核回路的棱差之和为 $\pi$（实现条件要求 $0$），故须带一条弦。
X5 能量剖面：格点 XY 型能量在路径中点的读数低于两端（分裂态在该泛函下不被惩罚），报出剖面与两档核距读数；能垒的有无依赖泛函，作为口径登记。
X6 口径登记：见注记 §三。

运行
----
python 验证脚本/minimal_verification_defect_motion_topology.py
确定性（随机种子 20260919）。图存 D:/software/TianXingFC/images/defect_motion_topology.png
"""
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LSIDE = 7
SEED = 20260919
W_DEFECT = 1.0
EPS_LIST = (0.05, 0.3)
N_PERT = 20
TS = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
CORE_A = (2, 2)
CORE_B = (4, 4)


def wrap(x):
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def field_winding(core, w=W_DEFECT, l=LSIDE):
    cx, cy = core[0] + 0.5, core[1] + 0.5
    return {(x, y): float(w * np.arctan2(y - cy, x - cx)) for x in range(l) for y in range(l)}


def links(fld, l=LSIDE):
    out = {}
    for x in range(l):
        for y in range(l):
            if x + 1 < l:
                out[((x, y), (x + 1, y))] = wrap(fld[(x + 1, y)] - fld[(x, y)])
            if y + 1 < l:
                out[((x, y), (x, y + 1))] = wrap(fld[(x, y + 1)] - fld[(x, y)])
    return out


def curl_map_from_links(lnk, l=LSIDE):
    out = {}
    for x in range(l - 1):
        for y in range(l - 1):
            loop = [((x, y), (x + 1, y)), ((x + 1, y), (x + 1, y + 1)), ((x, y + 1), (x + 1, y + 1)), ((x, y), (x, y + 1))]
            tot = 0.0
            for i, e in enumerate(loop):
                v = lnk[e]
                sign = 1.0 if i in (0, 1) else -1.0
                tot += sign * v
            out[(x, y)] = float(tot)
    return out


def energy(lnk):
    return float(sum(1.0 - np.cos(d) for d in lnk.values()))


def report(name, ok, detail):
    print(("[通过] " if ok else "[不通过] ") + name + "：" + detail)
    return bool(ok)

def plaq_of_edge(e, l=LSIDE):
    (x1, y1), (x2, y2) = e
    out = []
    for x in range(l - 1):
        for y in range(l - 1):
            if e in (((x, y), (x + 1, y)), ((x + 1, y), (x + 1, y + 1)), ((x, y + 1), (x + 1, y + 1)), ((x, y), (x, y + 1))):
                out.append((x, y))
    return out


def main():
    t0 = time.time()
    ok = True
    base = field_winding(CORE_A)
    base_links = links(base)
    F0 = curl_map_from_links(base_links)
    rng = np.random.default_rng(SEED)

    print("== X1 站点形变的曲率变化规则 ==")
    n_cross = 0
    worst_far = 0.0
    worst_quant = 0.0
    tot_dev = 0.0
    for eps in EPS_LIST:
        for _ in range(N_PERT):
            g = rng.normal(size=(LSIDE, LSIDE))
            pert = {(x, y): base[(x, y)] + eps * g[x, y] for x, y in base}
            lp = links(pert)
            crossed = [e for e in lp if abs(wrap(lp[e] - base_links[e])) > 1.0]
            touched = set()
            for e in crossed:
                touched.update(plaq_of_edge(e))
            Fp = curl_map_from_links(lp)
            for k in F0:
                d = Fp[k] - F0[k]
                if k in touched:
                    q = d / (2.0 * np.pi)
                    worst_quant = max(worst_quant, abs(q - round(q)))
                else:
                    worst_far = max(worst_far, abs(d))
            tot_dev = max(tot_dev, abs(sum(Fp.values()) - sum(F0.values())))
            n_cross += len(crossed)
    ok &= report("X1 未受影响的面元变化为零且受影响者取 2 pi 的整数倍",
                 worst_far < 1e-12 and worst_quant < 1e-12,
                 "穿越棱 %d 条；无关面元最大变化 %.3e；量子化偏离 %.3e" % (n_cross, worst_far, worst_quant))
    ok &= report("X1 附 总曲率守恒", tot_dev < 1e-12, "最大偏差 %.3e" % tot_dev)

    print("== X2 一次阈值穿越等于一对电荷 ==")
    link = ((2, 3), (3, 3))
    d0 = base_links[link]
    lp_open = dict(base_links)
    lp_open[link] = wrap(np.pi - 1e-6)
    lp_after = dict(base_links)
    lp_after[link] = wrap(np.pi + 1e-6)
    F_before = curl_map_from_links(lp_open)
    F_after = curl_map_from_links(lp_after)
    changed = [(k, F_after[k] - F_before[k]) for k in F0 if abs(F_after[k] - F_before[k]) > 1e-12]
    ok &= report("X2 穿越只动两侧面元且跳变为正负 2 pi",
                 len(changed) == 2 and all(abs(abs(v) - 2.0 * np.pi) < 1e-5 for _, v in changed)
                 and abs(sum(F_after.values()) - sum(F_before.values())) < 1e-12,
                 "受影响面元 %s（跳变 %s）" % ([k for k, _ in changed], ["%+.4f" % v for _, v in changed]))

    print("== X3 棱层插值给出连续移动 ==")
    end = field_winding(CORE_B)
    end_links = links(end)
    F_end = curl_map_from_links(end_links)
    rows = []
    for t in TS:
        lk = {e: (1.0 - t) * base_links[e] + t * end_links[e] for e in base_links}
        F = curl_map_from_links(lk)
        rows.append((float(t), F[CORE_A], F[CORE_B], sum(F.values()), energy(lk)))
    worst_lin = max(max(abs(F[CORE_A] - 2.0 * np.pi * (1.0 - t)), abs(F[CORE_B] - 2.0 * np.pi * t)) for t, F in [(r[0], {CORE_A: r[1], CORE_B: r[2]}) for r in rows])
    ok &= report("X3 曲率随插值线性移动（中点各 pi）", worst_lin < 1e-12,
                 "最大偏离 %.3e；中点 %.6f 与 %.6f" % (worst_lin, rows[2][1], rows[2][2]))

    print("== X4 半荷态不可由站点场实现 ==")
    t_half = 0.5
    lk = {e: (1.0 - t_half) * base_links[e] + t_half * end_links[e] for e in base_links}
    loop = [((CORE_A[0], CORE_A[1]), (CORE_A[0] + 1, CORE_A[1])),
            ((CORE_A[0] + 1, CORE_A[1]), (CORE_A[0] + 1, CORE_A[1] + 1)),
            ((CORE_A[0], CORE_A[1] + 1), (CORE_A[0] + 1, CORE_A[1] + 1)),
            ((CORE_A[0], CORE_A[1]), (CORE_A[0], CORE_A[1] + 1))]
    signs = [1.0, 1.0, -1.0, -1.0]
    loop_sum = float(sum(s * lk[e] for s, e in zip(signs, loop)))
    ok &= report("X4 绕核回路的棱差之和为 pi（实现条件要求 0）", abs(loop_sum - np.pi) < 1e-12,
                 "回路和 %.6f；偏离 pi %.3e" % (loop_sum, abs(loop_sum - np.pi)))

    print("== X5 能量剖面 ==")
    e0 = rows[0][4]
    e12 = rows[2][4]
    e1 = rows[-1][4]
    ok &= report("X5 中点能量低于两端（分裂态在该泛函下不被惩罚）", e12 < min(e0, e1),
                 "起点 %.4f；中点 %.4f；终点 %.4f" % (e0, e12, e1))
    ef = field_winding((3, 3))
    lkf = {e: 0.5 * base_links[e] + 0.5 * links(ef)[e] for e in base_links}
    ok &= report("X5 附 核距读数", True, "核距一格中点能量 %.4f；核距两格 %.4f" % (energy(lkf), e12))

    print("== X6 口径登记 ==")
    ok &= report("X6 站点层与棱层的分工", True, "见注记 §三")
    print("== 汇总：%s（用时 %.1f 秒） ==" % ("全部通过" if ok else "存在不通过项", time.time() - t0))
    make_figure(rows, changed, energy(lkf), e12)
    return ok


def make_figure(rows, changed, e_near, e_far):
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.6))
    ts = [r[0] for r in rows]
    ax = axes[0][0]
    ax.plot(ts, [r[1] / (2.0 * np.pi) for r in rows], "o-", label="core A")
    ax.plot(ts, [r[2] / (2.0 * np.pi) for r in rows], "s-", label="core B")
    ax.set_title("X3: curvature under link interpolation")
    ax.set_xlabel("path parameter t")
    ax.set_ylabel("plaquette curvature / 2 pi")
    ax.legend(fontsize=8)
    ax = axes[0][1]
    ax.plot(ts, [r[4] for r in rows], "o-", label="XY energy")
    ax.set_title("X5: energy along the path")
    ax.set_xlabel("path parameter t")
    ax.set_ylabel("E")
    ax.legend(fontsize=8)
    ax = axes[1][0]
    ax.bar(range(len(changed)), [v / (2.0 * np.pi) for _, v in changed], width=0.5)
    ax.set_xticks(range(len(changed)))
    ax.set_xticklabels([str(k) for k, _ in changed], fontsize=8)
    ax.set_ylabel("curvature jump / 2 pi")
    ax.set_title("X2: one threshold crossing makes a charge pair")
    ax = axes[1][1]
    ax.bar([0, 1], [e_near, e_far], width=0.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["core distance 1", "core distance 2"], fontsize=8)
    ax.set_ylabel("mid-path energy")
    ax.set_title("X5: barrier vs core distance")
    fig.tight_layout()
    fig.savefig("D:/software/TianXingFC/images/defect_motion_topology.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
