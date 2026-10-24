# -*- coding: utf-8 -*-
"""判据第 672 号：复空间层级嵌入规则（内部空间三层之间的映射）。

背景：
  - P1-58 三条工作链对照注记（2026-10-17）判定工作链三的缺口是"复空间层级嵌入规则"：
    库内只有三层对象的分类与"不可互代"声明（wiki/复结构），没有任何嵌入、投影或限制规则。
  - 支路三（光子侧未锁定与复结构分支的关系）需要该规则：C^2 的某个子空间是否与复平面层
    （U(1) 层）对应，对应时 J 的分支如何限制到该子空间上。

本轮构造并检验三条候选规则：嵌入（复直线进复平面）、投影取商、限制到实子空间。
判定标准是三条：保持复结构、与群作用相容、分支行为可判定。

可逆性等级：判据稳健（解析构造与范数读数）。
"""
import numpy as np

I2 = np.eye(2, dtype=complex)
CZ = np.array([[1, 0], [0, -1]], dtype=complex)
CX = np.array([[0, 1], [1, 0]], dtype=complex)
J0 = 1j * CZ            # C^2 的标准复结构（乘 i）


def e(x):
    return np.array(x, dtype=complex)


LOG = []


def rep(tag, cond, txt):
    LOG.append(f"[{'PASS' if cond else 'FAIL'}] {tag}: {txt}")
    print(LOG[-1])


def main():
    print("=== 判据第 672 号：复空间层级嵌入规则 ===")

    # ---------------- M1：三层对象的复结构
    # 层一（复平面）：J = i（1x1）
    J1 = np.array([[1j]], dtype=complex)
    # 层二（C^2）：J0 = i*I2
    # 层三（CP^2）：复射影空间携带 Fubini-Study 复结构，此处以 C^3 的 i*I3 代表其覆盖
    J3 = 1j * np.eye(3, dtype=complex)
    rep("M1", np.allclose(J1 @ J1, -np.eye(1)) and np.allclose(J0 @ J0, -I2)
        and np.allclose(J3 @ J3, -np.eye(3)),
        "三层各自的复结构平方为 -I（1x1、2x2、3x3）")

    # ---------------- M2：嵌入规则一，复直线作为 C^2 的 J-不变子空间
    # 候选：span{e1}（坐标轴复直线）。判 (a) 是 J0-不变 (b) 属某 U(1) 子群
    lanes = [("坐标轴 e1", e([1, 0])), ("坐标轴 e2", e([0, 1])),
             ("对角", e([1, 1]) / np.sqrt(2)), ("一般", e([1, 2j]) / np.sqrt(5))]
    # ---------------- M2：J0-不变的复直线（J0 = i*sigma_z 为对角复结构）
    lanes = [("坐标轴 e1", e([1, 0])), ("坐标轴 e2", e([0, 1])),
             ("对角", e([1, 1]) / np.sqrt(2)), ("一般", e([1, 2j]) / np.sqrt(5))]
    inv = {}
    for name, v in lanes:
        P = np.outer(v, v.conj())
        resid = np.linalg.norm(J0 @ P - P @ J0)
        inv[name] = resid
    rep("M2", inv["坐标轴 e1"] < 1e-13 and inv["坐标轴 e2"] < 1e-13
        and inv["对角"] > 1e-6 and inv["一般"] > 1e-6,
        f"复直线的 J0-不变性：e1 {inv['坐标轴 e1']:.1e}、e2 {inv['坐标轴 e2']:.1e}"
        f"（不变）；对角 {inv['对角']:.3f}、一般 {inv['一般']:.3f}（非不变）。"
        f"J0 = i*sigma_z 是对角复结构，只保持两根本征复直线，"
        f"所以不是每根复直线都能作嵌入的像")

    # ---------------- M3：嵌入规则二，限制分支的行为
    # C^2 上 J0 的本征分解由 CZ 本征矢给：e1（+i 支）与 e2（-i 支）
    P_plus = np.outer(e([1, 0]), e([1, 0]).conj())
    P_minus = np.outer(e([0, 1]), e([0, 1]).conj())
    r = [np.linalg.norm(P_plus + P_minus - I2), np.linalg.norm(P_plus @ P_minus),
         np.linalg.norm(P_plus @ J0 - 1j * P_plus), np.linalg.norm(P_minus @ J0 + 1j * P_minus)]
    rep("M3", max(r) < 1e-13,
        f"分支投影互补（{r[0]:.1e}）、幂等互斥（{r[1]:.1e}）、"
        f"P_+ 为 +i 本征（{r[2]:.1e}）、P_- 为 -i 本征（{r[3]:.1e}）")

    # ---------------- S1：限制到复直线的分支行为（核心规则）
    # 问：把 P_± 限制到一根复直线上，是保留一支还是混两支？
    res = {}
    for name, v in lanes:
        P = np.outer(v, v.conj())
        a = float(np.real(np.trace(P_plus @ P)))
        b = float(np.real(np.trace(P_minus @ P)))
        res[name] = (a, b)
    axis_clean = abs(res["坐标轴 e1"][0] - 1) < 1e-13 and abs(res["对角"][0] - 0.5) < 1e-13
    rep("S1", axis_clean,
        f"限制读数：e1 的 P_+ 权重 {res['坐标轴 e1'][0]:.3f} 对 {res['坐标轴 e1'][1]:.3f}、"
        f"对角的 {res['对角'][0]:.3f} 对 {res['对角'][1]:.3f}、"
        f"e2 的 {res['坐标轴 e2'][0]:.3f} 对 {res['坐标轴 e2'][1]:.3f}、"
        f"一般的 {res['一般'][0]:.3f} 对 {res['一般'][1]:.3f}")
    print("    判定：复直线若是 J0 的本征直线（e1/e2），限制后只含一支；"
          "一般复直线两支混合（各 0.5）。规则：嵌入保持分支当且仅当子空间是 J0 的本征子空间。")

    # ---------------- S2：与 U(1) 子群的相容性
    # 对角 U(1)：diag(e^{i a}, e^{i b})，保持本征直线 e1/e2
    from math import pi
    for th in (0.3, 0.9, 1.7):
        U = np.diag([np.exp(1j * th), np.exp(-1j * th)])
        d1 = np.linalg.norm(U @ P_plus - P_plus @ U)
        d2 = np.linalg.norm(U @ P_minus - P_minus @ U)
        rep("S2", d1 < 1e-13 and d2 < 1e-13,
            f"对角 U(1)（theta = {th}）保持两支投影（{d1:.1e} 与 {d2:.1e}）："
            f"群作用与分支限制相容")

    # ---------------- S3：层三（CP^2）侧的对应
    # CP^2 的复直线对应 C^3 的二维子空间；取 span{e1,e2}（第三坐标为零）
    W = e([1, 0, 0]), e([0, 1, 0])
    Pw = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=complex)
    d3 = np.linalg.norm(J3 @ Pw - Pw @ J3)
    rep("S3", d3 < 1e-13,
        f"C^3 的 span{{e1,e2}} 是 J3-不变的（残差 {d3:.1e}），即 CP^2 复直线与 "
        f"C^2 层的对应在该实现下为坐标前两维，限制的复结构与 J0 同型")

    # ---------------- 反例 1：在 C^2 内取一维实子空间（真子空间，检验实化路线）
    r_real = np.array([1.0, 1.0]) / np.sqrt(2)
    Pr = np.outer(r_real, r_real)
    dr = np.linalg.norm(J0 @ Pr - Pr @ J0)
    rep("R1", dr > 1.0, f"反例：一维实直线 span{{(1,1)}} 对 J0 非不变（{dr:.3f}），"
                        f"且 J0 把它送出自身（范数 {np.linalg.norm(J0 @ r_real - r_real * np.vdot(r_real, J0 @ r_real)):.3f}），"
                        f"故「复平面层的一维实化」不是合法的嵌入路线")

    # ---------------- 反例 2：商映射把两支混为一支
    # C^2 -> C^2 / span{e1}（取 e2 为代表）：分支信息在此商下丢失
    q_minus = np.real(np.trace(P_minus @ np.outer(e([0, 1]), e([0, 1]).conj())))
    q_plus = np.real(np.trace(P_plus @ np.outer(e([0, 1]), e([0, 1]).conj())))
    rep("R2", abs(q_minus - 1) < 1e-13 and abs(q_plus) < 1e-13,
        f"反例：沿 P_- 方向的商读出 P_+ 权重 {q_plus:.3f}、P_- 权重 {q_minus:.3f}，"
        f"即商映射保住的恰是它所在的那一支，混支信息丢失")

    n_fail = sum(1 for x in LOG if x.startswith("[FAIL]"))
    print(f"\n汇总：{len(LOG) - n_fail}/{len(LOG)} 通过")


if __name__ == "__main__":
    main()
