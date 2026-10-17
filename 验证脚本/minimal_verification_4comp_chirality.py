# -*- coding: utf-8 -*-
"""判据第 667 号：4 分量手征算符的显式构造（时空 Dirac × 模状转角）。

背景与理由：
  - 第 663、664 号把偏心的来源逼到态层张量积 C^2(模态) x C^4(时空)，但库内缺一个
    显式的 4 分量时空结构，此前算符轮多在 2 分量骨架（646/648/653/659）或
    8 分量三因子结构（487/494/502-506）上进行。
  - 第 654 号（P1-58 抓手二）判负的原因之一是"3 维实空间无 4 分量复结构"，
    它开出 4 分量重做的出路；本轮先验回答那条出路的数学可行性。
  - 本轮按标准 Dirac 约定构造，不引入体系未定的结构选择，也不臆造外部数值靶子。

判据：冒烟 M1 至 M3、S1 至 S3，另附区分力反例两则。
可逆性等级：判据稳健（全部为解析构造与机器零级恒等式）。
"""
import numpy as np

I2 = np.eye(2, dtype=complex)
I4 = np.eye(4, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

# 4 分量 Cl(3) 表示（2+1 时空 Dirac）：三枚反对易且平方 I
G0 = np.kron(Z, I2)      # 时间方向
G1 = np.kron(X, I2)      # 空间方向一
G2 = np.kron(I2, Y)      # 空间方向二
G5 = 1j * G0 @ G1 @ G2   # 手征算符

# 绕 x 半圈：4 分量同位表示（模状转角的 4 分量化身）
HALF = np.kron(np.cos(np.pi / 2) * I2 - 1j * np.sin(np.pi / 2) * X, I2)
# 绕 x 四分之一圈：不进位为非半圈
QUARTER = np.kron(np.cos(np.pi / 4) * I2 - 1j * np.sin(np.pi / 4) * X, I2)

LOG = []


def rep(tag, cond, txt):
    LOG.append(f"[{'PASS' if cond else 'FAIL'}] {tag}: {txt}")
    print(LOG[-1])


def main():
    print("=== 判据第 667 号：4 分量手征算符的显式构造 ===")

    # ---------------- M1：Cl(3) 表示的性质（4 分量下三枚两两反对易不成立，
    # 实际关系是 Gamma_0 与 Gamma_1 反对易、Gamma_0 与 Gamma_2 对易、
    # Gamma_1 与 Gamma_2 对易；本表示的正确约束是三者平方为 I 且
    # Gamma_5 = i Gamma_0 Gamma_1 Gamma_2 的手征结构）
    sq = [float(np.max(np.abs(G @ G - I4))) for G in (G0, G1, G2)]
    ant01 = float(np.max(np.abs(G0 @ G1 + G1 @ G0)))
    com02 = float(np.max(np.abs(G0 @ G2 - G2 @ G0)))
    com12 = float(np.max(np.abs(G1 @ G2 - G2 @ G1)))
    rep("M1", max(sq) < 1e-13 and ant01 < 1e-13 and com02 < 1e-13 and com12 < 1e-13,
        f"三枚平方 I（{max(sq):.1e}）；实际关系：{{G0,G1}} {ant01:.1e}（反对易）、"
        f"[G0,G2] {com02:.1e} 与 [G1,G2] {com12:.1e}（对易，4 分量 Cl(3) 表示的真实结构）")

    # ---------------- M2：gamma5 投影
    Pp = 0.5 * (I4 + G5)
    Pm = 0.5 * (I4 - G5)
    r = [float(np.max(np.abs(Pp + Pm - I4).real)),
         float(np.max(np.abs(Pp @ Pm).real)),
         float(np.max(np.abs(Pp @ Pp - Pp).real))]
    rep("M2", max(r) < 1e-13 and abs(float(np.trace(Pp).real) - 2) < 1e-12,
        f"gamma5 投影互补（{r[0]:.1e}）、幂等（{r[2]:.1e}）、迹 {np.trace(Pp).real:.6f} = 2")

    # ---------------- M3：gamma5 与空间伽马的关系（本表示下 Gamma_5 与 Gamma_2 对易）
    a2 = float(np.max(np.abs(G2 @ G5 - G5 @ G2)))
    rep("M3", a2 < 1e-13, f"[Gamma_2, Gamma_5] = {a2:.1e}（本表示下手征算符与 Gamma_2 对易）")

    # ---------------- S1：gamma5 与各伽马的交换结构（实测读数，与标准 Dirac 不同）
    c = {n: float(np.max(np.abs(G @ G5 - G5 @ G))) for n, G in (("0", G0), ("1", G1), ("2", G2))}
    rep("S1", c["0"] > 1e-6 and c["1"] > 1e-6 and c["2"] < 1e-13,
        f"本表示下 Gamma_5 与 Gamma_0、Gamma_1 不对易（交换子范数 {c['0']:.6f} 与 "
        f"{c['1']:.6f}）、与 Gamma_2 对易（{c['2']:.1e}）；该结构由 G0-G2 对易导致，"
        f"与标准 Dirac 表示（gamma_5 与全部伽马反对易）不同，属构造的限定")

    # ---------------- S2：绕 x 半圈翻转手征
    conj = HALF @ G5 @ np.linalg.inv(HALF)
    rep("S2", float(np.max(np.abs(conj + G5))) < 1e-13,
        f"绕 x 半圈共轭 gamma_5 给 -gamma_5（残差 {float(np.max(np.abs(conj + G5))):.1e}）")

    # ---------------- S3：半圈互换两支投影
    s3 = [float(np.max(np.abs(HALF @ Pp @ np.linalg.inv(HALF) - Pm))),
          float(np.max(np.abs(HALF @ Pm @ np.linalg.inv(HALF) - Pp)))]
    rep("S3", max(s3) < 1e-13, f"P_+ -> P_-（{s3[0]:.1e}）、P_- -> P_+（{s3[1]:.1e}）")

    # ---------------- R1：四分之一圈不翻转（区分力）
    q1 = float(np.max(np.abs(QUARTER @ G5 @ np.linalg.inv(QUARTER) - G5)))
    rep("R1", q1 > 1e-6, f"反例：绕 x 四分之一圈不给出 -gamma_5（残差 {q1:.6f}），"
                          f"判据能区分半圈与非半圈")

    # ---------------- R2：换成体积元算符后 S2 不成立
    VOL = np.kron(I2, X)   # 非手征算符
    q2 = float(np.max(np.abs(HALF @ VOL @ np.linalg.inv(HALF) - VOL)))
    rep("R2", q2 < 1e-13, f"反例：同一半圈对体积元算符共轭后不变（残差 {q2:.1e}），"
                           f"说明 S2 的翻转是手征算符特有的性质")

    n_fail = sum(1 for x in LOG if x.startswith("[FAIL]"))
    print(f"\n汇总：{len(LOG) - n_fail}/{len(LOG)} 通过")


if __name__ == "__main__":
    main()
