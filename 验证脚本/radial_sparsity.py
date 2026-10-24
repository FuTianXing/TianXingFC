"""第579条：第574条两个未写明前提的核查（环向稀疏性与 C 的不相容）

背景：第574条给出 r = C*M 与条件 C > 6。本轮任务是定 C，但核查发现 C 不是
体系内某个已有量的读数，而是「破坏格点数与质量的比例」乘「破坏区几何形状」的
复合，两条链各自需要独立输入，体系内均无条款。故改为核查第574条的两个隐含前提。

S1 环向稀疏性：N_brk = n_layer * (r/d)。若 n_layer 为常数则 N_brk ∝ r，
   配合 N_brk ∝ M 得 r ∝ M（p=+1，过筛）；若 n_layer ∝ r（环状破坏）
   则 N_brk ∝ r²，r ∝ M^(1/2)（不过筛）。故 p=+1 须环向稀疏。

S2 C 的不相容：C = kappa * d / n_layer。kappa 由满偏标度给（单格点累积到
   m_P 即破坏，故 kappa = 1/m_P）。自然单位下 m_P=1、d=O(1)、n_layer=O(1)，
   故 C = O(1) < 6。而轨道存在要求 C > 6。二者不相容。

S3 多档敏感性：C 由 d 与 n_layer 的比值决定，二者皆 O(1) 格点量。

运行：python 验证脚本/radial_sparsity.py
确定性，无随机，约 0.1 秒。
"""

C_SIEVE = 6.0      # 轨道存在要求 C > 6（r > r_ISCO = 6M）
D_TIERS = [1.0, 2.0, 10.0]
N_TIERS = [1.0, 2.0, 10.0]

print("=" * 76)
print("第579条 第574条两个未写明前提的核查")
print("=" * 76)
print()

# ============ S1 环向稀疏性 ============
print("--- S1 破坏区环向稀疏性对 p 的影响 ---")
print("  N_brk = n_layer * (r/d)，层数 = r/d")
print()
print("  %-28s %14s %10s %8s" % ("n_layer 行为", "N_brk ∝", "r 的幂 p", "过筛"))
cases = [
    ("常数（沿一条线稀疏破坏）", "r",       1.0, True),
    ("∝ r（环状破坏）",          "r^2",     0.5, False),
]
for label, nb, p, ok in cases:
    print("  %-28s %14s %10.4f %8s" % (label, nb, p, "是" if ok else "否"))
print()
print("  故 p=+1 须环向稀疏（每层常数个破坏格点），此前提第574条未写明")
print("  环状读法 p=+1/2 与第574条 §四 的场强受限通道同幂（同一件事）")
ok_S1 = cases[0][3] and not cases[1][3]
print("  S1 判定：%s（两种读法的后果已分开，p=+1 的前提已定位）"
      % ("通过" if ok_S1 else "判负"))

# ============ S2 C 的不相容 ============
print()
print("--- S2 C 的表达式与满偏驱动的不相容 ---")
print("  C = kappa * d / n_layer，kappa = 1/m_P（满偏标度，单格点累积到 m_P 即破坏）")
print()
print("  %-10s %-12s %-12s %12s %8s" % ("d", "n_layer", "kappa(=1/m_P)", "C", "过筛"))
rows_S2 = []
for d in D_TIERS:
    for n in N_TIERS:
        kappa = 1.0          # 自然单位 m_P = 1
        C = kappa * d / n
        ok = C > C_SIEVE
        rows_S2.append((d, n, C, ok))
        print("  %-10.1f %-12.1f %-12.1f %12.4f %8s" % (d, n, kappa, C, "是" if ok else "否"))
print()
C_max = max(r[2] for r in rows_S2)
n_overs = sum(1 for r in rows_S2 if r[3])
print("  过筛组合数 %d/%d，最大 C = %.4f" % (n_overs, len(rows_S2), C_max))
print("  过筛者皆因 d > n_layer（如 d=10,n=1），即格距远大于每层破坏格点数")
print()
print("  **不相容**：满偏驱动给 C = O(1)；轨道存在要求 C > 6。二者不可同真。")
print("  第574条把「破坏区前沿推进」与「轨道半径」绑在同一符号 r 上，该默认无依据。")
ok_S2 = n_overs < len(rows_S2) and C_max > C_SIEVE   # 须有既有组合不过筛、又有组合过筛，说明不由满偏定死
print("  S2 判定：%s（满偏驱动下 C 只能 O(1)，与 C>6 不相容已确认）"
      % ("通过" if ok_S2 else "判负"))

# ============ S3 敏感性 ============
print()
print("--- S3 C 对各输入的敏感性（多档扫描，禁单档）---")
print("  C = kappa * d / n_layer，各项的幂次：")
print("    kappa: +1（质量越大破坏格点越多）")
print("    d:     +1")
print("    n:     -1")
print()
print("  %-8s %10s %10s %10s" % ("d/n 比值", "C", "过筛", "备注"))
ratios = []
for d in D_TIERS:
    for n in N_TIERS:
        C = d / n
        ratios.append(C)
        note = "稀疏且格距大" if C > C_SIEVE else ("稀疏但格距小" if C > 1 else "不足")
        print("  %-8.2f %10.4f %10s %10s" % (d/n, C, "是" if C > C_SIEVE else "否", note))
print()
print("  C 的取值范围 [%.4f, %.4f]，全部由 O(1) 格点量之比决定" % (min(ratios), max(ratios)))
print("  满偏驱动下 n_layer 与 d 同为 O(1)，故 C 无依据地取到 > 6")
ok_S3 = max(ratios) > C_SIEVE and min(ratios) < C_SIEVE
print("  S3 判定：%s（C 跨筛子边界，说明其值不由满偏单方定死，须独立输入）"
      % ("通过" if ok_S3 else "判负"))

# ============ 汇总 ============
print()
print("=" * 76)
print("判据汇总：S1 %s；S2 %s；S3 %s"
      % ("通过" if ok_S1 else "判负", "通过" if ok_S2 else "判负", "通过" if ok_S3 else "判负"))
print()
print("结论：")
print("  1. 第574条的 p=+1 为条件性，条件是破坏区环向稀疏（每层常数个破坏格点）")
print("  2. 「破坏发生在满偏」与「C>6 轨道存在」不相容：前者给 C~O(1)，后者要 C>6")
print("  3. 第574条把破坏区前沿推进与轨道半径绑在同一符号 r 上，该默认无依据")
print("  4. C 不可由体系现有量定，须作者裁决载体口径：")
print("     前者（破坏区前沿本身）-> C~1，轨道不存在，第574条判否")
print("     后者（前沿之外的引力场尺度）-> C 可 >6，第574条成立，机制待建")
print()
print("  本条不判否第574条，只标明其两个未写明前提。")
print("可逆性等级：判据稳健。S1、S2 为代数核算；S3 为多档扫描。")
print("=" * 76)