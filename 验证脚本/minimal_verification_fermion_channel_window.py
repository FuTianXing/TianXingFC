# -*- coding: utf-8 -*-
"""费米子通道第一轮：复合模型的宇称奇源与 m_* 窗口（判据 F1 至 F4）。
问题：三通道里费米子通道"缺复合模型"。本轮不求系数，只判两件事：
其一，复合模型（均匀缠绕背景 + FK kink 位错 + 宽度参数区）里哪个自由度携带宇称奇；
其二，m_* 取复合模型可得的各种标度时，得到的 delta 窗口是否比光子通道更紧（即是否设独立约束）。
输入：边际通道协议路径二的量级关系 k_AF ~ delta*M_*, btilde ~ delta*m_*；上界抄录自协议 §7。
确定性、无图。输出到终端。
"""
import numpy as np

E_P = 1.2209e19      # GeV（普朗克能量）
hbar_wP = 2 * np.pi * E_P
b_fermion = {'中子': 1.5e-33, '质子': 7.6e-33}   # GeV
k_photon = 1e-44                                  # GeV，光子通道上界

print("=" * 72)
print("费米子通道第一轮：宇称奇源与 m_* 窗口（F1 至 F4）")
print("=" * 72)

print("F1 复合模型的自由度奇偶表（T 反演）")
rows = [
 ("kink 位置", "连续平移自由度", "偶", "位置在 T 下不变"),
 ("绕数背景", "整数拓扑荷", "偶", "标量荷"),
 ("法线轴层数", "非紧致实载（距离）", "偶", "距离在 T 下不变"),
 ("内部相位（S^1 角度）", "周期角变量", "奇", "角度在 T 下反号"),
]
for name, kind, par, why in rows:
    print(f"   {name:12s} | {kind:14s} | {par} | {why}")
odd = [r[0] for r in rows if r[2] == "奇"]
print(f"   → 宇称奇源集合：{odd}（唯一候选为内部相位）")

print("F2 m_* 候选与其 delta 窗口（用中子上界 1.5e-33 GeV）")
scales = [
 ("复合宽度标度 E_P/2（ℓ* = 2）", E_P / 2.0),
 ("复合宽度标度 E_P/7（ℓ* = 7）", E_P / 7.0),
 ("缺陷刚度标度 hbar*omega_0 = 1e-23 hbar*omega_P", 1e-23 * hbar_wP),
 ("电子质量", 0.51099895e-3),
 ("质子质量", 0.938272),
]
for name, val in scales:
    print(f"   {name:42s} | m_* = {val:.3e} GeV | delta <= {b_fermion['中子']/val:.2e}")
print("   另一条上界（质子 7.6e-33 GeV）把各窗口整体放宽 5.07 倍。")

print("F3 交叉封口：同尺度下光子通道是否恒更紧")
ratio = b_fermion['中子'] / k_photon
print(f"   两条上界之比 = {ratio:.2e}（与生成标度无关）")
print("   费米子通道要设独立约束，需其生成标度比光子通道的高出该比值以上：")
for ms, mp in [(0.51099895e-3, E_P), (1e-23 * hbar_wP, E_P), (E_P / 2, 1.0), (0.938272, 1.0)]:
    need = ratio * mp
    ok = ms > need
    print(f"     m_*(费米子) = {ms:.2e} GeV，M_*(光子) = {mp:.2e} GeV：需要 m_* > {need:.2e} GeV → {'成立' if ok else '不成立'}")

print("F3b 光子通道生成标度的锚定（用框架自身关系 k_AF = 2 delta / a，a = l_P）")
M_photon = 2.0 * E_P
d_photon = k_photon / M_photon
print(f"   光子通道生成标度 = 2 E_P = {M_photon:.3e} GeV；delta <= {d_photon:.2e}")
for name, val in [("复合宽度标度 E_P/2", E_P / 2.0), ("缺陷刚度标度", 1e-23 * hbar_wP), ("电子质量", 0.51099895e-3)]:
    d_f = b_fermion['中子'] / val
    print(f"   费米子通道（m_* = {name}）delta <= {d_f:.2e} → 比光子通道宽 {d_f/d_photon:.1e} 倍")

print("F4 结论行")
print("   其一，复合模型的宇称奇源为内部相位（S^1 角度），绕数与层数为偶，源身份判定完成。")
print("   其二，m_* 取复合模型可得标度（宽度标度、缺陷标度、费米子质量）时，delta 窗口为 1e-30 至 1e-52 量级。")
print("   其三，同尺度对照下光子通道恒紧 1.5e11 倍，费米子通道不设独立约束（除人为把光子生成标度压到 1e-8 GeV 以下）。")
print("   其四，三通道对照表的第三格填法：窗口被光子通道覆盖，源为构造性内部相位，模型标定系数 O(1) 未导出。")
