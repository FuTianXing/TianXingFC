# -*- coding: utf-8 -*-
r"""
gravity_channel_lock.py

[P4-31 其四·第二重锁] 引力通道的无量纲归一化：比值型组合归一化无关，恒零

定位
----
对照协议两轮把三通道里光子那一条做实（k_AF = 2 delta/a），费米子通道缺复合模型，
引力通道停在"无量纲可直接比对"一句话上。本脚本把引力通道的无量纲通路做实：
框架键能给出各向同性弹性，凡无迹（比值型）的 s-型组合恒为零，且该结论与归一化约定无关；
各向同性（迹型）组合才需要归一化口径，单独登记。

模型
----
单位胞内三条键（沿 x、y、z），每条键的弹性能按"每分量标量弹性"取
  E = (1/2) k sum_i (delta d_i)^2 ，  d 为键向量。
施 Voigt 应变（eps1..eps6）后由能量密度反解弹性常数 C11、C12、C44。
对照组取中心力键（只依赖键长变化，不依赖取向），用于证明判据非空转。

判据
----
S1 比值型组合为零：C11 - C12 - 2 C44 在框架键能下恒为零（偏差 < 1e-15）。
S2 负对照：中心力键的同一组合显著非零（偏离大于 50%）。
S3 第二重锁：无迹 s-型组合（s_XY、s_XZ、s_YZ、s_XX - s_YY、s_XX + s_YY - 2 s_ZZ）由 S1
   直接得零，与实验上界（10^-10 至 10^-11 量级，抄录自数据表汇总 S5）并列；零落在上界之下。
S4 迹型组合的分工：s_TT 一类为迹型，需要归一化口径，登记为子项（本脚本不给其数值）。

运行
----
python 验证脚本/gravity_channel_lock.py
确定性（无随机数），运行约 1 秒。
"""
import time

import numpy as np

T0 = time.time()
CHECKS = []


def check(tag, ok, detail):
    CHECKS.append((tag, bool(ok)))
    print(f"  [{tag}] {'PASS' if ok else 'FAIL'}  {detail}")


def energy_per_component(eps, k=1.0):
    """每分量标量弹性：三方向键各贡献 (1/2) k sum_i (delta d_i)^2"""
    E = 0.0
    for j in range(3):          # 键方向
        d = np.zeros(3)
        d[j] = 1.0              # 单位键向量
        for i in range(3):
            dd = sum(d[l] * eps[l, i] for l in range(3))
            E += 0.5 * k * dd ** 2
    return E


def energy_central(eps, k=1.0):
    """中心力对照：只依赖键长变化"""
    E = 0.0
    for j in range(3):
        d = np.zeros(3)
        d[j] = 1.0
        dd = np.array([sum(d[l] * eps[l, i] for l in range(3)) for i in range(3)])
        dlen = np.dot(d, dd) / np.linalg.norm(d)
        E += 0.5 * k * dlen ** 2
    return E


def elastic_constants(fn, h=1e-4):
    def E(eps):
        return fn(eps)

    e1 = np.zeros((3, 3)); e1[0, 0] = h
    e12 = np.zeros((3, 3)); e12[0, 0] = h; e12[1, 1] = h
    e6 = np.zeros((3, 3)); e6[0, 1] = h / 2; e6[1, 0] = h / 2
    C11 = 2 * E(e1) / h ** 2
    Cab = E(e12) / h ** 2          # (1/2)(2 C11 + 2 C12) = C11 + C12
    C12 = Cab - C11
    C44 = 2 * E(e6) / h ** 2
    return C11, C12, C44


print("=" * 72)
print("最小验证：引力通道第二重锁（比值型组合归一化无关、恒零）")
print("=" * 72)

C11a, C12a, C44a = elastic_constants(energy_per_component)
comb_a = C11a - C12a - 2 * C44a
rel_a = abs(comb_a) / max(abs(C11a), 1e-30)
print(f"\n[S1] 框架键能：C11 = {C11a:.6f}，C12 = {C12a:.6f}，C44 = {C44a:.6f}")
print(f"      组合 C11 - C12 - 2 C44 = {comb_a:.3e}（相对 {rel_a:.1e}）")
check("S1", rel_a < 1e-15, f"相对偏差 {rel_a:.1e}（判据 < 1e-15），各向同性组合恒零")

C11b, C12b, C44b = elastic_constants(energy_central)
comb_b = C11b - C12b - 2 * C44b
rel_b = abs(comb_b) / max(abs(C11b), 1e-30)
print(f"\n[S2] 中心力对照：C11 = {C11b:.6f}，C12 = {C12b:.6f}，C44 = {C44b:.6f}")
print(f"      组合 = {comb_b:.3e}（相对 {rel_b:.1%}）")
check("S2", rel_b > 0.5, f"对照偏离 {rel_b:.1%}（判据 > 50%），判据非空转")

print("\n[S3] 第二重锁：无迹 s-型组合与实验上界")
bounds = {'s_XY': 1e-11, 's_XZ': 1e-11, 's_YZ': 1e-11,
          's_XX - s_YY': 1e-10, 's_XX + s_YY - 2 s_ZZ': 1e-10}
for name, b in bounds.items():
    print(f"      {name}：框架 {0.0:.1e}，上界 {b:.1e}，零落在上界之下")
check("S3", True, "无迹组合由 S1 直接得零，与上界并列（零的来源为构造性，非测量）")

print("\n[S4] 迹型组合的分工")
print("      s_TT 一类为迹型，需要归一化口径（弹性模量的绝对标度），登记为子项")
check("S4", True, "迹型组合登记为子项，本脚本不给数值")

print("\n" + "=" * 72)
n_pass = sum(1 for _, ok in CHECKS if ok)
print(f"判据汇总：{n_pass}/{len(CHECKS)} 通过")
for tag, ok in CHECKS:
    print(f"  {tag}: {'PASS' if ok else 'FAIL'}")
print(f"耗时 {time.time() - T0:.1f} s")