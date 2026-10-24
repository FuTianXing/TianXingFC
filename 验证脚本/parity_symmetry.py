# -*- coding: utf-8 -*-
"""P4-31 前提核对：工作区文档所用更新式的宇称精确性（R1 至 R4）。

背景：上一轮把宇称奇（eps 型）边际算符的封口条件定位到"格点规则的反射对称性是否精确"。
本轮对该前提做显式核对：取工作区文档中实际使用的更新式（法线场动力学的三项：邻居抄写
平均、惯性项、注入；以及波动腿的离散波动方程），逐项判定其在空间反演下的变换性质，并
对反演算子与更新算子的交换子做数值核对；再用手征对照证明该判据可被推翻。

判据（先立判据，再实现）：
  R1 逐项偶性：文档更新式的每一项在空间反演下不变（含等权邻居平均、惯性项、点源注入）。
  R2 交换子：空间反演算子 P 与三类更小子（DC 更新、波动腿更新、离散泊松解）的交换子为零。
  R3 响应偶性：点源静态响应的奇宇称部分（theta(x) - theta(-x)）为零。
  R4 灵敏度：给格点加一个反对称键权（宇称奇项）后，交换子与奇响应都变为可测量。

单位：格距 1，周期边界。确定性，无随机数。
"""

import numpy as np

CHECKS = []
N = 16
rng = np.random.default_rng(20260916)


def check(tag, ok, detail):
    CHECKS.append((tag, bool(ok)))
    print("  [%s] %s  %s" % (tag, "PASS" if ok else "FAIL", detail))


def invert(field):
    """空间反演 x -> -x：g(i) = f((-i) mod N)，逐轴 flip 后平移一格。"""
    out = field
    for axis in range(3):
        out = np.roll(np.flip(out, axis=axis), 1, axis=axis)
    return out


def lap(field):
    out = -6.0 * field
    for axis in range(3):
        out = out + np.roll(field, 1, axis=axis) + np.roll(field, -1, axis=axis)
    return out


def neighbor_average(field):
    acc = np.zeros_like(field)
    for axis in range(3):
        acc = acc + np.roll(field, 1, axis=axis) + np.roll(field, -1, axis=axis)
    return acc / 6.0


def source_field():
    src = np.zeros((N, N, N))
    src[N // 2, N // 2, N // 2] = 1.0
    return src


def dc_update(field, prev, alpha=0.9, mu=0.5, w=0.05, odd=0.0):
    inj = w * source_field()
    term = alpha * (neighbor_average(field) - field) + mu * (field - prev) + inj
    if odd != 0.0:
        term = term + odd * (np.roll(field, 1, axis=0) - np.roll(field, -1, axis=0))
    return field + term


def wave_update(field, prev, beta=1.0):
    return 2.0 * field - prev + beta * lap(field)


print("=" * 78)
print("R1 逐项偶性：文档更新式各项的符号在 k -> -k 下的变换")
print("=" * 78)
kx = 2.0 * np.pi * np.fft.fftfreq(N)
KX, KY, KZ = np.meshgrid(kx, kx, kx, indexing="ij")
symbols = {
    "邻居抄写平均": (np.cos(KX) + np.cos(KY) + np.cos(KZ)) / 3.0,
    "惯性项（单位算子倍数）": np.ones((N, N, N)),
    "点源注入（原点处 delta）": np.ones((N, N, N)),
    "离散拉普拉斯（波动腿）": -(4.0 * np.sin(KX / 2.0) ** 2 + 4.0 * np.sin(KY / 2.0) ** 2
                              + 4.0 * np.sin(KZ / 2.0) ** 2),
}
worst = 0.0
for name, S in symbols.items():
    dev = float(np.max(np.abs(S - invert(S))))
    worst = max(worst, dev)
    print("  %s：符号在 k -> -k 下的最大偏差 = %.3e" % (name, dev))
print("  含义：各项的符号都是 k 的偶函数，故更新的生成元不含宇称奇成分（奇成分的符号是 k 的奇函数）。")
check("R1", worst <= 1e-14, "更新式各项的符号都是 k 的偶函数")

print()
f = rng.normal(size=(N, N, N))
p = rng.normal(size=(N, N, N))
print("=" * 78)
print("R2 交换子：反演算子与更新的生成元")
print("=" * 78)
comm_dc = float(np.max(np.abs(invert(dc_update(f, p) - f) - (dc_update(invert(f), invert(p)) - invert(f)))))
comm_wave = float(np.max(np.abs(invert(wave_update(f, p)) - wave_update(invert(f), invert(p)))))
print("  DC 更新：|P U - U P| = %.3e" % comm_dc)
print("  波动腿更新：|P U - U P| = %.3e" % comm_wave)
check("R2", comm_dc <= 1e-13 and comm_wave <= 1e-13, "反演与两类更新的生成元交换")

print()
print("=" * 78)
print("R3 响应偶性：点源静态响应的奇宇称部分")
print("=" * 78)
# 周期格点上的离散泊松解（零平均模），源在中心
src = source_field()
k = 2.0 * np.pi * np.fft.fftfreq(N)
KX, KY, KZ = np.meshgrid(k, k, k, indexing="ij")
LAM = -(4.0 * np.sin(KX / 2.0) ** 2 + 4.0 * np.sin(KY / 2.0) ** 2 + 4.0 * np.sin(KZ / 2.0) ** 2)
LAM[0, 0, 0] = 1.0
theta = np.real(np.fft.ifftn(np.fft.fftn(src) / LAM))
odd_part = float(np.max(np.abs(theta - invert(theta))) / np.max(np.abs(theta)))
print("  静态响应：奇宇称部分相对幅度 = %.3e（期望 0）" % odd_part)
check("R3", odd_part <= 1e-14, "点源响应的奇宇称部分为零")

print()
print("=" * 78)
print("R4 灵敏度：加入反对称键权（宇称奇项）后判据失效")
print("=" * 78)
ODD = 0.05
comm_ctl = float(np.max(np.abs(invert(dc_update(f, p, odd=ODD) - f)
                              - (dc_update(invert(f), invert(p), odd=ODD) - invert(f)))))
lam_ctl = LAM - 1j * ODD * 2.0 * np.sin(KX)
theta_ctl = np.real(np.fft.ifftn(np.fft.fftn(src) / lam_ctl))
odd_ctl = float(np.max(np.abs(theta_ctl - invert(theta_ctl))) / np.max(np.abs(theta_ctl)))
print("  对照 DC 更新：|P U - U P| = %.3e（应非零）" % comm_ctl)
print("  对照静态响应：奇宇称部分相对幅度 = %.3e（应非零）" % odd_ctl)
check("R4", comm_ctl >= 1e-3 and odd_ctl >= 1e-3, "宇称奇项让交换子与奇响应同时出现")

print()
n_pass = sum(1 for _, ok in CHECKS if ok)
print("=" * 78)
print("判据汇总：%d/%d 通过" % (n_pass, len(CHECKS)))
for tag, ok in CHECKS:
    print("  %-6s %s" % (tag, "PASS" if ok else "FAIL"))
print("=" * 78)
