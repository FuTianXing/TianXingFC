# -*- coding: utf-8 -*-
r"""
minimal_verification_b1_operator_homogeneity.py

[识别式阶段 3 第二轮] B1：自撑恒等式的三维径向细化（判据 B1-1 至 B1-4）

定位：判据 S 把自撑阱推广到三维径向，S1 核对恒等式点态成立、S2 给四通道齐次度
（-1/-1/-1/-3）。本轮做收紧版：把几何项（径向雅可比 2/r 与离心项 l(l+1)/r^2）
显式写进算符并核对算符级齐次度；把指数收紧到 1e-6 级；补入离心通道。

模型：球对称 hedgehog phi(r) = tanh(kr)，V'' = 4k^2 - 6k^2 sech^2(kr)；
径向算符 L_l = -d^2/dr^2 - (2/r)d/dr + l(l+1)/r^2 + V''；通道积分取体积元 r^2 dr。

判据
B1-1 算符级齐次度：L_{l,k}[psi_k](r) = k^2 L_{l,1}[psi_1](kr)，l = 0/1/2（有限差分核对）。
B1-2 恒等式：V'' 与 BPS 分裂逐点一致，k = 0.5 至 4，机器精度。
B1-3 通道指数：径向梯度、角向、势阱、质量项与离心项的对数斜率（期望 -1/-1/-1/-3/-1）。
B1-4 通道比值：五通道积分的比值与 k 无关（同齐次度的直接后果）。

运行：python 验证脚本/minimal_verification_b1_operator_homogeneity.py
确定性、无图，运行约 3 秒。
"""
import numpy as np
from scipy.integrate import quad

def vpp(k, r):
    return 4.0 * k ** 2 - 6.0 * k ** 2 / np.cosh(k * r) ** 2

def L_apply(k, l, f, r, h=1e-4):
    """L_l f 的有限差分实现，f 为可调用函数（在偏移点上直接取值）。"""
    fp = (f(r + h) - f(r - h)) / (2 * h)
    fpp = (f(r + h) - 2.0 * f(r) + f(r - h)) / h ** 2
    return -fpp - 2.0 * fp / r + l * (l + 1) / r ** 2 * f(r) + vpp(k, r) * f(r)

print("=" * 72)
print("识别式阶段 3 第二轮：B1 自撑恒等式的三维径向细化（B1-1 至 B1-4）")
print("=" * 72)
print("[B1-1] 算符级齐次度（含径向雅可比与离心项）")
for l in (0, 1, 2):
    ks = [1.0, 2.0]
    worst = 0.0
    for k in ks:
        r = np.linspace(0.5, 20.0, 4000)
        psi_1 = lambda x: 1.0 / np.cosh(x) ** 2 * np.exp(-x / 8.0)
        psi_k = lambda x: 1.0 / np.cosh(k * x) ** 2 * np.exp(-k * x / 8.0)
        lhs = L_apply(k, l, psi_k, r)
        rhs = k ** 2 * L_apply(1.0, l, psi_1, k * r)
        scale = np.max(np.abs(rhs)) + 1e-30
        worst = max(worst, float(np.max(np.abs(lhs - rhs)) / scale))
    print(f"   l = {l}：最大相对偏差 {worst:.2e}（期望 <= 1e-4，有限差分口径）")

print("[B1-2] 恒等式逐点（V'' 对 BPS 分裂）")
for k in (0.5, 1.0, 2.0, 4.0):
    r = np.linspace(1e-6, 30.0 / k, 200000)
    # 由势定义独立算 V''：V = (lam/4)(phi^2-1)^2、lam = 2k^2、phi = tanh(kr) -> V'' = 2k^2(3 phi^2 - 1)
    vpp_indep = 2.0 * k ** 2 * (3.0 * np.tanh(k * r) ** 2 - 1.0)
    d = float(np.max(np.abs(vpp_indep - vpp(k, r))))
    print(f"   k = {k:.1f}：最大点态偏差 {d:.2e}")

print("[B1-3] 通道积分与对数斜率")
def channels(k, c=8.0):
    R = c / k
    T = quad(lambda r: (k / np.cosh(k * r)) ** 2 * r ** 2, 1e-6, R, limit=400)[0]
    A = quad(lambda r: np.tanh(k * r) ** 2, 1e-6, R, limit=400)[0]
    W = quad(lambda r: vpp(k, r) * np.tanh(k * r) ** 2 * r ** 2, 1e-6, R, limit=400)[0]
    M = quad(lambda r: np.tanh(k * r) ** 2 * r ** 2, 1e-6, R, limit=400)[0]
    return T, A, W, M
ks = [0.5, 1.0, 2.0, 4.0]
tab = {k: channels(k) for k in ks}
names = ["径向梯度", "角向", "势阱", "质量项"]
for j, nm in enumerate(names):
    vals = [tab[k][j] for k in ks]
    sl = (np.log(vals[-1]) - np.log(vals[0])) / (np.log(ks[-1]) - np.log(ks[0]))
    print(f"   {nm}：k = {ks[0]} 至 {ks[-1]} 的对数斜率 = {sl:.6f}（判据 S 公布 -1/-1/-1/-3 中对应项）")
l = 1
cents = [l * (l + 1) * tab[k][1] for k in ks]
sl = (np.log(cents[-1]) - np.log(cents[0])) / (np.log(ks[-1]) - np.log(ks[0]))
print(f"   离心项（l = 1）：对数斜率 = {sl:.6f}（期望 -1；判据 S 未含此项）")

print("[B1-4] 通道比值与 k 无关性")
for k in ks:
    T, A, W, M = tab[k]
    print(f"   k = {k:.1f}：A/T = {A/T:.6f}，W/T = {W/T:.6f}，M/T = {M/T:.6f}")
