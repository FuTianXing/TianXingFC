# -*- coding: utf-8 -*-
"""波粒互补温度：粒侧无序度 p = 1 - xi 与等相干平衡的数值自检（探索级）

判据（理论期望）：
1. 相位侧（wrapped Gaussian）：eta = e^{-s^2/2} 精确，w = 1 - eta ≈ s^2/2
2. 粒侧 P = Pbar(1+eps)：xi = 1/(1+sigma^2) 精确（矩比），p ≈ sigma^2
3. 平衡恒等式 w = p（小涨落）-> sigma^2 = s^2/2，此时 E_w = E_p（能量均分）
4. 饱和形态：p -> 1 仅在定居率 q -> 0（P 几乎处处为零、偶发闪现）
5. 尺度不变：xi 只依赖分布形状，不依赖 Pbar 绝对标度
6. 二维熵：S = C + 1/2 ln w + 1/2 ln p；平衡下 S = C' + ln(1-eta)（对数熵系数 1）
"""
import numpy as np

rng = np.random.default_rng(20260908)
print("=" * 70)
print("1. 相位侧：wrapped Gaussian 的相干度与无序度")
print("=" * 70)
for s in [0.01, 0.05, 0.1, 0.2, 0.5]:
    phi = rng.normal(0.0, s, 2000000)
    eta = np.abs(np.mean(np.exp(1j * phi)))
    w = 1.0 - eta
    print(f"s={s:5.2f}  eta={eta:.8f} (期望 e^(-s2/2)={np.exp(-s * s / 2):.8f})  "
          f"w={w:.8f} (期望 s^2/2={s * s / 2:.8f})")

print("=" * 70)
print("2. 粒侧：P = Pbar(1+eps) 的矩比相干度与无序度")
print("=" * 70)
for sigma in [0.01, 0.05, 0.1, 0.2]:
    eps = rng.normal(0.0, sigma, 2000000)
    P = 1.0 + eps
    xi = np.mean(P) ** 2 / np.mean(P ** 2)
    p = 1.0 - xi
    print(f"sigma={sigma:5.2f}  xi={xi:.8f} (期望 1/(1+s2)={1 / (1 + sigma * sigma):.8f})  "
          f"p={p:.8f} (期望 s2/(1+s2)={sigma * sigma / (1 + sigma * sigma):.8f})")

print("=" * 70)
print("3. 平衡恒等式：w = p（小涨落）-> sigma^2 = s^2/2")
print("=" * 70)
for s in [0.02, 0.05, 0.1, 0.2]:
    sigma = s / np.sqrt(2.0)
    w = 1.0 - np.exp(-s * s / 2.0)
    p = 1.0 - 1.0 / (1.0 + sigma * sigma)
    print(f"s={s:5.2f} sigma={sigma:.6f}  w={w:.8f}  p={p:.8f}  w/p={w / p:.6f}")

print("=" * 70)
print("4. 饱和形态：粒侧完全无序 p -> 1 的途径（定居率 q -> 0）")
print("=" * 70)
for q in [0.5, 0.1, 0.01, 1e-4]:
    r = rng.random(2000000)
    P = np.where(r < q, 1.0, 0.0)
    xi = np.mean(P) ** 2 / np.mean(P ** 2)
    print(f"q={q:8.1e}  xi={xi:.8f} (期望 q={q:.1e})  p={1 - xi:.8f} (期望 1-q={1 - q:.1e})")
for name, P in [("均匀[0,1]", rng.uniform(0.0, 1.0, 1000000)),
                ("瑞利 sigma=1", np.sqrt(rng.exponential(2.0, 1000000)))]:
    xi = np.mean(P) ** 2 / np.mean(P ** 2)
    print(f"{name}: xi={xi:.8f} (均匀期望 3/4, 瑞利期望 pi/4), p={1 - xi:.8f}")

print("=" * 70)
print("5. 尺度不变：xi 不依赖 Pbar")
print("=" * 70)
for Pbar in [0.1, 1.0, 100.0]:
    eps = rng.normal(0.0, 0.1, 2000000)
    P = Pbar * (1.0 + eps)
    xi = np.mean(P) ** 2 / np.mean(P ** 2)
    print(f"Pbar={Pbar:6.1f}  xi={xi:.8f}  p={1 - xi:.8f}")

print("=" * 70)
print("6. 二维熵系数：平衡下 S = C + ln(1-eta)（系数 1 = 1/2 + 1/2）")
print("=" * 70)
for s in [0.01, 0.05, 0.1, 0.2]:
    sigma = s / np.sqrt(2.0)
    Sw = 0.5 * np.log(2 * np.pi * np.e * s * s) - s * s / 2.0   # wrapped Gaussian 相位熵
    Sp = 0.5 * np.log(2 * np.pi * np.e * sigma * sigma)          # 高斯幅度熵
    S = Sw + Sp
    w = 1.0 - np.exp(-s * s / 2.0)
    C = S - np.log(w)  # 若 S = C + 1*ln w，则 C 应稳定
    print(f"s={s:5.2f}  S={S:.8f}  ln w={np.log(w):.8f}  C=S-ln w={C:.8f}")

print("=" * 70)
print("结论：判据 1-6 全部满足，则双侧激发度结构数值自检通过")
print("=" * 70)
