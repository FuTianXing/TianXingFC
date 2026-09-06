# -*- coding: utf-8 -*-
"""
弹性身份裁决实验：cos（紧致）vs 平方（非紧致）弹性下的位错宽度
同一初条件（均匀缠绕 + 位错）、同一参数区（r = K/k）、同一宽度定义
cos 型：E = κ_∥ Σ(1-cos 2πΔu) + K Σ(1-cos 2πu)
平方型：E = κ_∥ Σ ½(Δu-1)² + K Σ(1-cos 2πu)
预期：cos 型位错塌缩（单键隐藏，宽度 ~ 0）；平方型落窗（ℓ* >= 2）
结果只打印终端，不写任何文档。
"""
import numpy as np

NS = 400
K_EL = 1.0

def relax(u, r, cos_elastic, steps=120000):
    K_pn = r * K_EL
    eta = 0.005
    twopi = 2.0 * np.pi
    for t in range(steps):
        grad = np.zeros(NS)
        if cos_elastic:
            # d/du_n Σ(1-cos 2πΔu) = 2π[sin 2π(u_n-u_{n-1}) - sin 2π(u_{n+1}-u_n)]
            grad[1:-1] += twopi * (np.sin(twopi * (u[1:-1] - u[:-2]))
                                   - np.sin(twopi * (u[2:] - u[1:-1])))
        else:
            # d/du_n Σ ½(Δu-1)² = (2u_n - u_{n-1} - u_{n+1})
            grad[1:-1] += K_EL * (2.0 * u[1:-1] - u[:-2] - u[2:])
        grad += K_pn * twopi * np.sin(twopi * u)
        u[1:-1] -= eta * grad[1:-1]
        u[0] = 0.0
        u[-1] = float(NS)
    return u

def width_and_E(u, r, cos_elastic):
    K_pn = r * K_EL
    twopi = 2.0 * np.pi
    du = np.diff(u)
    if cos_elastic:
        E = K_EL * np.sum(1.0 - np.cos(twopi * du)) + K_pn * np.sum(1.0 - np.cos(twopi * u))
        dens = 1.0 - np.cos(twopi * du)          # 弹性超量（平衡 Δu=1 时 = 0）
    else:
        E = 0.5 * K_EL * np.sum((du - 1.0) ** 2) + K_pn * np.sum(1.0 - np.cos(twopi * u))
        dens = 0.5 * (du - 1.0) ** 2
    idx = np.arange(NS - 1) + 0.5
    s = dens.sum()
    if s < 1e-12:
        return 0.0, E
    c0 = (dens * idx).sum() / s
    r2 = (dens * (idx - c0) ** 2).sum() / s
    return 2.0 * np.sqrt(r2), E

print("=" * 68)
print("弹性身份裁决实验：cos（紧致）vs 平方（非紧致）弹性")
print(f"1D 链 N = {NS}，均匀缠绕 + 位错初条件，同一参数区 r = K/k")
print("=" * 68)
print("\n  r=K/k     cos弹性 ℓ*    cos弹性 E    平方弹性 ℓ*   平方弹性 E")
for r in (0.0005, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1):
    u0 = np.arange(NS, dtype=float)
    half = NS // 2
    u0[half:] += 1.0
    u0[half] = half + 0.5
    uc = relax(u0.copy(), r, True)
    us = relax(u0.copy(), r, False)
    wc, Ec = width_and_E(uc, r, True)
    ws, Es = width_and_E(us, r, False)
    print(f"  {r:8.4f}   {wc:10.2f}   {Ec:10.3f}   {ws:10.2f}   {Es:10.3f}")

print("\n判定：cos 弹性下位错塌缩（ℓ* ≈ 0，尖锐位错隐藏）则身份裁决倾向非紧致；")
print("平方弹性落窗（ℓ* >= 2）且 cos 塌缩 → 宽度机制依赖非紧致弹性")
print("裁决实验完成（仅打印）。")
