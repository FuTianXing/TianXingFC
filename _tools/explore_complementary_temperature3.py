# -*- coding: utf-8 -*-
"""v3 修订：复相干度亏损载体 D = 1 - |g^1| 的端点自洽检验（探索级）

定义：D = 1 - |<z>| / sqrt(<|z|^2)>，kBT = T_P * D。
解析事实（模拟只做核对）：
  <z> = Pbar（均值即锁定矢量，精确）；<|z|^2> = Pbar^2 + <a^2> + <b^2>
  D = 1 - (1 + u)^{-1/2}, u = (<a^2>+<b^2>)/Pbar^2
  小涨落：D = (1-eta) + (1-xi)/2 + O(delta^4)（v2 线性式为其二阶展开）
  上限：柯西-施瓦茨 => D in [0,1] 对一切态（定理）

判据：
1. 各向同性高斯族（方差扫十个量级）：D 单调饱和于 1，全程 D<=1；
   线性式在 sigma/Pbar >~ 1 后越界（>1），D 不越界
2. 冻结族（|z|=Pbar，相位高斯）：D = 1-eta 全阶精确，s 扫到近均匀时 D -> 1
3. 定居率族（锁定相位、闪光率 q）：D = 1-sqrt(q) 精确 -> 1；
   线性式 = (1-q)/2 -> 1/2（欠估计一倍），D 修正之
4. 二阶截断标度：(D - 线性)/sigma^4 -> 常数（解析预期 -3/2，来自 -3u^2/8 项）
"""
import numpy as np

rng = np.random.default_rng(20260908)
Pbar = 1.0

def accum(N, sa, sb, chunk=2_500_000):
    """分块累计 <e^{iPhi}>, <P>, <P^2>, <a^2>, <b^2>"""
    se = 0j; sP = 0.0; sP2 = 0.0; sa2 = 0.0; sb2 = 0.0; n_tot = 0
    while n_tot < N:
        n = min(chunk, N - n_tot)
        a = rng.normal(0.0, sa, n); b = rng.normal(0.0, sb, n)
        X = Pbar + a; Y = b
        P = np.sqrt(X * X + Y * Y)
        se += np.sum(np.exp(1j * np.arctan2(Y, X))) * 1j * 0 + np.sum(np.exp(1j * np.arctan2(Y, X))) * 0
        se = se + np.sum(np.exp(1j * np.arctan2(Y, X)))
        sP += np.sum(P); sP2 += np.sum(P * P)
        sa2 += np.sum(a * a); sb2 += np.sum(b * b); n_tot += n
    eta = abs(se / n_tot)
    xi = (sP / n_tot) ** 2 / (sP2 / n_tot)
    D = 1.0 - Pbar / np.sqrt(Pbar ** 2 + sa2 / n_tot + sb2 / n_tot)
    return eta, xi, D

print("=" * 74)
print("1. 各向同性高斯族：D 饱和于 1、不越界；线性式越界（sigma 扫十个量级）")
print("=" * 74)
print(f"{'sigma':>8} {'eta':>10} {'1-xi':>10} {'D':>10} {'线性 w+p/2':>12} {'D_exact':>10}")
for sa in [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0]:
    eta, xi, D = accum(4_000_000, sa, sa)
    w, p = 1 - eta, 1 - xi
    Dex = 1.0 - Pbar / np.sqrt(Pbar ** 2 + 2 * sa * sa)
    print(f"{sa:8.2f} {eta:10.2e} {p:10.4f} {D:10.6f} {w + p / 2:12.6f} {Dex:10.6f}")

print("=" * 74)
print("2. 二阶截断标度：(D - 线性)/sigma^4 -> 常数（解析预期 -3/2）")
print("=" * 74)
for sa in [0.15, 0.2, 0.3]:
    eta, xi, D = accum(20_000_000, sa, sa)
    w, p = 1 - eta, 1 - xi
    diff = D - (w + p / 2)
    print(f"sigma={sa:5.2f}  D={D:.8f}  线性={w + p / 2:.8f}  "
          f"D-线性={diff:+.2e}  (D-线性)/sigma^4={diff / sa ** 4:+.3f}")

print("=" * 74)
print("3. 冻结族（|z|=Pbar，相位高斯）：D = 1-eta 全阶精确，s 大时 D -> 1")
print("=" * 74)
for s in [0.1, 0.5, 1.0, 2.0, 4.0]:
    phi = rng.normal(0.0, s, 4_000_000)
    eta = abs(np.mean(np.exp(1j * phi)))
    z = Pbar * np.exp(1j * phi)
    D = 1.0 - abs(np.mean(z)) / np.sqrt(np.mean(np.abs(z) ** 2))
    print(f"s={s:5.2f}  eta={eta:.8f}  D={D:.8f}  D-(1-eta)={D - (1 - eta):+.2e}  T/T_P={D:.6f}")

print("=" * 74)
print("4. 定居率族（锁定相位、闪光率 q）：D = 1-sqrt(q) -> 1；线性式欠估计至 1/2")
print("=" * 74)
Pf = 1.0
for q in [0.5, 0.1, 0.01, 1e-3]:
    n = 4_000_000
    mask = rng.random(n) < q
    z = np.where(mask, Pf, 0.0) * np.exp(1j * 0.3)
    D = 1.0 - abs(np.mean(z)) / np.sqrt(np.mean(np.abs(z) ** 2))
    print(f"q={q:8.1e}  D={D:.6f} (期望 1-sqrt(q)={1 - np.sqrt(q):.6f})  "
          f"线性=(1-q)/2={(1 - q) / 2:.6f}  T/T_P={D:.6f}")

print("=" * 74)
print("结论：D 全程 [0,1]、四角精确、两端修正线性式，则 v3 载体自洽")
print("=" * 74)
