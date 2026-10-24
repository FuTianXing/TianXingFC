# -*- coding: utf-8 -*-
'''非最近邻合成权重与源分配权重一致性（P1-5 移交两项，第 474 条）。

判据 N1 至 N4：更大模板下合成权重的唯一性与各向同性阶数；
判据 S1 至 S3：源分配权重与合成权重一致性（不一致引入的方向性残差与允许上界）。
承接第 463 条（最近邻六方向，w = 1/6 被二阶各向同性加中心对称唯一钉住）。
确定性脚本，无随机数。
'''
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

IMG = 'D:/software/TianXingFC/images/nonlocal_composition_weights.png'


def unit(v):
    v = np.array(v, dtype=float)
    return v / np.linalg.norm(v)


def faces():
    return [unit([1, 0, 0]), unit([-1, 0, 0]), unit([0, 1, 0]),
            unit([0, -1, 0]), unit([0, 0, 1]), unit([0, 0, -1])]


def edges():
    out = []
    for a in range(3):
        for b in range(a + 1, 3):
            for sa in (1, -1):
                for sb in (1, -1):
                    v = [0.0, 0.0, 0.0]
                    v[a] = sa
                    v[b] = sb
                    out.append(unit(v))
    return out


def corners():
    out = []
    for sx in (1, -1):
        for sy in (1, -1):
            for sz in (1, -1):
                out.append(unit([sx, sy, sz]))
    return out


def T2(ws, ds):
    A = np.zeros((3, 3))
    for w, d in zip(ws, ds):
        A += w * np.outer(d, d)
    return A


def inv4(ws, ds):
    xxxx = sum(w * d[0] ** 4 for w, d in zip(ws, ds))
    xxyy = sum(w * d[0] ** 2 * d[1] ** 2 for w, d in zip(ws, ds))
    return xxxx, xxyy


def inv6(ws, ds):
    x6 = sum(w * d[0] ** 6 for w, d in zip(ws, ds))
    x4y2 = sum(w * d[0] ** 4 * d[1] ** 2 for w, d in zip(ws, ds))
    return x6, x4y2


def probe(name, ws, ds):
    A = T2(ws, ds)
    r2 = float(np.max(np.abs(A - np.trace(A) / 3.0 * np.eye(3))))
    xxxx, xxyy = inv4(ws, ds)
    r4 = abs(xxxx - 3.0 * xxyy)
    print('%-26s T2 偏离各向同性 %.3e   T4 残差 %.3e' % (name, r2, r4))
    return r2, r4


print('==== 判据 N1 至 N4：更大模板的合成权重 ====')
f = faces()
e = edges()
c = corners()

r = probe('N1 面心 6，w = 1/6', [1.0 / 6] * 6, f)
print('   期望：T2 残差 0；T4 残差 = 1/3（立方四阶各向异性的来源，非缺陷）')

w12 = 1.0 / 12
r = probe('N2 棱心 12，w = 1/12', [w12] * 12, e)
print('   期望：T2 残差 0；T4 残差 = 1/12（仍非四阶各向同性）')

M18 = np.array([[2.0, 4.0], [2.0, -1.0]])
rhs = np.array([1.0 / 3, 0.0])
a18, b18 = np.linalg.solve(M18, rhs)
print('N3 面心加棱心 18 方向：解 a = %.12f（1/30 = %.12f），b = %.12f（1/15 = %.12f）'
      % (a18, 1.0 / 30, b18, 1.0 / 15))
print('   rank = %d（唯一解），行列式 = %.6f' % (np.linalg.matrix_rank(M18), np.linalg.det(M18)))
r = probe('N3 18 方向二阶加四阶', [a18] * 6 + [b18] * 12, f + e)
print('   期望：T2 与 T4 残差均到机器精度')

M14 = np.array([[2.0, 8.0 / 3], [2.0, -16.0 / 9]])
a14, c14 = np.linalg.solve(M14, np.array([1.0 / 3, 0.0]))
print('N4 面心加顶点 14 方向：解 a = %.12f（1/15 = %.12f），c = %.12f（3/40 = %.12f）'
      % (a14, 1.0 / 15, c14, 3.0 / 40))
r = probe('N4 14 方向二阶加四阶', [a14] * 6 + [c14] * 8, f + c)

M26 = np.array([[2.0, 4.0, 8.0 / 3],
                [2.0, -1.0, -16.0 / 9],
                [2.0, -13.0 / 2, -112.0 / 27]])
sol26 = np.linalg.solve(M26, np.array([1.0 / 3, 0.0, 0.0]))
print('N4 续 26 方向（二阶加四阶加六阶）：解 a = %.6e，b = %.6e，c = %.6e'
      % (sol26[0], sol26[1], sol26[2]))
print('   判负依据：三阶条件联立后出现负权重（min = %.6e < 0），物理上不可采纳' % sol26.min())

print('')
print('==== 判据 S1 至 S3：源分配权重与合成权重一致性 ====')
base = np.array([1.0 / 6] * 6)
dvec = np.array([1.0, -0.2, -0.2, -0.2, -0.2, -0.2]) / 5.0


def residue(v, w, ds):
    A = T2(v * w, ds)
    tr = np.trace(A) / 3.0
    return float(np.max(np.abs(A - tr * np.eye(3))) / abs(tr))


print('S1 一致口径（源分配 = 合成 = 1/6）：方向性残差 = %.3e（期望 0）' % residue(base, base, f))
eps = np.linspace(0.0, 0.30, 13)
res = np.array([residue(base + t * dvec, base, f) for t in eps])
slope = np.polyfit(eps[1:], res[1:], 1)
lin_err = float(np.max(np.abs(np.polyval(slope, eps[1:]) - res[1:])))
print('S2 不一致（源分配偏移 eps，合成保持 1/6）：残差 = %.6f eps（斜率 %.6f，线性拟合残差 %.2e）'
      % (slope[0], slope[0], lin_err))
print('   线性通过原点：截距 = %.3e' % slope[1])
bound = 1.0e-6 / slope[0]
print('S3 允许上界：方向性残差不超过 1e-6 时，源分配权重相对偏差不超过 %.3e' % bound)

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
agrid = np.linspace(0.0, 1.0 / 30, 400)
bgrid = (1.0 / 6 - agrid) / 2.0
r4grid = []
for a in agrid:
    b = (1.0 / 6 - a) / 2.0
    ws = [a] * 6 + [b] * 12
    xxxx, xxyy = inv4(ws, f + e)
    r4grid.append(abs(xxxx - 3.0 * xxyy))
ax[0].plot(agrid, r4grid, color='tab:blue')
ax[0].axvline(1.0 / 30, color='tab:red', ls='--', label='a = 1/30')
ax[0].set_yscale('log')
ax[0].set_xlabel('face weight a (edge weight b = (1/6 - a)/2)')
ax[0].set_ylabel('fourth-order residual')
ax[0].set_title('N3 uniqueness along isotropy line')
ax[0].legend()
ax[1].plot(eps, res, 'o-', color='tab:green', label='measured')
ax[1].plot(eps, np.polyval(slope, eps), 'r--', label='slope %.3f' % slope[0])
ax[1].set_xlabel('source-weight deviation eps')
ax[1].set_ylabel('directional residual')
ax[1].set_title('S2 residual vs deviation (linear, through zero)')
ax[1].legend()
plt.tight_layout()
plt.savefig(IMG, dpi=150)
print('')
print('图已保存：' + IMG)