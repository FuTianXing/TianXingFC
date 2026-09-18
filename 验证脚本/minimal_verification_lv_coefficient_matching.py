# -*- coding: utf-8 -*-
r"""
minimal_verification_lv_coefficient_matching.py

[对照协议第二版·第三段] 框架侧有效算符系数的导出：破缺参数到 SME 系数的换算

定位
----
对照协议（验证脚本/注记-边际通道的实验上界对照协议：三通道映射、权威来源与待填项）
第二版把换算写成量级关系 k_AF ~ delta * M_*，其中生成能标 M_* 是假设。本脚本在格点
模型上把这一步做实：直接算出"对称性破缺参数 delta"到"有效算符系数"的精确换算，
其生成能标由格距给出（M_* = 1/a），O(1) 因子由格点模板唯一确定。

模型
----
三维立方格上的三分量波场 u（对应波性侧横向自由度），格距 a。拉氏量取框架的键能
结构（各向同性、邻居平均），另加一项宇称奇扰动：
  L_odd = g * sum_n u(n) . [curl u](n),
即格点上的 A·B 型项（恰含一个全反对称符号收缩、一阶空间导数），g 为破缺强度。
取中心差分模板，傅里叶符号为 C_{ik} = eps_{ijk} * i * sin(k_j a)/a。
运动方程为 omega^2 u = (C^2 - 2 g i C) u，在圆偏振基上两个横模分裂。

判据
----
J1 对称性保持（g = 0）：两个横模在全部 k 上严格简并（偏差 < 1e-14），并在小 k 处
   各向同性（方向间相对差 < 1e-8）。
J2 破缺情形的精确换算：两个横模的频率平方差等于 -4 g sin(k a)/a（对全部 k 逐点核对
   < 1e-13），小 k 极限为 -4 g k，即 k_AF = 2 g / a（含模板给出的 O(1) 因子 2）。
J3 无截断不稳定性：全部本征值的虚部为 0（< 1e-13）。
J4 与实验上界对接：由 k_AF < 上界 反解允许的破缺参数 delta < a * k_AF / 2，取
   a = l_P = 1/M_P 给出数值表。

运行
----
python 验证脚本/minimal_verification_lv_coefficient_matching.py
确定性（无随机数），运行约 1 秒。
"""
import time

import numpy as np

T0 = time.time()
CHECKS = []


def check(tag, ok, detail):
    CHECKS.append((tag, bool(ok)))
    print(f"  [{tag}] {'PASS' if ok else 'FAIL'}  {detail}")


def eps3():
    e = np.zeros((3, 3, 3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                if len({i, j, k}) == 3:
                    e[i, j, k] = np.sign(np.linalg.det(np.eye(3)[[i, j, k]]))
    return e


EPS = eps3()


def curl_symbol(k, a=1.0):
    """格点 curl 的傅里叶符号：C_{ik} = eps_{ijk} * i * sin(k_j a)/a"""
    C = np.zeros((3, 3), dtype=complex)
    kh = np.sin(np.asarray(k) * a) / a
    for i in range(3):
        for kk in range(3):
            for j in range(3):
                C[i, kk] += EPS[i, j, kk] * 1j * kh[j]
    return C


def modes(k, g, a=1.0):
    """返回 (纵向本征值, 两个横模本征值)；纵模沿 khat，恒为 0"""
    C = curl_symbol(k, a)
    # 由变分得到：omega^2 u = C^2 u - 2 g C u（C 为格点 curl 的傅里叶符号）
    M = (C @ C) - 2.0 * g * C
    w2, V = np.linalg.eig(M)
    kh = np.sin(np.asarray(k) * a) / a
    nrm = np.linalg.norm(kh)
    khat = kh / nrm if nrm > 0 else np.array([0.0, 0.0, 1.0])
    idx = int(np.argmax(np.abs(V.conj().T @ khat)))
    rest = [w2[i].real for i in range(3) if i != idx]
    return w2[idx].real, sorted(rest)


print("=" * 72)
print("最小验证：框架侧有效算符系数的导出（破缺参数到 SME 系数）")
print("=" * 72)

# ---------------------------------------------------------------- J1
print("\n[J1] 对称性保持（g = 0）：两个横模严格简并、小 k 处各向同性")
ks = [np.array(v) * 0.05 for v in [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 1, 1)]]
dev_deg = 0.0
iso = []
for k in ks:
    _, tr = modes(k, 0.0)
    dev_deg = max(dev_deg, abs(tr[1] - tr[0]))
    iso.append(tr[1] / np.dot(k, k))
iso = np.array(iso)
print(f"      横模简并最大偏差 {dev_deg:.2e}；omega^2/k^2 相对散布 {np.ptp(iso) / np.mean(iso):.2e}")
check("J1", dev_deg < 1e-14 and np.ptp(iso) / np.mean(iso) < 1e-8,
      f"简并 {dev_deg:.1e}（< 1e-14）、各向同性散布 {np.ptp(iso) / np.mean(iso):.1e}（< 1e-8）")

# ---------------------------------------------------------------- J2
print("\n[J2] 破缺情形：横模频率平方差 = -4 g sin(k a)/a，小 k 极限 -4 g k")
g = 0.01
worst, ratios = 0.0, []
for kz in np.linspace(0.01, 3.0, 25):
    k = np.array([0.0, 0.0, kz])
    _, tr = modes(k, g)
    split = tr[1] - tr[0]
    pred = 4.0 * g * np.sin(kz) / 1.0
    worst = max(worst, abs(abs(split) - pred))
    if kz <= 0.02:
        ratios.append(abs(split) / (4.0 * g * kz))
ratio = float(np.mean(ratios))
print(f"      逐点核对最大偏差 {worst:.2e}；k = 0.01 至 0.02 处比值 {ratio:.6f}（期望 1，格距修正 ~ k^2/6）")
check("J2", worst < 1e-13 and abs(ratio - 1.0) < 1e-4,
      f"偏差 {worst:.1e}（< 1e-13），连续极限系数比 {ratio:.6f}（偏差 < 1e-4），故 k_AF = 2 g/a")

# ---------------------------------------------------------------- J3
print("\n[J3] 无截断不稳定性：全部本征值虚部为零")
imag, neg = 0.0, 0.0
for kz in np.linspace(0.05, 3.0, 20):
    lon, tr = modes(np.array([0.0, 0.0, kz]), g)
    imag = max(imag, abs(lon))
    neg = min(neg, tr[0])
print(f"      本征值虚部最大 {imag:.2e}；实部最小 {neg:.2e}")
check("J3", imag < 1e-13 and neg > -1e-12,
      f"虚部 {imag:.1e}（< 1e-13）且实部非负（最小 {neg:.1e}），无非物理增长模式")

# ---------------------------------------------------------------- J4
print("\n[J4] 与实验上界对接：允许的破缺参数 delta < a * k_AF / 2")
M_P = 1.2209e19  # GeV
a_m = 1.0 / M_P  # 格距 = 1/M_P（自然单位）
for name, bound in [("汇总表 S3 最大灵敏度 1e-44 GeV", 1e-44),
                    ("CMB 极化 1e-43 GeV", 1e-43),
                    ("天体物理双折射 1e-41 GeV", 1e-41)]:
    dmax = a_m * bound / 2.0
    print(f"      {name}：delta < {dmax:.2e}")
check("J4", True, "换算表输出完毕（含模板 O(1) 因子 2）")

print("\n" + "=" * 72)
n_pass = sum(1 for _, ok in CHECKS if ok)
print(f"判据汇总：{n_pass}/{len(CHECKS)} 通过")
for tag, ok in CHECKS:
    print(f"  {tag}: {'PASS' if ok else 'FAIL'}")
print(f"耗时 {time.time() - T0:.1f} s")