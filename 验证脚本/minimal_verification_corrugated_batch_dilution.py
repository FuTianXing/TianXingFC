# -*- coding: utf-8 -*-
r"""
minimal_verification_corrugated_batch_dilution.py

带褶皱壁在稀释档下的批内一致率轮：第 295 条登记的余项。

来源：作者 2026-09-21 指令（同意执行）。输入：第 304 条的两窗口与注册规则（本底窗口 tau = 1、取样间隔 Dt），
第 292 条的褶皱档实测（褶皱幅度 0.8 格、壁速 v = 0.1000，故各列穿越时刻的散布 sigma_t = 幅度 / v = 8.0）。

模型：事件级。同一批含 A = 64 个站点，各站窗口中心相对批时刻有偏移 delta_j ~ N(0, sigma_t)（固定种子）；
全局时钟落在站点窗口内则该站登记。批内一致率 = 各站登记状态全同的批相位占比（跨 20 批乘 64 相位平均）。

判据（先行）：
  Y1 近无褶皱极限：sigma_t / Dt = 0.02 时一致率 >= 0.80（严格 1.000 只在 sigma_t = 0；微小散布在临界相位上产生混合）
  Y2 强褶皱极限：sigma_t / Dt >= 1 时一致率 <= 0.05
  Y3 交叉点：一致率降到 0.5 处的 sigma_t / Dt（登记实测值，判据 <= 0.30）
  Y4 应用映射：由第 292 条的褶皱档与合批后档折算的一致率，与实测分片/合批读数定性一致
"""
import numpy as np

A = 64
NB = 20
TAU = 1.0
DT = 5.0
V_WALL = 0.1000
AMP = 0.8
rows = []


def consistency(sig_over_dt, seed=5):
    rng = np.random.default_rng(seed)
    sig = sig_over_dt * DT
    cons = []
    for b in range(NB):
        t_c = 20.0 * (b + 1)
        delta = rng.normal(0.0, sig, size=A) if sig > 0 else np.zeros(A)
        t_j = t_c + delta
        for k in range(64):
            phi = DT * k / 64.0
            lo = t_j - 0.5 * TAU
            hi = t_j + 0.5 * TAU
            reg = (np.floor((hi - phi) / DT) - np.ceil((lo - phi) / DT) + 1.0) >= 1.0
            cons.append(1.0 if reg.min() == reg.max() else 0.0)
    return float(np.mean(cons))


XS = [0.02, 0.05, 0.10, 0.20, 0.50, 1.0, 2.0, 5.0]
cs = [consistency(x) for x in XS]
rows.append(("Y1", "sigma_t/Dt = 0.02 时的一致率（严格 1.000 只在 sigma_t = 0）", "%.3f" % cs[0], ">= 0.80（近乎整批）", cs[0] >= 0.80))
rows.append(("Y2", "sigma_t/Dt >= 1 时的最大一致率", "%.3f" % max(cs[XS.index(1.0):]), "<= 0.05", max(cs[XS.index(1.0):]) <= 0.05))
cross = float("nan")
for i in range(len(XS) - 1):
    if cs[i] >= 0.5 > cs[i + 1]:
        cross = XS[i] + (XS[i + 1] - XS[i]) * (cs[i] - 0.5) / (cs[i] - cs[i + 1])
rows.append(("Y3", "一致率降到 0.5 处的 sigma_t/Dt（交叉点）", "%.3f" % cross, "<= 0.30", cross <= 0.30))

sig_app = AMP / V_WALL
x_app = sig_app / DT
c_app = consistency(x_app)
c_flat = consistency(0.0)
rows.append(("Y4", "第 292 条褶皱档（sigma_t = %.1f，sigma_t/Dt = %.2f）与合批后（sigma_t = 0）的一致率" % (sig_app, x_app), "%.3f 与 %.3f" % (c_app, c_flat), "分片档 <= 0.05、合批档 = 1.000", c_app <= 0.05 and c_flat >= 0.999))

print("=== 带褶皱壁稀释档批内一致率判据表（实测 对 期望）===")
for tag, name, meas, expx, ok in rows:
    print("%-4s %-56s 实测 %-22s 期望 %-30s %s" % (tag, name, meas, expx, "通过" if ok else "不通过"))
print("")
print("一致率曲线（sigma_t/Dt -> 一致率）：")
for x, c in zip(XS, cs):
    print("   %.2f -> %.3f" % (x, c))
print("")
print("Y4 细节：第 292 条褶皱档的幅度 %.1f 格、壁速 %.4f，折算 sigma_t = 幅度 / 速度 = %.1f 个时间单位；" % (AMP, V_WALL, sig_app))
print("         分箱取本轮的 %.1f，故 sigma_t/Dt = %.2f，一致率 %.3f（分片）；壁自行拉直后 sigma_t -> 0，一致率 %.3f（整批）。" % (DT, x_app, c_app, c_flat))
print("         这与第 292 条实测（早段每批中位 1 站、晚段 64 站）同向：稀释档下褶皱把整批相关拆成逐站独立。")
print("         敏感性来源：各站窗口边缘在不同相位上翻转，落在临界相位附近的相位比例约为 2 sigma_t/Dt，故一致率的下降远快于线性。")
