# -*- coding: utf-8 -*-
r"""
c1_derrick.py

[C1-3] Derrick 复查：三形态的 E(R) 标度律、格点钉扎临界曲线与脱钉阈值。

模型与 C1-2 同（平方型非紧致弹性，法线轴高度差）：

    E = (k/2) * sum_bonds (du - 1)^2 + K * sum_sites (1 - cos 2*pi*u)

背景 u0 = i + j + k，边界钉 u|bnd = u0；GF（eta=0.01）主协议。
形态：环（半球泡，Rc 扫描）、点（球泡，r0 扫描）、壁（z 向平板，厚 d 扫描）。

判据（先立判据，再实现；预期写错时修正判据并记录教训）
----
G0 静态标度合法：E0(R) 严格单调增；环与点的双对数斜率 s ∈ [1.5, 2.5]（面积主导预期）。
J1 连续极限无规避：K = 0.02 下环 R 扫描（2.5 至 5.0，步 0.5）全部塌缩
   （E_f/E0 <= 0.05）。
J2 临界曲线：K ∈ {0.10, 0.15, 0.20, 0.25} 下环的 fate(R)（R 2.5 至 5.0）存在临界
   R_c(K)（最小驻留半径）；预期 R_c 随 K 减小而增大（单调不减）。
J3 脱钉阈值有限：K = 0.25、R = 4 的驻留态加多模扰动 alpha*pert 后 GF 松弛，
   脱钉阈值 alpha* 有限（0 < alpha* <= 0.4）。
J4 对照：点（K = 0.25）的临界 r_c 存在（预期 r_c <= 3.5）；壁（K = 0.25）在
   d ∈ {4, 8, 16} 全部驻留（预期无临界）。

运行
----
python 验证脚本/c1_derrick.py
确定性（无随机数），运行约 2 分钟。图存
D:/software/TianXingFC/images/c1_derrick.png。
"""

import time
import numpy as np

T_START = time.time()

k = 1.0
N = 32
ETA = 0.01

print("=" * 72)
print("[C1-3] Derrick 复查：E(R) 标度律、钉扎临界曲线与脱钉阈值")
print("=" * 72)

IX, IY, IZ = np.indices((N, N, N))
u0 = (IX + IY + IZ).astype(float)
CX = CY = CZ = N // 2
XC = IX - CX
YC = IY - CY
ZC = IZ - CZ
RR = np.sqrt(XC * XC + YC * YC + ZC * ZC)

CHECKS = []


def check(tag, ok, detail):
    CHECKS.append((tag, bool(ok)))
    print(f"  [{tag}] {'PASS' if ok else 'FAIL'}  {detail}")


def energy(u, Kp):
    E = 0.0
    for ax in range(3):
        d = np.diff(u, axis=ax) - 1.0
        E += 0.5 * k * float(np.sum(d ** 2))
    E += Kp * float(np.sum(1.0 - np.cos(2.0 * np.pi * u)))
    return E


def grad(u, Kp):
    p = np.pad(u, 1, mode="edge")
    ctr = (slice(1, -1),) * 3
    s = (p[2:, 1:-1, 1:-1] + p[:-2, 1:-1, 1:-1] + p[1:-1, 2:, 1:-1]
         + p[1:-1, :-2, 1:-1] + p[1:-1, 1:-1, 2:] + p[1:-1, 1:-1, :-2])
    return k * (6.0 * u - s) + 2.0 * np.pi * Kp * np.sin(2.0 * np.pi * u)


def pin(u):
    u[0, :, :] = u0[0, :, :]
    u[-1, :, :] = u0[-1, :, :]
    u[:, 0, :] = u0[:, 0, :]
    u[:, -1, :] = u0[:, -1, :]
    u[:, :, 0] = u0[:, :, 0]
    u[:, :, -1] = u0[:, :, -1]


def readout(u):
    h = u - u0
    return float(h.max()), int(np.count_nonzero(h > 0.5))


def run_gf(u_init, Kp, steps, snap=200):
    u = u_init.copy()
    tr = []
    for t in range(steps):
        u = u - ETA * grad(u, Kp)
        pin(u)
        if (t % snap == 0) or (t == steps - 1):
            p, e = readout(u)
            tr.append((energy(u, Kp), p, e))
    return u, tr


def fate_of(tr, peak0, ext0):
    peakf, extf = tr[-1][1], tr[-1][2]
    drift = abs(tr[-1][0] - tr[-3][0]) / max(abs(tr[-1][0]), 1e-12)
    if peakf <= 0.15 * peak0 or extf <= 0.15 * ext0:
        return "collapse"
    if peakf >= 0.7 * peak0 and extf > 0.3 * ext0 and drift < 0.01:
        return "reside"
    return "partial"


def make_ring(Rc, t=1.0, w=2.5):
    g = 0.5 * (1.0 - np.tanh((RR - Rc) / w))
    return 0.5 * (1.0 - np.tanh(ZC / t)) * g


def make_point(r0, t=1.2):
    return 0.5 * (1.0 - np.tanh((RR - r0) / t))


def make_wall(d, t1=1.2, margin=3.0, mw=1.5):
    z1, z2 = -d / 2.0, d / 2.0
    h = 0.5 * (np.tanh((ZC - z1) / t1) - np.tanh((ZC - z2) / t1))
    dx = np.minimum(IX, N - 1 - IX)
    dy = np.minimum(IY, N - 1 - IY)
    g = 0.5 * (1.0 - np.tanh((margin - np.minimum(dx, dy)) / mw))
    return h * g


# ----------------------------------------------------------------------
# G0 静态标度
# ----------------------------------------------------------------------
print("\n[G0] 静态标度 E0(R)")
RING_RS = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
ring_E0 = []
for Rc in RING_RS:
    ring_E0.append(energy(u0 + make_ring(Rc), 0.25))
POINT_RS = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
point_E0 = [energy(u0 + make_point(r0), 0.25) for r0 in POINT_RS]
WALL_DS = [4.0, 6.0, 8.0, 10.0, 12.0, 16.0]
wall_E0 = [energy(u0 + make_wall(d), 0.25) for d in WALL_DS]

s_ring = float(np.polyfit(np.log(RING_RS), np.log(ring_E0), 1)[0])
s_point = float(np.polyfit(np.log(POINT_RS), np.log(point_E0), 1)[0])
mono_ring = all(b > a for a, b in zip(ring_E0, ring_E0[1:]))
mono_point = all(b > a for a, b in zip(point_E0, point_E0[1:]))
print(f"    环 E0(R)（R=2..6）：{' '.join(f'{v:.0f}' for v in ring_E0)}，双对数斜率 {s_ring:.3f}")
print(f"    点 E0(r)：{' '.join(f'{v:.0f}' for v in point_E0)}，斜率 {s_point:.3f}")
print(f"    壁 E0(d)：{' '.join(f'{v:.0f}' for v in wall_E0)}")
check("G0", mono_ring and mono_point and 1.5 <= s_ring <= 2.5 and 1.5 <= s_point <= 2.5,
      f"单调增成立；斜率 环 {s_ring:.3f}、点 {s_point:.3f}（期望 1.5-2.5）")

# ----------------------------------------------------------------------
# D1 命运图（环）：K × R
# ----------------------------------------------------------------------
print("\n[D1/J1-J2] 命运图（环）：K × R 扫描，GF 2400 步")
K_LIST = [0.02, 0.10, 0.15, 0.20, 0.25]
RR_LIST = [2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
FATE = {}
UREF = {}
for Kp in K_LIST:
    row = []
    for Rc in RR_LIST:
        u_init = u0 + make_ring(Rc)
        peak0, ext0 = readout(u_init)
        E0 = energy(u_init, Kp)
        uf, tr = run_gf(u_init, Kp, 2400)
        f = fate_of(tr, peak0, ext0)
        FATE[(Kp, Rc)] = f
        if Kp == 0.25 and abs(Rc - 4.0) < 1e-9:
            UREF["res"] = uf.copy()
        row.append(f[0].upper() if f != "partial" else "P")
    print(f"    K={Kp:5.2f}: R=2.5..5.0 → {' '.join(row)}（C=塌缩, R=驻留, P=部分）")

# J1：K=0.02 全塌
j1_ok = all(FATE[(0.02, Rc)] == "collapse" for Rc in RR_LIST)
check("J1", j1_ok, f"K=0.02 环全塌：{[FATE[(0.02, Rc)] for Rc in RR_LIST]}")


def rcrit(Kp):
    res = [Rc for Rc in RR_LIST if FATE[(Kp, Rc)] == "reside"]
    return min(res) if res else None


rc_table = {Kp: rcrit(Kp) for Kp in K_LIST}
print(f"    临界半径 R_c(K)：{ {Kp: (rc_table[Kp] if rc_table[Kp] else '>5') for Kp in K_LIST} }")
# J2：R_c 随 K 减小而增大（单调不减；None 视为 >5）
ks_desc = K_LIST[::-1]   # 从大 K 到小 K
rcs_desc = [rc_table[Kp] if rc_table[Kp] is not None else 99 for Kp in ks_desc]
j2_ok = all(b >= a for a, b in zip(rcs_desc, rcs_desc[1:])) and rc_table[0.25] is not None
check("J2", j2_ok,
      f"R_c 随 K 减小而增大（K 降序：{ks_desc} → R_c {rcs_desc}，None 记 >5）："
      + '；'.join(f'K={Kp}: {rc_table[Kp] if rc_table[Kp] else chr(62) + chr(53)}' for Kp in ks_desc))

# ----------------------------------------------------------------------
# D2 对照：点与壁
# ----------------------------------------------------------------------
print("\n[D2/J4] 对照：点（K=0.25 与 0.10）与壁（K=0.25）")
po_rc = None
for Kp in (0.25, 0.10):
    row = []
    for r0 in (2.0, 2.5, 3.0, 3.5, 4.0):
        u_init = u0 + make_point(r0)
        peak0, ext0 = readout(u_init)
        uf, tr = run_gf(u_init, Kp, 2400)
        f = fate_of(tr, peak0, ext0)
        row.append((r0, f))
    res = [r0 for r0, f in row if f == "reside"]
    rc = min(res) if res else None
    if Kp == 0.25:
        po_rc = rc
    print(f"    点 K={Kp}: " + ' '.join(f'r{r0}={f[0].upper()}' for r0, f in row) +
          f"；r_c = {rc if rc else '>4'}")
wall_rows = []
for d in (4.0, 8.0, 16.0):
    u_init = u0 + make_wall(d)
    peak0, ext0 = readout(u_init)
    uf, tr = run_gf(u_init, 0.25, 2400)
    f = fate_of(tr, peak0, ext0)
    wall_rows.append((d, f))
print("    壁 K=0.25: " + ' '.join(f'd={d:.0f}={f[0].upper()}' for d, f in wall_rows))
j4_ok = (po_rc is not None and po_rc <= 3.5) and all(f == "reside" for _, f in wall_rows)
check("J4", j4_ok, f"点 r_c = {po_rc}（期望 <= 3.5）；壁三档 {[f for _, f in wall_rows]}（期望全驻留）")

# ----------------------------------------------------------------------
# D3 脱钉阈值
# ----------------------------------------------------------------------
print("\n[D3/J3] 脱钉阈值：K=0.25、R=4 驻留态 + alpha*pert")
u_res = UREF["res"]
PERT = (np.sin(2.0 * np.pi * XC / 8.0) * np.sin(2.0 * np.pi * ZC / 6.0)
        * np.sin(2.0 * np.pi * YC / 10.0))
PERT = PERT / float(np.abs(PERT).max())
ALPHAS = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40]
alpha_star = None
res_peak0, res_ext0 = readout(u_res)
for a in ALPHAS:
    u_try = u_res + a * PERT
    uf, tr = run_gf(u_try, 0.25, 2400)
    f = fate_of(tr, res_peak0, res_ext0)
    print(f"    alpha={a:.2f}: fate = {f}")
    if f == "collapse" and alpha_star is None:
        alpha_star = a
check("J3", alpha_star is not None and alpha_star <= 0.4,
      f"脱钉阈值 alpha* = {alpha_star}（期望 0 < alpha* <= 0.4）")

# ----------------------------------------------------------------------
# 汇总
# ----------------------------------------------------------------------
n_pass = sum(1 for _, ok in CHECKS if ok)
print(f"\n判据汇总：{n_pass}/{len(CHECKS)} 通过；总运行 {time.time() - T_START:.0f} s")
for tag, ok in CHECKS:
    print(f"  {tag}: {'PASS' if ok else 'FAIL'}")

# 图
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axs = plt.subplots(2, 2, figsize=(12.5, 9))
    a = axs[0, 0]
    a.loglog(RING_RS, ring_E0, "o-", label=f"环 (slope {s_ring:.2f})")
    a.loglog(POINT_RS, point_E0, "s-", label=f"点 (slope {s_point:.2f})")
    a.set_xlabel("R"); a.set_ylabel("E0"); a.set_title("(a) 静态标度 E0(R)，K=0.25")
    a.legend()

    b = axs[0, 1]
    for i, Kp in enumerate(K_LIST):
        for j, Rc in enumerate(RR_LIST):
            f = FATE[(Kp, Rc)]
            col = {"collapse": "#d62728", "reside": "#2ca02c", "partial": "#ff7f0e"}[f]
            b.scatter(Rc, Kp, c=col, s=140, marker="s")
    b.set_xlabel("Rc"); b.set_ylabel("K")
    b.set_title("(b) 命运图（红=塌缩，绿=驻留）")

    c = axs[1, 0]
    c.plot(ALPHAS, [1 if a_ <= (alpha_star or 99) else 0 for a_ in ALPHAS], "o-")
    c.set_xlabel("扰动幅度 alpha"); c.set_ylabel("驻留(1)/脱钉(0)")
    c.set_title(f"(c) 脱钉扫描（alpha* = {alpha_star}）")

    d_ax = axs[1, 1]
    ks = [Kp for Kp in K_LIST if rc_table[Kp] is not None]
    rcs = [rc_table[Kp] for Kp in ks]
    d_ax.plot(ks, rcs, "o-")
    d_ax.set_xlabel("K"); d_ax.set_ylabel("R_c")
    d_ax.set_title("(d) 临界半径 R_c(K)")
    fig.tight_layout()
    fig.savefig("D:/software/TianXingFC/images/c1_derrick.png", dpi=150)
    print("图已保存：images/c1_derrick.png")
except Exception as exc:
    print(f"绘图失败（不影响判据）：{exc}")

print("\n[C1-3] 完成。")