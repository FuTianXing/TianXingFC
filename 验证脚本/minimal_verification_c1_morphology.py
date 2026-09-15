# -*- coding: utf-8 -*-
r"""
minimal_verification_c1_morphology.py

[C1-2] 形态枚举：位错环、畴壁、点三类形态的存在性与稳定性判定。
模型为体系裁定的键弹性原始形式（法线轴高度差，非紧致平方型）。

模型
----
三维格点高度场 u（法线轴层数），开放盒、边界钉背景（缠绕守恒的封闭环境）：

    E = (k/2) * sum_bonds (du - 1)^2 + K * sum_sites (1 - cos 2*pi*u)

k = 1；背景真空 u0 = i + j + k（三方向均匀缠绕，每键一层，K 项为零）。
边界钉 u|bnd = u0（额外层被困盒内；开放面由边界值承担）。

三类形态（多出层的区域 Omega 不同，过渡面即"膜/壁"）：
  位错环：Omega = 下半球（z<0 且 d<Rc）——膜为 z=0 圆盘，位错线 = 赤道环；
  畴壁  ：Omega = z 向平板（z1..z2，横向收进侧壁 3 格）——两个平行界面；
  点    ：Omega = 中心球（d<r0）——无棱闭合泡。

协议与参数
----
GF（梯度流 eta=0.01）与 ID（惯性阻尼 dt=0.1, gamma=0.2），与 C1-1 同族。
K 两档：0.02（宽位错区，l* ~ 7 格）与 0.25（窄位错区，l* ~ 2 格，落窗下界）。

判据（先立判据，再实现；预期写错时修正判据并记录教训。v0 预跑两处修正见
注记第六节：ID 的 v 边界同步、K=0.25 的协议敏感模式）
----
G0 构造合法：三形态 E0 有限、peak0 >= 0.6、ext0 >= 50 格。
J1（K = 0.02，宽位错区）：三形态在 GF 与 ID 下全部塌缩（fate = collapse）。
J2（K = 0.25，窄位错区，GF）：环 Rc = 4、5 驻留、Rc = 3 塌缩（钉扎的尺寸
   依赖）；壁与点驻留。
J3（K = 0.25，ID，协议敏感模式）：环与点脱钉塌缩、壁保持驻留（钉扎亚稳对
   惯性协议敏感；壁因界面驱动力小而两协议一致）。
J4（读数）：形态表与驻留能量比 E_f/E0 汇总打印，不设通过条件。

判读量：peak = max(u-u0)；ext = #{(u-u0) > 0.5}（形态体积）；
fate 分类：collapse（peak 或 ext <= 15% 初值）/ partial（显著但未消失）/
reside（peak >= 70% 初值、ext > 30% 初值、末段漂移 < 1%）。

运行
----
python 验证脚本/minimal_verification_c1_morphology.py
确定性（无随机数），运行约 4 分钟。图存
D:/software/TianXingFC/images/c1_morphology.png。
"""

import time
import numpy as np

T_START = time.time()

k = 1.0
N = 32
ETA = 0.01
DT, GAMMA = 0.1, 0.2
K_WIDE, K_NARROW = 0.02, 0.25

print("=" * 72)
print("[C1-2] 形态枚举：位错环、畴壁、点（平方型非紧致弹性）")
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


BND = np.zeros((N, N, N), dtype=bool)
BND[0, :, :] = True
BND[-1, :, :] = True
BND[:, 0, :] = True
BND[:, -1, :] = True
BND[:, :, 0] = True
BND[:, :, -1] = True


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


def run(u_init, Kp, steps, protocol, snap=200):
    u = u_init.copy()
    if protocol == "ID":
        v = np.zeros_like(u)
        a = -grad(u, Kp)
    tr = []
    for t in range(steps):
        if protocol == "GF":
            u = u - ETA * grad(u, Kp)
            teff = (t + 1) * ETA
        else:
            u = u + DT * v + 0.5 * DT * DT * a
            a1 = -grad(u, Kp)
            v = (v + 0.5 * DT * (a + a1)) / (1.0 + GAMMA * DT)
            a = a1
            teff = (t + 1) * DT
        pin(u)
        if protocol == "ID":
            v[BND] = 0.0
        if (t % snap == 0) or (t == steps - 1):
            p, e = readout(u)
            tr.append((teff, energy(u, Kp), p, e))
    return u, tr


def fate(peak0, ext0, tr):
    peakf, extf = tr[-1][2], tr[-1][3]
    drift = abs(tr[-1][1] - tr[-3][1]) / max(abs(tr[-1][1]), 1e-12)
    if peakf <= 0.15 * peak0 or extf <= 0.15 * ext0:
        return "collapse", peakf, extf, drift
    if peakf >= 0.7 * peak0 and extf > 0.3 * ext0 and drift < 0.01:
        return "reside", peakf, extf, drift
    return "partial", peakf, extf, drift


def make_ring(Rc, t=1.0, w=2.5):
    g = 0.5 * (1.0 - np.tanh((RR - Rc) / w))
    return 0.5 * (1.0 - np.tanh(ZC / t)) * g


def make_wall(t1=1.2, z1=-8.0, z2=8.0, margin=3.0, mw=1.5):
    h = 0.5 * (np.tanh((ZC - z1) / t1) - np.tanh((ZC - z2) / t1))
    dx = np.minimum(IX, N - 1 - IX)
    dy = np.minimum(IY, N - 1 - IY)
    g = 0.5 * (1.0 - np.tanh((margin - np.minimum(dx, dy)) / mw))
    return h * g


def make_point(r0=3.0, t=1.2):
    return 0.5 * (1.0 - np.tanh((RR - r0) / t))


# ----------------------------------------------------------------------
# G0 构造合法性
# ----------------------------------------------------------------------
print("\n[G0] 构造合法性（三形态初态）")
SHAPES = [("ring R4", make_ring(4.0)), ("wall", make_wall()), ("point", make_point())]
g0_ok = True
for name, h in SHAPES:
    u = u0 + h
    p, e = readout(u)
    E = energy(u, K_NARROW)
    ok = np.isfinite(E) and p >= 0.6 and e >= 50
    g0_ok &= ok
    print(f"    {name}：peak = {p:.3f}，ext = {e}，E0(K=0.25) = {E:.1f}")
check("G0", g0_ok, "三形态 peak >= 0.6、ext >= 50、E0 有限")

# ----------------------------------------------------------------------
# 主矩阵
# ----------------------------------------------------------------------
RESULTS = {}


def execute(tag, h, Kp, steps, protocol):
    u_init = u0 + h
    peak0, ext0 = readout(u_init)
    E0 = energy(u_init, Kp)
    t0 = time.time()
    uf, tr = run(u_init, Kp, steps, protocol)
    f, pf, ef, drift = fate(peak0, ext0, tr)
    Ef = energy(uf, Kp)
    RESULTS[(tag, Kp, protocol)] = dict(f=f, peak0=peak0, ext0=ext0, E0=E0, Ef=Ef,
                                        peakf=pf, extf=ef, drift=drift, uf=uf)
    print(f"    [{tag} K={Kp} {protocol}] E {E0:.1f} -> {Ef:.3f}（末段漂移 {drift:.2e}），"
          f"peak {peak0:.2f}->{pf:.2f}，ext {ext0}->{ef}，fate = {f}（{time.time() - t0:.0f} s）")
    return f, pf, ef, drift


print("\n[环] 位错环（半球泡，位错线 = 赤道环）")
print("    GF K=0.02（R 扫描 3/4/5）")
ring_gf02 = []
for Rc in (3.0, 4.0, 5.0):
    ring_gf02.append(execute(f"ring R{int(Rc)}", make_ring(Rc), K_WIDE, 2400, "GF"))
print("    GF K=0.25（R 扫描 3/4/5）")
ring_gf25 = []
for Rc in (3.0, 4.0, 5.0):
    ring_gf25.append(execute(f"ring R{int(Rc)}", make_ring(Rc), K_NARROW, 2400, "GF"))
print("    ID 对照（R = 4）")
ring_id02 = execute("ring R4", make_ring(4.0), K_WIDE, 4000, "ID")
ring_id25 = execute("ring R4", make_ring(4.0), K_NARROW, 4000, "ID")

print("\n[壁] 畴壁对（z 向平板，双界面）")
wall_gf02 = execute("wall", make_wall(), K_WIDE, 9600, "GF")
wall_gf25 = execute("wall", make_wall(), K_NARROW, 2400, "GF")
wall_id02 = execute("wall", make_wall(), K_WIDE, 4000, "ID")
wall_id25 = execute("wall", make_wall(), K_NARROW, 4000, "ID")

print("\n[点] 点缺陷（中心球泡）")
pt_gf02 = execute("point", make_point(), K_WIDE, 2400, "GF")
pt_gf25 = execute("point", make_point(), K_NARROW, 2400, "GF")
pt_id02 = execute("point", make_point(), K_WIDE, 4000, "ID")
pt_id25 = execute("point", make_point(), K_NARROW, 4000, "ID")

# ----------------------------------------------------------------------
# 判据
# ----------------------------------------------------------------------
print("\n" + "=" * 72)
print("判据")
print("=" * 72)

j1_ok = all(f == "collapse" for f, _, _, _ in ring_gf02)
j1_ok &= (ring_id02[0] == "collapse")
j1_ok &= (wall_gf02[0] == "collapse" and wall_id02[0] == "collapse")
j1_ok &= (pt_gf02[0] == "collapse" and pt_id02[0] == "collapse")
check("J1", j1_ok,
      f"K=0.02 全塌缩：环 GF{[x[0] for x in ring_gf02]}/ID {ring_id02[0]}，"
      f"壁 GF {wall_gf02[0]}/ID {wall_id02[0]}，点 GF {pt_gf02[0]}/ID {pt_id02[0]}")

j2_ok = (ring_gf25[0][0] == "collapse" and ring_gf25[1][0] == "reside"
         and ring_gf25[2][0] == "reside")
j2_ok &= (wall_gf25[0] == "reside" and pt_gf25[0] == "reside")
check("J2", j2_ok,
      f"K=0.25 GF：环 R3 {ring_gf25[0][0]}/R4 {ring_gf25[1][0]}/R5 {ring_gf25[2][0]}，"
      f"壁 {wall_gf25[0]}，点 {pt_gf25[0]}")

j3_ok = (ring_id25[0] == "collapse" and pt_id25[0] == "collapse"
         and wall_id25[0] == "reside")
check("J3", j3_ok,
      f"K=0.25 ID：环 {ring_id25[0]}、点 {pt_id25[0]}（脱钉），壁 {wall_id25[0]}")

print("  [J4] 读数：驻留能量比 E_f/E0")
for tag in ("ring R4", "ring R5", "wall", "point"):
    r = RESULTS.get((tag, K_NARROW, "GF"))
    if r:
        print(f"      {tag}: E {r['E0']:.1f} -> {r['Ef']:.3f}（比 {r['Ef'] / r['E0']:.3f}），"
              f"ext {r['ext0']} -> {r['extf']}")

print("\n形态表（fate：collapse 塌缩 / partial 部分衰减 / reside 驻留）")
print("  形态        K=0.02 GF/ID        K=0.25 GF/ID        读数")
for tag, h in SHAPES:
    r02g = RESULTS.get((tag.replace(' R4', ' R4'), K_WIDE, "GF")) or RESULTS.get((tag, K_WIDE, "GF"))
    a = RESULTS.get((tag, K_WIDE, "GF"), {}).get("f", "-")
    b = RESULTS.get((tag, K_WIDE, "ID"), {}).get("f", "-")
    c = RESULTS.get((tag, K_NARROW, "GF"), {}).get("f", "-")
    d = RESULTS.get((tag, K_NARROW, "ID"), {}).get("f", "-")
    key = (tag, K_NARROW, "GF")
    info = RESULTS.get(key, {})
    print(f"  {tag:10s}  {a:8s}/{b:8s}      {c:8s}/{d:8s}      "
          f"peak {info.get('peak0', 0):.2f}->{info.get('peakf', 0):.2f}, ext {info.get('ext0', 0)}->{info.get('extf', 0)}")

n_pass = sum(1 for _, ok in CHECKS if ok)
print(f"\n  合计：{n_pass}/{len(CHECKS)} 通过；总运行 {time.time() - T_START:.0f} s")

# ----------------------------------------------------------------------
# 切片图（环/壁/点 x-z 切片，y = CY）
# ----------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    titles = [("位错环 R4", "ring R4"), ("畴壁", "wall"), ("点缺陷", "point")]
    fig, axes = plt.subplots(3, 2, figsize=(11, 12))
    for row, (zh, tag) in enumerate(titles):
        for col, Kp in enumerate((K_WIDE, K_NARROW)):
            key = (tag, Kp, "GF")
            uf = RESULTS[key]["uf"]
            h = uf - u0
            sl = h[:, CY, :].T  # (z, x) 便于显示
            ax = axes[row, col]
            im = ax.imshow(sl, origin="lower", cmap="RdBu_r", vmin=-0.2, vmax=1.2)
            ax.set_title(f"{zh}，K={Kp}，GF 末态（{RESULTS[key]['f']}）")
            ax.set_xlabel("x"); ax.set_ylabel("z")
            fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig("D:/software/TianXingFC/images/c1_morphology.png", dpi=150)
    print("图已保存：images/c1_morphology.png")
except Exception as exc:
    print(f"绘图失败（不影响判据）：{exc}")

print("\n[C1-2] 完成。")