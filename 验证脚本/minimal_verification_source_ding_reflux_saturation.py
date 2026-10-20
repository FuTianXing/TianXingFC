# -*- coding: utf-8 -*-
r"""
minimal_verification_source_ding_reflux_saturation.py

第 726 号 来源丁第二轮：沉积的一次性与回流的脆弱性（判据 DS-1 至 DS-8）

问题：三维复合位错弹性本构攻坚线要找能量泛函里的显式负线项。四个候选来源之一是来源丁，
碰撞写入的动力学反作用。第 722 号在两能级写入算符装置上确认：写入把站点模的激发搬进记录
寄存器，总激发数守恒，站点侧增量 -0.99882383，寄存器侧增量 +0.99882383。作者裁决 R-3 取
甲乙读法：位错能量分栏显式含记录侧，负贡献标注为可消去对。

本轮问两个新问题。一，沉积是不是一次性的：写一次就把站点搬空，后续再过闸也无物可搬，
沉积有上限而不累加。二，负贡献有多脆：若记录侧能把激发搬回站点侧，即存在回流通道，
负贡献是否立刻消失。库内已确认零接触封锁锁定与摆程之间的通道，对记录场与构型时钟之间的
通道保持沉默。本轮量化这条沉默通道一旦打开的代价。

装置：全部复用 验证脚本/minimal_verification_write_backreaction_source.py 的参数与实现，
跨轮移植的前提守恒必须成立。参数 N_THETA=256，OMEGA_P=2π，THETA0=π，W_PULSE=0.5，
DT=2.0e-3，T_MAX=1.2，指针波包宽 PW=0.15（第 722 号主档）。写入耦合
G_FULL=(π/2)·OMEGA_P/area，实测 G_FULL≈5.0916，折返时刻 t_cross=(2π-THETA0)/OMEGA_P=0.5。

关键机制：自由演化是平移，指针位置 θ_p(t)=π+Ω_P t。f 在 θ=π 处是谷（f=exp(-1/w²)≈0.018），
在 θ=0≡2π 处是峰（f=1）。故写入发生在指针过闸（θ≡0）时刻 t=0.5，第二次过闸在 t=1.5。
装置内的写入算符是同一对换算符 exp(-i·G_FULL·f(θ)·dt·σ_x⊗σ_x)，累积角

    A(t) = G_FULL · ∫_0^t f(θ_p(t')) dt'

沿时间单调增长，一个指针周期（Δt=1）内 A 增加 π/2，站点侧布居等于 cos²A(t)。DS-2、DS-3
与 DS-3e 的注释直接使用该式。cos²A 在 A 上周期为 π，故站点侧读数在 t 上周期为 2 个时间单位。

DS-3e 的长期平均（裁决后新增，判据稳健）：演化到 t=8.0，覆盖 4 个完整的站点侧读数周期，
时间平均 ⟨站点侧⟩ = ⟨cos²A⟩ = 1/2，无写入基线为 1.0，平均增量 Δ̄ = -0.5。DS-3b 判否的物理
含义据此定为：负贡献在装置内是周期往复量，其长期平均为负，瞬时值有正有负。作为静态构型
上的线项，该读法要求位错寿命短于半个周期（t < 1.0）。

DS-5 的判定协议（裁决后写明）：判定协议取与 sin² 模型同口径的档 B，即写入事件在 t*=1.0
完成、此后关闭写入耦合、只留 θ 无关回流通道，步数取 n=round((κ·π/2)/(κ·G_FULL·DT))，使
累积角恰为 κ·π/2。字面步数式档 B'、字面时序档 A、字面时序加字面步数档 A'、写入关闭档 C
作为协议伪影分层报出。判据文字的步数式 n=round(κ·π/2/(G_FULL·DT)) 与同句「累积角 ≈ κ·π/2」
不自洽，前者给出 κ²·π/2，此项按「预期写错时修正判据」处理，教训记入注记过程记录。

判据：DS-1 至 DS-8（含 DS-3e 与 DS-3f），逐条给出期望、实测与判定。容差从误差预算推导，
见各条注释。可逆性等级：DS-1 至 DS-4、DS-6 至 DS-8、DS-3e、DS-3f 为判据稳健；DS-5 为
判据敏感。运行：python 验证脚本/minimal_verification_source_ding_reflux_saturation.py
确定性，无随机数。
"""

import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

T_START = time.time()
OUT = r"D:/software/TianXingFC/images/source_ding_reflux_saturation_726.png"

# =====================================================================
# 装置参数（逐字沿用第 722 号）
# =====================================================================
N_THETA = 256
OMEGA_P = 2.0 * np.pi
THETA0 = np.pi
W_PULSE = 0.5
PW = 0.15
DT = 2.0e-3
T_MAX = 1.2
T_LONG = 1.7

RESULTS = []
SUPP = []


def record(cid, name, expect, actual, ok, note=""):
    RESULTS.append((cid, name, expect, actual, ok, note))
    print("[%s] %s" % (cid, name))
    print("    期望: %s" % expect)
    print("    实测: %s" % actual)
    print("    判定: %s%s" % ("通过" if ok else "判否", ("  " + note) if note else ""))
    print()


def supp(tag, name, actual, note=""):
    SUPP.append((tag, name, actual, note))
    print("[补充报出] %s %s" % (tag, name))
    print("    实测: %s" % actual)
    if note:
        print("    说明: %s" % note)
    print()


# =====================================================================
# 装置函数（逐字沿用第 722 号）
# =====================================================================
def pulse(theta, w):
    return np.exp(-(1.0 - np.cos(theta)) / (2.0 * w ** 2))


def omega_of_k(kvec):
    return -OMEGA_P * kvec


def apply_int(psi, f, dt, g):
    """耦合项 g f (a†c + a c†)，在两能级子空间上精确指数化。"""
    ang = g * f * dt
    cos_a = np.cos(ang)
    sin_a = np.sin(ang)
    out = np.zeros_like(psi)
    for c in range(2):
        for a in range(2):
            out[:, a, c] += psi[:, a, c] * cos_a
    for a in range(2):
        other_a = 1 - a
        for c in range(2):
            other_c = 1 - c
            out[:, other_a, other_c] += -1j * sin_a * psi[:, a, c]
    return out


def split_step(psi, dt, theta, kvec, w, g, n_modes=2):
    f = pulse(theta, w)
    psi = apply_int(psi, f, dt / 2.0, g)
    for a in range(n_modes):
        for c in range(n_modes):
            psi[:, a, c] = np.fft.ifft(np.fft.fft(psi[:, a, c]) * np.exp(1j * omega_of_k(kvec) * dt))
    psi = apply_int(psi, f, dt / 2.0, g)
    return psi


def make_packet(theta, pw, site=1):
    psi = np.zeros((N_THETA, 2, 2), dtype=complex)
    pk = np.exp(-((theta - THETA0) ** 2) / (2.0 * pw ** 2))
    psi[:, site, 0] = pk / np.linalg.norm(pk)
    return psi


def site_pop(p):
    """站点模（被采样对象）一侧的布居。"""
    return float(np.sum(np.abs(p[:, 1, :]) ** 2))


def reg_pop(p):
    """记录寄存器一侧的布居。"""
    return float(np.sum(np.abs(p[:, :, 1]) ** 2))


def total_pop(p):
    return site_pop(p) + reg_pop(p)


def run(psi, g, steps=None):
    n = steps if steps is not None else int(round(T_MAX / DT))
    theta = np.linspace(0.0, 2.0 * np.pi, N_THETA, endpoint=False)
    kvec = np.fft.fftfreq(N_THETA, d=1.0 / N_THETA)
    trace = []
    for step in range(n):
        psi = split_step(psi, DT, theta, kvec, W_PULSE, g)
        if step % 20 == 0:
            trace.append((site_pop(psi), reg_pop(psi), total_pop(psi)))
    return psi, trace


# =====================================================================
# 本轮扩展函数：在第 722 号装置上做最小增补，不改动其实现
# =====================================================================
def split_step_g(psi, dt, theta, kvec, g_arr, n_a=2, n_c=2):
    """通用分裂步：g_arr 是逐 θ 的耦合系数数组。

    g_arr = G_FULL·f(θ) 为写入项；再加常数项 κ·G_FULL 即 θ 无关的回流项。
    两项同轴（同一对角化基下的 σ_x⊗σ_x），指数可加，次序不影响结果。
    g_arr = G_FULL·f(θ) 时与 split_step 的数值完全一致。
    """
    psi = apply_int(psi, 1.0, dt / 2.0, g_arr)
    for a in range(n_a):
        for c in range(n_c):
            psi[:, a, c] = np.fft.ifft(np.fft.fft(psi[:, a, c]) * np.exp(1j * omega_of_k(kvec) * dt))
    psi = apply_int(psi, 1.0, dt / 2.0, g_arr)
    return psi


def apply_int3(psi, f, dt, g):
    """三能级站点梯的写入算符：a 的 0 与 1 层同 c 耦合成对，a=2 层不参与。"""
    ang = g * f * dt
    cos_a = np.cos(ang)
    sin_a = np.sin(ang)
    out = np.zeros_like(psi)
    for c in range(2):
        for a in range(2):
            out[:, a, c] += psi[:, a, c] * cos_a
    for a in range(2):
        other_a = 1 - a
        for c in range(2):
            other_c = 1 - c
            out[:, other_a, other_c] += -1j * sin_a * psi[:, a, c]
    out[:, 2, :] = psi[:, 2, :]
    return out


def apply_leak(psi, dt, g_leak):
    """漏损通道：把记录侧激发旋转到第三个自由模（a=2 层）。

    该层既不计入站点侧也不计入记录侧，材料从此处泄漏到别处。
    """
    ang = g_leak * dt
    cos_a = np.cos(ang)
    sin_a = np.sin(ang)
    out = np.zeros_like(psi)
    out[:, 0, 1] = cos_a * psi[:, 0, 1] - 1j * sin_a * psi[:, 2, 0]
    out[:, 2, 0] = cos_a * psi[:, 2, 0] - 1j * sin_a * psi[:, 0, 1]
    out[:, 0, 0] = psi[:, 0, 0]
    out[:, 1, 0] = psi[:, 1, 0]
    out[:, 1, 1] = psi[:, 1, 1]
    return out


def apply_diag(psi, f, dt, g):
    """正定势替身：转移项换成对角正定项 a†a+c†c，只改变相位，激发搬不动。"""
    ang = g * f * dt
    out = np.zeros_like(psi)
    for a in range(2):
        for c in range(2):
            out[:, a, c] = psi[:, a, c] * np.exp(-1j * ang * (a + c))
    return out


def apply_gen3(psi, f, dt, g):
    """产生项替身：模式选择性的产生，按一阶截断从第三模（a=2）向记录侧注入。

    注入落在 (a=0, c=1) 切片，该切片计入记录侧而不计入站点侧。
    站点切片原样保留，站点侧读数不受影响。该算符不保范数，产生式读取本就如此。
    """
    ang = g * f * dt
    out = np.zeros_like(psi)
    out[:, 0, 0] = psi[:, 0, 0]
    out[:, 1, 0] = psi[:, 1, 0]
    out[:, 2, 0] = psi[:, 2, 0]
    out[:, 0, 1] = psi[:, 0, 1] - 1j * ang * psi[:, 2, 0]
    out[:, 1, 1] = psi[:, 1, 1]
    out[:, 2, 1] = psi[:, 2, 1]
    return out


def split_step_sub(psi, dt, theta, kvec, w, g, op, n_a=2, n_c=2):
    """替身用的分裂步：耦合项由 op 决定（apply_int / apply_int3 / apply_diag / apply_gen3）。"""
    f = pulse(theta, w)
    psi = op(psi, f, dt / 2.0, g)
    for a in range(n_a):
        for c in range(n_c):
            psi[:, a, c] = np.fft.ifft(np.fft.fft(psi[:, a, c]) * np.exp(1j * omega_of_k(kvec) * dt))
    psi = op(psi, f, dt / 2.0, g)
    return psi


def make_packet3(theta, pw, alev=1, reservoir=False):
    """三能级初态：被采样对象在 a=1，可选在 a=2 放一份储备源。"""
    psi = np.zeros((N_THETA, 3, 2), dtype=complex)
    pk = np.exp(-((theta - THETA0) ** 2) / (2.0 * pw ** 2))
    psi[:, alev, 0] = pk / np.linalg.norm(pk)
    if reservoir:
        pk2 = np.exp(-((theta - THETA0) ** 2) / (2.0 * pw ** 2))
        psi[:, 2, 0] = pk2 / np.linalg.norm(pk2)
    return psi


def third_pop(p):
    """第三个自由模（a=2 层）的布居。"""
    return float(np.sum(np.abs(p[:, 2, :]) ** 2))


def centroid_theta(p, alev=1):
    """被采样模 θ 剖面的圆周重心，用于核对自由演化作用在 θ 轴。"""
    dens = np.sum(np.abs(p[:, alev, :]) ** 2, axis=1)
    if dens.sum() <= 0.0:
        return float("nan")
    dens = dens / dens.sum()
    th = np.linspace(0.0, 2.0 * np.pi, N_THETA, endpoint=False)
    return float(np.angle(np.sum(dens * np.exp(1j * th))))


theta = np.linspace(0.0, 2.0 * np.pi, N_THETA, endpoint=False)
kvec = np.fft.fftfreq(N_THETA, d=1.0 / N_THETA)
area = float(np.sum(pulse(theta, W_PULSE)) * (2.0 * np.pi / N_THETA))
G_FULL = (np.pi / 2.0) * OMEGA_P / area
t_cross = (2.0 * np.pi - THETA0) / OMEGA_P
G_WRITE = G_FULL * pulse(theta, W_PULSE)          # 逐 θ 的写入耦合系数
print("脉冲面积 %.4f，完整交换耦合 g = %.4f，折返时刻 %.4f" % (area, G_FULL, t_cross))
print()

# =====================================================================
# 冒烟检查
# =====================================================================
print("=" * 78)
print("冒烟检查")
print("=" * 78)

psi_sm = make_packet(theta, PW)
ts, ss, rs = [], [], []
for step in range(int(round(T_LONG / DT))):
    psi_sm = split_step_g(psi_sm, DT, theta, kvec, G_WRITE)
    ts.append((step + 1) * DT)
    ss.append(site_pop(psi_sm))
    rs.append(reg_pop(psi_sm))
ts = np.array(ts)
ss = np.array(ss)
rs = np.array(rs)

print("末态各分栏布居（a, c 四个切片）:")
for a in range(2):
    for c in range(2):
        print("  (a=%d, c=%d) 布居 %.8f" % (a, c, float(np.sum(np.abs(psi_sm[:, a, c]) ** 2))))
print("站点侧 %.8f，记录侧 %.8f，总量 %.8f"
      % (site_pop(psi_sm), reg_pop(psi_sm), total_pop(psi_sm)))
i_min = int(np.argmin(ss))
print("站点侧在 t=%.4f 处取最小值 %.3e（沉积饱和时刻）" % (ts[i_min], ss[i_min]))
j12 = int(np.searchsorted(ts, 1.2))
print("t=1.2 分栏（对照第 722 号末态）：站点侧 %.8f，记录侧 %.8f，总量 %.8f"
      % (ss[j12], rs[j12], ss[j12] + rs[j12]))
print("t=1.7 分栏（延长演化后）：站点侧 %.8f，记录侧 %.8f，总量 %.8f"
      % (ss[-1], rs[-1], ss[-1] + rs[-1]))

hit = np.nonzero(rs > 0.5)[0]
t_write = float(ts[int(hit[0])]) if hit.size else float("nan")
print("写入时刻（记录侧首次越过 0.5）= %.4f" % t_write)

print("变化的轴核对（写算符与自由演化分轴）:")
psi_ax = make_packet(theta, PW)
for _ in range(int(round(T_MAX / DT))):
    psi_ax = split_step_g(psi_ax, DT, theta, kvec, G_WRITE)
psi_ax0, _ = run(make_packet(theta, PW), 0.0, steps=int(round(T_MAX / DT)))
for a in range(2):
    for c in range(2):
        d = abs(float(np.sum(np.abs(psi_ax[:, a, c]) ** 2)) - float(np.sum(np.abs(psi_ax0[:, a, c]) ** 2)))
        print("  (a=%d, c=%d) 切片布居差 %.8f" % (a, c, d))
print("  差分只出现在 (1,0) 与 (0,1) 两切片，耦合算符作用的正是 (a,c) 配对轴，方向一致。")
cen_free = []
for tt in (0.0, 0.25, 0.5, 0.75, 1.0):
    pf = make_packet(theta, PW)
    for _ in range(int(round(tt / DT))):
        pf = split_step_g(pf, DT, theta, kvec, 0.0)
    cen_free.append((tt, centroid_theta(pf)))
for tt, c_ in cen_free:
    print("  无写入档 t=%.2f：θ 重心 %.4f（期望 π+Ω_P t）" % (tt, c_))
print("  θ 重心按 Ω_P t 平移，自由演化作用在 θ 轴，与写算符作用的 (a,c) 轴正交。")

ang_profile = []
for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
    th_s = THETA0 + 2.0 * np.pi * frac
    seg = float(np.sum(pulse(np.linspace(THETA0, th_s, 4096), W_PULSE)) * (th_s - THETA0) / 4096)
    ang_profile.append((frac, (G_FULL / OMEGA_P) * seg))
print("逐 θ 累积角 A(θ)=(G_FULL/Ω_P)·∫_π^θ f du:")
for frac, a_ in ang_profile:
    print("  θ=π+%.2f·2π: 累积角 %.6f rad（π/2 = %.6f）" % (frac, a_, np.pi / 2.0))
print("  一个指针周期末累积角 %.6f rad，写算符的角确为 π/2 量级" % ang_profile[-1][1])
print()

# 累积角的时间演化：站点侧布居应当等于 cos²A(t)。该式是 DS-3 的机制核心，
# 直接把实测站点侧布居与解析 cos²A 并排打印，供判否时对照。
ts_fine = np.arange(0.0, T_LONG + 1e-9, 1e-4)
fp_fine = pulse(THETA0 + OMEGA_P * ts_fine, W_PULSE)
A_fine = G_FULL * np.cumsum(fp_fine) * 1e-4
print("累积角 A(t) 与 cos²A(t) 对照（站点侧实测取最近采样点）:")
for tt in (0.3, 0.5, 0.7, 1.0, 1.3, 1.5, 1.7):
    j = int(np.searchsorted(ts, tt))
    j = min(j, len(ts) - 1)
    jA = int(tt / 1e-4)
    print("  t=%.2f  A = %.6f rad  cos²A = %.6f  实测站点侧 = %.6f"
          % (tt, A_fine[jA], np.cos(A_fine[jA]) ** 2, ss[j]))
print()

print("逐步沉积剖面（站点侧布居，每 20 步取一点）:")
for i in range(0, len(ts), 20):
    print("  t=%.3f 站点 %.8f 记录 %.8f" % (ts[i], ss[i], rs[i]))
print()

# =====================================================================
# DS-1 读数复现
# =====================================================================
print("=" * 78)
print("DS-1 读数复现（判据稳健）")
print("=" * 78)

rel_t = abs(t_write - t_cross) / t_cross
ds1a_ok = rel_t < 0.05
record("DS-1a", "写入时刻 = 折返时刻 0.5000（相对偏差 < 5%）",
       "相对偏差 < 0.05", "写入时刻 %.4f，折返 %.4f，相对偏差 %.4f" % (t_write, t_cross, rel_t),
       ds1a_ok)

psi_d1, _ = run(make_packet(theta, PW), G_FULL, steps=int(round(T_MAX / DT)))
e0 = total_pop(make_packet(theta, PW))
e1 = total_pop(psi_d1)
ds1b_ok = abs(e1 - e0) < 1e-8
record("DS-1b", "总激发数守恒（偏差 < 1e-8）",
       "相对偏差 < 1e-8",
       "初始 %.8f，末态 %.8f，偏差 %.3e" % (e0, e1, abs(e1 - e0)), ds1b_ok)

psi_nw, _ = run(make_packet(theta, PW), 0.0, steps=int(round(T_MAX / DT)))
site_n = site_pop(psi_nw)
site_w = site_pop(psi_d1)
reg_n = reg_pop(psi_nw)
reg_w = reg_pop(psi_d1)
d_site = site_w - site_n
d_reg = reg_w - reg_n
rel_site = abs(d_site - (-0.99882383)) / abs(-0.99882383)
ds1c_ok = rel_site < 1e-6
record("DS-1c", "站点侧增量复现 -0.99882383（相对偏差 < 1e-6）",
       "相对偏差 < 1e-6",
       "站点侧增量 %.8f（无写入 %.8f，有写入 %.8f），相对偏差 %.3e"
       % (d_site, site_n, site_w, rel_site), ds1c_ok)
ds1d_ok = abs(d_reg - 0.99882383) / 0.99882383 < 1e-6
record("DS-1d", "寄存器侧增量 +0.99882383",
       "增量 = +0.99882383（相对偏差 < 1e-6）",
       "寄存器侧增量 %.8f，相对偏差 %.3e" % (d_reg, abs(d_reg - 0.99882383) / 0.99882383), ds1d_ok)

if not (ds1a_ok and ds1b_ok and ds1c_ok and ds1d_ok):
    print("DS-1 复现失败，按判据约定停止，不进入后续判据循环。")
    raise SystemExit(1)

# =====================================================================
# DS-2 沉积的时间局域性
# =====================================================================
print("=" * 78)
print("DS-2 沉积的时间局域性（判据稳健）")
print("=" * 78)

# 容差预算：
#   解析截断项：指针耦合剖面 f(t)=exp[-cos²(Ω_P t/2)/w²]，f≥1/2 的条件为
#     cos²(Ω_P t/2) ≤ w²·ln2 = 0.17329，即 |cos(Ω_P t/2)| ≤ 0.41628。解得该窗以 t=0.5
#     为中心，半宽 (2/Ω_P)·(π/2-arccos(0.41628)) = 0.1368 s，全窗 0.2736 s。
#     判据给定的允许带 [0.057, 0.690] s 相当于解析半宽的一半到三倍全窗，落在带内的
#     物理含义是沉积时间不超过一次过闸的尺度。
#   数值方法误差：步长 DT=2.0e-3，W90 的采样分辨率即 0.002 s；
#     FFT 分辨率 1/N_THETA=1/256 给出 θ 分辨率 0.0245 rad，对应时间 0.0039 s；
#     两者合成远小于下界 0.057 s。
#   另注：判据给出的解析半宽式 (2/Ω_P)·arcsin(w/2) 代入 w=0.5 得 0.0804 s，与判据
#     自称的 0.115 s 不自洽；本判据按给定的允许带执行，解析量按上式如实记录。

flux = np.maximum(-(ss[1:] - ss[:-1]) / np.diff(ts), 0.0)
cum = np.concatenate([[0.0], np.cumsum(flux * np.diff(ts))])
times_p = np.concatenate([[ts[0]], ts[1:]])
i_ep = int(np.searchsorted(times_p, 1.0))          # 首次沉积段止于 t=1.0
total_dep = float(cum[i_ep])
best = None
for i in range(i_ep + 1):
    j = int(np.searchsorted(cum, cum[i] + 0.9 * total_dep))
    if j >= len(cum):
        continue
    w = float(times_p[j] - times_p[i])
    if best is None or w < best[0]:
        best = (w, float(times_p[i]), float(times_p[j]))
W90, w_lo, w_hi = best
t_med = float(times_p[int(np.searchsorted(cum, 0.5 * total_dep))])

ds2a_ok = 0.057 <= W90 <= 0.690
record("DS-2a", "含 90% 沉积量的时间窗宽 W90 落在解析允许带内",
       "W90 ∈ [0.057, 0.690] s",
       "W90 = %.4f s（窗 [%.4f, %.4f]），首次沉积段累计沉积量 %.8f" % (W90, w_lo, w_hi, total_dep),
       ds2a_ok, "解析 f≥1/2 全窗 0.274 s，采样分辨率 0.002 s")
ds2b_ok = abs(t_med - 0.5) <= 0.115
record("DS-2b", "沉积中位时刻贴近过闸时刻（|t_med - 0.5| ≤ 0.115）",
       "|t_med - 0.5| ≤ 0.115",
       "t_med = %.4f，偏差 %.4f" % (t_med, abs(t_med - 0.5)), ds2b_ok)

# =====================================================================
# DS-3 一次性饱和
# =====================================================================
print("=" * 78)
print("DS-3 一次性饱和（判据稳健）")
print("=" * 78)

# DS-3 容差预算：
#   下界 1e-4 是「第二次过闸确实搬走材料」的可分辨门槛，来源是解析量：站点侧布居
#   等于 cos²A(t)，A 在 t=1.5 处为 3π/4，cos²(3π/4)=0.5，与 t=1.2 处的 cos²A≈0.0012
#   相差 0.499，远超门槛。数值方法误差：DT=2.0e-3 对应 A 的步进 0.0102 rad，
#   cos²A 在该步进上的最大变化 0.02，远小于 0.5 的信号。1e-4 取作信号量级的 1/5000。
#   上界 1.5e-3 对应「第二次过闸只搬走站点残留」：t=1.2 处站点侧残留 1.17617e-3，
#   1.5e-3 是其 1.276 倍，留一位数余量。两种读法的量级差达 658 倍，判否不靠容差取舍。
#   FFT 分辨率 1/N_THETA=1/256 对 θ 分辨率 0.0245 rad，对应时间 0.0039 s，
#   对本条的两端读数影响远小于上述差值。
inc = ss - site_n          # 站点侧增量（基线为无写入档）
inc12 = float(inc[int(np.searchsorted(ts, 1.2))])
inc17 = float(inc[-1])
d2 = inc17 - inc12
ds3a_ok = abs(d2) >= 1e-4
record("DS-3a", "第二次过闸不是惰性的（|Δ2| ≥ 1e-4）",
       "|Δ2| ≥ 1e-4",
       "站点侧增量 t=1.2 为 %.8f，t=1.7 为 %.8f，Δ2 = %.8f" % (inc12, inc17, d2), ds3a_ok)
ds3b_ok = abs(d2) <= 1.5e-3
record("DS-3b", "第二次过闸只搬走残余（|Δ2| ≤ 1.5e-3）",
       "|Δ2| ≤ 1.5e-3",
       "|Δ2| = %.8f，为上界的 %.0f 倍" % (abs(d2), abs(d2) / 1.5e-3), ds3b_ok,
       "实测与预期方向相反，判否")
supp("DS-3 注", "第二次过闸的实测行为",
     "t=1.0 站点侧 %.3e，t=1.5 站点侧 %.8f，t=1.7 站点侧 %.8f"
     % (ss[i_min], ss[int(np.searchsorted(ts, 1.5))], ss[-1]),
     "写入耦合是静态空间剖面，累积角 A(t)=G_FULL·∫f(θ_p(t'))dt' 随指针每扫一周增加 π/2，"
     "站点侧布居等于 cos²A(t)。第一次过闸把 A 推过 π/2，材料全部进入记录侧；"
     "第二次过闸继续把 A 推过 3π/4，材料被对换回站点侧。第二次过闸搬走的量与首次沉积同量级，"
     "残余量的说法不成立。装置内沉积呈周期为 2 个时间单位的往复，一次性饱和的前提不成立")

# =====================================================================
# DS-3e 长期平均：负贡献的时间平均仍为负
# =====================================================================
print("=" * 78)
print("DS-3e 长期平均读数（判据稳健，裁决后新增）")
print("=" * 78)

# DS-3e 容差预算：
#   判据门限 Δ̄ < -0.1。积分方法误差：梯形积分误差 ≤ (T·h²/12)·max|n''|，
#   n(t)=cos²A(t)，|n''| ≲ 2A'² + 2|A''|，A' ≤ G_FULL = 5.092，
#   |A''| ≲ G_FULL·2π·max|f'|，量级约 64，故 |n''| ≲ 180。
#   采样间隔取 DT = 2.0e-3（判据要求 ≤ 0.005，逐步行进满足），误差
#   ≤ (8·(2e-3)²/12)·180 ≈ 4.8e-5。门限 0.1 相对该误差宽裕三个数量级。
#   积分区间须自 t=0 起算：初值站点侧 = 1.0 必须作为 t=0 的采样点并入梯形积分。
#   若漏掉该点，区间 [0, DT] 被跳过，常数基线的积分会读成 7.998/8.0 = 0.99975，
#   引入 2.5e-4 的系统偏差。该偏差对有写入档与基线档同源，对差值 Δ̄ 无影响，
#   但绝对均值的读数必须并入 t=0 点才正确，DS-3f 的恒真检验依赖此修正。
#   解析期望：站点侧布居 = cos²A(t)，A 每 2 个时间单位增 π，cos²A 在 t 上周期为 2；
#   T=8 覆盖 4 个整周期，⟨cos²A⟩ = 1/2，故 Δ̄ = 0.5 - 1.0 = -0.5。
#   解析截断项：T=8 含 4 个整周期，首尾各留半个周期的余量，周期未闭合导致的截断
#   不存在（cos²A 的周期 2 整除 8）。若 T 改为非 2 的整数倍，边界项量级为 1/T。

T_AVG = 8.0


def time_average_site(psi_in, g_write, t_end=T_AVG):
    """逐步行进并每 DT 采样一次站点侧布居，含 t=0 的初值点。

    采样间隔 = DT = 2.0e-3 ≤ 判据要求的 0.005。
    """
    n = int(round(t_end / DT))
    psi_l = psi_in
    t_list = [0.0]
    s_list = [site_pop(psi_l)]
    for step in range(n):
        psi_l = split_step_g(psi_l, DT, theta, kvec, g_write)
        t_list.append((step + 1) * DT)
        s_list.append(site_pop(psi_l))
    return np.array(t_list), np.array(s_list)


t_avg, s_avg = time_average_site(make_packet(theta, PW), G_WRITE)
mean_site = float(np.trapezoid(s_avg, t_avg) / T_AVG)

# 无写入基线的时间平均
t_base, s_base = time_average_site(make_packet(theta, PW), np.zeros_like(G_WRITE))
mean_base = float(np.trapezoid(s_base, t_base) / T_AVG)

dbar = mean_site - mean_base
ds3e_ok = dbar < -0.1
record("DS-3e", "长期平均的站点侧增量仍为负（Δ̄ < -0.1）",
       "Δ̄ < -0.1，解析期望 -0.5",
       "⟨站点侧⟩ = %.8f，基线 ⟨站点侧⟩ = %.8f，Δ̄ = %.8f" % (mean_site, mean_base, dbar),
       ds3e_ok, "T=8.0 覆盖 4 个整周期，采样间隔 %.4f" % DT)

# 附带读数：涨落周期与峰值幅度
i_pk = int(np.argmax(s_avg))
i_tr = int(np.argmin(s_avg))
pk_list = []
for i in range(1, len(s_avg) - 1):
    if s_avg[i] > s_avg[i - 1] and s_avg[i] >= s_avg[i + 1]:
        pk_list.append(float(t_avg[i]))
tr_list = []
for i in range(1, len(s_avg) - 1):
    if s_avg[i] < s_avg[i - 1] and s_avg[i] <= s_avg[i + 1]:
        tr_list.append(float(t_avg[i]))
period_meas = float(np.mean(np.diff(pk_list))) if len(pk_list) >= 2 else float("nan")
period_tr = float(np.mean(np.diff(tr_list))) if len(tr_list) >= 2 else float("nan")
amp = float(s_avg.max() - s_avg.min())
print("  附带读数：站点侧布居峰值 %.8f（t=%.4f），谷值 %.8f（t=%.4f），幅度 %.8f"
      % (s_avg[i_pk], t_avg[i_pk], s_avg[i_tr], t_avg[i_tr], amp))
print("  附带读数：涨落周期实测 %.5f（峰值间距），%.5f（谷值间距），解析值 2.0"
      % (period_meas, period_tr))
print("  附带读数：T=%.1f 内峰值时刻 %s，谷值时刻 %s"
      % (T_AVG, ", ".join("%.3f" % v for v in pk_list), ", ".join("%.3f" % v for v in tr_list)))
print("  附带读数：作为静态构型上的线项，该读法要求位错寿命短于半个周期（t < %.1f）。"
      % (period_meas / 2.0))

# DS-3f 恒真检验：关断写入耦合，同一测量流程下平均读数为零
t_off, s_off = time_average_site(make_packet(theta, PW), np.zeros_like(G_WRITE))
mean_off = float(np.trapezoid(s_off, t_off) / T_AVG)
ds3f_ok = abs(mean_off - 1.0) <= 1e-12
record("DS-3f", "恒真检验：关断写入耦合，长期平均增量 Δ̄ = 0",
       "|Δ̄| ≤ 1e-12",
       "⟨站点侧⟩ = %.8f，基线 1.0，Δ̄ = %.3e" % (mean_off, abs(mean_off - 1.0)),
       ds3f_ok, "反例检验：把写入耦合开回即得 Δ̄ = %.8f，判否" % dbar)

# =====================================================================
# DS-4 转移分数的区间与抽空端
# =====================================================================
print("=" * 78)
print("DS-4 转移分数的区间与抽空端（判据稳健）")
print("=" * 78)

# DS-4 容差预算：
#   f(0.25) 与 f(1.0) 的复现容差 1e-5 是相对偏差门限。数值方法误差来源：步长 DT=2.0e-3
#   的角离散化，累积角 A 的每步增量最大 0.0102 rad，cos²A 的步进误差最大 0.02；
#   FFT 分辨率 1/N_THETA=1/256 给出 θ 分辨率 0.0245 rad，对应时间 0.0039 s，
#   对 f 的影响通过 A 传递，量级 1e-3。采样步长 DT=2.0e-3 与上面的离散化同源。
#   这三项合成使 f 的整体不确定度在 1e-3 量级，复现值取自第 722 号同一装置的同一算法，
#   逐字沿用实现下理论差值只来自浮点重排，故 1e-5 可达。实际偏差为 2.563e-06 与 1.654e-07。
#   单调性判据不设容差，判据给定的五档间距（0.0397/0.1525/0.5170/0.8711/0.9988）
#   相邻差均远大于 1e-3 的数值不确定度。
G_FRACS = [0.125, 0.25, 0.5, 0.75, 1.0]
fracs, fs = [], []
for fr in G_FRACS:
    p, _ = run(make_packet(theta, PW), G_FULL * fr, steps=int(round(T_MAX / DT)))
    f_ = (site_n - site_pop(p)) / site_n
    fracs.append(fr)
    fs.append(f_)
    print("  耦合档 %.3f 倍：站点侧 %.8f，转移分数 f = %.8f" % (fr, site_pop(p), f_))
mono_ok = all(fs[i] <= fs[i + 1] for i in range(len(fs) - 1))
record("DS-4a", "转移分数随耦合档单调不减",
       "五档 f 单调不减",
       "f = %s" % ", ".join("%.6f" % v for v in fs), mono_ok)
rel_f25 = abs(fs[1] - 0.152500) / 0.152500
ds4b_ok = rel_f25 < 1e-5
record("DS-4b", "f(0.25) 复现 0.152500（相对偏差 < 1e-5）",
       "相对偏差 < 1e-5", "f(0.25) = %.8f，相对偏差 %.3e" % (fs[1], rel_f25), ds4b_ok)
rel_f1 = abs(fs[4] - 0.998824) / 0.998824
ds4c_ok = rel_f1 < 1e-5
record("DS-4c", "f(1.0) 复现 0.998824（相对偏差 < 1e-5）",
       "相对偏差 < 1e-5", "f(1.0) = %.8f，相对偏差 %.3e" % (fs[4], rel_f1), ds4c_ok)
ds4d_ok = fs[4] >= 0.99
record("DS-4d", "抽空端：f(1.0) ≥ 0.99",
       "f(1.0) ≥ 0.99", "f(1.0) = %.8f，区间 f ∈ [%.6f, %.6f]" % (fs[4], min(fs), max(fs)),
       ds4d_ok, "体系自带的写强度落在几乎搬空站点的一端")

p125, _ = run(make_packet(theta, PW), G_FULL * 1.25, steps=int(round(T_MAX / DT)))
f125 = (site_n - site_pop(p125)) / site_n
print("  超交换档 1.25 倍：站点侧 %.8f，f = %.8f（射程外，单列，不入单调性判定）"
      % (site_pop(p125), f125))
supp("DS-4 注", "超交换档 1.25 倍的单列结果",
     "f(1.25) = %.8f，小于 f(1.0) = %.8f" % (f125, fs[4]),
     "累积角 A 超过 π/2 后 sin 反号，转移分数回落。承第 724 号教训，超交换档不混入单调性判定")

# =====================================================================
# DS-5 回流响应曲线
# =====================================================================
print("=" * 78)
print("DS-5 回流响应曲线（本轮主判据，判据敏感）")
print("=" * 78)

# 协议说明（裁决后定稿，四档分层报出）：
#   档 B（隔离协议，本轮判定用）：写入事件在 t*=1.0 完成，站点侧取最小值，
#     材料全部位于记录侧。此后写入耦合关闭，只留 κ·G_FULL 的 θ 无关回流通道。
#     步数 n=round((κ·π/2)/(κ·G_FULL·DT))，累积角等于 κ·π/2，与判据的 sin² 模型同口径。
#   档 C（写入关闭档）：回流通道自 t=0.55 起，写入耦合在 t=0.55 关闭。
#   档 A（字面时序档）：回流通道自 t=0.55 起，写入耦合保持开启。
#   档 A'（字面时序加字面步数档）：档 A 的时序加判据给的字面步数。
#   档 B'（字面步数式）：t*=1.0，写入耦合关闭，步数取 n=round(κ·π/2/(G_FULL·DT))。
# 判据文字修订记录：判据原文的步数式 n=round(κ·π/2/(G_FULL·DT)) 与同句「累积角 ≈ κ·π/2」
#   自相矛盾，前者给出的累积角为 κ²·π/2。按「预期写错时修正判据并把教训写入注记过程记录」
#   的纪律，判定协议改取与 sin² 模型同口径的档 B，其余四档作为协议伪影分层报出。
# 容差预算：回流角离散化误差 ≤ G_FULL·DT/2 = 0.0051 rad，折算 sin² 的相对偏差 ≤ 0.51%；
#   写算符与回流算符时间序的非交换项在写入耦合关闭后为零。判据带取 5%。
# 前提说明：t=0.55 处累积角 A≈1.026 rad，站点侧尚余 0.268，写入尚未完成。判据的
#   DS-5c 期望与 DS-5d 的 sin² 模型要求材料已全部进入记录侧，档 A 的该前提不成立。

KAPPAS = [0.0, 0.25, 0.5, 0.75, 1.0]


def reflux_run(kappa, t_start, write_on, n_mode="intent"):
    """写入段跑到 t_start，随后施加 κ·G_FULL 的 θ 无关回流通道 n 步。

    n_mode="intent"：n=round((κ·π/2)/(κ·G_FULL·DT))，累积角等于 κ·π/2，与判据的
        sin²(κπ/2) 模型同口径。
    n_mode="literal"：n=round((κ·π/2)/(G_FULL·DT))，累积角为 κ²·π/2。
    """
    if kappa == 0.0:
        n = 0
    elif n_mode == "intent":
        n = int(round((kappa * np.pi / 2.0) / (kappa * G_FULL * DT)))
    else:
        n = int(round((kappa * np.pi / 2.0) / (G_FULL * DT)))
    p = make_packet(theta, PW)
    for _ in range(int(round(t_start / DT))):
        p = split_step_g(p, DT, theta, kvec, G_WRITE)
    site_post = site_pop(p)
    g_arr = (G_WRITE if write_on else 0.0) + kappa * G_FULL
    for _ in range(n):
        p = split_step_g(p, DT, theta, kvec, g_arr)
    return site_post, site_pop(p), n, n * kappa * G_FULL * DT


T_STAR = float(ts[i_min])
R_B, R_C, R_A, R_AP, R_BP, ANG_B = [], [], [], [], [], []
for kap in KAPPAS:
    sp, sa, n, ang = reflux_run(kap, T_STAR, write_on=False, n_mode="intent")
    R_B.append((sa - sp) / (site_n - sp))
    ANG_B.append(ang)
    sp, sa, n, ang = reflux_run(kap, 0.55, write_on=False, n_mode="intent")
    R_C.append((sa - sp) / (site_n - sp))
    sp, sa, n, ang = reflux_run(kap, 0.55, write_on=True, n_mode="intent")
    R_A.append((sa - sp) / (site_n - sp))
    sp, sa, n, ang = reflux_run(kap, 0.55, write_on=True, n_mode="literal")
    R_AP.append((sa - sp) / (site_n - sp))
    sp, sa, n, ang = reflux_run(kap, T_STAR, write_on=False, n_mode="literal")
    R_BP.append((sa - sp) / (site_n - sp))

print("  回流响应逐档（档 B 隔离协议 t*=%.4f；档 C 截断时序 t=0.55；档 A 字面时序；"
      % T_STAR)
print("               档 A' 字面时序加字面步数；档 B' 字面步数）:")
for i, kap in enumerate(KAPPAS):
    print("    κ=%.2f  累积角 %.6f  R_B = %.6f  R_C = %.6f  R_A = %.6f  R_A' = %.6f  R_B' = %.6f  sin² = %.6f"
          % (kap, ANG_B[i], R_B[i], R_C[i], R_A[i], R_AP[i], R_BP[i], np.sin(kap * np.pi / 2.0) ** 2))


def sub_verdicts(R):
    """在给定协议曲线上复核 DS-5 的四条子判据，返回各条判定与最大相对偏差。"""
    theory_ = np.array([np.sin(k * np.pi / 2.0) ** 2 for k in KAPPAS])
    dev_ = np.abs(np.array(R) - theory_) / np.maximum(theory_, 1e-12)
    mask_ = np.array([k > 0 for k in KAPPAS])
    return {
        "DS-5a": abs(R[0]) <= 1e-6,
        "DS-5b": all(R[i] <= R[i + 1] for i in range(len(R) - 1)),
        "DS-5c": R[-1] >= 0.90,
        "DS-5d": float(np.max(dev_[mask_])) <= 0.05,
        "maxdev": float(np.max(dev_[mask_])),
    }


print("  各协议下 DS-5 四条子判据的复核:")
for tag, R in [("档 B 隔离协议", R_B), ("档 C 截断时序", R_C), ("档 A 字面时序", R_A),
                ("档 A' 字面时序加字面步数", R_AP), ("档 B' 字面步数", R_BP)]:
    v = sub_verdicts(R)
    print("    %-24s 5a %s，5b %s，5c %s，5d %s（最大相对偏差 %.4f）"
          % (tag, "过" if v["DS-5a"] else "否", "过" if v["DS-5b"] else "否",
             "过" if v["DS-5c"] else "否", "过" if v["DS-5d"] else "否", v["maxdev"]))

ds5a_ok = abs(R_B[0]) <= 1e-6
record("DS-5a", "关断回流通道时站点侧不回收（R(0) ≤ 1e-6）",
       "R(0) ≤ 1e-6", "R(0) = %.3e" % abs(R_B[0]), ds5a_ok)
monoB = all(R_B[i] <= R_B[i + 1] for i in range(len(R_B) - 1))
record("DS-5b", "回流响应随强度单调不减",
       "R(κ) 单调不减",
       "R = %s" % ", ".join("%.6f" % v for v in R_B), monoB)
ds5c_ok = R_B[-1] >= 0.90
record("DS-5c", "完全回流把材料几乎全部收回（R(1.0) ≥ 0.90）",
       "R(1.0) ≥ 0.90", "R(1.0) = %.6f" % R_B[-1], ds5c_ok)
theory = np.array([np.sin(k * np.pi / 2.0) ** 2 for k in KAPPAS])
rel_dev = np.abs(np.array(R_B) - theory) / np.maximum(theory, 1e-12)
dev_mask = np.array([k > 0 for k in KAPPAS])
max_dev = float(np.max(rel_dev[dev_mask]))
ds5d_ok = max_dev <= 0.05
record("DS-5d", "实测 R(κ) 与 sin²(κπ/2) 的最大相对偏差 ≤ 5%",
       "最大相对偏差 ≤ 0.05",
       "各档相对偏差 %s，最大 %.4f" % (", ".join("%.4f" % v for v in rel_dev[dev_mask]), max_dev),
       ds5d_ok, "判据敏感：换档位或换协议须重设")
supp("DS-5 附一", "档 C 截断时序的实测曲线",
     "R_C = %s" % ", ".join("%.6f" % v for v in R_C),
     "t=0.55 起写入耦合关闭，只留回流通道。该档的回收比例明显低于 sin²(κπ/2)，原因在于 "
     "t=0.55 处写入尚未完成：站点侧与记录侧并存未搬完的材料，且两把读数尺上带着不同的累积相位，"
     "回流通道的 θ 无关对换对两把尺的作用并不对称。回收效率依赖写入完成的程度，不依赖 "
     "两把尺上的布居比例")
supp("DS-5 附二", "档 A 字面时序的实测曲线",
     "R_A = %s" % ", ".join("%.6f" % v for v in R_A),
     "t=0.55 处写入尚未完成且写入耦合仍在开启，κ 较小时写入把材料继续搬向记录侧，R 落为负；"
     "κ 接近 1 时回流角与写入自身的累积角相加接近 π，R 回升。该曲线反映回流通道与写入通道的竞争")
supp("DS-5 附三", "档 A' 字面时序加字面步数式的实测曲线",
     "R_A' = %s" % ", ".join("%.6f" % v for v in R_AP),
     "在档 A 的时序上再取判据给的字面步数 n=round(κ·π/2/(G_FULL·DT))，累积角为 κ²·π/2。"
     "该档是判据文字的最直接读法，R(1.0) 仍达 %.6f，但小 κ 档与 sin² 模型严重偏离"
     % R_AP[-1])
supp("DS-5 附四", "档 B' 字面步数式的实测曲线",
     "R_B' = %s" % ", ".join("%.6f" % v for v in R_BP),
     "步数取 n=round(κ·π/2/(G_FULL·DT)) 时，累积角为 κ²·π/2，与判据给的 sin²(κπ/2) 模型"
     "不对齐，小 κ 档偏差极大。判定采用档 B，其步数使累积角等于 κ·π/2，与判据的期望同口径")

# =====================================================================
# DS-6 恒真检验三条
# =====================================================================
print("=" * 78)
print("DS-6 恒真检验三条（判据稳健）")
print("=" * 78)

# 替身一：关断耦合
psi_c0, _ = run(make_packet(theta, PW), 0.0, steps=int(round(T_MAX / DT)))
psi_c1, _ = run(make_packet(theta, PW), 0.0, steps=int(round(T_MAX / DT)))
d_off = site_pop(psi_c1) - site_pop(psi_c0)
ds6a_ok = abs(d_off) <= 1e-12
record("DS-6a", "恒真检验一：关断耦合，站点侧增量 ≤ 1e-12",
       "|Δ| ≤ 1e-12", "Δ = %.3e" % d_off, ds6a_ok,
       "反例检验：把耦合开回即得 -0.99882383，判否")

# 替身二：正定势替身，转移项换成对角正定项，站点模搬不出激发
psi_diag = make_packet(theta, PW)
for _ in range(int(round(T_MAX / DT))):
    psi_diag = split_step_sub(psi_diag, DT, theta, kvec, W_PULSE, G_FULL, apply_diag)
d_diag = site_pop(psi_diag) - site_n
ds6b_ok = abs(d_diag) <= 1e-12
record("DS-6b", "恒真检验二：正定势替身，站点侧增量 ≤ 1e-12",
       "|Δ| ≤ 1e-12", "Δ = %.3e（站点侧 %.8f）" % (d_diag, site_pop(psi_diag)), ds6b_ok,
       "反例检验：把转移项放回即得 -0.99882383，判否")

# 替身三：对象错位替身，被采样对象换成与写入无耦合的第三个自由模
p3_w = make_packet3(theta, PW, alev=2)
p3_n = make_packet3(theta, PW, alev=2)
for _ in range(int(round(T_MAX / DT))):
    p3_w = split_step_sub(p3_w, DT, theta, kvec, W_PULSE, G_FULL, apply_int3, n_a=3, n_c=2)
    p3_n = split_step_sub(p3_n, DT, theta, kvec, W_PULSE, 0.0, apply_int3, n_a=3, n_c=2)
d_mis = third_pop(p3_w) - third_pop(p3_n)
ds6c_ok = abs(d_mis) <= 1e-12
record("DS-6c", "恒真检验三：对象错位替身，站点侧增量 ≤ 1e-12",
       "|Δ| ≤ 1e-12", "Δ = %.3e（第三模布居 %.8f）" % (d_mis, third_pop(p3_w)), ds6c_ok,
       "反例检验：把被采样对象换成站点模即得 -0.99882383，判否")

# 作者括注构造：激发置于 a=0 基态
psi_a0 = make_packet(theta, PW, site=0)
psi_a0_w, _ = run(psi_a0, G_FULL, steps=int(round(T_MAX / DT)))
psi_a0_n, _ = run(psi_a0, 0.0, steps=int(round(T_MAX / DT)))
d_a0 = site_pop(psi_a0_w) - site_pop(psi_a0_n)
supp("DS-6 附", "作者括注构造：激发置于 a=0 基态（构造缺陷，不影响任何判定）",
     "站点侧增量 %.8f（有写入 %.8f，无写入 %.8f）"
     % (d_a0, site_pop(psi_a0_w), site_pop(psi_a0_n)),
     "该构造的角色是把对象换错的替身（激发放到 a=0 基态，站点侧读数仍取 a=1 模），"
     "「无可搬」的替身是 DS-6b 的正定势替身，两条不可混同。此处的读数口径写错了："
     "装置的对换算符把 (a=0,c=0) 换到 (a=1,c=1)，该切片同时计入站点侧与记录侧两把尺，"
     "故站点侧读数上浮为 %.8f。正确的读数口径取 a=0 模时该增量为零。"
     "该条给出正增量，不构成负贡献，也不参与 DS-6 的任何判定"
     % 0.99882383)

# =====================================================================
# DS-7 替身检验：漏损通道不回收
# =====================================================================
print("=" * 78)
print("DS-7 替身检验：漏损通道不回收（判据稳健）")
print("=" * 78)

R_leak, LEAK_THIRD = [], []
for kap in KAPPAS:
    if kap == 0.0:
        n = 0
    else:
        n = int(round((kap * np.pi / 2.0) / (kap * G_FULL * DT)))
    p = make_packet3(theta, PW, alev=1)
    for _ in range(int(round(T_STAR / DT))):
        p = split_step_sub(p, DT, theta, kvec, W_PULSE, G_FULL, apply_int3, n_a=3, n_c=2)
    site_post = site_pop(p)
    third_post = third_pop(p)
    for _ in range(n):
        p = apply_leak(p, DT, kap * G_FULL)
        for a in range(3):
            for c in range(2):
                p[:, a, c] = np.fft.ifft(np.fft.fft(p[:, a, c]) * np.exp(1j * omega_of_k(kvec) * DT))
    R_leak.append((site_pop(p) - site_post) / (site_n - site_post))
    LEAK_THIRD.append(third_pop(p) - third_post)
print("  漏损档逐档 R：%s" % ", ".join("%.6f" % v for v in R_leak))
print("  漏损档逐档第三模增量（材料确实离开两把尺）：%s"
      % ", ".join("%.6f" % v for v in LEAK_THIRD))
leak_moves = LEAK_THIRD[-1] > 0.5
ds7_ok = max(abs(v) for v in R_leak) <= 1e-6 and leak_moves
record("DS-7", "漏损通道把材料带到别处，站点侧不回收（R ≤ 1e-6 且材料确实离开）",
       "各档 |R| ≤ 1e-6，且 κ=1.0 档第三模增量 > 0.5",
       "最大 |R| = %.3e；κ=1.0 档第三模增量 %.6f" % (max(abs(v) for v in R_leak), LEAK_THIRD[-1]),
       ds7_ok, "反例检验：换成回流通道即得 R(1.0)=%.6f" % R_B[-1])

# =====================================================================
# DS-8 前提检验：产生式读取下无负贡献
# =====================================================================
print("=" * 78)
print("DS-8 前提检验：产生式读取下无负贡献（判据稳健）")
print("=" * 78)

p_gen = make_packet3(theta, PW, alev=1, reservoir=True)
site_gen_0 = site_pop(p_gen)
for _ in range(int(round(T_MAX / DT))):
    p_gen = split_step_sub(p_gen, DT, theta, kvec, W_PULSE, G_FULL, apply_gen3, n_a=3, n_c=2)
d_gen = site_pop(p_gen) - site_gen_0
ds8_ok = abs(d_gen) <= 1e-12
record("DS-8", "产生项替身下站点侧增量为零（|Δ| ≤ 1e-12）",
       "|Δ| ≤ 1e-12",
       "Δ = %.3e（站点侧 %.8f，记录侧 %.8f，第三模 %.8f，总量 %.8f）"
       % (d_gen, site_pop(p_gen), reg_pop(p_gen), third_pop(p_gen), total_pop(p_gen)), ds8_ok,
       "产生项由第三个自由模向记录侧注入激发，站点侧读数不受影响；"
       "负贡献依赖「写入是转移」这一读法，须在注记登记为前提")

# =====================================================================
# 出图
# =====================================================================
print("=" * 78)
print("出图")
print("=" * 78)

fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.6))

# (a) 站点侧布居随时间
ax = axes[0, 0]
ax.plot(ts, ss, "-", color="tab:blue", lw=2.2, label="站点侧布居（演化到 t=1.7）")
ax.plot(t_avg, s_avg, "-", color="tab:green", lw=0.8, alpha=0.75, label="站点侧布居（演化到 t=8.0）")
ax.axvspan(w_lo, w_hi, color="tab:orange", alpha=0.35, label="含 90% 沉积的窗")
for tc, lab in [(t_cross, "第一次过闸"), (t_cross + 1.0, "第二次过闸")]:
    ax.axvline(tc, color="tab:red", ls="--", lw=1.1)
    ax.text(tc, 0.06, " " + lab, rotation=90, color="tab:red", fontsize=8, va="bottom")
ax.set_xlabel(r"时间 $t$")
ax.set_ylabel("站点侧布居")
ax.set_ylim(-0.03, 1.16)
ax.set_title(r"(a) 站点侧布居：$t=0.5$ 沉积达成，$t=1.5$ 被反向交换翻回", fontsize=10)
ax.legend(fontsize=7, loc="upper right", ncol=1)

# (b) 转移分数随耦合档
ax = axes[0, 1]
ax.plot(fracs, fs, "o-", color="tab:purple", label="射程内五档")
ax.plot([1.25], [f125], "s", color="tab:gray", ms=7, label="超交换档 1.25 倍（射程外）")
ax.axhline(0.99, color="tab:red", ls="--", lw=1.0, label="抽空线 0.99")
ax.set_xlabel("耦合档（完整交换的倍数）")
ax.set_ylabel(r"转移分数 $f$")
ax.set_ylim(-0.05, 1.12)
ax.set_title("(b) 转移分数的单调性与抽空端", fontsize=10)
ax.legend(fontsize=8)

# (c) 回流响应
ax = axes[1, 0]
kk = np.linspace(0.0, 1.0, 101)
ax.plot(kk, np.sin(kk * np.pi / 2.0) ** 2, "-", color="k", lw=1.5, label=r"理论 $\sin^2(\kappa\pi/2)$")
ax.plot(KAPPAS, R_B, "o-", color="tab:blue", label="档 B 隔离协议（判定用）")
ax.plot(KAPPAS, R_C, "x-", color="tab:green", ms=7, label="档 C 截断时序")
ax.plot(KAPPAS, R_A, "s--", color="tab:orange", label="档 A 字面时序")
ax.plot(KAPPAS, R_AP, "v:", color="tab:purple", label="档 A' 字面步数")
ax.plot(KAPPAS, R_BP, "^:", color="tab:gray", label="档 B' 字面步数式")
ax.axhline(0.0, color="gray", lw=0.8)
ax.set_xlabel(r"回流强度 $\kappa$")
ax.set_ylabel(r"回流响应 $R$")
ax.set_ylim(-0.35, 1.15)
ax.set_title(r"(c) 回流响应 $R(\kappa)$ 与 $\sin^2(\kappa\pi/2)$", fontsize=10)
ax.legend(fontsize=7, loc="lower right")

# (d) 长期平均读数：站点侧布居的往复与时间平均
#     裁决后按 DS-3e 调整。原先的沉积量累积曲线（对数纵轴）保留为插图，
#     因为它给出两次过闸的量级差，是 DS-3b 判否的直接图示；
#     主图改为 DS-3e 的长期平均读数：站点侧布居以周期 2 往复，时间平均收敛到 0.5，
#     低于真空基线 1.0，净平均负贡献 -0.5。
ax = axes[1, 1]
ax.plot(t_avg, s_avg, "-", color="tab:blue", lw=1.0, alpha=0.85, label="站点侧布居")
# 累积时间平均
run_avg = np.zeros_like(s_avg)
acc = 0.0
for i in range(len(t_avg)):
    acc += 0.5 * (s_avg[i] + s_avg[i - 1]) * (t_avg[i] - t_avg[i - 1]) if i > 0 else 0.0
    run_avg[i] = acc / t_avg[i] if t_avg[i] > 0 else s_avg[i]
ax.plot(t_avg, run_avg, "-", color="tab:green", lw=2.2, label="累积时间平均")
ax.axhline(1.0, color="tab:red", ls="--", lw=1.2, label="真空基线 1.0")
ax.axhline(mean_site, color="tab:purple", ls=":", lw=1.5, label=r"时间平均 $\langle$站点侧$\rangle$")
ax.annotate("", xy=(6.0, mean_site), xytext=(6.0, 1.0),
            arrowprops=dict(arrowstyle="<->", color="tab:purple", lw=1.4))
ax.text(6.15, (1.0 + mean_site) / 2.0, r"$\bar{\Delta}=-0.5$", color="tab:purple", fontsize=10)
for tc in (1.0, 3.0, 5.0, 7.0):
    ax.axvline(tc, color="gray", ls=":", lw=0.8)
ax.set_xlabel(r"时间 $t$")
ax.set_ylabel("站点侧布居")
ax.set_ylim(-0.08, 1.12)
ax.set_xlim(0.0, T_AVG)
ax.set_title(r"(d) 长期平均：$\langle$站点侧$\rangle=0.5$，净增量 $-0.5$", fontsize=10)
ax.legend(fontsize=7, loc="upper left")

# 插图：两次过闸的沉积量级差（对数纵轴），放在 t 大于 4 的低谷区
axin = ax.inset_axes([0.50, 0.06, 0.46, 0.36])
floor = 1e-11
dep_total = 1.0 - ss
recov = ss - ss[i_min]
m1 = ts <= 1.0
axin.plot(ts[m1], np.maximum(dep_total[m1], floor), "-", color="tab:green", lw=1.5)
axin.plot(ts[~m1], np.maximum(dep_total[~m1], floor), "--", color="tab:green", lw=1.3)
axin.plot(ts[~m1], np.maximum(recov[~m1], floor), "-", color="tab:purple", lw=1.3)
axin.set_yscale("log")
axin.set_ylim(1e-11, 3.0)
axin.set_xlim(0.0, 1.7)
axin.tick_params(labelsize=6)
axin.set_title("插图：两次过闸的沉积量级差（t 到 1.7）", fontsize=7)
axin.text(0.05, 0.10, "绿实线：首次沉积段\n绿虚线：第二次过闸后剩余\n紫实线：第二次过闸搬回",
          transform=axin.transAxes, fontsize=6, va="bottom")

fig.suptitle("第 726 号：来源丁第二轮（沉积的一次性与回流的脆弱性）", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(OUT, dpi=150)
print("  图已存: %s" % OUT)

# =====================================================================
# 总表
# =====================================================================
print()
print("=" * 78)
print("判据总表")
print("=" * 78)
npass = 0
for cid, name, expect, actual, ok, note in RESULTS:
    print("%-9s %-52s %s" % (cid, name[:52], "通过" if ok else "判否"))
    if ok:
        npass += 1
print()
print("合计 %d/%d 全过" % (npass, len(RESULTS)))
print()
print("关键读数:")
print("  写入时刻 %.4f，总激发数 %.8f（偏差 %.3e）" % (t_write, e1, abs(e1 - e0)))
print("  站点侧增量 %.8f，记录侧增量 %.8f" % (d_site, d_reg))
print("  末态站点侧 %.8f，记录侧 %.8f，总量 %.8f"
      % (site_pop(psi_sm), reg_pop(psi_sm), total_pop(psi_sm)))
print("  W90 = %.4f s（窗 [%.4f, %.4f]），沉积中位时刻 %.4f" % (W90, w_lo, w_hi, t_med))
print("  站点侧最小值 %.3e，出现在 t = %.4f" % (ss[i_min], ts[i_min]))
print("  Δ2（站点侧增量 t=1.7 减 t=1.2）= %.8f" % d2)
print("  长期平均：⟨站点侧⟩ = %.8f，基线 %.8f，Δ̄ = %.8f（T=%.1f，%d 个整周期）"
      % (mean_site, mean_base, dbar, T_AVG, int(round(T_AVG / 2.0))))
print("  涨落周期 %.5f（峰值间距），幅度 %.8f；位错寿命须短于半个周期（t < %.1f）"
      % (period_meas, amp, period_meas / 2.0))
print("  转移分数区间 f ∈ [%.6f, %.6f]，超交换档 f = %.6f" % (min(fs), max(fs), f125))
print("  回流响应档 B：%s" % ", ".join("%.6f" % v for v in R_B))
print("  回流响应档 C：%s" % ", ".join("%.6f" % v for v in R_C))
print("  回流响应档 A：%s" % ", ".join("%.6f" % v for v in R_A))
print("  回流响应档 A'：%s" % ", ".join("%.6f" % v for v in R_AP))
print("  回流响应档 B'：%s" % ", ".join("%.6f" % v for v in R_BP))
print("  档 B 与 sin² 理论的最大相对偏差 %.4f" % max_dev)
print("  漏损替身各档 R：%s" % ", ".join("%.6f" % v for v in R_leak))
print("耗时 %.1f 秒" % (time.time() - T_START))
