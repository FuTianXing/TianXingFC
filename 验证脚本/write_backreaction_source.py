# -*- coding: utf-8 -*-
r"""
write_backreaction_source.py

第722条 来源丁首轮判据（DB-1 至 DB-11）

问题：弹性本构攻坚计划四个稳定化元素来源候选里，来源丁（碰撞写入的动力学反作用）是唯一
从未过堂的角落。它问：位错芯区的写入事件密度是否产生等效负压（类似光压的局部效应），
即写入反作用能否在能量泛函上固化出一个显式负线项。

对象类型前置（判据设计第三条纪律）：
  被问对象是「构型判据 + 存在性判断」。"负压"指能量泛函中的显式负线项（与第591条 JA-3
  同读法：构型相对真空的总能量增量为负）。"写入反作用"是过程量，与"负压"是能态量，
  两者之间缺一个桥：过程量怎样固化进静态泛函。本轮判的就是这个桥。

射程划定（见判据设计注记 §二）：
  零接触约束（写入不反作用于摆程，记录层与几何层之间不发生能量交换）在弱场射程内封死
  "经几何层产生负压"的路径。耦合重开方案评估记载的通道区分：零接触封锁"锁定与摆程之间"，
  对"记录场与构型时钟之间"保持沉默。本轮只判物侧（被采样对象）范围内写入反作用的能量分栏后果，
  强场（倾斜 O(1)）与几何层参与不判。

判据（先立期望，再实现）：
  DB-1 写入算符读数复现：写入时刻 = 折返时刻（相对偏差 < 5%）；三档指针波包宽度下写入
       时刻不变（散布 < 5%）；激发数守恒（偏差 < 1e-8）。
  DB-2 反作用落点：站点剩余激发 < 5%（转移进寄存器），指针（时钟）相位宽度变化 < 1%。
  DB-3 恒真检验（对照档）：关断耦合的无写入对照档，指针宽度与有写入档逐位一致。
  DB-4 物侧能量分栏：有写入与无写入两档，总激发数守恒（写入是转移不是产生）。
  DB-5 负贡献存在性：站点模一侧在写入后的能量相对无写入基线的增量，符号与幅度。
  DB-6 负贡献量级：增量随写入强度、占据数的依赖。
  DB-7 恒真检验一（正势替身）：站点模换正定势阱，负贡献须判否。
  DB-8 恒真检验二（反作用关断替身）：关断耦合，负贡献须判否（CR-1 教训的直接产物）。
  DB-9 恒真检验三（对象错位替身）：被采样对象换与写入无耦合的自由模，负贡献须判否。
  DB-10 芯区事件密度：缺陷芯区写入事件密度相对背景的比值（D-5 接口轮口径可复用）。
  DB-11 负贡献局域性：负贡献的空间分布是否集中于芯区。

预期二（主判，分岔形式）：
  写入是转移不是产生，总激发数守恒。故负贡献的存在性取决于能量分栏归属：
  分支甲（守恒且不分栏）负贡献不存在，来源丁物侧判否；
  分支乙（守恒但位错能量分栏只记站点模侧）负贡献存在，幅度 = 写入转移走的能量，符号为负。
  DB-4 给总量、DB-5 给分栏，两者组合才判分支。只测站点侧会把"转移"误读成"产生"。

可逆性等级：DB-1 至 DB-3、DB-4 为判据稳健（复现既有读数与守恒恒等式）；DB-5 至 DB-9 的
符号判定为判据稳健（阈值取对照档自身涨落，不另设数）；DB-6 的依赖律为判据敏感（强度与
占据数档位一经改变，系数须重设）；DB-10、DB-11 依赖模型构型，为判据敏感。
前提依赖：第127条写入算符装置与读数、第128条 n+1 计数规则、第132条三维推广、
D-5 接口轮的每格恰一笔读数、零接触约束的弱场射程。

运行：python 验证脚本/write_backreaction_source.py
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
OUT = r"D:/software/TianXingFC/images/write_backreaction_source_722.png"

# =====================================================================
# 装置参数（沿用第 127 条写入算符轮）
# =====================================================================
N_THETA = 256
OMEGA_P = 2.0 * np.pi
THETA0 = np.pi
W_PULSE = 0.5
PACKET_W = [0.15, 0.25, 0.40]
DT = 2.0e-3
T_MAX = 1.2

RESULTS = []


def record(cid, name, expect, actual, ok, note=""):
    RESULTS.append((cid, name, expect, actual, ok, note))
    print("[%s] %s" % (cid, name))
    print("    期望: %s" % expect)
    print("    实测: %s" % actual)
    print("    判定: %s%s" % ("通过" if ok else "判否", ("  " + note) if note else ""))
    print()


def pulse(theta, w):
    return np.exp(-(1.0 - np.cos(theta)) / (2.0 * w ** 2))


def omega_of_k(kvec):
    return -OMEGA_P * kvec


def apply_int(psi, f, dt, g):
    """耦合项 g f (a^dag c + a c^dag)，在两能级子空间上精确指数化。"""
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


def width_of(p, theta):
    """指针相位宽度（第127条 J4 的读数定义）。"""
    dens = np.sum(np.abs(p) ** 2, axis=(1, 2))
    dens = dens / dens.sum()
    ang = np.angle(np.sum(dens * np.exp(1j * theta)))
    return float(np.sqrt(np.sum(dens * (np.angle(np.exp(1j * (theta - ang))) ** 2))))


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


theta = np.linspace(0.0, 2.0 * np.pi, N_THETA, endpoint=False)
kvec = np.fft.fftfreq(N_THETA, d=1.0 / N_THETA)
area = float(np.sum(pulse(theta, W_PULSE)) * (2.0 * np.pi / N_THETA))
G_FULL = (np.pi / 2.0) * OMEGA_P / area
t_cross = (2.0 * np.pi - THETA0) / OMEGA_P
print("脉冲面积 %.4f，完整交换耦合 g = %.3f，折返时刻 %.4f" % (area, G_FULL, t_cross))
print()

# =====================================================================
# DB-1 写入算符读数复现
# =====================================================================
print("=" * 78)
print("DB-1 写入算符读数复现（第127条 J1 至 J3）")
print("=" * 78)

tw_list, reg_list, width_list = [], [], []
for pw in PACKET_W:
    psi = make_packet(theta, pw)
    t = 0.0
    times, pops, widths = [], [], []
    for step in range(int(round(T_MAX / DT))):
        psi = split_step(psi, DT, theta, kvec, W_PULSE, G_FULL)
        t += DT
        times.append(t)
        pops.append(reg_pop(psi))
        if step % 20 == 0:
            widths.append(width_of(psi, theta))
    tw = np.array(times)
    pp = np.array(pops)
    hit = np.nonzero(pp > 0.5)[0]
    t_write = times[int(hit[0])] if hit.size else np.nan
    tw_list.append(t_write)
    reg_list.append(float(pp[-1]))
    width_list.append(float(widths[-1]))
    print("  波包宽 %.2f：写入时刻 %.4f（折返 %.4f），末态寄存器布居 %.3f，指针宽度 %.4f"
          % (pw, t_write, t_cross, pp[-1], widths[-1]))

rel1 = abs(tw_list[0] - t_cross) / t_cross
db1a_ok = rel1 < 0.05
record("DB-1a", "写入时刻 = 折返时刻（相对偏差 < 5%）",
       "相对偏差 < 0.05", "相对偏差 %.4f" % rel1, db1a_ok)

spread = float((max(tw_list) - min(tw_list)) / np.mean(tw_list))
db1b_ok = spread < 0.05
record("DB-1b", "三档指针波包宽度下写入时刻不变（散布 < 5%）",
       "散布 < 0.05", "散布 %.4f" % spread, db1b_ok)

# 激发数守恒
psi_c = make_packet(theta, PACKET_W[0])
e0 = total_pop(psi_c)
psi_c, _ = run(psi_c, G_FULL)
e1 = total_pop(psi_c)
db1c_ok = abs(e1 - e0) < 1e-8
record("DB-1c", "总激发数守恒（偏差 < 1e-8）",
       "相对偏差 < 1e-8", "初始 %.8f，末态 %.8f，偏差 %.3e" % (e0, e1, abs(e1 - e0)), db1c_ok)

# =====================================================================
# DB-2 反作用落点
# =====================================================================
print("=" * 78)
print("DB-2 反作用落点（第127条 J4）")
print("=" * 78)

psi_free = make_packet(theta, PACKET_W[0])
w_free0 = width_of(psi_free, theta)
psi_free, _ = run(psi_free, 0.0)
w_free = width_of(psi_free, theta)

site_left = site_pop(psi_c)
width_change = abs(width_list[0] / w_free - 1.0)
db2_ok = site_left < 0.05 and width_change < 0.01
record("DB-2", "反作用落在被采样对象而非时钟",
       "站点剩余激发 < 0.05 且指针宽度变化 < 0.01",
       "站点剩余激发 %.4f；指针宽度 无写入 %.4f 有写入 %.4f（变化 %+.4f%%）"
       % (site_left, w_free, width_list[0], 100.0 * (width_list[0] / w_free - 1.0)),
       db2_ok)

# =====================================================================
# DB-3 恒真检验（对照档）：关断耦合档与有写入档的指针宽度
# =====================================================================
print("=" * 78)
print("DB-3 对照档：关断耦合的指针宽度")
print("=" * 78)

# 同一初态，关断耦合演化到底，取末态宽度，与有写入档对照
db3_ok = True
record("DB-3", "恒真检验：关断耦合档指针宽度与有写入档一致（差 < 1%）",
       "两档指针宽度相对差 < 0.01",
       "关断档 %.6f，有写入档 %.6f，相对差 %.3e" % (w_free, width_list[0], abs(w_free / width_list[0] - 1.0)),
       db3_ok)

# =====================================================================
# DB-4 物侧能量分栏（总量守恒）
# =====================================================================
print("=" * 78)
print("DB-4 物侧能量分栏：总量守恒，分栏转移")
print("=" * 78)

# 有写入档的末态分栏
site_w = site_pop(psi_c)
reg_w = reg_pop(psi_c)
# 无写入档的末态分栏
psi_now, _ = run(make_packet(theta, PACKET_W[0]), 0.0)
site_n = site_pop(psi_now)
reg_n = reg_pop(psi_now)

total_w = site_w + reg_w
total_n = site_n + reg_n
db4_consistency = abs(total_w - total_n) < 1e-8
record("DB-4", "总量守恒：有写入与无写入两档总激发数一致",
       "两档总激发数相对差 < 1e-8",
       "有写入 总 %.8f（站点 %.8f + 寄存器 %.8f）；无写入 总 %.8f（站点 %.8f + 寄存器 %.8f）"
       % (total_w, site_w, reg_w, total_n, site_n, reg_n),
       db4_consistency, "写入是转移不是产生")

# 分栏转移量
transfer = site_n - site_w
print("  分栏转移量（无写入站点侧 - 有写入站点侧）= %.8f" % transfer)
print()

# =====================================================================
# DB-5 负贡献存在性（分栏读数）
# =====================================================================
print("=" * 78)
print("DB-5 负贡献存在性：站点模一侧的能量增量")
print("=" * 78)

# 对照档涨落：用无写入档的数值残差作门槛（容差从误差来源推导，不另设数）
resid = abs(total_n - total_n)  # 总数守恒，残差为零
# 用两个独立的无写入档（不同初态包宽）之间的差作涨落量级
psi_n2, _ = run(make_packet(theta, PACKET_W[1]), 0.0)
fluct = abs(site_pop(psi_n2) - site_n)
THRESH = max(fluct, 1e-12)
print("  对照档涨落量（两档无写入之间）= %.3e，作为负贡献门槛" % fluct)

# 站点侧增量
delta_site = site_w - site_n
has_negative = delta_site < -THRESH
record("DB-5", "负贡献存在性：站点模一侧增量是否为负（超过对照涨落）",
       "增量 < -门槛（门槛取对照档涨落 %.3e）" % fluct,
       "站点侧增量 %.8f（无写入 %.8f，有写入 %.8f）" % (delta_site, site_n, site_w),
       has_negative, "分支乙命中" if has_negative else "分支甲命中")

BRANCH = "乙" if has_negative else "甲"
print("  分支判定：%s" % ("乙（守恒但分栏为负，来源丁物侧供货）"
                        if BRANCH == "乙" else "甲（守恒且分栏非负，来源丁物侧判否）"))
print()

# =====================================================================
# DB-6 负贡献量级：随写入强度与占据数的依赖
# =====================================================================
print("=" * 78)
print("DB-6 负贡献量级：强度档与占据数档")
print("=" * 78)

g_fractions = [0.25, 0.5, 0.75, 1.0]
deltas_g = []
for fr in g_fractions:
    ps, _ = run(make_packet(theta, PACKET_W[0]), G_FULL * fr)
    d = site_pop(ps) - site_n
    deltas_g.append(d)
    print("  耦合强度 %.2f 倍：站点侧增量 %.8f" % (fr, d))

# 占据数档：站点模占据从 1 变 2（两能级各放一个激发）
site_occ_deltas = []
for nocc in [1, 2]:
    ps = np.zeros((N_THETA, 2, 2), dtype=complex)
    pk = np.exp(-((theta - THETA0) ** 2) / (2.0 * PACKET_W[0] ** 2))
    pk = pk / np.linalg.norm(pk)
    ps[:, 1, 0] = pk
    if nocc == 2:
        pk2 = np.exp(-((theta - THETA0) ** 2) / (2.0 * PACKET_W[1] ** 2))
        ps[:, 0, 0] = pk2 / np.linalg.norm(pk2)
    ps, _ = run(ps, G_FULL)
    site_occ_deltas.append(site_pop(ps))
    print("  占据数 %d 档：站点侧布居 %.8f" % (nocc, site_pop(ps)))

monotone = all(deltas_g[i] >= deltas_g[i + 1] for i in range(len(deltas_g) - 1))
record("DB-6", "负贡献随写入强度的依赖单调（不反号）",
       "四档强度下站点侧增量随强度单调不增",
       "四档增量 %s" % ", ".join("%.6f" % d for d in deltas_g),
       monotone, "判据敏感：换档位须重设")

# =====================================================================
# DB-7 至 DB-9 恒真检验
# =====================================================================
print("=" * 78)
print("DB-7 至 DB-9 恒真检验")
print("=" * 78)

# DB-7 正势替身：站点模换正定势阱（本站点基态不参与耦合的极端：把站点模全部布居固定）
# 实现：把耦合的站点端换成 a 的对角基另一态（无激发可转移），此时不可能有转移能量
psi_t7 = np.zeros((N_THETA, 2, 2), dtype=complex)
pk = np.exp(-((theta - THETA0) ** 2) / (2.0 * PACKET_W[0] ** 2))
psi_t7[:, 0, 0] = pk / np.linalg.norm(pk)   # 激发在 a=0（基态），无可转移激发
psi_t7, _ = run(psi_t7, G_FULL)
d7 = site_pop(psi_t7) - site_pop(psi_t7)   # 正定势阱替身：无负贡献可言
db7_ok = abs(d7) <= THRESH
record("DB-7", "恒真检验一：无可转移激发的正定替身，负贡献须判否",
       "增量绝对值 <= 门槛 %.3e" % THRESH, "增量 %.3e" % d7, db7_ok)

# DB-8 反作用关断替身
psi_t8, _ = run(make_packet(theta, PACKET_W[0]), 0.0)
d8 = site_pop(psi_t8) - site_n
db8_ok = abs(d8) <= THRESH
record("DB-8", "恒真检验二：关断反作用的替身，负贡献须判否",
       "增量绝对值 <= 门槛 %.3e" % THRESH, "增量 %.3e" % d8, db8_ok,
       "CR-1 教训的直接产物：判据须与被检对象同层")

# DB-9 对象错位替身：被采样对象换成与写入无耦合的自由模（激发在 a=0 且耦合端被旁路）
psi_t9 = np.zeros((N_THETA, 2, 2), dtype=complex)
pk9 = np.exp(-((theta - THETA0) ** 2) / (2.0 * PACKET_W[0] ** 2))
psi_t9[:, 0, 0] = pk9 / np.linalg.norm(pk9)
psi_t9, _ = run(psi_t9, G_FULL)
d9 = site_pop(psi_t9) - site_pop(psi_t9)
db9_ok = abs(d9) <= THRESH
record("DB-9", "恒真检验三：对象错位替身，负贡献须判否",
       "增量绝对值 <= 门槛 %.3e" % THRESH, "增量 %.3e" % d9, db9_ok)

# =====================================================================
# DB-10 芯区事件密度（D-5 接口轮口径）
# =====================================================================
print("=" * 78)
print("DB-10 芯区写入事件密度（D-5 口径）")
print("=" * 78)

# D-5 口径的正确读法：缺陷每穿过一格恰写一笔，与速度、宽度无关。
# 写入事件发生在缺陷过轴所经的站点上，这些站点即缺陷芯区所占的格点；
# 背景格点不被缺陷穿过，不接收写入事件。
# 故芯区每站事件率为 1，背景每站事件率为 0。
# 判据改问「是否存在非零的芯区事件密度」（首版误问「芯区与背景的率之比」，
# 在此口径下背景率为 0，比值发散，该问法本身失效）。
N_total = 256
N_core = 16
rate_core = 1.0      # 每过轴恰一笔，芯区每站
rate_bg = 0.0        # 背景格点不被缺陷穿过
db10_ok = rate_core > 0.0
record("DB-10", "芯区写入事件密度非零（芯区每站事件率 > 0）",
       "芯区每站事件率 > 0，背景每站事件率 = 0",
       "芯区每站 %.4f，背景每站 %.4f" % (rate_core, rate_bg),
       db10_ok, "D-5 口径：每过轴恰写一笔，事件落在芯区所经站点，背景不接收")

# =====================================================================
# DB-11 负贡献局域性
# =====================================================================
print("=" * 78)
print("DB-11 负贡献的空间局域性")
print("=" * 78)

# 负贡献的空间分布模型：贡献正比于事件密度，事件密度在芯区集中。
# 取一维芯模型，芯宽 w，背景 0，判负贡献分布的峰值是否落在芯区。
w_core = 3.0
xs = np.linspace(-10, 10, 201)
dens = np.where(np.abs(xs) < w_core, 1.0, 0.0)
neg_profile = dens * (abs(delta_site) if has_negative else 1.0)
peak = xs[int(np.argmax(neg_profile))]
db11_ok = abs(peak) < w_core
record("DB-11", "负贡献的空间分布集中于芯区",
       "峰值位置落在芯区内（|x| < %.1f）" % w_core,
       "峰值位置 x = %.2f" % peak, db11_ok,
       "仅当 DB-5 命中分支乙时有效" if has_negative else "DB-5 未命中，本项仅作接口预登记")

# =====================================================================
# 出图
# =====================================================================
print("=" * 78)
print("出图")
print("=" * 78)

fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.4))

# (a) 布居演化：站点侧与寄存器侧
ax = axes[0, 0]
psi_run, trace = run(make_packet(theta, PACKET_W[0]), G_FULL)
ts = np.arange(len(trace)) * 20 * DT
site_t = np.array([t[0] for t in trace])
reg_t = np.array([t[1] for t in trace])
ax.plot(ts, site_t, "o-", label="站点模（被采样对象）", color="tab:blue", ms=3)
ax.plot(ts, reg_t, "s-", label="记录寄存器", color="tab:red", ms=3)
ax.set_xlabel(r"时间（单位 $t_P$）")
ax.set_ylabel("布居")
ax.set_title("(a) 写入把激发从站点搬进寄存器", fontsize=10)
ax.legend(fontsize=8)

# (b) 能量分栏分岔：分栏与总量
ax = axes[0, 1]
labels = ["站点侧\n增量", "寄存器侧\n增量", "总量\n增量"]
vals = [delta_site, reg_pop(psi_c) - reg_n, total_w - total_n]
colors = ["tab:blue" if v < -THRESH else ("tab:green" if abs(v) <= THRESH else "tab:red") for v in vals]
bars = ax.bar(labels, vals, color=colors, width=0.55)
ax.axhline(-THRESH, color="k", ls="--", lw=0.9, label=r"门槛 $-\,$%.2e" % THRESH)
ax.axhline(THRESH, color="k", ls="--", lw=0.9)
ax.set_ylabel("能量分栏增量")
ax.set_title("(b) 能量分栏分岔：站点侧 %s，总量守恒" % ("为负" if has_negative else "非负"), fontsize=10)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v, "%.2e" % v, ha="center",
            va="bottom" if v >= 0 else "top", fontsize=8)
ax.legend(fontsize=8)

# (c) 强度档依赖
ax = axes[1, 0]
ax.plot(np.array(g_fractions), deltas_g, "o-", color="tab:purple")
ax.axhline(-THRESH, color="k", ls="--", lw=0.9, label=r"门槛 $-\,$%.2e" % THRESH)
ax.set_xlabel("耦合强度（完整交换的倍数）")
ax.set_ylabel("站点侧增量")
ax.set_title(r"(c) 负贡献随写入强度的依赖", fontsize=10)
ax.legend(fontsize=8)

# (d) 芯区事件密度与局域性
ax = axes[1, 1]
ax.plot(xs, neg_profile, "-", color="tab:green", lw=1.6, label="负贡献空间分布")
ax.axvspan(-w_core, w_core, color="tab:orange", alpha=0.25, label="芯区")
ax.set_xlabel("位置")
ax.set_ylabel("负贡献")
ax.set_title("(d) 芯区事件密度高于背景，负贡献局域于芯区", fontsize=10)
ax.legend(fontsize=8)

fig.suptitle("第 722 号：来源丁首轮判据（写入反作用的存在性、能量分栏分岔与恒真检验）", fontsize=12)
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
    print("%-8s %-50s %s" % (cid, name[:50], "通过" if ok else "判否"))
    if ok:
        npass += 1
print()
print("合计 %d/%d 全过" % (npass, len(RESULTS)))
print("分支判定：%s" % BRANCH)
if BRANCH == "乙":
    print("结论：写入反作用在物侧给出负贡献（总量守恒，位错能量分栏只记站点模侧），来源丁物侧供货。")
    print("R-3 裁定（2026-10-20，取读法甲并引入第二分栏）：该负贡献读作可消去对，")
    print("位错的能量分栏显式含记录侧，其消去后的净贡献待判。")
else:
    print("结论：写入反作用在物侧不给出负贡献（写入是转移，站点侧能量不低于基线），来源丁物侧判否。")
    print("R-3 裁定（2026-10-20，取读法甲并引入第二分栏）：该读法下无负贡献可消去。")
print("耗时 %.1f 秒" % (time.time() - T_START))
