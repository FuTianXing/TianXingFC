# -*- coding: utf-8 -*-
r"""
minimal_verification_construction_case_a_table.py

第 735 号 构造轮案甲：标度对表（判据 甲-1 至 甲-7）

问题：三维复合位错弹性本构攻坚线要找能量泛函里的显式负线项。来源丁（碰撞写入的
动力学反作用）在判据第 726、726 号已验证：写入把激发从站点搬进记录寄存器，总量守恒；
负贡献在装置内是周期往复量，长期平均净增量 -0.50000000；回流通道打开时按
sin^2(kappa pi/2) 律消失。本轮所有读数都是无量纲计数，与 C1 侧的凝聚能量纲不同类，
第 724 号已判三条量纲桥全不成立。

本轮做构造轮方案提要的案甲（标度对表）。承第 725 号确立的原则「只比区间，不比能量」：
标度指数与窗口区间端点都是无量纲的，绕开量纲桥这个死结。案甲不建新装置，在第 726 号
写入侧与第 725 号 C1 侧各自已有的读数上做标度与端点区间对表，给相容性判语。

装置：写入侧复用 验证脚本/minimal_verification_source_ding_reflux_saturation.py，
C1 侧复用 验证脚本/minimal_verification_amplitude_channel_w5_w6.py。两脚本的装置函数
以 AST 逐字摘取，不重造实现。本轮只取 C1 侧 AW-1 与 AW-3 所需那组函数（core_width、
slope_scan、anchor、solve_R、mean_slope、comp_at）；channel_coeffs 与 bound_gap_profile
属 AW-9 至 AW-13 的复标量通道，与本轮对表无关，未取用。

判据：甲-1 至 甲-7，逐条给期望、实测与判定。容差从误差预算推导，见各条注释。
可逆性等级：甲-1 与甲-2 为判据稳健；甲-3 至 甲-7 为判据敏感（甲-3 的型态标签、
甲-4 的档位阈值、甲-5 至 甲-7 的替身构型都带裁量，换裁量会翻）。

运行：python 验证脚本/minimal_verification_construction_case_a_table.py
确定性，无随机数。
"""

import ast
import inspect
import io
import time

import numpy as np
from scipy.optimize import minimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

T_START = time.time()
OUT = r"D:/software/TianXingFC/images/construction_case_a_table_735.png"

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
# 装置摘取工具：从既有脚本的 AST 逐字取出函数与模块级常量
# =====================================================================
def harvest(path, const_names, func_names):
    """按源码顺序取出指定模块级常量与函数定义，原样编译执行。

    只取赋值语句与函数定义，不执行源脚本的顶层流程，故不产生图文件，
    也不触发源脚本自身的判据循环。
    """
    src = io.open(path, encoding="utf-8").read()
    tree = ast.parse(src)
    lines = src.splitlines(True)
    chunks = []
    const_src = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in func_names:
            chunks.append("".join(lines[node.lineno - 1:node.end_lineno]))
        elif isinstance(node, ast.Assign):
            names = []

            def _collect(t):
                if isinstance(t, ast.Name):
                    names.append(t.id)
                elif isinstance(t, (ast.Tuple, ast.List)):
                    for e in t.elts:
                        _collect(e)

            for t in node.targets:
                _collect(t)
            hit = [nm for nm in names if nm in const_names]
            if hit:
                text = "".join(lines[node.lineno - 1:node.end_lineno])
                chunks.append(text)
                for nm in hit:
                    const_src[nm] = text.strip()
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id in const_names:
                text = "".join(lines[node.lineno - 1:node.end_lineno])
                chunks.append(text)
                const_src[node.target.id] = text.strip()
    code = "import numpy as np\nfrom scipy.optimize import minimize\n" + "\n\n".join(chunks)
    ns = {}
    exec(compile(code, path, "exec"), ns)
    ns["_const_src"] = const_src
    return ns


def sig_of(ns, name):
    return str(inspect.signature(ns[name]))


P726 = "验证脚本/minimal_verification_source_ding_reflux_saturation.py"
P725 = "验证脚本/minimal_verification_amplitude_channel_w5_w6.py"

CONST_726 = {"N_THETA", "OMEGA_P", "THETA0", "W_PULSE", "PW", "DT", "T_MAX", "T_LONG",
             "theta", "kvec", "area", "G_FULL", "t_cross", "G_WRITE"}
FUNC_726 = {"pulse", "omega_of_k", "apply_int", "split_step", "make_packet",
             "site_pop", "reg_pop", "total_pop", "run"}

CONST_725 = {"N", "N3", "COORD", "_X", "_Y", "_Z", "RHO", "IC", "UPT", "DNT", "RS",
             "LAMS", "SHALLOW_D", "DEEP_D", "lam_grid", "Ra0", "d0", "w0"}
FUNC_725 = {"ring_phase", "cos_cache", "make_fun", "anchor", "solve_R", "mean_slope",
             "core_width", "slope_scan", "comp_at"}

NS726 = harvest(P726, CONST_726, FUNC_726)
NS725 = harvest(P725, CONST_725, FUNC_725)

theta = NS726["theta"]
PW = NS726["PW"]
DT = NS726["DT"]
T_MAX = NS726["T_MAX"]
G_FULL = NS726["G_FULL"]
run = NS726["run"]
make_packet = NS726["make_packet"]
site_pop = NS726["site_pop"]
reg_pop = NS726["reg_pop"]
total_pop = NS726["total_pop"]

solve_R = NS725["solve_R"]
anchor = NS725["anchor"]
core_width = NS725["core_width"]
slope_scan = NS725["slope_scan"]
mean_slope = NS725["mean_slope"]
RS = NS725["RS"]
N_C1 = NS725["N"]

# =====================================================================
# 前提守恒核对（三条）
# =====================================================================
print("=" * 78)
print("前提守恒核对（跨轮移植纪律）")
print("=" * 78)

# 守恒一：写入侧的耦合强度参数与第 726 号一致
EXPECT_726 = {
    "N_THETA": 256,
    "OMEGA_P": 2.0 * np.pi,
    "THETA0": np.pi,
    "W_PULSE": 0.5,
    "PW": 0.15,
    "DT": 2.0e-3,
    "T_MAX": 1.2,
}
print("守恒一，写入侧参数（对照第 726 号）：")
keep1 = True
for k, v in EXPECT_726.items():
    got = NS726[k]
    same = abs(float(got) - float(v)) <= 1e-12 * max(1.0, abs(float(v)))
    keep1 = keep1 and same
    print("  %-10s 期望 %-14s 实测 %-14s %s"
          % (k, repr(v), repr(got), "一致" if same else "不一致"))
g_ref_726 = 5.092
g_dev = abs(G_FULL - g_ref_726) / g_ref_726
keep1 = keep1 and g_dev < 1e-3
print("  %-10s 期望 %-14s 实测 %-14s %s（相对偏差 %.3e）"
      % ("G_FULL", "5.092（第 726 号引用）", "%.6f" % G_FULL,
         "一致" if g_dev < 1e-3 else "不一致", g_dev))
print("  结论: %s" % ("写入侧参数与第 726 号一致，可移植" if keep1 else "写入侧参数漂移"))
print()

# 守恒二：C1 侧的 lambda 档位与锚核参数与第 725 号一致
EXPECT_725 = {
    "N": 32,
    "LAMS": [0.25, 0.5, 1.0],
    "SHALLOW_D": 0.05,
    "DEEP_D": 0.5,
    "lam_grid": [0.25, 0.5, 1.0, 2.0, 4.0],
    "Ra0": 5.0,
    "d0": 0.5,
    "w0": 1.5,
}
print("守恒二，C1 侧档位与锚核参数（对照第 725 号）：")
keep2 = True
for k, v in EXPECT_725.items():
    got = NS725[k]
    if isinstance(v, list):
        same = (list(np.asarray(got, dtype=float)) == list(np.asarray(v, dtype=float)))
    else:
        same = abs(float(got) - float(v)) <= 1e-12
    keep2 = keep2 and same
    print("  %-10s 期望 %-28s 实测 %-28s %s"
          % (k, str(v), str(got), "一致" if same else "不一致"))
rs_ref = list(np.arange(3.0, 8.01, 0.5))
same_rs = (list(np.asarray(RS, dtype=float)) == rs_ref)
keep2 = keep2 and same_rs
print("  %-10s 期望 %-28s 实测 %-28s %s"
      % ("RS", "arange(3,8.01,0.5)（11 档）",
         "arange(3,8.01,0.5)（%d 档）" % len(RS), "一致" if same_rs else "不一致"))
print("  结论: %s" % ("C1 侧档位与锚核参数与第 725 号一致，可移植"
                      if keep2 else "C1 侧参数漂移"))
print()

# 守恒三：两脚本扫描轴不同质
AXIS_WRITE = "耦合强度倍数 g / G_FULL（无量纲，来源丁写入侧）"
AXIS_C1 = "锚核强度参数 lambda（无量纲，C1 侧振幅通道）"
print("守恒三，两脚本扫描轴：")
print("  写入侧扫描轴: %s" % AXIS_WRITE)
print("  C1 侧扫描轴  : %s" % AXIS_C1)
print("  轴不同质：写入侧扫的是「写入算符把材料搬走多少」的驱动倍数，")
print("  C1 侧扫的是「钉扎井多深」的结构参数，两轴无共同的物理同构关系。")
print("  故本案只在无量纲标度型态与无量纲端点档位这一层对表，")
print("  「同一参数上的窗口重合」在本案不成立。这是「只比区间，不比能量」原则能成立的根本理由。")
print()
KEEP_ALL = keep1 and keep2
supp("前提守恒", "三条前提守恒汇总",
     "守恒一 %s；守恒二 %s；守恒三 轴不同质已写明" %
     ("通过" if keep1 else "判否", "通过" if keep2 else "判否"),
     "本轮读数可移植的前提全部成立" if KEEP_ALL else "存在漂移，移植前提不成立")

# =====================================================================
# 冒烟检查
# =====================================================================
print("=" * 78)
print("冒烟检查")
print("=" * 78)

# 冒烟一：写入侧复现装置的末态分栏布居
psi_nw, _ = run(make_packet(theta, PW), 0.0, steps=int(round(T_MAX / DT)))
psi_w, _ = run(make_packet(theta, PW), G_FULL, steps=int(round(T_MAX / DT)))
site_n = site_pop(psi_nw)
site_w = site_pop(psi_w)
reg_w = reg_pop(psi_w)
print("写入侧末态分栏布居（t = %.1f，满档 g = G_FULL）：" % T_MAX)
print("  站点侧 %.8f（期望约 0.00117617），记录侧 %.8f（期望约 0.99882383）"
      % (site_w, reg_w))
print("  无写入基线站点侧 %.8f，末态总量 %.8f" % (site_n, total_pop(psi_w)))
smoke_pop_ok = (abs(site_w - 0.00117617) < 1e-6 and abs(reg_w - 0.99882383) < 1e-6)
print("  与第 726 号末态读数一致: %s" % ("是" if smoke_pop_ok else "否"))
print()

# 冒烟二：C1 侧五档补偿比的原始数值
LAM_GRID = [float(v) for v in NS725["lam_grid"]]


def comp_and_tension(lam):
    """补偿与线张力斜率，按第 725 号 AW-3 的口径逐字实现。

    补偿 = 对照档步进斜率均值 - 锚核档步进斜率均值；
    线张力 = 对照档步进斜率均值。
    """
    ctrl = slope_scan(5.0, 0.0, 1.5, lam)
    anch = slope_scan(5.0, 0.5, 1.5, lam)
    sc = np.diff(ctrl) / np.diff(RS)
    sa = np.diff(anch) / np.diff(RS)
    tension = float(np.mean(sc))
    compens = float(np.mean(sc) - np.mean(sa))
    return compens, tension, float(sa.min())


c1_comp = []
c1_tension = []
c1_minstep = []
for lam in LAM_GRID:
    cm, tn, ms = comp_and_tension(lam)
    c1_comp.append(cm / tn)
    c1_tension.append(tn)
    c1_minstep.append(ms)
    print("  C1 侧 lambda = %.2f：补偿 %.4f，线张力 %.4f，补偿比 %.6f，最小步进斜率 %+.4f"
          % (lam, cm, tn, cm / tn, ms))
print()

# 冒烟三：两脚本各函数的调用签名
print("调用签名核对（摘取自源脚本，逐字一致）：")
for nm in sorted(FUNC_726):
    print("  第 726 号 %-14s %s" % (nm, sig_of(NS726, nm)))
for nm in sorted(FUNC_725):
    print("  第 725 号 %-14s %s" % (nm, sig_of(NS725, nm)))
print("  实际调用示例：core_width(A, 5.0) 收三维振幅场与管壁半径；")
print("  slope_scan(Ra, depth, w, lam) 收锚核半径、深度、几何宽度与 lambda。")
print()

# 冒烟四：变化的轴 = 扫描的轴
G_FRACS = [0.125, 0.25, 0.5, 0.75, 1.0]
f_true = []
for fr in G_FRACS:
    p, _ = run(make_packet(theta, PW), G_FULL * fr, steps=int(round(T_MAX / DT)))
    f_true.append((site_n - site_pop(p)) / site_n)
f_zero = (site_n - site_pop(psi_nw)) / site_n
print("变化的轴核对，写入侧（扫描轴 = 耦合强度倍数）：")
print("  关闭写入耦合档（g = 0）转移分数 f = %.8f（应为 0）" % f_zero)
print("  五档 f 随耦合档单调不减: %s" % all(f_true[i] <= f_true[i + 1]
                                          for i in range(len(f_true) - 1)))
ctrl_comp_noanchor = comp_and_tension(1.0)[0]
print("变化的轴核对，C1 侧（扫描轴 = lambda）：")
print("  无锚核档补偿 %.3e（应为 0，补偿由锚核携带）" % ctrl_comp_noanchor)
print("  补偿比随 lambda 变化: %s（极差 %.6f）"
      % (max(c1_comp) - min(c1_comp) > 1e-6, max(c1_comp) - min(c1_comp)))
print()

# =====================================================================
# 型态标签与判据工具
# =====================================================================
def shape_label(vals, tol_const=1e-9, sat_ratio=0.5):
    """给一列读数打型态标签。

    规则（从两侧实测型态定义，逐条可核）：
      一，极差不超过容差者判常数型。
      二，存在方向反转者判非单调型；峰值落在序列内部者记「峰值在中段」。
      三，单调者若末端增量比（末段增量除以前一段增量）不超过 sat_ratio，
          判饱和型：曲线在末端已弯平，趋向上界。
      四，其余单调者判单调线性型。
    返回标签字符串与诊断量（末端增量比、幂次拟合斜率、拟合优度）。
    """
    y = np.asarray(vals, dtype=float)
    d = np.diff(y)
    rng = float(y.max() - y.min())
    inc = float(np.sum(d > 0))
    dec = float(np.sum(d < 0))
    if rng <= tol_const:
        return "常数型", {"range": rng, "d_ratio": float("nan"),
                          "power": float("nan"), "r2": float("nan")}
    peak = int(np.argmax(y))
    n = len(y)
    if inc > 0 and dec > 0:
        pos = "峰值在中段" if 0 < peak < n - 1 else "峰值在端点"
        return "非单调型（%s）" % pos, {"range": rng, "d_ratio": float("nan"),
                                       "power": float("nan"), "r2": float("nan")}
    d_ratio = float(d[-1] / d[-2]) if (len(d) >= 2 and abs(d[-2]) > 1e-15) else float("nan")
    xs = np.arange(1.0, n + 1.0)
    if y.min() > 0:
        p = float(np.polyfit(np.log(xs), np.log(np.abs(y) + 1e-300), 1)[0])
        pred = np.polyval(np.polyfit(np.log(xs), np.log(np.abs(y) + 1e-300), 1), np.log(xs))
        sst = float(np.sum((np.log(np.abs(y) + 1e-300)
                            - np.mean(np.log(np.abs(y) + 1e-300))) ** 2))
        sse = float(np.sum((np.log(np.abs(y) + 1e-300) - pred) ** 2))
        r2 = 1.0 - sse / sst if sst > 0 else float("nan")
    else:
        p, r2 = float("nan"), float("nan")
    if not np.isnan(d_ratio) and d_ratio <= sat_ratio:
        return "饱和型", {"range": rng, "d_ratio": d_ratio, "power": p, "r2": r2}
    return "单调线性型", {"range": rng, "d_ratio": d_ratio, "power": p, "r2": r2}


def mag_tier(a, b):
    """数量级档对照：两值比值不超过一个十倍即记同档。"""
    lo, hi = (a, b) if a <= b else (b, a)
    ratio = hi / lo
    return ratio, ("同档" if ratio < 10.0 else "分离")


# =====================================================================
# 甲-1 写入侧标度型态
# =====================================================================
print("=" * 78)
print("甲-1 写入侧标度型态")
print("=" * 78)

REF_F = [0.039701, 0.152500, 0.516971, 0.871081, 0.998824]
print("写入侧五档转移分数（强度分数对 G_FULL 的倍数）：")
rel_f = []
for fr, v, ref in zip(G_FRACS, f_true, REF_F):
    rel = abs(v - ref) / ref
    rel_f.append(rel)
    print("  档 %.3f：f = %.8f，第 726 号 %.6f，相对偏差 %.3e" % (fr, v, ref, rel))
mono_f = all(f_true[i] <= f_true[i + 1] for i in range(len(f_true) - 1))
f_lo, f_hi = min(f_true), max(f_true)
label_w, diag_w = shape_label(f_true)
print("  五档单调不减: %s" % mono_f)
print("  型态标签: %s（末端增量比 %.4f，幂次拟合 %+0.4f，拟合优度 %.4f）"
      % (label_w, diag_w["d_ratio"], diag_w["power"], diag_w["r2"]))
print("  端点区间 f ∈ [%.6f, %.6f]" % (f_lo, f_hi))

p125, _ = run(make_packet(theta, PW), G_FULL * 1.25, steps=int(round(T_MAX / DT)))
f125 = (site_n - site_pop(p125)) / site_n
print("  超交换档 1.25 倍：f = %.8f（射程外单列，不入单调性判定）" % f125)

ok_1 = (max(rel_f) < 1e-5 and mono_f and label_w == "饱和型")
record("甲-1", "写入侧标度型态：复现五档 f 并给型态与端点区间",
       "五档相对偏差 < 1e-5，型态为饱和型，给端点区间",
       "五档 f = %s；相对偏差 %s；标签 %s；区间 [%.6f, %.6f]；超交换档 %.6f 单列"
       % (", ".join("%.6f" % v for v in f_true),
          ", ".join("%.1e" % r for r in rel_f), label_w, f_lo, f_hi, f125),
       ok_1, "复现容差 1e-5 与第 726 号同源")

# =====================================================================
# 甲-2 C1 侧标度型态复现
# =====================================================================
print("=" * 78)
print("甲-2 C1 侧标度型态复现")
print("=" * 78)

# 容差预算（甲-2）：
#   宽度幂次的引用值给到六位小数，复现值逐位相同，相对偏差在 1e-13 量级，
#   取相对容差 1e-5 稳过。
#   补偿比五档的引用值只给到四位小数，其自身舍入半宽为 5e-5，相对宽度约 2.4e-3。
#   相对容差 1e-5 与该参照精度不自洽（参照值本身的舍入误差已大于 1e-5 的相对量），
#   故补偿比一项按绝对容差 5e-5 取，即参照值的舍入半宽。
#   这是容差从误差预算推导的纪律：容差下限由参照值的给出精度决定。
#   实测最大绝对偏差 4.876e-05，落在 5e-5 内。
LAMS_W = [float(v) for v in NS725["LAMS"]]
SHALLOW_D = float(NS725["SHALLOW_D"])
DEEP_D = float(NS725["DEEP_D"])

widths = []
for lam in LAMS_W:
    A0 = anchor(5.0, SHALLOW_D, 1.5)
    _, x = solve_R(5.0, A0, lam)
    widths.append(core_width(x.reshape(N_C1, N_C1, N_C1), 5.0))
p_width = float(np.polyfit(np.log(np.array(LAMS_W)), np.log(np.array(widths)), 1)[0])
bridge = [float(widths[i] / (1.0 / np.sqrt(4.0 * LAMS_W[i]))) for i in range(3)]
deep_widths = []
for lam in LAMS_W:
    A0 = anchor(5.0, DEEP_D, 1.5)
    _, x = solve_R(5.0, A0, lam)
    deep_widths.append(core_width(x.reshape(N_C1, N_C1, N_C1), 5.0))
p_deep = float(np.polyfit(np.log(np.array(LAMS_W)), np.log(np.array(deep_widths)), 1)[0])
print("AW-1 复现：浅锚档（d = %.2f）三档宽度 %s" % (SHALLOW_D, ", ".join("%.6f" % w for w in widths)))
print("  幂次拟合 %+0.6f（第 725 号 -0.286038），桥因子 %s"
      % (p_width, ", ".join("%.6f" % b for b in bridge)))
print("  深锚对照档（d = %.2f）幂次 %+0.6f（第 725 号 -0.193393）" % (DEEP_D, p_deep))
rel_pw = abs(p_width - (-0.286038)) / 0.286038
rel_pd = abs(p_deep - (-0.193393)) / 0.193393
print("  两幂次相对偏差 %.3e 与 %.3e" % (rel_pw, rel_pd))

bridge_mean = float(np.mean(bridge))
lam_window_max = (bridge_mean / 4.0) ** 2
print("AW-2 复现：桥因子均值 %.6f，落窗上限 lambda <= %.6f（第 725 号 1.189473）"
      % (bridge_mean, lam_window_max))
rel_lw = abs(lam_window_max - 1.189473) / 1.189473

REF_C1 = [0.0208, 0.0295, 0.0275, 0.0197, 0.0095]
abs_c1 = [abs(v - r) for v, r in zip(c1_comp, REF_C1)]
print("AW-3 复现：补偿比五档 %s" % ", ".join("%.6f" % v for v in c1_comp))
print("  第 725 号引用值 %s" % ", ".join("%.4f" % r for r in REF_C1))
print("  逐档绝对偏差 %s（容差 5e-5，参照值舍入半宽）"
      % ", ".join("%.3e" % a for a in abs_c1))
print("  AW-3 那一档：补偿 %.4f，线张力 %.4f，最小步进斜率 %+.4f"
      % (comp_and_tension(1.0)[0], comp_and_tension(1.0)[1], comp_and_tension(1.0)[2]))
label_c, diag_c = shape_label(c1_comp)
print("  型态标签: %s" % label_c)

ok_2 = (rel_pw < 1e-5 and rel_pd < 1e-5 and max(abs_c1) < 5e-5
        and label_c == "非单调型（峰值在中段）")
record("甲-2", "C1 侧标度型态复现：宽度幂次与补偿比五档走势",
       "宽度幂次相对偏差 < 1e-5，补偿比五档绝对偏差 < 5e-5，型态为非单调且峰值在中段",
       "浅锚幂次 %+0.6f（偏差 %.1e），深锚幂次 %+0.6f（偏差 %.1e）；补偿比 %s，"
       "最大绝对偏差 %.3e；落窗上限 lambda <= %.6f（偏差 %.1e）；标签 %s"
       % (p_width, rel_pw, p_deep, rel_pd, ", ".join("%.6f" % v for v in c1_comp),
          max(abs_c1), lam_window_max, rel_lw, label_c),
       ok_2, "补偿比一项的容差按参照值四位小数的舍入半宽取 5e-5")

# 引用既有轮次的线张力斜率三档（W1 轮给区间，第 724、725 号只是引用）
TENSION_REF = [(0.25, 129.0), (1.0, 193.0), (4.0, 239.7)]
tension_slope_3 = []
print("线张力斜率三档（引用振幅通道径向曲线轮 W1 的读数，本轮用同一装置同函数复算）：")
for lam, ref in TENSION_REF:
    v = mean_slope(slope_scan(5.0, 0.0, 1.5, lam))
    tension_slope_3.append((lam, v, ref))
    print("  lambda = %.2f：mean_slope %.4f，W1 轮引用 %.1f" % (lam, v, ref))
tension_lo = min(v for _, v, _ in tension_slope_3)
tension_hi = max(v for _, v, _ in tension_slope_3)
supp("AW-2", "C1 侧其余端点：落窗上限与线张力斜率区间",
     "落窗上限 lambda <= %.6f；线张力斜率区间 [%.4f, %.4f]（W1 轮给区间 %.1f 至 %.1f）"
     % (lam_window_max, tension_lo, tension_hi, 129.0, 251.8),
     "线张力斜率与落窗上限与 f 的无量纲端点不同类，只在同侧内部作区间列示，不入档位判定")

# =====================================================================
# 甲-3 型态对表
# =====================================================================
print("=" * 78)
print("甲-3 型态对表")
print("=" * 78)

same_type = (label_w == label_c)
print("写入侧型态标签: %s" % label_w)
print("C1 侧型态标签  : %s" % label_c)
print("主判定: %s" % ("同型" if same_type else "不同型"))

# 次级判定：上包络形态对照（本轮新增读数，方案提要未写，属裁量）
# 判据期望的来源：两侧实测型态定义本身。
#   写入侧的饱和型在「高段趋平」上表现为高段读数单调不减且末段增量收缩。
#   C1 侧的「补偿比在低 lambda 段先升」表现为该段读数先升后降、峰值在内部。
#   取两侧各自的上包络子集，比较这两个子集的形态标签是否相同。
env_w_idx = [i for i, v in enumerate(f_true) if v >= 0.87]
env_c_idx = [i for i, v in enumerate(c1_comp) if v >= 0.0197]
env_w = [f_true[i] for i in env_w_idx]
env_c = [c1_comp[i] for i in env_c_idx]
env_w_label, _ = shape_label(env_w)
env_c_label, _ = shape_label(env_c)
print()
print("次级判定，上包络形态对照（本轮新增读数）：")
print("  写入侧上包络子集（f >= 0.87，档 %s）：%s，标签 %s"
      % (env_w_idx, ", ".join("%.6f" % v for v in env_w), env_w_label))
print("  C1 侧上包络子集（补偿比 >= 0.0197，档 %s）：%s，标签 %s"
      % (env_c_idx, ", ".join("%.6f" % v for v in env_c), env_c_label))
print("  上包络判定: %s" % ("同型" if env_w_label == env_c_label else "不同型"))
print("  裁量说明：写入侧按数值 f >= 0.87 只取到两档（0.87 以上的档仅 0.75 与 1.0），")
print("  不足以与 C1 侧的三档作对称比较；作为补充，取强度档的上三档")
print("  （0.5、0.75、1.0）重判一次，结论并入下条读数。")
env_w_sym = f_true[2:]
env_w_sym_label, _ = shape_label(env_w_sym)
print("  补充口径（强度档 0.5、0.75、1.0）：%s，标签 %s"
      % (", ".join("%.6f" % v for v in env_w_sym), env_w_sym_label))

ok_3 = same_type
record("甲-3", "型态对表：两侧标度型态是否同型",
       "型态标签相等即判同型（对表假设 H3）；给出主判定与上包络次级判定",
       "写入侧 %s 对 C1 侧 %s，主判定 %s；上包络次级判定 %s（写入侧 %s 对 C1 侧 %s）"
       % (label_w, label_c, "同型" if same_type else "不同型",
          "同型" if env_w_label == env_c_label else "不同型", env_w_label, env_c_label),
       ok_3, "真实读数下两侧型态不同型，本案的标度对表在这对对象上不成立")

# =====================================================================
# 甲-4 端点档位对表
# =====================================================================
print("=" * 78)
print("甲-4 端点档位对表")
print("=" * 78)

c1_hi = max(c1_comp)
c1_lo = min(c1_comp)
ratio_low, tier_low = mag_tier(f_lo, c1_hi)
ratio_high, tier_high = mag_tier(f_hi, c1_lo)
print("对照对象的选择依据：f 是转移分数，量的对象是材料被搬走多少；")
print("补偿比是钉扎能补偿住多少。两者都是无量纲比例，故取同向端点相配。")
print("  下端配对：f 下端 %.6f 对补偿比上端 %.6f，比值 %.4f 倍，%s" % (f_lo, c1_hi, ratio_low, tier_low))
print("  上端配对：f 上端 %.6f 对补偿比下端 %.6f，比值 %.4f 倍，%s" % (f_hi, c1_lo, ratio_high, tier_high))
print("  上端一对是最不利对：f 上端对应几乎搬空，补偿比下端对应钉扎最弱。")
print("  档位判据：两值比值不超过一个十倍记同档，十倍恰为一个数量级。")
all_same_tier = (tier_low == "同档" and tier_high == "同档")
print("  整体判定: %s" % ("全部同档" if all_same_tier else "下端同档、上端分离"))
ok_4 = all_same_tier
record("甲-4", "端点档位对表：两侧区间端点是否落在同一数量级档",
       "两配对比值均小于 10 即全部同档（对表假设 H4）",
       "下端配对 %.6f 对 %.6f，比值 %.4f 倍，%s；上端配对 %.6f 对 %.6f，比值 %.4f 倍，%s；"
       "整体为下端同档、上端分离"
       % (f_lo, c1_hi, ratio_low, tier_low, f_hi, c1_lo, ratio_high, tier_high),
       ok_4, "上端一对是最不利对，相差约 %.1f 倍，已跨过两个数量级" % ratio_high)

# =====================================================================
# 甲-5 恒真检验一：写入侧线性替身
# =====================================================================
print("=" * 78)
print("甲-5 恒真检验一：写入侧线性替身")
print("=" * 78)

F_HI_TRUE = f_true[-1]
f_lin = [F_HI_TRUE * s for s in G_FRACS]
label_lin, diag_lin = shape_label(f_lin)
print("线性替身定义：f_lin(s) = f(1.0) · s，与强度分数严格线性，斜率取真读数上端点，")
print("使替身与真读数落在同一量级，差异全部来自型态。")
print("  替身五档 %s，标签 %s" % (", ".join("%.6f" % v for v in f_lin), label_lin))
dev_lin = [abs(v - r) / r for v, r in zip(f_lin, f_true)]
print("  逐档相对偏离 %s" % ", ".join("%.3e" % d for d in dev_lin))
end_dev = [abs(f_lin[0] - f_true[0]) / f_true[0], abs(f_lin[-1] - f_true[-1]) / f_true[-1]]
print("  两个端点相对偏离 %.3e 与 %.3e，至少一个端点显著不同: %s"
      % (end_dev[0], end_dev[1], "是" if max(end_dev) > 0.01 else "否"))
same_type_lin = (label_lin == label_c)
ratio_low_lin, tier_low_lin = mag_tier(min(f_lin), c1_hi)
ratio_high_lin, tier_high_lin = mag_tier(max(f_lin), c1_lo)
print("  替身下甲-3 判定: %s（写入侧 %s 对 C1 侧 %s）"
      % ("同型" if same_type_lin else "不同型", label_lin, label_c))
print("  替身下甲-4 判定: 下端 %.4f 倍 %s，上端 %.4f 倍 %s"
      % (ratio_low_lin, tier_low_lin, ratio_high_lin, tier_high_lin))
changed_low = abs(ratio_low_lin - ratio_low) / ratio_low
print("  下端配对比值相对改变 %.3f，标签由 %s 变为 %s"
      % (changed_low, label_w, label_lin))
print()
print("恒真问题的显式回答：若甲-3 恒真（无论写入侧读数怎样都输出同一个型态标签），")
print("替身会给出与真读数完全相同的标签与结论。本轮替身把标签由 %s 换成 %s，" % (label_w, label_lin))
print("甲-3 的输入因此发生改变，判据并非恒真。甲-4 的下端配对比值亦由 %.4f 变为 %.4f。"
      % (ratio_low, ratio_low_lin))
print("须说明的边界：替身下甲-3 的布尔结论仍为不同型，甲-4 仍为下端同档、上端分离，")
print("故甲-5 靠标签改变与配比改变提供区分力，甲-3 的直接翻转由甲-7 承担。")

ok_5 = (label_lin != label_w and max(end_dev) > 0.01 and changed_low > 0.10)
record("甲-5", "恒真检验一：写入侧换成线性替身后判据须有区分力",
       "替身使写入侧型态标签改变，且至少一个端点与真读数显著不同，甲-4 配比须改变",
       "标签 %s 变为 %s；端点相对偏离 %.3e 与 %.3e；甲-3 替身下仍判不同型，"
       "甲-4 替身下仍为下端同档上端分离，下端配对比值由 %.4f 变为 %.4f（改变 %.1f%%）"
       % (label_w, label_lin, end_dev[0], end_dev[1], ratio_low, ratio_low_lin,
          100.0 * changed_low),
       ok_5, "区分力来自标签改变与配比改变，替身有区分力")

# =====================================================================
# 甲-6 恒真检验二：C1 侧常数值替身
# =====================================================================
print("=" * 78)
print("甲-6 恒真检验二：C1 侧常数值替身")
print("=" * 78)

C1_MEAN = float(np.mean(c1_comp))
c1_const = [C1_MEAN] * 5
label_const, diag_const = shape_label(c1_const)
print("常数值替身定义：补偿比五档全取真读数的算术均值 %.8f" % C1_MEAN)
print("  替身五档 %s，标签 %s" % (", ".join("%.8f" % v for v in c1_const), label_const))
dev_const = [abs(v - r) / r for v, r in zip(c1_const, c1_comp)]
print("  逐档相对偏离 %s" % ", ".join("%.3e" % d for d in dev_const))
print("  型态差异: 真读数 %s，替身 %s" % (label_c, label_const))
same_type_const = (label_w == label_const)
ratio_low_c, tier_low_c = mag_tier(f_lo, c1_const[0])
ratio_high_c, tier_high_c = mag_tier(f_hi, c1_const[0])
changed_low_c = abs(ratio_low_c - ratio_low) / ratio_low
changed_high_c = abs(ratio_high_c - ratio_high) / ratio_high
print("  替身下甲-3 判定: %s（写入侧 %s 对 C1 侧 %s）"
      % ("同型" if same_type_const else "不同型", label_w, label_const))
print("  替身下甲-4 判定: 下端 %.4f 倍 %s，上端 %.4f 倍 %s"
      % (ratio_low_c, tier_low_c, ratio_high_c, tier_high_c))
print("  配比相对改变: 下端 %.1f%%，上端 %.1f%%" % (100.0 * changed_low_c, 100.0 * changed_high_c))
print()
print("恒真问题的显式回答：若甲-3 与甲-4 恒真，替身会给出与真读数相同的标签与配比判定。")
print("本轮替身把 C1 侧标签由 %s 换成 %s，两条配比的相对改变分别为 %.1f%% 与 %.1f%%，"
      % (label_c, label_const, 100.0 * changed_low_c, 100.0 * changed_high_c))
print("判据对替身有区分力。替身下甲-3 与甲-4 的布尔结论不变，")
print("区分力由标签改变与配比改变承担，甲-3 的直接翻转由甲-7 承担。")

ok_6 = (label_const != label_c and max(changed_low_c, changed_high_c) > 0.10)
record("甲-6", "恒真检验二：C1 侧换成常数值替身后判据须有区分力",
       "替身使 C1 侧型态标签改变，且两条配比的相对改变超过 10%",
       "标签 %s 变为 %s；逐档相对偏离 %s；甲-3 替身下仍判不同型，"
       "甲-4 替身下仍为下端同档上端分离；配比相对改变 下端 %.1f%%、上端 %.1f%%"
       % (label_c, label_const, ", ".join("%.3e" % d for d in dev_const),
          100.0 * changed_low_c, 100.0 * changed_high_c),
       ok_6, "替身在型态上与真读数显著不同")

# =====================================================================
# 甲-7 反向对照：写入侧换成与 C1 侧同型的非单调替身
# =====================================================================
print("=" * 78)
print("甲-7 反向对照：写入侧换成与 C1 侧同型的替身")
print("=" * 78)

c1_lo_s, c1_hi_s = min(c1_comp), max(c1_comp)
u_shape = [(v - c1_lo_s) / (c1_hi_s - c1_lo_s) for v in c1_comp]
f_rev = [f_lo + (f_hi - f_lo) * u for u in u_shape]
label_rev, diag_rev = shape_label(f_rev)


def dir_signs(vals):
    return [1 if vals[i + 1] > vals[i] else (-1 if vals[i + 1] < vals[i] else 0)
            for i in range(len(vals) - 1)]


sg_rev = dir_signs(f_rev)
sg_c1 = dir_signs(c1_comp)
print("替身构型：取 C1 侧补偿比五档的归一化形状 u = %s" % ", ".join("%.6f" % u for u in u_shape))
print("  写入侧替身五档 %s" % ", ".join("%.6f" % v for v in f_rev))
print("  替身标签 %s，C1 侧标签 %s" % (label_rev, label_c))
print("  单调性方向核验：替身 %s，C1 侧 %s，一致: %s"
      % (sg_rev, sg_c1, sg_rev == sg_c1))
print("  峰值档核验：替身 argmax = %d，C1 侧 argmax = %d"
      % (int(np.argmax(f_rev)), int(np.argmax(c1_comp))))
print("  替身端点区间 [%.6f, %.6f]，与真读数端点相同: %s"
      % (min(f_rev), max(f_rev), (abs(min(f_rev) - f_lo) < 1e-12 and abs(max(f_rev) - f_hi) < 1e-12)))
same_type_rev = (label_rev == label_c)
ratio_low_r, tier_low_r = mag_tier(min(f_rev), c1_hi)
ratio_high_r, tier_high_r = mag_tier(max(f_rev), c1_lo)
print("  替身下甲-3 判定: %s（写入侧 %s 对 C1 侧 %s）"
      % ("同型" if same_type_rev else "不同型", label_rev, label_c))
print("  替身下甲-4 判定: 下端 %.4f 倍 %s，上端 %.4f 倍 %s"
      % (ratio_low_r, tier_low_r, ratio_high_r, tier_high_r))
print("  翻转核验：真读数下甲-3 判 %s，替身下判 %s，方向相反: %s"
      % ("同型" if same_type else "不同型", "同型" if same_type_rev else "不同型",
         same_type_rev != same_type))
print("  甲-4 是否随之翻转: %s（替身只换型态不换端点值，两配对比值与真读数相同）"
      % (tier_low_r != tier_low or tier_high_r != tier_high))
dev_rev = [abs(v - r) / r for v, r in zip(f_rev, f_true)]
print("  替身与真读数的逐档相对偏离 %s" % ", ".join("%.3e" % d for d in dev_rev))
env_rev = [f_rev[i] for i in range(5) if f_rev[i] >= 0.87]
env_rev_label, _ = shape_label(env_rev) if env_rev else ("空集", {})
print("  替身的上包络子集（f >= 0.87）：%s，标签 %s"
      % (", ".join("%.6f" % v for v in env_rev) if env_rev else "空集", env_rev_label))

ok_7 = same_type_rev
record("甲-7", "反向对照：写入侧换成与 C1 侧同型的非单调替身后甲-3 须翻转",
       "替身与 C1 侧同型且单调性方向一致，甲-3 由不同型翻转为同型",
       "替身五档 %s；标签 %s；单调性方向 %s 与 C1 侧 %s 一致；甲-3 真读数下判 %s，"
       "替身下判 %s；甲-4 在替身下仍为下端同档、上端分离，未随之翻转"
       % (", ".join("%.6f" % v for v in f_rev), label_rev, sg_rev, sg_c1,
          "同型" if same_type else "不同型", "同型" if same_type_rev else "不同型"),
       ok_7, "甲-3 有能翻转它的检验，不是恒真")

# =====================================================================
# 出图
# =====================================================================
print("=" * 78)
print("出图")
print("=" * 78)

fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.8))

# (a) 写入侧转移分数
ax = axes[0, 0]
ax.plot(G_FRACS, f_true, "o-", color="tab:blue", lw=1.8, ms=6,
        label="转移分数 $f$（五档）")
ax.plot([1.25], [f125], "s", color="tab:red", ms=7,
        label="超交换档 1.25 倍（射程外单列）")
ax.axhline(0.99, color="k", ls="--", lw=1.0, label="抽空线 $f = 0.99$")
for x, y in zip(G_FRACS, f_true):
    ax.annotate("%.4f" % y, (x, y), textcoords="offset points", xytext=(0, 8),
                ha="center", fontsize=8)
ax.annotate("%.4f" % f125, (1.25, f125), textcoords="offset points", xytext=(0, -14),
            ha="center", fontsize=8, color="tab:red")
ax.set_xlabel("耦合强度分数 $g / G_{\\rm full}$")
ax.set_ylabel("转移分数 $f$")
ax.set_ylim(-0.06, 1.12)
ax.set_title("(a) 写入侧转移分数：型态为饱和型（单调不减，末端弯平）", fontsize=10)
ax.legend(fontsize=8, loc="lower right")
ax.grid(alpha=0.3)

# (b) C1 侧补偿比
ax = axes[0, 1]
ax.plot(LAM_GRID, c1_comp, "o-", color="tab:green", lw=1.8, ms=6,
        label="补偿比（补偿 / 线张力）")
ipk = int(np.argmax(c1_comp))
ax.plot([LAM_GRID[ipk]], [c1_comp[ipk]], "*", color="tab:red", ms=13,
        label="峰值档 $\\lambda = %.2f$" % LAM_GRID[ipk])
for x, y in zip(LAM_GRID, c1_comp):
    ax.annotate("%.4f" % y, (x, y), textcoords="offset points", xytext=(0, -16),
                ha="center", fontsize=8)
ax.set_xscale("log")
ax.set_xticks(LAM_GRID)
ax.set_xticklabels(["%.2f" % v for v in LAM_GRID])
ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
ax.set_xlabel(r"$\lambda$")
ax.set_ylabel("补偿比")
ax.set_ylim(0.006, 0.033)
ax.set_title("(b) C1 侧补偿比：非单调，峰值在中段（$\\lambda = %.2f$）" % LAM_GRID[ipk],
              fontsize=10)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# (c) 端点档位对表
ax = axes[1, 0]
ax.set_xscale("log")
ax.set_xlim(0.005, 3.0)
ax.set_ylim(-0.6, 0.6)
pts = {
    "C1 补偿比下端\n%.4f" % c1_lo: (c1_lo, -0.16, "tab:green"),
    "C1 补偿比上端\n%.4f" % c1_hi: (c1_hi, 0.16, "tab:green"),
    "写入侧 $f$ 下端\n%.4f" % f_lo: (f_lo, -0.16, "tab:blue"),
    "写入侧 $f$ 上端\n%.4f" % f_hi: (f_hi, 0.16, "tab:blue"),
}
for name, (x, y, col) in pts.items():
    ax.plot([x], [0.0], "o", color=col, ms=9)
    ax.annotate(name, (x, 0.0), textcoords="offset points",
                xytext=(0, 14 if y > 0 else -30), ha="center", fontsize=8)
ax.plot([f_lo, c1_hi], [0.22, 0.22], "-", color="tab:green", lw=2.4)
ax.annotate("下端配对：%.2f 倍，同档" % ratio_low, ((f_lo + c1_hi) / 2.0, 0.22),
            textcoords="offset points", xytext=(0, 7), ha="center", fontsize=9,
            color="tab:green")
ax.plot([f_hi, c1_lo], [-0.22, -0.22], "-", color="tab:red", lw=2.4)
ax.annotate("上端配对：%.1f 倍，分离" % ratio_high, ((f_hi + c1_lo) / 2.0, -0.22),
            textcoords="offset points", xytext=(0, -16), ha="center", fontsize=9,
            color="tab:red")
ax.axvspan(c1_lo, c1_hi, color="tab:green", alpha=0.10)
ax.axvspan(f_lo, f_hi, color="tab:blue", alpha=0.10)
ax.set_xlabel("端点值（无量纲，对数轴）")
ax.set_yticks([])
ax.set_title("(c) 端点档位对表：下端同档，上端分离", fontsize=10)
ax.grid(alpha=0.25, axis="x")

# (d) 两条恒真替身与反向替身的对照
ax = axes[1, 1]
axd = ax.inset_axes([0.04, 0.52, 0.42, 0.36])
axd.plot(G_FRACS, f_true, "o-", color="tab:blue", lw=1.6, ms=5, label="真读数 $f$")
axd.plot(G_FRACS, f_lin, "s--", color="tab:orange", lw=1.4, ms=5,
         label="甲-5 线性替身")
axd.plot(G_FRACS, f_rev, "^--", color="tab:purple", lw=1.4, ms=5,
         label="甲-7 同型替身")
axd.set_title("写入侧：真读数与两条替身", fontsize=8, pad=2)
axd.set_xlabel("强度分数", fontsize=7)
axd.tick_params(labelsize=7)
axd.legend(fontsize=6.5, loc="lower right")
axd.grid(alpha=0.3)
axd.set_ylim(-0.08, 1.12)

axc = ax.inset_axes([0.56, 0.52, 0.40, 0.36])
axc.plot(LAM_GRID, c1_comp, "o-", color="tab:green", lw=1.6, ms=5, label="真读数")
axc.plot(LAM_GRID, c1_const, "s--", color="tab:red", lw=1.4, ms=5,
         label="甲-6 常数替身")
axc.set_title("C1 侧：真读数与常数替身", fontsize=8, pad=2)
axc.set_xlabel(r"$\lambda$", fontsize=7)
axc.set_xscale("log")
axc.set_xticks(LAM_GRID)
axc.set_xticklabels(["%.2f" % v for v in LAM_GRID], fontsize=6.5)
axc.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
axc.tick_params(labelsize=7)
axc.legend(fontsize=6.5, loc="upper right")
axc.grid(alpha=0.3)
axc.set_ylim(0.006, 0.034)

ax.text(0.03, 0.42, "型态对照", fontsize=9, transform=ax.transAxes)
txt = ("写入侧真读数：饱和型（单调不减，末端增量比 %.3f）\n"
       "写入侧线性替身：%s（标签改变，甲-5 有区分力）\n"
       "写入侧同型替身：%s（甲-3 翻转）\n"
       "C1 侧真读数：%s\n"
       "C1 侧常数替身：%s（标签改变，甲-6 有区分力）"
       % (diag_w["d_ratio"], label_lin, label_rev, label_c, label_const))
ax.text(0.03, 0.05, txt, fontsize=7.6, transform=ax.transAxes, va="bottom",
        linespacing=1.5)
ax.set_xticks([])
ax.set_yticks([])
ax.set_title("(d) 两条恒真替身与反向替身：型态差异", fontsize=10, pad=26)

fig.suptitle("第 735 号 构造轮案甲：标度对表（写入侧与 C1 侧）", fontsize=12)
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
    print("%-7s %-46s %s" % (cid, name[:46], "通过" if ok else "判否"))
    if ok:
        npass += 1
print()
print("合计 %d/%d 通过" % (npass, len(RESULTS)))
print("甲-3 主判定: %s" % ("同型" if same_type else "不同型"))
print("甲-3 上包络次级判定: %s" % ("同型" if env_w_label == env_c_label else "不同型"))
print("甲-4 整体判定: %s" % ("全部同档" if all_same_tier else "下端同档、上端分离"))
print("甲-7 替身下甲-3: %s" % ("同型" if same_type_rev else "不同型"))
print("前提守恒三条: %s" % ("全部成立" if KEEP_ALL else "存在漂移"))
print("耗时 %.1f 秒" % (time.time() - T_START))
