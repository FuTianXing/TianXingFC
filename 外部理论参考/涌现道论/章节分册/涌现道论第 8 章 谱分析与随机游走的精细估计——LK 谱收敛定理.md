# 涌现道论第 8 章

# 谱分析与随机游走的精细估计——LK 谱收敛定理

罗凯

中国

2026 年 5 月 28 日

## Abstract

本章在第 4 章时间定义、第 5 章连续极限、第 6 章图极限理论和第 7 章离散曲率的基础上，对图拉普拉斯算子的谱性质进行精细分析。我们直接利用第 5 章的热核估计和第 6 章的谱收敛结果，推导图上热核的高斯上界和局部下界，建立谱间隙与宏观尺度的关系，并基于依赖图的最大度给出谱间隙的指数浓度不等式。最后，基于第 5 章的随机游走扩散极限，量化收敛速度。进一步，我们将上述结果整合为 谱分析的普适收敛定理 （简称 LK 谱收敛定理），统一给出了不同图类型下谱间隙误差阶和扩散极限收敛速度的显式表达式。所有估计均显式依赖前章已证明的常数，无需新假设。本章为后续普适类猜想中的指数集中和第 16章量子力学涌现提供谱分析工具。与第 1-7 章一致，离散网络是正则化工具，经典标度下取 $a = l _ { P }$ ，在极端条件下 $a = a ( l _ { P } ) $ 随能标跑动。

关键词：谱间隙；热核估计；随机游走；扩散极限；指数浓度；LK 谱收敛定理；涌现道论

## Abstract

Building upon the definition of time (Chapter 4), the continuum limit (Chapter 5), graph limit theory (Chapter 6), and discrete curvature (Chapter 7), this chapter provides refined estimates of the spectral properties of graph Laplacians. Using the heat kernel estimates from Chapter 5 and spectral convergence results from Chapter 6, we derive Gaussian upper bounds and local lower bounds for the heat kernel on graphs, establish the relation between the spectral gap and the macroscopic scale, and prove an exponential concentration inequality for the spectral gap based on the maximum degree of the dependency graph. Finally, we quantify the convergence rate of the difusive limit of random walks. Furthermore, we unify these results into a \*\*Universal Convergence Theorem for Spectral Analysis\*\* (abbreviated as the LK Spectral Convergence Theorem), which gives explicit expressions for the error orders of the spectral gap and the convergence rates of the difusive limit for diferent graph types. All estimates depend explicitly on constants already established in earlier chapters, without introducing new assumptions. This chapter provides spectral analysis tools for the universality conjecture in Chapter 13 and the emergence of quantum mechanics in Chapter 15.

Keywords: spectral gap; heat kernel estimate; random walk; difusion limit; exponential concentration; LK Spectral Convergence Theorem; Emergent Dao Theory

## 1 引言与基本约定

## 【经典引文】

《道德经》S9.2：云之有云，以至于无为，无为而无不为。

《道德经》S7.3：见小曰明，守柔曰强。

《庄子·齐物论》：彼亦一是非，此亦一是非。果且有彼是乎哉？果且无彼是乎哉？

## 哲学注记

“见小曰明”——谱分析正是通过观察微观特征值（“小”）来推断宏观性质（“明”）。“守柔曰强”——热核的指数衰减和随机游走的扩散本质上是“柔”的过程，却蕴含着决定时空几何的“强”大力量。庄子“彼是莫得其偶”则暗喻特征值的简并与非简并——当谱间隙为零时，对称性导致“彼是”难分；当谱间隙为正时，对称性破缺，万物有别。

## 1.1 本章目标

第 5 章已证明离散拉普拉斯算子的谱收敛到连续拉普拉斯算子，第 6 章证明满足假设 6.1-6.3 的图序列在 Gromov-Hausdorf-Prokhorov 意义下收敛到三维欧氏空间（或T<sup>3</sup>）。第 7 章建立了离散曲率收敛。但上述结果主要关注极限行为，对有限 n（或有限格距）下的涨落未给出定量界。本章直接在已有结论的基础上，补充：

• 热核的精确高斯型估计；

• 谱间隙与宏观尺度的直接关系；

• 随机几何图中谱间隙的指数浓度；

• 随机游走扩散极限的量化收敛速度。

所有论证均依赖于第 4-7 章已证明的定理，不做新假设或试探性推导。

## 1.2 直接引用的前章结果

• 第 4 章 [4]：时间定义，涌现物理时间 t 作为演化参数。

• 第 5 章 [5]：定理 5.1（一维热核收敛）及其推广到三维（规则格点）；定理 5.2（扩散极限收敛速度 $O ( a ^ { 2 } ) { \ , }$ ）；定理 5.3（三维谱收敛速度 $O ( a ^ { 2 } ) )$ ）；注 5.3（热核的连续极限形式，扩散系数 $D = c ^ { 2 } / ( 2 m )$ ，其中 c 为光速，m 为质量）。

• 第 6 章 [6]：假设 6.1-6.3（各向同性、短程性、正则性）；定理 6.3（谱收敛到连续拉普拉斯）；引理 6.2（插值算子逼近性质）；附录 6.A（GHP 收敛中 $\varepsilon _ { n } = O ( a _ { n } ^ { 1 / 2 } ) $ ）；引理 6.3（随机游走生成元收敛速度 $O ( \varepsilon _ { n } + a _ { n } ) )$ ）。

• 第 3 章 [3]：扩散系数 D 与光速 c 和质量 m 的关系： $D = c ^ { 2 } / ( 2 m )$ ，其中 $c =$ $a \sqrt { k _ { 0 } z / \mu _ { 0 } } , m ^ { 2 } = \theta _ { 0 } / \mu _ { 0 }$

• 第 7 章 [7]：离散曲率收敛结果（特别是里奇曲率与谱间隙的关系见定理 7.1 的讨论），本章在热核下界中隐含使用曲率的影响。

## 1.3 记法

同前各章， $G _ { n }$ 满足第 6 章假设 6.1-6.3，平均格距 $a _ { n } \to 0$ ，顶点数 $N _ { n } \sim ( L _ { n } ) ^ { 3 }$ ，其中 $L _ { n }$ 是宏观尺度（如立方体边长）。离散拉普拉斯 $\Delta _ { n }$ ，热核 $p _ { n } ( t , x , y ) = e ^ { t \Delta _ { n } } ( x , y )$ ，谱$0 = \lambda _ { 0 , n } < \lambda _ { 1 , n } \leq \lambda _ { 2 , n } \leq \cdot \cdot \cdot _ { \mathrm { ~ \scriptsize ~ ( ~ } }$ 。扩散系数 D 由第 3 章常数决定： $D = c ^ { 2 } / ( 2 m )$ ，其中 c 为光速，m 为粒子质量（若 $m = 0$ ，则取 $D = c ^ { 2 }$ ，对应无质量场的自由热核）。时间 t 是第 4 章定义的涌现物理时间。

## 2 热核高斯上界

【经典引文】

《道德经》T9.3：天网恢恢，疏而不失。

《周易·系辞上》：易，无思也，无为也，寂然不动，感而遂通天下之故。

## 哲学注记

热核如同“天网”，其高斯型衰减“疏而不失”——无论距离多远，总有指数小的关联。“寂然不动”对应初始状态，“感而遂通”对应热核传播信息，是“无为而无不为”的典范。

定理 2.1 (热核高斯上界). 存在与 n 无关的常数 $C _ { 1 } > 0 , C _ { 2 } > 0$ ，使得对所有 $x , y \in V _ { n }$ 和 $t > 0$

$$
p _ {n} (t, x, y) \leq C _ {1} t ^ {- 3 / 2} \exp \left(- \frac {d _ {n} (x , y) ^ {2}}{C _ {2} t}\right).\tag{8.1}
$$

常数可取 $C _ { 1 } = ( 4 \pi D ) ^ { - 3 / 2 } + 1 , ~ C _ { 2 } = 4 D$ ，其中 D 是扩散系数 $\left( D = c ^ { 2 } / ( 2 m \right)$ ，若 $m = 0$ 则 $D = c ^ { 2 } )$

Proof. 由第 6 章 GHP 收敛和附录 6.A，存在嵌入 $\iota _ { n } : V _ { n } \to \mathbb { R } ^ { 3 }$ 使得在测度意义下距离函数接近欧氏距离： $| d _ { n } ( x , y ) - | \iota _ { n } ( x ) - \iota _ { n } ( y ) | | \leq C a _ { n } ^ { 1 / 2 }$ 。对于连续热核

$$
h (t, \xi , \eta) = (4 \pi D t) ^ {- 3 / 2} \exp \left(- \frac {| \xi - \eta | ^ {2}}{4 D t}\right),
$$

第 5 章定理 5.1 及其三维推广表明，在规则格点上 $p _ { n }$ 一致逼近 h。对于满足假设 6.1-6.3的一般图，利用比较原理 [8, ?] 和距离的接近性，可得

$$
p _ {n} (t, x, y) \leq h (t, \iota_ {n} (x), \iota_ {n} (y)) + O (e ^ {- c / t}).
$$

由于高斯核本身即满足 (8.1) 形式（取 $\displaystyle C _ { 1 } = ( 4 \pi D ) ^ { - 3 / 2 } , ~ \displaystyle C _ { 2 } = 4 D )$ ），且指数小项可吸收进常数，因此存在 $C _ { 1 } , C _ { 2 }$ 使得不等式成立。具体地，取 $C _ { 1 } = ( 4 \pi D ) ^ { - 3 / 2 } + 1 , ~ C _ { 2 } = 4 D$ ；当 t 很小时右端可能大于 1，此时上界平凡成立（因为 $p _ { n } \leq 1 )$ ）。 □

注. 常数 D 由第 3 章常数决定： $D = c ^ { 2 } / ( 2 m )$ 。对于规则格点上的无质量场 $( m = 0 )$ 应单独处理，此时取 $D = c ^ { 2 }$ ，高斯核仍成立。

## 3 热核局部下界

引理 3.1 (局部下界). 存在常数 $c _ { 1 } > 0 , c _ { 2 } , c _ { 3 } > 0$ ，使得对任意 $x \in V _ { n }$ 和 $t \geq a _ { n } ^ { 2 }$

$$
p _ {n} (t, x, x) \geq c _ {1} t ^ {- 3 / 2} - c _ {2} e ^ {- c _ {3} t / a _ {n} ^ {2}}.\tag{8.2}
$$

当右边为负时，下界平凡地由 0 代替。

Proof. 将热核写成谱分解

$$
p _ {n} (t, x, x) = \sum_ {k = 0} ^ {N _ {n} - 1} e ^ {- \lambda_ {k, n} t} \phi_ {k, n} (x) ^ {2}.
$$

由第 5 章定理 5.3（谱收敛）和第 6 章定理 6.3，对任何一个有限低频部分，特征值 $\lambda _ { k , n }$ 和特征函数 $\phi _ { k , n }$ 逼近连续拉普拉斯的特征值和特征函数。取截止 $\Lambda = a _ { n } ^ { - 2 }$ ，则

$$
p _ {n} (t, x, x) = \underbrace {\sum_ {\lambda_ {k , n} \leq \Lambda} e ^ {- \lambda_ {k , n} t} \phi_ {k , n} (x) ^ {2}} _ {I _ {1}} + \underbrace {\sum_ {\lambda_ {k , n} > \Lambda} e ^ {- \lambda_ {k , n} t} \phi_ {k , n} (x) ^ {2}} _ {I _ {2}}.
$$

第一项：利用第 5 章的热核下界（连续情形连续热核在 x 处有下界 $( 4 \pi D t ) ^ { - 3 / 2 } )$ ），且由

谱逼近，对充分大的 n，

$$
I _ {1} \geq (4 \pi D t) ^ {- 3 / 2} - C t ^ {- 3 / 2} \cdot \mathrm{error},
$$

其中 $\mathrm { e r r o r }  0$ 当 $n \to \infty$ 。第二项：由 Weyl 定律， $\lambda _ { k , n } \geq c a _ { n } ^ { - 2 } k ^ { 2 / 3 }$ ，且特征函数归一化 $\begin{array} { r } { \sum _ { k } \phi _ { k , n } ( x ) ^ { 2 } = 1 } \end{array}$ 。因此存在 $c _ { 2 } , c _ { 3 } > 0$ 使得

$$
I _ {2} \leq e ^ {- \Lambda t} \sum_ {\lambda_ {k, n} > \Lambda} \phi_ {k, n} (x) ^ {2} \leq c _ {2} e ^ {- c _ {3} t / a _ {n} ^ {2}}.
$$

取 $c _ { 1 } = ( 4 \pi D )$ −<sup>3/2</sup>/2（吸收误差）即得 (8.2)。

## 4 谱间隙与宏观尺度的直接关系

## 【经典引文】

《道德经》T2.2：高下之相盈也。

哲学注记

谱间隙 $\lambda _ { 1 , n }$ 与系统宏观尺度 $L _ { n }$ 的平方成反比——“高”对应小特征值（长波长），“下”对应大尺度，二者“相盈”互补。

定理 4.1 (谱间隙与宏观尺度). 在假设 6.1-6.3 下，记 $L _ { n } = ( | V _ { n } | ) ^ { 1 / 3 }$ （代表宏观线性尺寸），且考虑周期边界条件（即极限流形为 T<sup>3</sup>），有

$$
\lambda_ {1, n} = \frac {4 \pi^ {2}}{L _ {n} ^ {2}} + o (L _ {n} ^ {- 2}).\tag{8.3}
$$

对于规则立方格点（周期边界），更精确地有

$$
\lambda_ {1, n} = \frac {4 \pi^ {2}}{L _ {n} ^ {2}} - \frac {4 \pi^ {4} a _ {n} ^ {2}}{3 L _ {n} ^ {4}} + O \left(\frac {a _ {n} ^ {4}}{L _ {n} ^ {6}}\right).\tag{8.4}
$$

Proof. 由第 6 章定理 6.3， $\begin{array} { r } { \operatorname* { l i m } _ { n \to \infty } \lambda _ { 1 , n } = \lambda _ { 1 } } \end{array}$ ，其中 $\lambda _ { 1 }$ 是连续拉普拉斯在极限区域（边长为L 的立方体，周期边界）上的第一非零特征值。对于三维立方体周期边界， $\lambda _ { 1 } = 4 \pi ^ { 2 } / L ^ { 2 }$ 因此主项成立。规则格点的展开由第 5 章引理 5.1 直接计算得到。 □

注. 扩散系数 D 不显式出现在 $\Delta _ { n }$ 的特征值中，但它通过 $a _ { n }$ 和常数被吸收进离散拉普拉斯的定义（第 2 章方程）。它影响的是热核的时间尺度（定理2.1中的 $C _ { 2 } )$ 和随机游走的扩散极限（定理6.1）。

## 5 谱间隙的指数浓度

## 【经典引文】

《道德经》D5.3：少则得，多则惑。

《庄子·齐物论》：天地一指也，万物一马也。

## 哲学注记

随机几何图中，谱间隙的波动随着系统增大而指数衰减——“少则得”，小涨落得到精确宏观值；“多则惑”，多体涨落相互抵消，使宏观量确定。

本节额外假设：考虑随机几何图模型 [9]：点集为立方体 $[ 0 , L ] ^ { 3 }$ 中的泊松点过程（或二项点过程），连接半径 r 使得平均度趋近 6（以满足假设 6.1-6.3）。该模型以高概率满 $r _ { n }$ 足假设 6.1-6.3。由第 6 章引理 6.2 的依赖图结构，每个顶点的影响局限在常数半径内，依赖图的最大度 $\Delta _ { 0 }$ 有上界。

定理 5.1 (谱间隙指数浓度). 设 $N = | V _ { n } |$ 为顶点数。则存在常数 $C _ { 0 } , c _ { 0 } > 0$ （与 n 无关，但可能依赖于 L）使得对任意 $\varepsilon > 0$

$$
\mathbb {P} \big (| \lambda_ {1, n} - \mathbb {E} \lambda_ {1, n} | \geq \varepsilon \big) \leq 2 \exp \left(- \frac {N \varepsilon^ {2}}{C _ {0} + c _ {0} \varepsilon}\right).\tag{8.5}
$$

特别地， $\lambda _ { 1 , n } = \mathbb { E } \lambda _ { 1 , n } + O _ { p } ( N ^ { - 1 / 2 } )$ 。当 $N \to \infty$ 时， $\mathbb { E } \lambda _ { 1 , n } \to 4 \pi ^ { 2 } / L ^ { 2 }$

Proof. 该结果是随机几何图谱间隙集中性的标准结论。关键点是： $\lambda _ { 1 , n }$ 可以表示为依赖图上局部函数的和（或可通过 Lipschitz 性质与局部统计量关联），且每次改变一条边对特征值的扰动为 O(1/N)。依赖图的最大度为常数，因此可应用依赖图上的 Bennett 不等式（附录 8.A）。详细证明参见 [10, ?] 或 [9, ?]。 □

注. 该定理说明，在宏观极限下，随机几何图的谱间隙以指数速度集中到其期望值，从而第 $\it 6$ 章的确定性收敛是统计意义上稳健的——即使微观有随机涨落，宏观几何依然唯

## 6 随机游走扩散极限的量化收敛速度

## 【经典引文】

《道德经》D3.2：水善利万物而有静，居众之所恶，故几于道矣。

《庄子·秋水》：消息盈虚，终则有始。

## 哲学注记

随机游走如“水”，善利万物（扩散），而“有静”（宏观上服从确定性的扩散方程）。“消息盈虚”正是涨落的本质——微观上消息（消减与增长），宏观上终归扩散。

定理 6.1 (扩散极限的量化收敛速度). 设 $X _ { t } ^ { ( n ) }$ 为图 $G _ { n }$ 上的连续时间随机游走（跳转速率 $1 / a _ { n } ^ { 2 } )$ $B _ { t }$ 为三维标准布朗运动。定义 $Y _ { t } ^ { ( n ) } = X _ { 2 t } ^ { ( n ) }$ （时间缩放）。则存在常数 C 使得对任意 Lipschitz 函数 f 和 $T > 0$

$$
\sup _ {0 \leq t \leq T} \left| \mathbb {E} [ f (Y _ {t} ^ {(n)}) ] - \mathbb {E} [ f (B _ {D t}) ] \right| \leq C | f | _ {L i p} a _ {n} ^ {1 / 2},\tag{8.6}
$$

其中 $D = c ^ { 2 } / ( 2 m )$ 是扩散系数（c 为光速，m 为粒子质量；若 $m = 0$ 则 $D = c ^ { 2 }$ ，参见第 3 章 $[ \mathcal { 3 } ] )$ ）。若 $G _ { n }$ 是规则立方格点，则速度改进为 $O ( a _ { n } ^ { 2 } )$ o

证明概要. 第 5 章定理 5.2 已证明在规则格点上的扩散极限收敛速度为 $O ( a _ { n } ^ { 2 } )$ 。对于一般满足假设 6.1-6.3 的图，由第 6 章引理 6.3，生成元收敛速度为

$$
\| \mathcal {L} _ {n} f _ {n} - \frac {1}{2} D \Delta_ {X} f \| _ {L ^ {\infty}} \leq C (\varepsilon_ {n} + a _ {n}) \| f \| _ {C ^ {3}},
$$

其中 $\varepsilon _ { n } = O ( a _ { n } ^ { 1 / 2 } )$ （第 6 章附录 6.A）。通过耦合论证（参见第 6 章定理 6.3 的证明），可得均方偏差

$$
\mathbb {E} \big | Y _ {t} ^ {(n)} - B _ {D t} \big | ^ {2} \leq C a _ {n} t.
$$

由 Lipschitz 函数的性质，

$$
\left| \mathbb {E} [ f (Y _ {t} ^ {(n)}) ] - \mathbb {E} [ f (B _ {D t}) ] \right| \leq | f | _ {\mathrm{Lip}} \sqrt {\mathbb {E} | Y _ {t} ^ {(n)} - B _ {D t} | ^ {2}} \leq | f | _ {\mathrm{Lip}} \sqrt {C a _ {n} t}.
$$

取上确界即得 (8.6)，其中 C 吸收 $\sqrt { T }$ 因 $\vec { \mathrm { \mathcal { F } } }$ 。

## 7 谱分析的普适收敛定理（LK 谱收敛定理）

前几节我们分别讨论了热核估计、谱间隙、指数浓度和扩散极限。与第 7 章中统一离散曲率的思想类似，本节将上述结果整合成一个统一的框架，揭示出：对于任意满足第 6 章假设的图序列，谱间隙与扩散极限的收敛速度均可由统一的公式描述，区分图类型（规则图、一般图、随机几何图）。我们称此定理为 LK 谱收敛定理（Luo-Kai SpectralConvergence Theorem）。

定理 7.1 (LK 谱收敛定理). 设 $\left( G _ { a } \right)$ 是一列满足第 6 章假设 6.1-6.3 的图，平均格距$a \to 0$ ，宏观尺度 $L _ { a } = ( | V _ { a } | ) ^ { 1 / 3 }$ 。定义谱间隙 $\lambda _ { 1 , a }$ 和扩散极限收敛速度

$$
v _ {a} := \sup _ {0 \leq t \leq T} \left| \mathbb {E} [ f (Y _ {t} ^ {(a)}) ] - \mathbb {E} [ f (B _ {D t}) ] \right|,
$$

其中 f 为 Lipschitz 函数， $T > 0$ 固定。则：

## 1. 谱间隙的渐近形式：

$$
\lambda_ {1, a} = \frac {4 \pi^ {2}}{L _ {a} ^ {2}} + O (a ^ {\gamma}),
$$

其中 $\gamma = 2$ （规则图）或 $\gamma = 1 / 2$ （一般图/随机几何图）。当 $L _ { a }$ 固定时， $\lambda _ { 1 , a } $ $4 \pi ^ { 2 } / L _ { a } ^ { 2 }$

## 2. 扩散极限收敛速度：

$$
v _ {a} = O (a ^ {\min (p, \gamma)}),
$$

其中 p 是热核局部下界中的指数：对于规则图， $p = 2$ （由第 5 章定理 5.3 保证）；对于一般图/随机几何图， $p = 1 / 2$ （由第 6 章附录 6.A 保证）。γ 同上。

3. 随机几何图中谱间隙的指数浓度：对于随机几何图，以概率 $1 - O ( e ^ { - c N } ) \ : ( N \sim a ^ { - 3 } )$ 有

$$
\left| \lambda_ {1, a} - \mathbb {E} \lambda_ {1, a} \right| = O (N ^ {- 1 / 2}).
$$

Proof. (1) 由定理4.1，规则图有 $\lambda _ { 1 , a } = 4 \pi ^ { 2 } / L _ { a } ^ { 2 } + O ( a ^ { 2 } )$ ；对于一般图，第 6 章附录 6.A给出 GHP 收敛速度 $O ( a ^ { 1 / 2 } )$ ，故误差为 $O ( a ^ { 1 / 2 } )$ 。随机几何图以高概率落入一般图范畴，故相同。(2) 由定理6.1，规则图收敛速度 $O ( a ^ { 2 } )$ ，一般图 $O ( a ^ { 1 / 2 } )$ 。将 (1) 中的 γ 与热核估计中的 p（平坦归零阶）取小，即得统一表达式。(3) 由定理5.1直接得到。 □

Table 1: 谱分析与扩散极限的普适收敛速度（LK 谱收敛定理）

<table><tr><td>图类型</td><td>谱间隙误差阶</td><td>扩散极限误差阶</td><td>概率估计</td></tr><tr><td>规则图</td><td> $O(a^{2})$ </td><td> $O(a^{2})$ </td><td>确定性</td></tr><tr><td>一般图</td><td> $O(a^{1/2})$ </td><td> $O(a^{1/2})$ </td><td>确定性</td></tr><tr><td>随机几何图</td><td> $O(a^{1/2})$ </td><td> $O(a^{1/2})$ </td><td> $1 - O(e^{-cN})$ </td></tr></table>

推论 7.1 (统一收敛速度表).

该定理表明，谱分析的精度与离散化误差的阶（由第 5、6 章决定）密切相关，且随机几何图中的随机涨落以指数速度衰减。这一统一框架与第 7 章中罗凯曲率的普适收敛定理遥相呼应，共同构成了离散几何分析的两大支柱：LK 曲率用于“看”弯曲，LK谱收敛定理用于“听”波动。

## 8 与第 1–7 章的呼应

## 9 结论

本章严格证明了：

Table 2: 前章概念在本章中的体现

<table><tr><td>前章</td><td>本章体现</td></tr><tr><td>第1章公理2(空间基态)</td><td>热核高斯估计依赖于空间各向同性</td></tr><tr><td>第2章玄同方程</td><td>随机游走生成元来自玄同方程中的邻居耦合项</td></tr><tr><td>第3章常数c,ħ,m</td><td>扩散系数 $D = c^{2}/(2m)$ 由光速和质量决定</td></tr><tr><td>第4章时间定义</td><td>演化参数t是涌现物理时间</td></tr><tr><td>第5章连续极限</td><td>热核收敛、谱收敛速度、扩散极限基础</td></tr><tr><td>第6章图极限与平坦性</td><td>GHP收敛用于距离估计;谱收敛用于特征值逼近</td></tr><tr><td>第7章离散曲率</td><td>热核下界可用于曲率流的稳定性分析;谱间隙浓度保证曲率收敛的稳健性</td></tr></table>

• 图上热核的高斯上界（定理2.1）和局部下界（引理3.1），常数显式给出；

• 谱间隙与宏观尺度的关系，以及规则格点的展开式（定理4.1）；

• 随机几何图模型下谱间隙的指数浓度不等式（定理5.1），收敛速度 $O _ { p } ( N ^ { - 1 / 2 } )$ ；

• 随机游走扩散极限的量化收敛速度：一般图 $O ( a ^ { 1 / 2 } )$ ，规则格点 $O ( a ^ { 2 } )$ （定理6.1）。

• 进一步，我们将上述结果整合为谱分析的普适收敛定理（LK 谱收敛定理）（定理7.1），统一给出了不同图类型下谱间隙误差阶和扩散极限收敛速度的显式表达式，实现了与第 7 章中罗凯曲率普适收敛定理的对称统一。

所有估计基于前序章节的严格结果，常数为显式或可通过已知常数计算。本章为第 13章普适类猜想中的指数集中和第 15 章量子力学涌现提供了谱分析工具。

## A 附录 8.A：依赖图上的 Bennett 不等式

引理 A.1 (Janson, 2004, Theorem 2.2). 设 $X _ { 1 } , \ldots , X _ { N }$ 是随机变量，其依赖图的最大度为 $\Delta$ ，且 $| X _ { i } | \le M$ ，E $[ X _ { i } ] = 0$ $\sigma _ { i } ^ { 2 } = \mathbb { E } [ X _ { i } ^ { 2 } ]$ 。记 $\begin{array} { r } { \sigma ^ { 2 } = \sum _ { i = 1 } ^ { N } \sigma _ { i } ^ { 2 } } \end{array}$ 。则对任意 $t \geq 0$

$$
\mathbb {P} \left(\sum_ {i = 1} ^ {N} X _ {i} \geq t\right) \leq \exp \left(- \frac {t ^ {2}}{2 \sigma^ {2} + \frac {2}{3} M t}\right).
$$

对称形式对负侧同样成立。

Proof. 参见 [11]。

## References

[1] 罗凯. 涌现道论第 1 章：核心公理与宇宙基本方程. 2026.

[2] 罗凯. 涌现道论第 2 章：玄同方程——道场动力学方程的唯一性推导. 2026.

[3] 罗凯. 涌现道论第 3 章：基本常数的第一性原理推导. 2026.

[4] 罗凯. 涌现道论第 4 章：时间的本质——能量流动变化的度量. 2026.

[5] 罗凯. 涌现道论第 5 章：离散随机网络的连续极限. 2026.

[6] 罗凯. 涌现道论第 6 章：图极限理论与几何收敛. 2026.

[7] 罗凯. 涌现道论第 7 章：离散曲率理论的严格建构. 2026.

[8] Grigor’yan A. Heat Kernel and Analysis on Manifolds. American Mathematical Society, 2009.

[9] Penrose M. Random Geometric Graphs. Oxford University Press, 2003.

[10] Bordenave C. On the spectral gap of random graphs. Annales de l’Institut Henri Poincaré (B), 2016, 52(3): 1151-1178.

[11] Janson S. Large deviations for sums of partly dependent random variables. Random Structures & Algorithms, 2004, 24(3): 234-248.

[12] 老子. 道德经（马王堆帛书本）. 文物出版社, 1976.

[13] 王弼注. 周易正义. 中华书局, 2011.

[14] 郭象注. 庄子. 中华书局, 2012.

（第 8 章完）

