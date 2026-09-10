# -*- coding: utf-8 -*-
"""开放问题总览交叉引用检查器（常驻）。

与 `_tools/check_latex_tears.py` 配套：前者查文本损坏，本器查引用图谱。

检查项：
1. 条目数与唯一键（发现编号重复或区块界定错误）。
2. 前向引用总数与单向引用对。
3. 零入引条目（无任何其他条目引用它）。
4. 裸引用「第 N 条」（跨区有歧义，且图谱识别不到）。

用法：
    python _tools/check_cross_refs.py            # 默认查开放问题总览
    python _tools/check_cross_refs.py <路径>      # 指定文件
    python _tools/check_cross_refs.py --full     # 额外打印单向与零入引明细

退出码：0 无异常，1 检出编号重复或裸引用。

关键陷阱（2026-09-11 踩过）：
- `## 结构观察` 与 `## 已关闭区` 也用编号列表（1. 2. 3.），若按「最近的 ## 标题」
  归层会覆盖真实的 P4 前几条，条目数少 4 且产生假阳性引用。必须显式界定区间。
- 裸引用「承接第 4 条」在跨区场景有歧义（P4 区亦有第 4 条），且正则识别不到，
  会造成引用图谱缺边。
"""
import re
import sys

DEFAULT = '内部草稿/开放问题总览：元相子理论待解决问题清单.md'

# 活跃条目区：从 P0 区标题到「结构观察」标题之间
SEC_START = '## P0'
SEC_END = ('## 结构观察', '## 已关闭区')


def load(path):
    with open(path, 'rb') as f:
        return f.read().decode('utf-8')


def build(path):
    lines = load(path).split('\n')
    lo = next((i for i, l in enumerate(lines) if l.startswith(SEC_START)), None)
    if lo is None:
        return None, '未找到 P0 区标题'
    hi = len(lines)
    for tag in SEC_END:
        for i, l in enumerate(lines):
            if l.startswith(tag) and i > lo:
                hi = min(hi, i)
    sub = lines[lo:hi]

    starts = [(i, l) for i, l in enumerate(sub) if re.match(r'^\d+\. ', l)]
    sec = None
    blocks = []
    for k, (i, l) in enumerate(starts):
        for j in range(i, -1, -1):
            m = re.match(r'^## (P\d)', sub[j])
            if m:
                sec = m.group(1)
                break
        end = len(sub)
        for j in range(i + 1, len(sub)):
            if re.match(r'^#{2,3} ', sub[j]):
                end = j
                break
        if k + 1 < len(starts) and starts[k + 1][0] < end:
            end = starts[k + 1][0]
        key = '%s-%s' % (sec, re.match(r'^(\d+)\.', l).group(1))
        blocks.append((key, '\n'.join(sub[i:end])))
    return blocks, None


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    full = '--full' in sys.argv
    path = args[0] if args else DEFAULT

    blocks, err = build(path)
    if err:
        print('错误：%s' % err)
        return 1

    keys = [b[0] for b in blocks]
    dup = sorted({k for k in keys if keys.count(k) > 1})
    print('条目 %d 条 | 唯一 %d' % (len(keys), len(set(keys))))
    if dup:
        print('编号重复：%s' % '、'.join(dup))

    ks = set(keys)
    fwd = {}
    for key, body in blocks:
        refs = {m.group(1) for m in re.finditer(r'(P\d-\d+)', body)}
        fwd[key] = {r for r in refs if r != key and r in ks}

    one = [(a, b) for a in fwd for b in fwd[a] if a not in fwd.get(b, set())]
    inbound = {k: 0 for k in ks}
    for a in fwd:
        for b in fwd[a]:
            inbound[b] += 1
    zero_in = sorted(k for k in ks if inbound[k] == 0)

    print('前向引用 %d | 单向对 %d | 零入引 %d'
          % (sum(len(v) for v in fwd.values()), len(one), len(zero_in)))

    # 裸引用：无文档名限定的「第 N 条」
    text = load(path)
    bare = re.findall(r'[承接见联动]第 ?\d+ ?条', text)
    if bare:
        print('裸引用（跨区有歧义）：%s' % '、'.join(sorted(set(bare))))

    if full:
        print('\n--- 单向引用 ---')
        for a, b in sorted(one):
            print('  %s -> %s' % (a, b))
        print('\n--- 零入引 ---')
        print('  ' + '、'.join(zero_in))

    ok = not dup and not bare
    print('\n%s' % ('通过' if ok else '需处理'))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
