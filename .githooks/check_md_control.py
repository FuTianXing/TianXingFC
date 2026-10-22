#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""提交前检查：拒绝把控制字符写入 .md 文件。

背景：用 Python 普通字符串字面量拼含反斜杠的 LaTeX 时，\\t \\v \\d \\b \\n \\x 等
会被解释成控制字符，落盘后 \theta 变成 TAB+heta、\times 变成 TAB+imes，肉眼
难辨、渲染出错。此脚本作为 pre-commit 钩子，拦住这类提交。
"""
import subprocess
import sys

ALLOWED = {0x09, 0x0A, 0x0D}  # 但 .md 中 TAB 亦视为违规，见下
HARD_FAIL = {0x09, 0x0B, 0x0C, 0x08, 0x07, 0x00, 0x1B}


def staged_md():
    out = subprocess.run(
        ['git', 'diff', '--cached', '--name-only', '--diff-filter=ACM'],
        capture_output=True, text=True, encoding='utf-8'
    )
    return [f for f in out.stdout.splitlines() if f.strip().endswith('.md')]


def scan(path):
    try:
        data = open(path, 'rb').read()
    except OSError:
        return []
    hits = []
    line = 1
    for i, byte in enumerate(data):
        if byte == 0x0A:
            line += 1
            continue
        if byte in HARD_FAIL:
            hits.append((line, byte, i))
    return hits


def main():
    files = staged_md()
    if not files:
        return 0
    bad = []
    for f in files:
        hits = scan(f)
        if hits:
            bad.append((f, hits))
    if not bad:
        return 0
    print('=' * 68)
    print('提交被拒绝：以下文件含控制字符（多为反斜杠加 t/v/d 等被误解释的产物）')
    print('=' * 68)
    for f, hits in bad:
        print('\n%s  (%d 处)' % (f, len(hits)))
        for line, byte, pos in hits[:8]:
            print('  行 %d  字节 0x%02X  (偏移 %d)' % (line, byte, pos))
        if len(hits) > 8:
            print('  ... 另有 %d 处' % (len(hits) - 8))
    print('\n修复：把受污染处还原为反斜杠加命令名（如 TAB+heta -> \theta）。')
    print('预防：含反斜杠的文本一律用 r-string，或改用 Edit/Write 工具落盘。')
    return 1


if __name__ == '__main__':
    sys.exit(main())
