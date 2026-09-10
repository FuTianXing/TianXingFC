# -*- coding: utf-8 -*-
r"""工作区 LaTeX 撕裂检查器。

背景：向 md 文件写入含 LaTeX 的内容时，若脚本用普通字符串而非 raw string，
`\ne`、`\nabla`、`\langle` 的 `\n` 与 `\rangle` 的 `\r` 会被当作转义符执行，
命令被吃掉前两字符，断口处留下余部（`\ne` 剩 `e`，`\nabla` 剩 `abla`，
`\rangle` 剩 `angle`），原行被拆成两行。这类损坏不会报错，只会静默积累。

判据（双条件交叉，避免误报）：
  1. 行首为已知 LaTeX 命令的余部；
  2. 该行 `$` 计数为奇数（撕裂把一对 `$` 分到两行）。

用法：python _tools/check_latex_tears.py [--all]
默认跳过 _tools/（其中的 .bak 是修复前快照，理应保留损坏），--all 则全查。
退出码：0 表示无残留，1 表示有残留。
"""
import io
import os
import re
import sys

# 行首余部：LaTeX 命令去掉被吃掉的转义字符后剩下的部分
REMNANTS = [
    'e 0', 'e0', 'e ', 'angle', 'rangle', 'langle', 'abla', 'eq ', 'eqno',
    'u ', 'u_', 'ot ', 'mid', 'orm', 'atural', 'u m', 'abla',
]
PAT = re.compile(r'^(' + '|'.join(re.escape(r) for r in REMNANTS) + r')')

SKIP_DIRS = {'.git', 'node_modules', 'images', '__pycache__', '.trae'}
if '--all' not in sys.argv:
    SKIP_DIRS.add('_tools')


def scan(root='.'):
    hits = []
    for cur, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for name in files:
            if not name.endswith('.md'):
                continue
            path = os.path.join(cur, name)
            try:
                lines = io.open(path, encoding='utf-8', errors='replace').read().split('\n')
            except OSError:
                continue
            for i, s in enumerate(lines):
                if not PAT.match(s) or not (0 < len(s) < 300):
                    continue
                if s.count('$') % 2 == 0:
                    continue  # 双条件不满足，视为误报
                hits.append((path, i + 1, s[:70], lines[i - 1][-40:] if i else ''))
    return hits


if __name__ == '__main__':
    hits = scan()
    if not hits:
        print('无 LaTeX 撕裂残留。')
        sys.exit(0)
    print('发现 %d 处疑似撕裂：' % len(hits))
    for path, ln, head, prev in hits:
        print('  %s L%d' % (path, ln))
        print('     行首: %r' % head)
        print('     前一行尾: %r' % prev)
    sys.exit(1)
