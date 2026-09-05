#!/usr/bin/env python3
"""Find markdown paragraphs where emphasis parsing would eat the underscores of inline math.

GitHub reads markdown before it reads math, so an underscore inside $...$ can act as an emphasis
delimiter. In \\hat{y}_i a brace precedes the underscore, which lets it open emphasis; in SS_{tot} a
letter precedes it, which lets it close. When both sit in one paragraph markdown pairs them, removes
both underscores and italicises the text between, so the rendered formulas lose their subscripts.

This script reports the paragraphs where such a pair exists. It exits 1 when it finds any, so it can
gate a commit.
"""
__author__ = 'yRocket'
__version__ = "0.0.0.2026.9.4"  # Semantic Versioning: Major.Minor.Patch.Date(YYYY.M.D)

import argparse
import dataclasses
import pathlib
import re
import sys
from typing import Iterator

__all__ = ['Finding', 'flanking', 'prose_paragraphs', 'find_emphasis_pairs', 'check_paths']

# the characters CommonMark counts as punctuation for the flanking rules, restricted to those that
# actually turn up beside an underscore in TeX
PUNCTUATION: frozenset = frozenset('{}()[]\\^,.;:!?/|+-=<>~"\'`*_&%$#@')
FENCED_CODE: re.Pattern = re.compile(r'```.*?```', re.S)
DISPLAY_MATH: re.Pattern = re.compile(r'\$\$.*?\$\$', re.S)
CODE_SPAN: re.Pattern = re.compile(r'`[^`]*`')


@dataclasses.dataclass
class Finding:
    """One paragraph in which an opening underscore is followed by a closing one."""
    path: pathlib.Path
    paragraph: int
    text: str


def flanking(text: str = None, position: int = None) -> tuple:
    """Whether the underscore at position can open and whether it can close emphasis.

    Returns a (can_open, can_close) pair of bool, following the CommonMark flanking rules. An
    underscore between two alphanumerics is intraword and is neither, which is what makes y_i safe
    and \\hat{y}_i unsafe.
    """
    if text is None or position is None:
        raise ValueError('text and position are both required.')
    if not 0 <= position < len(text) or text[position] != '_':
        raise ValueError(f"position {position} does not hold an underscore.")
    before = text[position - 1] if position else ' '
    after = text[position + 1] if position + 1 < len(text) else ' '
    if before.isalnum() and after.isalnum():
        return False, False
    can_open = not after.isspace() and (after not in PUNCTUATION or before.isspace() or before in PUNCTUATION)
    can_close = not before.isspace() and (before not in PUNCTUATION or after.isspace() or after in PUNCTUATION)
    return can_open, can_close


def prose_paragraphs(source: str = None) -> Iterator[tuple]:
    """The prose of each paragraph, as (index, text) pairs.

    Fenced code, display math, code spans, table rows and headings are removed first, because
    markdown does not parse emphasis inside them. Soft line breaks are joined, since the paragraph
    rather than the line is what emphasis is resolved over.
    """
    if source is None:
        raise ValueError('source is required.')
    source = FENCED_CODE.sub(' ', source)
    source = DISPLAY_MATH.sub(' ', source)
    for index, block in enumerate(source.split('\n\n')):
        lines = [line for line in block.splitlines() if not line.startswith(('|', '#'))]
        yield index, CODE_SPAN.sub(' ', ' '.join(lines))


def find_emphasis_pairs(path: pathlib.Path = None) -> list:
    """The paragraphs of one markdown file that carry an emphasis pair.

    Returns a list of Finding. A pair needs an underscore that can open and a later one that can
    close, which is the order markdown resolves them in.
    """
    if path is None:
        raise ValueError('path is required.')
    findings = []
    for index, text in prose_paragraphs(source=path.read_text(encoding='utf-8')):
        underscores = [match.start() for match in re.finditer('_', text)]
        opens = [at for at in underscores if flanking(text=text, position=at)[0]]
        closes = [at for at in underscores if flanking(text=text, position=at)[1]]
        if opens and any(at > opens[0] for at in closes):
            findings.append(Finding(path=path, paragraph=index, text=text.strip()))
    return findings


def check_paths(paths: list = None) -> list:
    """Every finding across the given markdown files and folders, folders searched recursively.

    Returns a list of Finding. Raises when the paths hold no markdown file at all, since an empty
    sweep is a mistyped path rather than a clean result.
    """
    if not paths:
        raise ValueError('paths is required and must not be empty.')
    files = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(item for item in path.rglob('*.md') if '.git' not in item.parts))
        elif path.suffix == '.md':
            files.append(path)
        else:
            raise ValueError(f"{path} is neither a folder nor a markdown file.")
    if not files:
        raise ValueError(f"no markdown file under {', '.join(str(p) for p in paths)}.")
    findings = []
    for path in sorted(set(files)):
        findings.extend(find_emphasis_pairs(path=path))
    print(f'checked {len(set(files))} markdown files', file=sys.stderr)
    return findings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog=pathlib.Path(__file__).name,
        description=f'{pathlib.Path(__file__).name} {__version__}\n'
                    'Report markdown paragraphs where emphasis parsing would eat the underscores of '
                    'inline math.',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-v', '--version', action='version', version=__version__)
    parser.add_argument('paths', type=pathlib.Path, nargs='+',
                        help='markdown files, or folders searched recursively for them')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    results = check_paths(paths=args.paths)
    for finding in results:
        print(f'{finding.path}  paragraph {finding.paragraph}')
        print(f'    {finding.text[:160]}')
    print(f'flagged paragraphs: {len(results)}')
    sys.exit(1 if results else 0)
