#!/usr/bin/env python3
"""Compare a markdown document with its Korean mirror and report where the two have drifted apart.

A mirror pair is <stem>.md and <stem>-ko.md. Everything but the prose has to match: the headings,
the display equations, the table and figure labels, the shape of every table, the citations and
their targets, the fenced code blocks byte for byte, the terminology of Appendix A, the list items
under each heading, and the link targets. The prose itself is the translation and is not compared.

The script exits 1 when it finds a difference, so it can gate a commit.
"""
__author__ = 'yRocket'
__version__ = "0.0.0.2026.9.5"  # Semantic Versioning: Major.Minor.Patch.Date(YYYY.M.D)

import argparse
import dataclasses
import pathlib
import re
import sys

__all__ = ['Mismatch', 'mirror_pairs', 'structure', 'compare_pair', 'check_paths']

KOREAN_SUFFIX: str = '-ko'
TITLE_MARKER: str = ' (Korean)'  # the mirror marks its H1 with this and nothing else may differ
FENCED_CODE: re.Pattern = re.compile(r'^```(\w*)\n(.*?)^```', re.M | re.S)
HEADING: re.Pattern = re.compile(r'^#{1,6} .*$', re.M)
DISPLAY_MATH: re.Pattern = re.compile(r'^\$\$.*\$\$$', re.M)
LABEL: re.Pattern = re.compile(r'^(Table|Fig) (\d+)\.', re.M)
CITATION: re.Pattern = re.compile(r'\[\[(\d+)\]\(#(ref-\d+)\)\]')
REF_ANCHOR: re.Pattern = re.compile(r'<a id="(ref-\d+)"></a>')
TERM: re.Pattern = re.compile(r'^- \*\*(.+?)\*\*:', re.M)
LIST_ITEM: re.Pattern = re.compile(r'^\s*(?:-|\d+\.) ', re.M)
LINK: re.Pattern = re.compile(r'\]\(([^)]+)\)')
IMAGE_SRC: re.Pattern = re.compile(r'<img [^>]*src="([^"]+)"')
VERSION_LINE: re.Pattern = re.compile(r'^Rev\. (\d+) \| Created: (\S+) \| Updated: (.+)$', re.M)
TERMINOLOGY_HEADING: str = 'Appendix A. Terminology'


@dataclasses.dataclass
class Mismatch:
    """One element that differs between the two editions of a mirror pair."""
    english: pathlib.Path
    korean: pathlib.Path
    element: str
    english_value: str
    korean_value: str


def mirror_pairs(paths: list = None) -> list:
    """The mirror pairs reachable from the given files and folders, folders searched recursively.

    Returns a list of (english_path, korean_path) tuples, sorted. Either half of a pair may be
    named on the command line; the other half is looked up beside it. Raises when the paths hold no
    pair at all, since an empty sweep is a mistyped path rather than a clean result.
    """
    if not paths:
        raise ValueError('paths is required and must not be empty.')
    candidates = []
    for path in paths:
        if path.is_dir():
            candidates.extend(item for item in path.rglob('*.md') if '.git' not in item.parts)
        elif path.suffix == '.md':
            candidates.append(path)
        else:
            raise ValueError(f"{path} is neither a folder nor a markdown file.")
    pairs = set()
    for path in candidates:
        english = path.with_name(path.stem[:-len(KOREAN_SUFFIX)] + '.md') \
            if path.stem.endswith(KOREAN_SUFFIX) else path
        korean = english.with_name(english.stem + KOREAN_SUFFIX + '.md')
        if english.is_file() and korean.is_file():
            pairs.add((english, korean))
    if not pairs:
        raise ValueError(f"no mirror pair under {', '.join(str(path) for path in paths)}.")
    return sorted(pairs)


def _sections(body: str = None) -> list:
    """(heading, text) pairs of one document, the text before the first heading under an empty key."""
    sections, heading, lines = [], '', []
    for line in body.splitlines():
        if HEADING.fullmatch(line):
            sections.append((heading, '\n'.join(lines)))
            heading, lines = line, []
        else:
            lines.append(line)
    sections.append((heading, '\n'.join(lines)))
    return sections


def _tables(body: str = None) -> list:
    """The shape of every table in document order, as the pipe count of each of its rows.

    Every row is counted rather than only the first, so a row that lost or gained a cell is caught
    even when the table still has the right number of rows.
    """
    shapes, rows = [], []
    for line in body.splitlines():
        if line.startswith('|'):
            rows.append(line.count('|'))
        elif rows:
            shapes.append(tuple(rows))
            rows = []
    if rows:
        shapes.append(tuple(rows))
    return shapes


def structure(path: pathlib.Path = None) -> dict:
    """Everything about one document that a mirror has to reproduce.

    Returns a dict whose values are lists, keyed by the name the report uses for that element. The
    H1 keeps its ' (Korean)' marker, which compare_pair allows for and nothing else.
    """
    if path is None:
        raise ValueError('path is required.')
    source = path.read_text(encoding='utf-8')
    fences = FENCED_CODE.findall(source)
    body = FENCED_CODE.sub('<FENCE>', source)
    terminology = body.split(TERMINOLOGY_HEADING)[-1] if TERMINOLOGY_HEADING in body else ''
    return {
        'heading': HEADING.findall(body),
        'display equation': DISPLAY_MATH.findall(body),
        'table or figure label': [f'{kind} {number}.' for kind, number in LABEL.findall(body)],
        'table shape': [str(shape) for shape in _tables(body)],
        'fenced code block': [f'```{language}\n{code}```' for language, code in fences],
        'citation': [f'[[{number}](#{target})]' for number, target in CITATION.findall(body)],
        'reference anchor': REF_ANCHOR.findall(body),
        'terminology entry': TERM.findall(terminology),
        'list items per section': [str(len(LIST_ITEM.findall(text))) for _, text in _sections(body)],
        'link target': LINK.findall(body) + IMAGE_SRC.findall(body),
        'thematic break': [line for line in body.splitlines() if line == '---'],
        'version line': [f'Updated: {match[2]}' for match in VERSION_LINE.findall(body)],
    }


def compare_pair(english: pathlib.Path = None, korean: pathlib.Path = None) -> list:
    """Every element in which the two editions of one pair differ.

    Returns a list of Mismatch, empty when the pair is consistent. The H1 of the mirror is expected
    to carry the ' (Korean)' marker and is compared with it removed.
    """
    if english is None or korean is None:
        raise ValueError('english and korean are both required.')
    left, right = structure(path=english), structure(path=korean)
    if right['heading'] and right['heading'][0].endswith(TITLE_MARKER):
        right['heading'][0] = right['heading'][0][:-len(TITLE_MARKER)]
    elif right['heading']:
        return [Mismatch(english=english, korean=korean, element='title marker',
                         english_value=f"expected the mirror H1 to end with '{TITLE_MARKER.strip()}'",
                         korean_value=right['heading'][0])]
    mismatches = []
    for element, values in left.items():
        others = right[element]
        for index in range(max(len(values), len(others))):
            here = values[index] if index < len(values) else '<missing>'
            there = others[index] if index < len(others) else '<missing>'
            if here != there:
                mismatches.append(Mismatch(english=english, korean=korean,
                                           element=f'{element} {index + 1}',
                                           english_value=here, korean_value=there))
    return mismatches


def check_paths(paths: list = None) -> list:
    """Every mismatch across the mirror pairs the paths reach.

    Returns a list of Mismatch. The pair count is reported on stderr so a sweep that found nothing
    to check is visible rather than silent.
    """
    pairs = mirror_pairs(paths=paths)
    mismatches = []
    for english, korean in pairs:
        mismatches.extend(compare_pair(english=english, korean=korean))
    print(f'checked {len(pairs)} mirror pairs', file=sys.stderr)
    return mismatches


def _report(value: str = None, other: str = None) -> tuple:
    """The two sides of one mismatch, narrowed to the first line that differs when both are blocks.

    A fenced code block runs to dozens of lines and printing its head hides the one line that
    changed, so multi-line values are reported by that line instead.
    """
    here, there = value.splitlines(), other.splitlines()
    if len(here) <= 1 and len(there) <= 1:
        return value, other
    for number in range(max(len(here), len(there))):
        left = here[number] if number < len(here) else '<missing>'
        right = there[number] if number < len(there) else '<missing>'
        if left != right:
            return f'line {number + 1}: {left}', f'line {number + 1}: {right}'
    return value, other


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog=pathlib.Path(__file__).name,
        description=f'{pathlib.Path(__file__).name} {__version__}\n'
                    'Report where a markdown document and its <stem>-ko.md mirror have drifted apart.',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-v', '--version', action='version', version=__version__)
    parser.add_argument('paths', type=pathlib.Path, nargs='+',
                        help='markdown files of either edition, or folders searched recursively')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    results = check_paths(paths=args.paths)
    for mismatch in results:
        left, right = _report(value=mismatch.english_value, other=mismatch.korean_value)
        print(f'{mismatch.english}  {mismatch.element}')
        print(f'    en: {left[:150]}')
        print(f'    ko: {right[:150]}')
    print(f'mismatched elements: {len(results)}')
    sys.exit(1 if results else 0)
