#!/usr/bin/env python3

"""
This script is like check-basics.py, but specific to the documentation.
It's split off into a separate script, so that it can be easily run on its own.
"""

import re
import sys
import urllib.parse
import urllib.request

from pathlib import Path

OMZ_ROOT = Path(__file__).resolve().parents[1]

sys.path.append(str(OMZ_ROOT / 'ci/lib'))

import omzdocs

HTML_FRAGMENT_RE = re.compile(r'</?([^>\s]+)', re.IGNORECASE)

# taken from https://www.doxygen.nl/manual/htmlcmds.html
ALLOWED_HTML_ELEMENTS = frozenset([
    'a', 'b', 'blockquote', 'br', 'caption', 'center', 'code', 'dd', 'del',
    'dfn', 'div', 'dl', 'dt', 'em', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
    'hr', 'i', 'img', 'ins', 'kbd', 'li', 'ol', 'p', 'pre', 's', 'small',
    'span', 'strike', 'strong', 'sub', 'sup', 'table', 'td', 'th', 'tr',
    'tt', 'u', 'ul', 'var',
])

def find_md_files():
    thirdparty_dir = OMZ_ROOT / 'demos' / 'thirdparty'

    for path in OMZ_ROOT.glob('**/*.md'):
        if thirdparty_dir in path.parents: continue
        yield path

def main():
    all_passed = True

    index_file_paths = (
        OMZ_ROOT / 'models/intel/index.md',
        OMZ_ROOT / 'models/public/index.md',
        OMZ_ROOT / 'demos/README.md',
    )

    all_md_files = tuple(find_md_files())

    def complain(message):
        nonlocal all_passed
        all_passed = False
        print(message, file=sys.stderr)

    index_child_md_links = {}
    for index_file_path in index_file_paths:
        if not index_file_path.exists():
            complain(f'{index_file_path}: file not found')
            continue

        required_md_links = []
        for md_file in all_md_files:
            if md_file.name == "README.md" and md_file.parent != index_file_path.parent:
                try:
                    md_rel_path = md_file.relative_to(index_file_path.parent)
                except ValueError:
                    continue

                md_intermediate_parents = list(md_rel_path.parents)[1:-1] # removed root and first parent dirs

                if not any((index_file_path.parent / parent_dir / 'README.md').exists()
                        for parent_dir in md_intermediate_parents):
                    required_md_links.append(md_file)

        index_child_md_links[index_file_path] = sorted(required_md_links)

    omz_reference_prefix = '<omz_dir>/'

    for md_path in sorted(all_md_files):
        referenced_md_files = set()

        md_path_rel = md_path.relative_to(OMZ_ROOT)

        doc_page = omzdocs.DocumentationPage(md_path.read_text(encoding='UTF-8'))

        # check local link validity

        for url in sorted([ref.url for ref in doc_page.external_references()]):
            try:
                components = urllib.parse.urlparse(url)
            except ValueError:
                complain(f'{md_path_rel}: invalid URL reference {url!r}')
                continue

            if components.scheme: # non-local URLs
                continue

            if components.netloc or components.path.startswith('/'):
                complain(f'{md_path_rel}: non-relative local URL reference "{url}"')
                continue

            if not components.path: # self-link
                continue

            target_path = (md_path.parent / urllib.request.url2pathname(components.path)).resolve()

            if OMZ_ROOT not in target_path.parents:
                complain(f'{md_path_rel}: URL reference "{url}" points outside the OMZ directory')
                continue

            if not target_path.is_file():
                complain(f'{md_path_rel}: URL reference "{url}" target'
                    ' does not exist or is not a file')
                continue

            if md_path in index_child_md_links:
                referenced_md_files.add(target_path)

        # check <omz_dir> reference validity

        for code_span in doc_page.code_spans():
            if code_span.startswith(omz_reference_prefix):
                target_path_rel = Path(code_span[len(omz_reference_prefix):])
                target_path = OMZ_ROOT / target_path_rel

                if ".." in target_path_rel.parts:
                    complain(f'{md_path_rel}: OMZ reference "{code_span}"'
                        ' contains a ".." component.')
                    continue

                if not target_path.exists():
                    complain(f'{md_path_rel}: OMZ reference "{code_span}" target'
                        ' does not exist')

        # check for existence of links to README.md files of models and demos

        if md_path in index_child_md_links:
            for md_file in index_child_md_links[md_path]:
                if md_file not in referenced_md_files:
                    complain(f"{md_path_rel}: {md_file.relative_to(OMZ_ROOT)} is not referenced")

        # check for HTML fragments that are unsupported by Doxygen

        for html_fragment in doc_page.html_fragments():
            match = HTML_FRAGMENT_RE.match(html_fragment)
            if not match:
                complain(f'{md_path_rel}: cannot parse HTML fragment {html_fragment!r}')
                continue

            if match.group(1).lower() not in ALLOWED_HTML_ELEMENTS:
                complain(f'{md_path_rel}: unknown/disallowed HTML element in {html_fragment!r}')
                continue

    sys.exit(0 if all_passed else 1)

if __name__ == '__main__':
    main()
