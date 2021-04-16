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

    def complain(message):
        nonlocal all_passed
        all_passed = False
        print(message, file=sys.stderr)

    for md_path in sorted(find_md_files()):
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
