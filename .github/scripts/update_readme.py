#!/usr/bin/env python3
import requests
from pathlib import Path
import re

REPO = Path('.')
README = REPO / 'README.md'
MARKER_START = '<!-- README_HEADER_START -->'
MARKER_END = '<!-- README_HEADER_END -->'

# Fetch latest PyPI version
def get_pypi_version(package='hypernix'):
    try:
        r = requests.get(f'https://pypi.org/pypi/{package}/json', timeout=10)
        r.raise_for_status()
        data = r.json()
        return data.get('info', {}).get('version', 'unknown')
    except Exception:
        return 'unknown'

version = get_pypi_version('hypernix')

# New header HTML (icon + badges)
new_header = f"""<!-- README_HEADER_START -->
<p align="center">
  <img src="https://raw.githubusercontent.com/trail-b1az3r/HyperNix-pip/main/assets/logo-new/hypernix-lockup-dark.svg" alt="hypernix logo" width="240" />
</p>

<p align="center">
  <img alt="PyPI" src="https://img.shields.io/badge/PyPI-v{version}-ff2d55?style=for-the-badge&logo=pypi&logoColor=white" />
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10--3.14-00c9ff?style=for-the-badge&logo=python&logoColor=white" />
  <img alt="License" src="https://img.shields.io/badge/License-HOS%20/%20LLU-00c853?style=for-the-badge" />
</p>
<!-- README_HEADER_END -->"""

text = README.read_text(encoding='utf-8')

if MARKER_START in text and MARKER_END in text:
    pattern = re.compile(
        re.escape(MARKER_START) + r'.*?' + re.escape(MARKER_END),
        re.DOTALL
    )
    new_text = pattern.sub(new_header, text)
else:
    # If markers not present, prepend header at top
    new_text = new_header + '\n\n' + text

if new_text != text:
    README.write_text(new_text, encoding='utf-8')
    print("README updated")
else:
    print("No update needed")
