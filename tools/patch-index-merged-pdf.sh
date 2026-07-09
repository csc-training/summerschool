#!/bin/bash

# SPDX-FileCopyrightText: 2026 CSC - IT Center for Science Ltd. <www.csc.fi>
#
# SPDX-License-Identifier: MIT

set -euo pipefail

if [[ -z "${1:-}" ]]; then
  echo "Error: no input file provided"
  echo "Usage: $0 <html-file>"
  exit 1
fi

INPUT_FILE="$1"

if [[ ! -f "$INPUT_FILE" ]]; then
  echo "Error: file '$INPUT_FILE' not found"
  exit 1
fi

# TODO: The merged PDF is built (by slidefactory --merge-pdf) before the
# "Fetch external slides" step below swaps in the externally-hosted Parallel
# Algorithms deck, so that chapter only has its placeholder link-slide in the
# merged PDF, not the real content. Note that on the download link, wherever
# slidefactory places it, until this is fixed.
echo "Warning: merged PDF does not include the externally-fetched Parallel Algorithms slides"

perl -0777 -i -pe '
s|(<c-link href="slides-merged\.pdf">Download a single merged PDF with a table of contents\.</c-link>)|$1 (Parallel Algorithms not included)|;
' "$INPUT_FILE"

echo "Patched: $INPUT_FILE"
