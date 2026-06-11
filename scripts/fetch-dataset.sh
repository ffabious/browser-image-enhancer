#!/usr/bin/env bash
# Download DIV2K into ml/data/. By default fetches the 100-image validation
# set (~450 MB) which is enough to train the parameter-regression model;
# pass --full to also fetch the 800-image training set (~3.5 GB).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA="$ROOT/ml/data"
BASE_URL="https://data.vision.ee.ethz.ch/cvl/DIV2K"

mkdir -p "$DATA"
cd "$DATA"

fetch() {
  local name="$1"
  if [ -d "$name" ]; then
    echo "$name already present, skipping"
    return
  fi
  echo "downloading $name ..."
  curl -L -o "$name.zip" --retry 3 -C - "$BASE_URL/$name.zip"
  unzip -q "$name.zip"
  rm "$name.zip"
}

fetch DIV2K_valid_HR
if [ "${1:-}" = "--full" ]; then
  fetch DIV2K_train_HR
fi
echo "done: $(ls "$DATA")"
