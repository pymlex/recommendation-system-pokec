#!/bin/bash
set -e

echo "1) Checking ~/.kaggle/kaggle.json"
if [ ! -f "$HOME/.kaggle/kaggle.json" ]; then
  echo "~/.kaggle/kaggle.json not found."
  exit 1
fi

echo "2) Checking Kaggle CLI"
if ! command -v kaggle >/dev/null 2>&1; then
  echo "Kaggle not found. Attempting to install with pip..."
  if command -v pip3 >/dev/null 2>&1; then
    pip3 install --user kaggle
    export PATH="$HOME/.local/bin:$PATH"
    echo "Kaggle has been installed at \$HOME/.local/bin"
  else
    echo "pip3 not found."
    exit 1
  fi
fi

mkdir -p data
echo "3) Downloading H-Pokec (~2.2 GB)..."
kaggle datasets download -d shakeelsial/pokec-full-data-set -p data --unzip

echo "Successfully saved to data/:"
ls -lh data | sed -n '1,200p'
