#!/bin/bash
set -e
SKILL_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Installing Python dependencies..."
pip3 install -r "$SKILL_DIR/requirements.txt"

echo "Importing pre-computed data into ChromaDB..."
python3 "$SKILL_DIR/scripts/setup.py" --data "$SKILL_DIR/data/chunks.json" --db-path "$SKILL_DIR/chroma_db"

echo "Linking skill to OpenClaw..."
ln -sfn "$SKILL_DIR" ~/.openclaw/skills/chinese-tax-law

echo "Done! chinese-tax-law skill installed."
