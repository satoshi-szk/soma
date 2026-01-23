#!/bin/bash
set -e

cd "$(dirname "$0")/.."

echo "==> Building frontend..."
cd frontend
npm run build
cd ..

echo "==> Cleaning previous builds..."
rm -rf build dist

echo "==> Running PyInstaller..."
uv run pyinstaller soma.spec

echo "==> Done! App created at: dist/SOMA.app"
