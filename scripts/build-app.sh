#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

echo "==> Building frontend..."
cd frontend
npm run build
cd ..

clean_dir() {
    local target="$1"
    local attempts=5
    local i

    if [ ! -e "$target" ]; then
        return 0
    fi

    for i in $(seq 1 "$attempts"); do
        if rm -rf "$target" 2>/dev/null; then
            if [ ! -e "$target" ]; then
                return 0
            fi
        fi
        # 一時ファイル生成競合を吸収するため、短時間だけ再試行する。
        sleep 0.2
    done

    if [ -d "$target" ]; then
        # rm -rf が失敗する環境向けのフォールバック（中身削除→ディレクトリ削除）
        find "$target" -mindepth 1 -depth -exec rm -rf {} + 2>/dev/null || true
        rmdir "$target" 2>/dev/null || true
    fi

    if [ -e "$target" ]; then
        echo "Error: failed to remove $target after ${attempts} attempts."
        ls -la "$target" || true
        return 1
    fi
}

echo "==> Cleaning previous builds..."
clean_dir build
clean_dir dist

echo "==> Running PyInstaller..."
uv run pyinstaller soma.spec

echo "==> Done! App created at: dist/SOMA.app"
