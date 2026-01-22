#!/bin/bash
set -e

cd "$(dirname "$0")/.."

APP_NAME="SOMA"
APP_PATH="dist/${APP_NAME}.app"
VERSION=$(grep '^version' pyproject.toml | head -1 | sed 's/.*"\(.*\)"/\1/')
DMG_NAME="${APP_NAME}-macOS-${VERSION}"
DMG_PATH="dist/${DMG_NAME}.dmg"
VOLUME_NAME="${APP_NAME}"
STAGING_DIR="dist/dmg-staging"

if [ ! -d "$APP_PATH" ]; then
    echo "Error: $APP_PATH not found. Run build-app.sh first."
    exit 1
fi

echo "==> Creating DMG..."

rm -rf "$STAGING_DIR"
mkdir -p "$STAGING_DIR"

cp -R "$APP_PATH" "$STAGING_DIR/"
ln -s /Applications "$STAGING_DIR/Applications"

rm -f "$DMG_PATH"

hdiutil create -volname "$VOLUME_NAME" \
    -srcfolder "$STAGING_DIR" \
    -ov -format UDZO \
    "$DMG_PATH"

rm -rf "$STAGING_DIR"

echo "==> Done! DMG created at: $DMG_PATH"
