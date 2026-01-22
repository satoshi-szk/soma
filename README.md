# SOMA

Sonic Observation, Musical Abstraction

詳細は [docs/spec.md](docs/spec.md) を参照してください。

## 開発構成

- Backend: Python + pywebview (uv)
- Frontend: React + TypeScript + Tailwind + Konva

## セットアップ

```sh
uv sync
cd frontend
npm install
```

## 開発起動

フロントの開発サーバーを起動してから、pywebview を起動します。

```sh
cd frontend
npm run dev
```

別ターミナルで:

```sh
SOMA_DEV=1 uv run soma
```

開発サーバーURLを明示する場合:

```sh
SOMA_DEV_SERVER_URL=http://localhost:5173 uv run soma
```

## フロントビルド

```sh
cd frontend
npm run build
```

ビルド済みファイルを表示する場合:

```sh
uv run soma
```

## PyInstaller

フロントをビルドしてから、単体バンドルを作成します。

```sh
cd frontend
npm run build
cd ..
uv run pyinstaller soma.spec
```

## Audio Preview (Dev)

現状のローダーは WAV のみに対応し、最初の 30 秒を軽量スペクトログラムとして生成します。
