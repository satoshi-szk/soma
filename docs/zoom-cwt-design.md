# ズーム時 CWT 再計算機能 設計書

## 概要

スペクトログラムをズームした際に、可視領域に対して CWT を再計算し、より詳細なスペクトログラムを表示する機能。

## 現状の問題

- スペクトログラムは初回ロード時に固定サイズ（768×320）で1回だけ生成
- ズーム時は画像を拡大しているだけで、解像度は向上しない
- 詳細な周波数成分を確認できない

## 設計方針

### アプローチ

**可視領域ベースの動的再計算**

ズーム/パン操作完了後に、現在の可視領域（時間範囲・周波数範囲）で高解像度の CWT を再計算する。

### パラメータ

| 項目 | 値 | 理由 |
|------|-----|------|
| デバウンス時間 | 500ms | CWT計算に時間がかかるため、操作完了を待つ |
| キャンセル処理 | あり | 連続ズーム時に古い結果が後から表示されるのを防ぐ |
| キャッシュ | なし | MVPでは不要、後から追加可能 |
| プログレッシブ表示 | なし | 全体プレビューを背景として活用 |

## アーキテクチャ

### レイヤー構成

```
┌─────────────────────────────────────────┐
│  Layer 3: パーシャル / UI要素           │
├─────────────────────────────────────────┤
│  Layer 2: 詳細プレビュー（可視領域）    │  ← 新規追加
├─────────────────────────────────────────┤
│  Layer 1: 全体プレビュー（低解像度）    │  ← 既存
└─────────────────────────────────────────┘
```

- 全体プレビュー: 常に表示、ズーム時は引き伸ばし
- 詳細プレビュー: 計算完了後に可視領域にオーバーレイ

### データフロー

```
[ユーザー: ズーム/パン操作]
         ↓
[フロントエンド: 500ms デバウンス]
         ↓
[フロントエンド: 可視領域を計算]
         ↓
[API: request_viewport_preview 呼び出し]
         ↓
[バックエンド: 前回リクエストをキャンセル（フラグ設定）]
         ↓
[バックエンド: CWT 再計算（非同期スレッド）]
         ↓
[フロントエンド: viewport_preview_status でポーリング]
         ↓
[フロントエンド: 詳細プレビューを Layer 2 に表示]
```

## API 設計

### request_viewport_preview

可視領域の高解像度スペクトログラムをリクエストする。

**リクエスト:**

```typescript
{
  time_start: number   // 開始時刻（秒）
  time_end: number     // 終了時刻（秒）
  freq_min: number     // 最低周波数（Hz）
  freq_max: number     // 最高周波数（Hz）
  width: number        // 出力幅（ピクセル）
  height: number       // 出力高さ（ピクセル）
}
```

**レスポンス:**

```typescript
{
  status: 'processing' | 'error'
  request_id: string   // キャンセル判定用
  message?: string     // エラー時
}
```

### viewport_preview_status

リクエストした詳細プレビューの状態を取得する。

**レスポンス:**

```typescript
{
  status: 'ok' | 'error'
  state: 'processing' | 'ready' | 'cancelled' | 'error'
  request_id: string
  preview?: {
    width: number
    height: number
    data: number[]     // グレースケール 0-255
    time_start: number
    time_end: number
    freq_min: number
    freq_max: number
  }
  message?: string
}
```

## モデル設計

### ViewportPreview（新規）

```python
@dataclass(frozen=True)
class ViewportPreview:
    width: int
    height: int
    data: list[int]
    time_start: float
    time_end: float
    freq_min: float
    freq_max: float
```

## バックエンド実装

### analysis.py

新関数 `make_viewport_spectrogram` を追加:

```python
def make_viewport_spectrogram(
    audio: np.ndarray,
    sample_rate: int,
    settings: AnalysisSettings,
    time_start: float,
    time_end: float,
    freq_min: float,
    freq_max: float,
    width: int,
    height: int,
) -> ViewportPreview:
    """
    指定された時間・周波数範囲のみを高解像度で計算する。

    1. time_start〜time_end に対応するオーディオサンプルを切り出し
    2. freq_min〜freq_max の周波数ビンのみを計算
    3. width×height にリサンプル
    """
```

### document.py

viewport preview 用のステート管理を追加:

```python
class SomaDocument:
    def __init__(self):
        # ... 既存 ...
        self._viewport_request_id: str | None = None
        self._viewport_preview: ViewportPreview | None = None
        self._viewport_state: str = "idle"  # idle, processing, ready, cancelled, error
        self._viewport_thread: threading.Thread | None = None
```

キャンセル処理:

```python
def start_viewport_preview_async(self, params) -> str:
    request_id = str(uuid.uuid4())

    # 前回のリクエストをキャンセル扱いにする
    with self._lock:
        self._viewport_request_id = request_id
        self._viewport_state = "processing"

    def _worker():
        # 計算前にキャンセルチェック
        if self._viewport_request_id != request_id:
            return  # 新しいリクエストが来たので終了

        preview = make_viewport_spectrogram(...)

        # 結果保存前にもキャンセルチェック
        with self._lock:
            if self._viewport_request_id != request_id:
                return
            self._viewport_preview = preview
            self._viewport_state = "ready"

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    return request_id
```

### app.py

API エンドポイントを追加:

```python
def request_viewport_preview(self, payload: dict[str, Any]) -> dict[str, Any]:
    request_id = self._doc.start_viewport_preview_async(
        time_start=payload["time_start"],
        time_end=payload["time_end"],
        freq_min=payload["freq_min"],
        freq_max=payload["freq_max"],
        width=payload["width"],
        height=payload["height"],
    )
    return {"status": "processing", "request_id": request_id}

def viewport_preview_status(self) -> dict[str, Any]:
    state, preview, request_id = self._doc.get_viewport_preview_status()
    return {
        "status": "ok",
        "state": state,
        "request_id": request_id,
        "preview": preview.to_dict() if preview else None,
    }
```

## フロントエンド実装

### types.ts

```typescript
interface ViewportPreview {
  width: number
  height: number
  data: number[]
  time_start: number
  time_end: number
  freq_min: number
  freq_max: number
}
```

### App.tsx

```typescript
const [viewportPreview, setViewportPreview] = useState<ViewportPreview | null>(null)
const [viewportRequestId, setViewportRequestId] = useState<string | null>(null)
const viewportDebounceRef = useRef<number | null>(null)

// ズーム/パン変更時にデバウンス付きでリクエスト
useEffect(() => {
  if (!preview) return

  if (viewportDebounceRef.current) {
    clearTimeout(viewportDebounceRef.current)
  }

  viewportDebounceRef.current = window.setTimeout(async () => {
    const viewport = calculateViewport(zoom, pan, stageSize, preview)
    const result = await api.request_viewport_preview(viewport)
    if (result.status === 'processing') {
      setViewportRequestId(result.request_id)
    }
  }, 500)

  return () => {
    if (viewportDebounceRef.current) {
      clearTimeout(viewportDebounceRef.current)
    }
  }
}, [zoom, pan, stageSize, preview])

// ポーリング
useEffect(() => {
  if (!viewportRequestId) return

  const poll = async () => {
    const result = await api.viewport_preview_status()
    if (result.request_id !== viewportRequestId) return  // 古いリクエスト

    if (result.state === 'ready' && result.preview) {
      setViewportPreview(result.preview)
      setViewportRequestId(null)
    } else if (result.state === 'error' || result.state === 'cancelled') {
      setViewportRequestId(null)
    }
  }

  const interval = setInterval(poll, 200)
  return () => clearInterval(interval)
}, [viewportRequestId])
```

### Workspace.tsx

詳細プレビューのレイヤーを追加:

```typescript
// viewportPreview を canvas に変換
const viewportImage = useMemo(() => {
  if (!viewportPreview) return null
  // ... canvas 生成ロジック（previewImage と同様）
}, [viewportPreview, settings])

// 詳細プレビューの位置を計算
const viewportPosition = useMemo(() => {
  if (!viewportPreview || !preview) return null
  return {
    x: (viewportPreview.time_start / duration) * preview.width,
    y: freqToY(viewportPreview.freq_max),
    width: ((viewportPreview.time_end - viewportPreview.time_start) / duration) * preview.width,
    height: freqToY(viewportPreview.freq_min) - freqToY(viewportPreview.freq_max),
  }
}, [viewportPreview, preview, duration])

// Layer 構成
<Layer>
  <Group x={pan.x} y={pan.y} scaleX={scale.x} scaleY={scale.y}>
    {/* 全体プレビュー */}
    {previewImage && <KonvaImage image={previewImage} ... />}

    {/* 詳細プレビュー（オーバーレイ） */}
    {viewportImage && viewportPosition && (
      <KonvaImage
        image={viewportImage}
        x={viewportPosition.x}
        y={viewportPosition.y}
        width={viewportPosition.width}
        height={viewportPosition.height}
      />
    )}
  </Group>
</Layer>
```

## 実装順序

1. **models.py**: `ViewportPreview` データクラスの追加
2. **analysis.py**: `make_viewport_spectrogram` 関数の実装
3. **document.py**: viewport preview の非同期生成・キャンセルロジック
4. **app.py**: `request_viewport_preview`, `viewport_preview_status` API
5. **types.ts**: `ViewportPreview` 型定義
6. **App.tsx**: デバウンス・ポーリング処理
7. **Workspace.tsx**: 詳細プレビューレイヤーの表示

## テスト観点

- [ ] ズーム後 500ms で API が呼ばれること
- [ ] 連続ズーム時に古いリクエストがキャンセルされること
- [ ] 詳細プレビューが正しい位置に表示されること
- [ ] ズームアウト時（zoom < 1）は詳細プレビューを要求しないこと
- [ ] オーディオ未ロード時にエラーにならないこと
