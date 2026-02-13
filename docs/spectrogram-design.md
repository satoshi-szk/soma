# SOMA Sub-System Design: Spectrogram Visualization

## 1. システム概要

本サブシステムは、オーディオ信号から周波数特性を可視化し、GUI（Frontend）へ配信する責務を持つ。

- **Core Concept:** **"Dynamic Server-Side Rendering"**
  - 静的な画像ファイルを作らず、リクエストのたびに生データ（`float32`）から画像を生成する。
  - これにより、ユーザーは「Gain（明るさ）」や「Dynamic Range（コントラスト）」を瞬時に変更できる。
- **Performance Goal:**
  - 1時間のオーディオファイル (44.1kHz Stereo) を快適に操作可能とする。
  - スクロール・ズーム時のレイテンシを 100ms 以内に抑える。

---

## 2. データ生成 (Generation Layer)

表示用データには、計算コストと視認性のバランスが良い **Multi-Resolution STFT** を採用する。

### 2.1 マルチ解像度合成ロジック

単一の窓関数ではなく、帯域ごとに最適な時間解像度を持つ3つのSTFT結果を合成し、1つの巨大なスペクトログラム行列を作成する。

| **Band** | **周波数範囲**     | **窓サイズ (Window)** | **目的**                           |
| -------- | ------------------ | --------------------- | ---------------------------------- |
| **Low**  | 20Hz ~ 200Hz       | 4096 samples          | 低音の音程感（周波数解像度）を確保 |
| **Mid**  | 200Hz ~ 2kHz       | 1024 samples          | ボーカル帯域のバランス             |
| **High** | 2kHz ~ Nyquist     | 256 samples           | アタック感（時間解像度）を確保     |

各バンドの結果は**重み付きブレンド**で合成される。帯域境界でのクロスオーバーは対数周波数スケール上の滑らかな重みで行い、不連続を防ぐ（`_band_weights`）。

### 2.2 Sparse STFT

従来のスライディングウィンドウ STFT ではなく、**出力列数分の離散的な中心位置でのみ FFT を計算する** Sparse STFT を採用している（`_sparse_stft_magnitude`）。音声全体を走査しないため、計算量は出力解像度に比例し、長時間音声でも一定のパフォーマンスを保つ。

### 2.3 データ構造 (Storage)

生成された行列は、メモリ効率のため **`numpy.memmap`** を用いて管理する。

- **File Format:** `.dat` (Raw Binary / Temporary file, `tempfile.TemporaryDirectory` に配置)
- **Shape:** `(1024, Time_Frames)`
  - 縦 1024 は固定 (`_BASE_HEIGHT`)
  - 横は `duration * 84 cols/sec` で算出し、2,048 ~ 320,000 の範囲にクランプ
- **Dtype:** `float32` (振幅の絶対値 `np.abs()` のみ保持)
- **Audio Memmap:** 元音声も `float32` の memmap として保持し、ローカル STFT 時に高速アクセスする

### 2.4 対数周波数軸

行列の周波数軸は**対数スケール (`np.geomspace`)** で構成される。線形周波数軸の FFT 結果を `_interpolate_freq_matrix` で対数軸へ線形補間している。

---

## 3. レンダリングパイプライン (Processing Layer)

`SpectrogramRenderer` が Frontend からのリクエストに応じて画像バイナリを生成する。グローバル行列とローカル STFT の 2 つのモードを持つ。

### 3.1 グローバル行列モード (`render_overview`, `render_tile`)

初期化時に構築済みの memmap 行列からスライスし、リサンプリングして描画する。

1. **Slice:** リクエストされた時間範囲に対応する列インデックスを算出し、memmap からスライス
2. **Time Resample:** `_resample_time_matrix` でリクエストされた `width` にリサイズ
   - ダウンサンプル時はピーク保持（`np.max` ベース）
   - アップサンプル時は線形補間
3. **Freq Resample:** `_interpolate_freq_matrix` でリクエストされた周波数範囲・高さに合わせて対数軸補間
4. **Normalize:** dB 正規化 (`_normalize_magnitude_db`)
   $$P_{dB} = 20 \cdot \log_{10}\left(\frac{P_{raw}}{P_{max}}\right)$$
   ダイナミックレンジ -60dB 〜 0dB を 0.0 〜 1.0 に正規化
5. **Tone Adjust:** `_apply_preview_tone` でユーザーパラメータ（`gain`, `gamma`, `min_db`, `max_db`）を適用
6. **Output:** `uint8` グレースケール (0–255) を `SpectrogramPreview` として返却

### 3.2 ローカル STFT モード (`render_tile` → `_render_local_tile`)

高ズーム時（1ピクセルあたりの時間がグローバル行列の解像度を超える場合）に自動で切り替わる。

- **切替条件:** `required_cols_per_sec` が `84 × 1.10 = 92.4` を超えたら Local へ遷移、`84 × 0.90 = 75.6` を下回ったら Global へ復帰（ヒステリシス）
- **処理:** リクエスト範囲 ± マージンの音声を memmap から読み出し、その場で Sparse STFT → バンド合成 → 正規化
- **キャッシュ:** LRU キャッシュ（最大 24 エントリ）でレンダリング結果を保持。キーは時間・周波数・サイズ・トーン設定の量子化値

### 3.3 着色 & エンコード (`preview_cache.py`)

`SpectrogramRenderer` から返されたグレースケール `uint8` データを、`PreviewCacheServer` 経由で配信するために着色・圧縮する。

1. **Colorization:** Magma 風の 6 点カラーストップによる線形補間カラーマップ (`_apply_magma_like_colormap`)
2. **Encoding:** JPEG (Quality: 82, optimize: True)
3. **Caching:** ディスク上の一時ディレクトリに JPEG ファイルを保存。最大 200 エントリの LRU 管理

---

## 4. インターフェース (Communication Layer)

### 4.1 プロトコル

`pywebview` の JS API (`window.pywebview.api`) を使用し、Python バックエンドの `SomaApi` クラスのメソッドを直接呼び出す。画像データは**ローカル HTTP サーバー**経由で配信する。

### 4.2 画像配信サーバー (`PreviewCacheServer`)

- `http://127.0.0.1:<port>/.soma-cache/` で JPEG ファイルを配信
- WSGI ベースのスレッド化サーバー（`wsgiref.simple_server` + `ThreadingMixIn`）
- ポートは OS が自動割当

### 4.3 API エンドポイント

#### A. タイル取得 (`request_spectrogram_tile`)

ズーム時に Frontend から呼ばれる。

- **Request Params:**
  - `time_start`, `time_end`: 時間範囲 (秒)
  - `freq_min`, `freq_max`: 周波数範囲 (Hz)
  - `width`, `height`: 生成する画像のピクセルサイズ
  - `gain`, `min_db`, `max_db`, `gamma`: 画質調整パラメータ (optional)
- **Response:** JSON
  ```json
  {
    "status": "ok",
    "quality": "high" | "local" | "low",
    "preview": {
      "width": 1024,
      "height": 392,
      "image_path": "http://127.0.0.1:PORT/.soma-cache/tile-xxx.jpg",
      "time_start": 0.0,
      "time_end": 5.0,
      "freq_min": 20.0,
      "freq_max": 12000.0,
      "duration_sec": 60.0
    }
  }
  ```
- **quality フィールド:**
  - `"high"`: グローバル行列から生成
  - `"local"`: ローカル STFT から生成（高ズーム時）

#### B. 全体図取得 (`request_spectrogram_overview`)

オーディオロード時・設定変更時に呼ばれる。

- **Request Params:**
  - `width` (default: 2048, 16–4096), `height` (default: 320, 16–2048)
  - `gain`, `min_db`, `max_db`, `gamma`: 画質調整パラメータ (optional)
- **Response:** JSON（同上の形式、`quality` は常に `"low"`）

### 4.4 設定モデル

設定は `analysis_settings` を以下 2 系統に分離する。

- `spectrogram`: GUI 描画用（STFT ベース）
  - `freq_min`, `freq_max`, `preview_freq_max`
  - `multires_blend_octaves`（Low/Mid/High のクロスフェード幅）
  - `gain`, `min_db`, `max_db`, `gamma`
- `snap`: スナップ解析用（CWT ベース）
  - `freq_min`, `freq_max`
  - `bins_per_octave`, `time_resolution_ms`
  - `wavelet_bandwidth`, `wavelet_center_freq`

---

## 5. フロントエンド制御 (Frontend Strategy)

### 5.1 仮想スクロールとタイリング (`useViewport`)

ブラウザのメモリ消費を抑えるため、**「必要な分だけ `<img>` を配置する」** 戦略をとる。

- **タイル分割:** 1タイルの幅は `1024 / zoomX` 秒。ズーム倍率に応じてタイル数が増減する
- **ビューポートキャッシュ:** 最大 8 エントリの LRU で、同じ時間・周波数・設定のタイルを再利用
- **リクエスト重複排除:** `requestedTileKeysRef` で発行済みリクエストを追跡し、同一タイルの多重リクエストを防止

### 5.2 Overscan (先読み)

ネットワークレイテンシ（生成時間）を隠蔽する。

- **Overscan 範囲:** 可視ビューポートの左右に `1024 / zoomX` 秒（= 1タイル分）のマージンを追加
- **Debounce:** ズーム/パン操作後 200ms 静止したタイミングでバッチリクエストを発行

### 5.3 Layer Composition (重ね合わせ)

Canvas (Konva.js / `react-konva`) 上でのレイヤー構成。

1. **Layer 0: Overview Image** (低解像度, opacity: 0.85)
   - 常に全体に引き伸ばして表示。詳細タイルがロードされるまでの「プレースホルダー」として機能し、白飛びを防ぐ
2. **Layer 1: High-Res Tiles** (高解像度, opacity: 0.95)
   - `request_spectrogram_tile` で取得した画像。ロード完了次第、Layer 0 の上に重ねる
   - `useViewportImageCache` で `Image` オブジェクトのライフサイクルを管理。不要な画像は即座に `src=""` でメモリ解放
3. **Layer 2: Spectrogram Dim** (半透明黒オーバーレイ)
   - Partial 視認性向上のため、スペクトログラムの輝度を抑制するオプショナルレイヤー
4. **Layer 3: Partials / Selection / Playhead**
   - 描画されたパーシャル曲線、選択ボックス、再生ヘッドなどのインタラクティブ要素
5. **Layer 4: Rulers / Automation Lane**
   - 時間ルーラー、周波数ルーラー、振幅オートメーションレーン

### 5.4 ズーム

- **X軸 (時間):** 0.05 ~ 10,000 px/sec、ステップ比 2.0x
- **Y軸 (周波数):** 1 ~ 10 倍、ステップ 1.0

---

## 6. 振幅リファレンス管理

スペクトログラムの明るさをズームレベル間で一貫させるため、振幅リファレンス値を管理する。

- `_stft_amp_reference`: グローバル行列のピーク値。Overview と Global タイルで共有
- `_snap_amp_reference`: CWT 解析（snap/trace）で使用するリファレンス値
- タイルレンダリング時にリファレンスが未設定の場合は、そのタイルのピーク値を使用し、以降のリクエストで再利用する
