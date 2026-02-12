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
| **Band** | **周波数範囲** | **窓サイズ (Window)** | **目的** |
| -------- | -------------- | --------------------- | ---------------------------------- |
| **Low** | 20Hz ~ 200Hz | 4096 samples | 低音の音程感（周波数解像度）を確保 |
| **Mid** | 200Hz ~ 2kHz | 1024 samples | ボーカル帯域のバランス |
| **High** | 2kHz ~ 22kHz | 256 samples | アタック感（時間解像度）を確保 |

### 2.2 データ構造 (Storage)

生成された行列は、メモリ効率のため **`numpy.memmap`** を用いて管理する。

- **File Format:** `.dat` (Raw Binary / Temporary file)
- **Shape:** `(Frequency_Bins, Time_Frames)`
- **Dtype:** `float32` (複素数ではなく、絶対値 `np.abs()` をとった振幅情報のみ保持)
  Python

#

`# 概念コード: コンポジット行列の作成

# 実際には Low/Mid/High を計算し、周波数軸で切り貼りして1枚にする

spectrogram_matrix = np.memmap(
filename='temp_spec.dat',
dtype='float32',
mode='w+',
shape=(1024, 300000) # 例: 縦1024px, 横30万px
)`

---

## 3. レンダリングパイプライン (Processing Layer)

Frontend からのリクエスト (`soma://`) を受け、画像バイナリを生成するまでのフロー。ここは **Hot Path** (頻繁に通る処理) なので、NumPy によるベクトル演算で最適化する。

### Step 1: Slice (切り出し)

リクエストされた時間範囲 (`start_time`, `end_time`) に対応する配列インデックスを計算し、`memmap` からスライスする。

- _Note:_ ディスクアクセスはこの瞬間のみ発生する。OSのページキャッシュが効くため、2回目以降は爆速。

### Step 2: Dynamic Adjustment (画質調整)

ユーザーの表示設定（Gain等）を適用する。ここが動的であるため、静的画像配信では代替できない。
$$P_{dB} = 10 \cdot \log_{10}(P_{raw} + \epsilon)$$
$$Pixel = \frac{P_{dB} - Min_{dB}}{Max_{dB} - Min_{dB}} \times Gain$$

- **Parameters:**
  - `min_db`, `max_db`: 表示するダイナミックレンジ（例: -80dB 〜 0dB）。
  - `gamma`: ガンマ補正（中間色を持ち上げる等）。

### Step 3: Colorization (着色)

正規化された `0.0 ~ 1.0` の値を、カラーマップ（Magma/Inferno等）を用いて RGBA に変換し、`uint8` (0~255) にキャストする。

### Step 4: Encoding (圧縮)

- **Format:** JPEG (Quality: 80-85)
  - PNGは圧縮処理が遅く、データ量も大きくなりがちなので、スペクトログラムのような連続階調画像には JPEG が最適。
- **Orientation:** 周波数軸は `Low` が配列の先頭（画像の上）に来るため、必要に応じて `np.flipud` で上下反転する。

---

## 4. インターフェース (Communication Layer)

`pywebview` のカスタムスキーム機能を使用し、オーバーヘッドなしでバイナリ転送を行う。

### プロトコル: `soma://`

### A. 詳細タイル取得 (`soma://tile`)

Frontend の `<img>` タグから直接呼ばれる。

- **Endpoint:** `soma://tile`
- **Query Params:**
  - `t`: 中心時間 (秒)
  - `w`: 要求するタイルの物理幅 (秒)
    - 例: `w=5.0` なら前後5秒分の画像を返す
  - `px_w`: 生成する画像の横幅ピクセル数 (LOD判定用)
  - `gain`, `range`: 画質調整パラメータ
- **Response:** `image/jpeg` バイナリ

### B. 全体図取得 (`soma://overview`)

ナビゲーションバー用。ロード時に一度だけ呼ばれることが多い。

- **Endpoint:** `soma://overview`
- **Response:** 全体を横幅 2048px 程度にダウンサンプリングした `image/jpeg`。

---

## 5. フロントエンド制御 (Frontend Strategy)

### 5.1 仮想スクロールとタイリング

ブラウザのメモリ死を防ぐため、**「必要な分だけ DOM に置く」** 戦略をとる。

- **Tile Manager:**
  - 画面幅を基準に、論理的な「タイル番号」を計算する。
  - 現在表示されている `Tile Index` ±1 の範囲のみ `<img>` タグを生成する。
  - 範囲外に出た `<img>` は即座に削除または `src=""` にしてメモリ開放する。

### 5.2 Overscan (先読み)

ネットワークレイテンシ（生成時間）を隠蔽する。

- **Viewport:** 画面に表示されている範囲。
- **Render Area:** Viewport の左右に **画面幅×1.0倍** ずつマージンを取った範囲。
- **挙動:**
  1. ユーザーがスクロールする。
  2. Viewport が Render Area の端に近づく。
  3. 静止 (`debounce` 200ms) したタイミングで、次のエリアの画像を裏でリクエストする。

### 5.3 Layer Composition (重ね合わせ)

Canvas (Konva.js) 上でのレイヤー構成。

1. **Layer 0: Background Color** (黒)
2. **Layer 1: Overview Image** (低解像度)
   - 常に全体に引き伸ばして表示しておく。詳細画像がロードされるまでの「プレースホルダー」として機能し、白飛びを防ぐ。
3. **Layer 2: High-Res Tiles** (高解像度)
   - `soma://tile` で取得した画像。ロード完了次第、Layer 1 の上に重ねる。

---

## 6. 実装ステップ (Recommended Path)

1. **Backend Core:**
   - `MultiResStft` クラスを作成し、WAV読込 → `memmap` 保存までを実装。
2. **Pipeline:**
   - `memmap` からスライスして JPEG bytes を返す関数を実装。
3. **Bridge:**
   - `window.register_custom_scheme('soma', ...)` を設定し、ブラウザで `soma://...` を叩いて画像が出るか確認。
4. **Frontend:**
   - Konva.js で単純に画像を並べる実装から始め、徐々に「不要な画像の削除（仮想スクロール）」を追加する。
