# SOMA技術設計

## 1. プロジェクト概要

### 1.1 目的

環境音やノイズ、楽器音、声などのスペクトログラムを可視化し、ユーザーが指定した成分（パーシャル）を「音楽的なノート」として抽出するスタンドアロンアプリケーション。抽出結果は MPE (MIDI Polyphonic Expression) および CV (Control Voltage) として出力し、シンセサイザー等での再構築を可能にする。

---

## 2. システムアーキテクチャ

### 2.1 技術スタック

- **Runtime / GUI Wrapper:** `Python 3.1x` + `pywebview`
    - OS標準のレンダリングエンジン (WebKit/WebView2) を使用し、軽量化と配布容易性を確保。
- **Frontend (View/Controller):**
    - `HTML5 / CSS3 / TypeScript`
    - `Konva.js`: スペクトログラム画像とベクター描画（パーシャル）の高速レンダリング（Canvas APIラッパー）。
    - `React`
    - `Tailwind CSS`
- **Backend (Model):**
    - `NumPy`: 行列演算、バッファ管理。
    - `SciPy` / `PyWavelets`: 信号処理（STFT/CWT）、補間。
    - `SoundDevice`: オーディオ再生。
- **Build / CI:**
    - `uv`: パッケージ・依存関係管理。
    - `PyInstaller`: 単体実行ファイル化。
    - `GitHub Actions (Self-hosted)`: ビルドパイプライン。

### 2.2 アーキテクチャ図 (Conceptual)

Frontend (JS) と Backend (Python) は `window.pywebview.api` を介して通信する。

補足:
- パーシャル抽出（スナップ）などの操作系コマンドは通常のリクエスト/レスポンス。
- スペクトログラムプレビューは計算が重く非同期になりやすいため、**フロントは fire & forget でリクエストし、結果は Backend から push 通知**する方式を採用する。

コード スニペット

```mermaid
graph TD
    User[User / UI] -->|Draw/Edit| JS[Frontend JS/Konva]
    JS -->|JSON Request| API[Backend API Python]
    
    subgraph Backend
        API --> Analyzer[Wavelet Analyzer]
        API --> Synth[Destructive Synthesizer]
        API --> Store[Partial Data Store]
        
        Analyzer -->|Snap Logic| Store
        Store -->|Update| Synth
        Synth -->|Audio Stream| AudioDev[Audio Interface]
    end
    
    API -->|JSON Response| JS
    API -.->|Preview Events (Push)| JS
```

---

## 3. モジュール詳細設計

### 3.1 Backend: 解析エンジン (`Analyzer`)

パーシャル抽出（スナップ処理）では、FFT/STFTではなく **Continuous Wavelet Transform (CWT)** を採用する。

- **アルゴリズム:** Complex Morlet Wavelet を使用。
- **理由:** 低域の周波数解像度と高域の時間解像度（過渡特性の捕捉）を両立させるため。また、対数周波数軸との親和性が高い。
- **スナップ処理 (Peak Snapping / Ridge Detection):**
    - **タイミング:** `MouseUp` 時（描画中は生の軌跡を表示し、操作終了後にバックグラウンドで計算して確定させる）。
    - **処理フロー:**
        1. Frontendから軌跡座標 $(t, f_{in})$ のリストを受け取る。
        2. **JIT解析:** 軌跡の周辺時間（$\pm$ 数100ms）のみ、高解像度 Wavelet 変換を行う。
        3. **Ridge Detection:** 各時刻において、$f_{in}$ 近傍の局所最大ピーク（Ridge）を探索し、最も確かな成分をつなぎ合わせて Partial を生成する。
    - **備考:** 描画中のリアルタイムスナップは行わないため、事前計算キャッシュは不要。常に最新・高精度のJIT解析結果を用いる。
- **重要:** STFT は GUI のスペクトログラムプレビュー専用であり、スナップ処理には **絶対に使用しない**。

### 3.2 Backend: 合成・再生エンジン (`Synthesizer`)

1万本以上のパーシャルをレイテンシなしで再生するため、**「破壊的加算 (Destructive Accumulation)」** 方式を採用する。

- **Master Buffer:**
    - `numpy.ndarray` (dtype=`float64`)
    - 長さ: プロジェクトの全尺分（例: 60秒 @ 44.1kHz ≈ 21MB）。
    - *※メモリ容量と精度のバランスを考慮し、64bit floatを採用。*
- **Add / Update Logic:**
    1. パーシャル追加時: サイン波を生成し、Master Buffer に加算 (`+=`)。
    2. パーシャル削除/編集時: 編集前のパラメータでサイン波を再生成し、Master Buffer から減算 (`-=`)。
    - **補足:** 浮動小数点演算の誤差蓄積による Undo/Redo 時の微細なノイズ残り（非可逆性）は許容する。
- **Playback:**
    - 再生ボタン押下時、Master Buffer を `float32` にキャストして `SoundDevice` に流すのみ。計算負荷はパーシャル数に依存しない。
    - 入力音声（原音）と再合成音をミックスして出力する。原音/再合成 は 1ノブで調整可能とする。
    - 再合成中は再生要求を受け付けず、完了後に再生可能とする（UIで待ち状態を明示）。

### 3.3 Data Structure (`Partial`)

パーシャル1本を表すデータモデル。

```python
@dataclass
class PartialPoint:
    time: float       # 秒
    freq: float       # Hz
    amp: float        # 0.0 - 1.0 (Linear amplitude from Wavelet coeff)

@dataclass
class Partial:
    id: str           # UUID
    points: List[PartialPoint]
    is_muted: bool    # 計算対象外フラグ
    # ※ Gain, Pan などの人工的なパラメータは持たない
```

### 3.4 Wavelet解析・計算戦略 (Computation Strategy)

1分以上のオーディオデータを扱う際のメモリ爆発を防ぐため、**「表示用」と「解析用」でデータを分離する二層アーキテクチャ**を採用する。

1. **Visualization Layer (表示用・低解像度)**
    - **目的:** スペクトログラムの全体描画。
    - **手法:** **STFT** を用いた高速プレビューをベースラインとして生成する（時間解像度は出力 `width` に合わせて調整）。
    - **プログレッシブ更新:** 表示窓長 `time_end - time_start` が閾値以下の場合、バックグラウンドで **CWT** による高品質版を生成し差し替える。
    - **タイミング:** 初期化時（overview）およびズーム/パン完了時（viewport）に fire & forget で生成を要求し、結果は Backend から push 通知する。
    - **Peak List Cache (局所最大リストのキャッシュ)**
        - **廃止:** スナップ処理は `MouseUp` 時に JIT 計算を行う方針に変更されたため、広域の Peak List 事前計算・キャッシュは行わない。
        - Visualization Layer は純粋に「背景のスペクトログラム画像」の描画のみを担当する。
2. **Interaction Layer (編集用・高解像度)**
    - **目的:** 描画時のピークスナップ、詳細解析。
    - **手法:** **On-the-fly (JIT) Computation**。全データをメモリに持たず、カーソル周辺の微小時間（例: $\pm 100\text{ms}$）のみを、ユーザー操作が発生した瞬間に局所的に計算する。
    - **データサイズ:** 数 KB 程度（一瞬で破棄）。
    - **タイミング:** マウスダウン / ドラッグ時のみ。
    - **重要:** Interaction Layer の解析は CWT のみで行う（STFT は使わない）。
3. **Persistence (永続化)**
    - 巨大なCWT行列データは `.soma` ファイルに保存しない。
    - 「ソース音声パス」と「解析パラメータ」のみを保存し、次回起動時に Visualization Layer を再計算する。

### 3.5 データの永続化とファイル形式 (Data Persistence & File Format)

- **フォーマット:** JSON形式（拡張子 `.soma`）。
- **設計方針:** "Lightweight Recipe" (軽量なレシピ)。
    - ソース音声へのパスと解析パラメータ、抽出済みのパーシャルデータのみを記録する。
    - **巨大なWavelet解析結果（行列）は保存しない。** ロード時に再計算する。
- **データ構造概略:**
    - `meta`: バージョン情報
    - `source`: ファイルパス、ハッシュ
    - `analysis_settings`: 再現用パラメータ
    - `data.partials`: `[t, f, amp]` の配列リスト

```json
{
  "meta": {
    "format_version": "1.0.0",
    "app_name": "SOMA",
    "created_at": "2023-10-27T10:00:00Z"
  },
  
  "source": {
    "file_path": "./rain.wav",
    "sample_rate": 44100,
    "duration_sec": 60.0,
    "md5_hash": "..." // ファイル整合性チェック用
  },

  "analysis_settings": {
    "freq_min": 20.0,
    "freq_max": 20000.0,
    "bins_per_octave": 48,
    "time_resolution_ms": 10.0
  },

  "data": {
    "partials": [
      {
        "id": "uuid-v4-string",
        "is_muted": false,
        // [Time(s), Freq(Hz), Amp(0.0-1.0)]
        // シンプルな数値配列の羅列
        "points": [
          [1.00, 440.0, 0.5],
          [1.01, 440.1, 0.5],
          [1.02, 440.2, 0.6]
        ]
      },
      // ... 繰り返し
    ]
  }
}
```

### 3.6 Partial Store: Hit Test Index (Selection Acceleration)

- **目的:** Overview上のクリック選択で「最寄りパーシャル」を高速に求める。
- **方式:** タイムラインを固定幅 Δt（例: 50ms〜200ms）で分割し、各チャンクに「その区間に存在するパーシャルID一覧」を保持する。
    - `index[chunk_id] -> List[partial_id]`
- **更新:**
    - partial追加・削除・分割（Erase後の再セグメンテーション）・延長・クロップのたびに、影響するチャンクのみ再構築する。
- **ヒットテスト:**
    - クリック時刻 `t` から該当チャンク `chunk_id` を求め、`index[chunk_id]`（必要に応じて隣接チャンクも）を候補集合として取り出す。
    - 候補集合の各partialについて、クリック点 `(t,f)` と partial（点列）の **最短距離**を計算し、最小のものをヒットとする。
- **備考:** これは再生用リアルタイム合成のためではなく、UI選択を高速にするためのインデックスである。

### 3.7 Export: MPE (SMF)

- 同時発音が 16 本以上になる場合は、SMF を自動分割して複数ファイルとして書き出す。
- ファイル名は同一ベース名に連番を付与する（例: `project_01.mid`, `project_02.mid`）。

---

## 4. 機能仕様とロジック

### 4.1 描画・編集 (Editing Logic)

「移動」や「ゲイン調整」は排除し、解析結果の抽出に特化する。

0. **選択 (Select / Hit Test):**
    - Command Name: `hit_test_partial`
    - **Frontend:**
        - Overview表示（Konva.Image）上でクリックされた座標を `(time_sec, freq_hz)` に変換し Backend へ送信する。
    - **Backend:**
        - クリック点に最も近いパーシャル（点列）を探索し、該当パーシャルIDを返す。
        - 返されたIDのパーシャルのみを Frontend が Edit Layer（Konva.Line）として生成・表示し、編集可能状態にする。
1. **新規描画 (Draw):**
    - Command Name: `trace_partial`
    - **Frontend:** ユーザーのマウス軌跡（Raw）を表示。`MouseUp` 時に軌跡座標列を Backend へ送信。
    - **Backend:** JITスナップ処理を行い、確定した Partial を返す。
    - **Frontend:** 戻ってきた Partial を表示し、Raw軌跡を消去。
2. **延長 (Extend):**
    - Command Name: `extend_partial`
    - 端点をドラッグ。
    - ドラッグした際のユーザーのマウス操作の軌跡をもとにスナップ処理してリッジを partial に追加。
3. **部分消去 (Erase):**
    - Command Name: `apply_eraser`
    - **Frontend:** ユーザーが消しゴムツールでなぞった軌跡（座標リストと許容幅）を `MouseUp` 時に Backend へ送信する。
        
        ```json
        {
          "command": "apply_eraser",
          "payload": {
            // 物理量に変換された消しゴムのサイズ (半分サイズ = 半径相当)
            "tolerance_time_sec": 0.1,    // 時間方向の許容幅 (±0.1秒)
            "tolerance_freq_oct": 0.2,    // 周波数方向の許容幅 (±0.2オクターブ)
        
            // なぞった軌跡のリスト
            // [秒, Hz]
            "path": [
              [1.20, 440.0],
              [1.25, 442.0],
              ...
            ]
          }
        }
        ```
        
        "tolerance_time_sec": 0.1,    // 時間方向の許容幅 (±0.1秒)
        "tolerance_freq_oct": 0.2,    // 周波数方向の許容幅 (±0.2オクターブ)
        
    - **Backend:**
        1. 全パーシャルに対して当たり判定を行う。
        2. 衝突した区間のポイントを削除する。
        3. **再セグメンテーション:** 削除によって不連続（ギャップ）が生じた場合、元のパーシャルを複数の新しいパーシャル（新規UUID付与）に分割する。
        4. 更新されたパーシャルリスト全体を返す。
4. **結合 (Merge):**
    - Command Name: `merge_partials`
    - 離れた2つのパーシャルを選択。
    - **ロジック:** 2点間を直線で結ぶガイドを引き、そのガイド周辺で**再スナップ処理**を行う（単なる直線補間ではない）。これにより、途切れた区間の音成分も正確に拾う。

### 4.2 エクスポート (Export Logic)

### MPE (MIDI Polyphonic Expression)

- **Channel Management:**
    - MPE Zone あたり Ch 2-16 (計15ボイス)。
    - パーシャル数が15を超える場合、**SMF (Standard MIDI File) を分割して出力**する（例: `output_01.mid`, `output_02.mid`...）。
- **Pitch Bend:**
    - Sensitivity: ±48 semitones (Configurable).
    - 各時刻の $f(t)$ を基準ノート（Note On時のNote Number）からの偏差として出力。
    - 偏差がレンジを超過した場合、Note Off/On (Retrigger) を挿入。
- **Amplitude Mapping:**
    - 設定により `Velocity` (Note Onのみ), `Channel Pressure`, `CC74` へマッピング。
    - 変換式: Linear Amplitude -> dB Scale -> MIDI Value (0-127)。

### Audio / CV Output

- **Format:** 1 channel WAV (32bit float) 複数ファイル.
- なるべく voice 数が少なくなるように、パーシャルをボイスに分けてから書き出し。
- 同時に鳴るパーシャルの最大数がボイス数
- 1 voice は 2 ファイル
    - L: Pitch CV (1V/Oct, 0V=C3基準).
    - R: Gate/Amp CV (Linear envelope, DC Coupled).

---

### 4.3 履歴管理システム (History System)

UX（応答速度）とリソース効率を両立するため、**Delta Transaction Model（差分トランザクションモデル）** を採用する。
重い計算（JITスナップ、交差判定、再セグメンテーション等）は「操作時」に一度だけ行い、Undo/Redo時にはその結果（差分）だけを適用する。

#### 対象範囲

- Undo/Redo の対象は「ドキュメント（Partial群・解析設定）」の変更のみとする。
- 選択状態、ズーム/パン、表示モード等の **純UI状態** は Undo/Redo に含めない（必要ならUI専用の別履歴として扱う）。

#### 所有権 (Backend as Source of Truth)

- 履歴（Undo/Redoスタック）とドキュメント状態（Partial Store / Analysis Settings）は Backend が管理する。
- Frontend はコマンド実行/Undo/Redo要求を送信し、Backend は差分（または更新後の必要情報）を返す。
    - これにより、非同期処理があっても「UIだけ戻ってBackendが戻らない」不整合を避ける。

#### イベント種別（ハイブリッド方式）

履歴スタックには2種類のイベントを混在させる。

**A. PartialDelta（軽量・高頻度）**

`trace` / `extend` / `erase` / `merge` / `crop` / `mute` 等の編集操作用。計算後に得られた「結果差分」だけを保持する。

```python
@dataclass(frozen=True)
class PartialDelta:
    event_type: str = "delta"
    added_partials: tuple[Partial, ...]
    removed_partials: tuple[Partial, ...]  # 復元用
```

**B. GlobalSnapshot（重量・低頻度）**

解析設定変更など、全体を無効化/置換する操作用。

```python
@dataclass(frozen=True)
class GlobalSnapshot:
    event_type: str = "snapshot"
    settings: AnalysisSettings
    partials: dict[str, Partial]  # 全データ（不変を前提に保持）
```

#### 処理フロー

**Do / Commit（操作実行）**

1. Logic Phase: JIT解析や幾何計算を実行し、`added` / `removed` を決定する。
2. Commit Phase: `PartialDelta`（または`GlobalSnapshot`）を Undo スタックに積み、Partial Store を更新する。
    - 新しい操作が入った時点で Redo スタックは破棄する。

**Undo（逆適用）**

- Delta: `added` を削除し、`removed` を復元する（計算は行わない）。処理時間は「変更されたPartial数 k」に比例する。
- Snapshot: スナップショット内容で丸ごと置換し、必要に応じて再解析/再合成をトリガーする。

**Redo（順適用）**

- Delta: `removed` を削除し、`added` を追加する。
- Snapshot: Undo と同様に丸ごと置換する。

#### 不変データ前提

- `Partial` は不変（immutable）として扱い、Undo/Redoで同一インスタンスを安全に再利用できる前提を置く。
- 科学計算系（NumPy等）の内部処理は可変でも良いが、Storeへ格納する成果物は不変（または書き換え不能）に寄せる。

---

## 5. UI/UX 設計

### 5.1 画面レイアウト

- **Canvas座標系:**
    - Backend (Hz/Sec) と Frontend (Pixel) の変換は Frontend 側で責務を持つ。
    - Backend は常に物理量 (Hz/Sec) のみを扱う。
- **レイヤー構成 (Konva.js)**
    - Layer 0 (Background): スペクトログラム画像（Konva.Image）
    - Layer 1 (Overview Partials / Raster Cache):
        - 確定済みパーシャル全体の表示。
        - **描画戦略:**
            - Frontend (Konva) 側で Canvas に描画し、それを画像としてキャッシュする (`Konva.Layer.toImage` 等の活用)。
            - **遅延更新 (Debounce):** ユーザーの描画操作中は更新せず、操作完了から一定時間（例: 1秒）経過後に、変更があった区間のみを部分的にラスタライズし直す（Dirty Rect）。
            - これにより、大量のベクタオブジェクトによる描画負荷を回避する。
    - Layer 2 (Interaction / Input):
        - 透明の受け皿（Konva.Rect等）としてポインタイベントを集約し、クリック・ドラッグ・ズームの入力を管理する。
        - OverviewのKonva.Image自体にはパーシャル単位のヒットテストを持たせない。
    - Layer 3 (Edit Partials / Vector):
        - **選択された少数のパーシャルのみ**を Konva.Line 等のベクタとして表示し、ハンドル編集や強調表示を行う。
    - Layer 4 (Playhead / HUD): 再生ヘッド、選択ハイライト、HUD等

詳細は [docs/ui.md](ui.md) を参照。

### 5.2 状態遷移 (State Machine)

非同期処理によるデータの不整合を防ぐ。

コード スニペット

```mermaid
stateDiagram-v2
    direction LR
    [*] --> IDLE

    state "IDLE" as IDLE
    state "EDITING" as EDITING {
        state "Drawing" as DRAW
        state "Modifying" as MOD
        state "Merging" as MRG
    }
    state "PROCESSING" as PROC {
        state "Snap & Calc" as CALC
        state "Buffer Update" as BUF
    }
    state "PLAYING" as PLAY

    IDLE --> DRAW : Mouse Down
    IDLE --> MOD : Drag Handle
    IDLE --> MRG : Select Pair
    IDLE --> CALC : Click (HitTest)

    DRAW --> CALC : Mouse Up
    MOD --> CALC : Mouse Up
    MRG --> CALC : Confirm
    
    CALC --> BUF : Snap Result
    CALC --> IDLE : HitTest Result (Select) 

    BUF --> IDLE : Complete
    
    IDLE --> PLAY : Play Click
    PLAY --> IDLE : Stop Click
```

---

## 6. インフラ・開発環境

### 6.1 依存管理・ビルド

- **Dependency Manager:** `uv`
    - `pyproject.toml` にてバージョンを厳密固定。
- **Packager:** `PyInstaller`
    - NumPy/SciPy の隠蔽された DLL/Dylib 依存関係を Hooks で解決。
    - `pywebview` のブラウザエンジン依存を解決。

### 6.2 CI/CD (GitHub Actions)

- **Runner:** **Self-hosted Runner** (macOS / Windows)
    - クラウドランナーの高コスト回避と、クリーンなビルド環境の維持（毎回 `uv sync --clean` を実行）。
- **Workflow:**
    1. Test (Unit Tests for DSP logic).
    2. Build Frontend (`npm run build`).
    3. Build Backend (`pyinstaller`).
    4. Artifact Upload.

---

## 7. リスクと対策

| **リスク項目** | **対策** |
| --- | --- |
| **メモリ消費** | 破壊的加算方式により、1万パーシャルでもバッファは1本分(20MB強)に抑制。描画パスデータは軽量なため問題なし。 |
| **計算誤差** | `float32` での多数回の加減算によるノイズ蓄積を防ぐため、Master Buffer は `float64` で保持する。 |
| **配布後の起動エラー** | `uv` による厳密なロックと、CI上でのクリーンビルド生成物のみを配布することで、DLL欠損等の環境依存エラーを防ぐ。 |
| **GUI描画負荷** | 確定済みパーシャル全体はベクタノードとして保持せず、まとめ描きして **Konva.Image（ラスタ）** として表示する。編集対象は選択された少数パーシャルのみを **Konva.Line等のベクタ**で表示する。Overview表示では編集操作を限定し、ズーム後に編集可能とする。 |
