# SOMA UI Structure Definition (v1.1)

**Philosophy:** Canvas First / Flat Access

## 1. Main Window Layout (常時表示エリア)

画面全体は3つの固定領域で構成されます。

### 1-A. Header Toolbar (高さ: 48px)

左から右への作業フローに基づいた配置です。

- **Left Group (System Access)**
    - `[Menu Button (☰)]`: クリックするとフラットなメニューリストを展開（詳細は後述）。
    - `[App Title]`: "SOMA" ロゴ。
- **Center Group (Transport)**
    - `[Stop (◼)]`: 停止＆先頭へ戻る。
    - `[Play/Pause (▶/II)]`: 再生トグル（再合成中は無効）。
    - `[Loop (↻)]`: ループ再生 ON/OFF。
    - `[原音/再合成]`: 入力音（原音）と再合成音のミックス比。
    - `[Time Display]`: `00:00:00.000` 形式の再生位置カウンター。
- **Right Group (Tools & Actions)**
    - **Tool Switcher:** (ラジオボタン形式 / ショートカット対応)
        - `[Select (V)]`: 選択、矩形選択。
        - `[Trace (P)]`: ペンシル描画（スナップ有効）。
        - `[Erase (E)]`: 部分削除。
        - `[Connect (C)]`: マージ（2点結合）。
    - **Primary Action:**
        - `[Export Button]`: 書き出しダイアログを開く。

### 1-B. Workspace (最大化)

- **Main Canvas:**
    - `Layer 0`: スペクトログラム画像 (Wavelet Heatmap / Konva.Image)
    - `Layer 1`: パーシャル表示（Overview / Raster Image）
        - 確定済みパーシャル全体はラスタ化して表示（性能優先）
    - `Layer 2`: 入力受付（透明レイヤ。クリック/ドラッグ/ズーム）
    - `Layer 3`: 編集対象パーシャル（Edit / Vector）
        - 選択中の少数パーシャルのみベクタ＋ハンドル表示
        - 端点ハンドルのドラッグで延長/トリム（クロップ）を行う
    - `Layer 4`: 再生ヘッド、選択ハイライト、HUD
- **Rulers:**
    - `Top`: 時間軸 (Time/Bars)。
    - `Right`: 周波数軸 (Log Hz / Note Name)。
- **Navigation:**
    - `Scrollbars`: 極細
    - `Zoom`: ホイール/ピンチ操作。

### 1-C. Status Bar

- **Left:** システム状態 (`Ready`, `Analyzing...`, `Resynthesizing...`, `Playing`).
- **Center:** カーソル情報 (`T: 12.3s | F: 440Hz (A4) | A: -6dB`).
- **Right:** デバッグ/リソース情報 (`Mem: 42MB`).

---

## 2. Main Menu Structure (フラットリスト)

ハンバーガーボタン [☰] を押した際に表示されるドロップダウンリスト。

階層（サブメニュー）を持たず、機能カテゴリごとに区切り線（Separator）でグループ化して表示します。

**[Project Operations]**

1. `New Project`
2. `Open Project...`
3. `Save Project`
4. `Save As...`
    - *(Separator)*

[Analysis & Settings]

5.  Analysis Settings... (モーダル起動)

6.  Plugin Manager... (将来用)

- (Separator)

[View Operations]

7.  Zoom In

8.  Zoom Out

9.  Reset View (全体表示)

- (Separator)

[System]

10. About SOMA

11. Quit

---

## 3. Modal Overlays (ポップアップ画面)

必要な時だけ中央に表示され、背景は暗転（Dimmed）します。

### 3-A. Analysis Settings Modal

プロジェクトの解析パラメータを定義します。

- **Frequency Range:** Min / Max (Hz)
- **Time Resolution:** Bins per Octave (48, 60, 96...)
- **Visualization:** Magma (fixed), Brightness, Contrast
- **[Cancel]** / **[Apply & Re-analyze]**

### 3-B. Export Modal

- **Tab: MPE (MIDI)**
    - Pitch Bend Range
    - Amplitude Mapping (Velocity / Pressure / CC74)
    - 同時発音が 16本以上の場合は自動的に複数のSMFを出力（連番付与）
- **Tab: Multi-Track MIDI**
    - Pitch Bend Range
    - Amplitude Mapping (Velocity / Pressure / CC74)
    - 1トラック = 1ボイス、全トラックは同一チャンネル
- **Tab: Monophonic MIDI**
    - Pitch Bend Range
    - Amplitude Mapping (Velocity / Pressure / CC74)
    - 1トラック1チャンネル、ノートは重なり可（モノシンセ側でレガート想定）
- **Tab: Audio / CV**
    - Output Type (Sine Synthesis / CV Control)
    - Sample Rate / Bit Depth
- **[Cancel]** / **[Export File...]**

---

### 4. Contextual UI (状況依存)

- **Selection HUD:**
    - キャンバス上でパーシャルを選択した際、カーソル付近に小さなフローティングパネルを表示。
    - 内容: `Info (ID/Freq)`, `[Mute]`, `[Delete]`。
