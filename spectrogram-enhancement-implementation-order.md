# Spectrogram Enhancement 実装順

## 前提（今回確定）

- 画像転送形式は `JPEG` に統一する（デバッグ用の生配列経路は作らない）。
- スペクトログラムの画像化（dB正規化・ゲイン/レンジ/ガンマ適用・カラーマップ）は **Python側で実施** する。
- Frontend は画像を表示するだけに寄せる（現在の `data[] -> RGBA` 変換処理は削除）。
- 制御プレーンは既存の pywebview API/event を基本維持し、データプレーンをタイル画像配信へ置き換える。

## 実装ステップ（完了）

1. [x] **Backend: タイルレンダラ基盤を追加**
- 追加: `src/soma/spectrogram_renderer.py`（新規）
- 実装:
  - Multi-Resolution STFT（Low/Mid/High）生成
  - `numpy.memmap` 保存・読込
  - `time/freq` 範囲スライス
  - `gain/min_db/max_db/gamma` 適用
  - カラーマップ適用
  - JPEGエンコード
- 目的: 「要求範囲 -> JPEG bytes」を1関数で返せる状態を作る。

2. [x] **Backend: セッションにレンダラ寿命管理を追加**
- 変更: `src/soma/session.py`, `src/soma/services/project_service.py`
- 実装:
  - 音声ロード時にレンダラ初期化（段階的ビルド開始）
  - プロジェクト切替/クローズ時にmemmap/一時ファイルを解放
- 目的: リソースリーク防止と再ロードの安定化。

3. [x] **Backend: 画像配信エンドポイントを追加**
- 変更: `src/soma/app.py`, `src/soma/api_schema.py`
- 実装:
  - `overview` 取得API（初期表示用）
  - `tile` 取得API（viewport用）
  - リクエストに `time range`, `freq range`, `px size`, `gain/min_db/max_db/gamma`, `quality` を含める
  - レスポンスは JPEG の参照URL（または直接バイナリ返却のラッパ）へ統一
- 目的: Frontend が data配列を扱わず画像のみ扱う構造へ移行。

4. [x] **Backend: プレビューキャッシュを JPEG前提に切替**
- 変更: `src/soma/preview_cache.py`, `src/soma/cache_server.py`, `src/soma/app.py`
- 実装:
  - `.bin` 前提処理を整理し、画像ファイルキャッシュ（`.jpg`）へ寄せる
  - 既存 `data_path/data_length` 依存を段階廃止
- 目的: 転送とキャッシュの責務を画像中心に一本化。

5. [x] **Frontend: 画像生成ロジックを削除し表示専用化**
- 変更:
  - `frontend/src/components/Workspace/internal/useWorkspaceController.ts`
  - `frontend/src/workers/viewportRenderer.ts`（削除）
  - `frontend/src/components/Workspace/internal/useViewportImageCache.ts`
  - `frontend/src/app/previewData.ts`
  - `frontend/src/app/utils.ts`（`mapColor` など不要部削除）
- 実装:
  - `preview.data` を前提にしたピクセル変換を除去
  - URLベースの画像ロードに置換
  - viewportタイル画像のキャッシュ・破棄をURL単位で管理
- 目的: ブラウザ側CPU/GC負荷を削減し、責務を明確化。

6. [x] **Frontend: タイルマネージャ実装**
- 変更: `frontend/src/hooks/useViewport.ts`, `frontend/src/components/Workspace/internal/WorkspaceCanvas.tsx`
- 実装:
  - 1024pxタイル、overscan 1.0x、debounce 200ms
  - 可視タイル ± 先読みのみ保持（6〜8枚）
  - パン/ズーム時に不足タイルを非同期要求
- 目的: 大規模音声でもDOM/メモリを一定に保つ。

7. [x] **Frontend+Backend: 表示パラメータを dB 系へ統一**
- 変更:
  - `src/soma/models.py`（設定項目）
  - `src/soma/api_schema.py`
  - `frontend/src/app/types.ts`
  - `frontend/src/app/apiSchemas.ts`
  - `frontend/src/components/modals/AnalysisSettingsModal.tsx`
- 実装:
  - `brightness/contrast` から `gain/min_db/max_db/gamma` へ置換
  - GUI文言を英語で更新
- 目的: 音響的に解釈可能な調整軸へ一本化。

8. [x] **旧プレビュー経路の削除と互換整理**
- 変更: `src/soma/services/preview_service.py`, `src/soma/workers.py`, 関連フロント受信コード
- 実装:
  - `SpectrogramPreview(data)` 前提のイベント経路を削除
  - タイル/overview取得フローに完全移行
- 目的: 二重経路をなくし保守コストを下げる。

9. [x] **最小限テスト追加**
- 変更: `tests/test_analysis.py`（または新規 `tests/test_spectrogram_renderer.py`）, フロント最小テスト
- 実装:
  - タイル切出し境界
  - dB正規化とパラメータ適用
  - JPEG生成の正常系
  - viewportタイル保持数上限
- 目的: 主要リグレッションのみを短時間で検知。

10. [x] **性能計測と閾値確認**
- 追加: ベンチ/ログ出力（`logs/`）
- 実装:
  - warm cache 条件で viewport 更新 p95 <= 100ms を測定
  - cold 条件は別指標で記録
- 目的: 目標値に対する達成/未達を定量判断できる状態にする。

## 完了条件

- Frontend でのスペクトログラム画像化コードが削除されている。
- Python側でタイルJPEGを生成し、overview + viewport で表示できる。
- 1時間音声で操作時、warm cache の viewport 更新 p95 が 100ms 以内。
