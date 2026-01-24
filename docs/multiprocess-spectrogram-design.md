# スペクトログラム計算のマルチプロセス化設計

## 1. 背景と問題

### 1.1 現状の問題

現在のスペクトログラム計算は `threading.Thread` を使用しているが、Python の GIL (Global Interpreter Lock) により以下の問題が発生している：

1. **UIのカクつき**: CWT計算中にGILを握るため、pywebviewのUIイベント処理がブロックされる
2. **操作の蓄積による性能劣化**: zoom/pan操作を繰り返すと、複数のスレッドがGILを奪い合い、徐々に動作が重くなる
3. **ファイル切り替え時の顕著な遅延**: 古い計算と新しい計算が同時に走り、競合が悪化

### 1.2 根本原因

```
┌─────────────────────────────────────────────────────────┐
│                  Python プロセス (GIL)                   │
│                                                         │
│  ┌──────────────┐    ┌──────────────┐                   │
│  │ Main Thread  │    │ CWT Thread   │                   │
│  │ (pywebview)  │    │ (NumPy/PyWT) │                   │
│  │              │    │              │                   │
│  │ UIイベント処理 │◄───┤ GIL 競合     │                   │
│  │ evaluate_js  │    │              │                   │
│  └──────────────┘    └──────────────┘                   │
│         ▲                   ▲                           │
│         └───────────────────┘                           │
│              GILを奪い合う                               │
└─────────────────────────────────────────────────────────┘
```

- `threading.Thread` はGILを回避できない
- NumPy/PyWTの一部はCレベルでGILを解放するが、Pythonレベルの処理ではGILを握る
- CWT計算は特に重く、数秒〜数十秒GILを握り続けることがある

## 2. 解決策

**multiprocessing** を使用して、重い計算を別プロセスで実行する。

```
┌─────────────────────────────────────────────────────────────┐
│                     Main Process (UI)                        │
│                                                             │
│  pywebview + SomaDocument                                   │
│      │                                                      │
│      ├─── Viewport Worker Process (最大1つ)                 │
│      │    - STFTまたはCWT計算                               │
│      │    - 新規リクエストで古いプロセスをterminate         │
│      │                                                      │
│      └─── Snap Worker Process (最大1つ)                     │
│           - トレースのCWTスナップ処理                       │
│           - Viewportと並列実行可能                          │
└─────────────────────────────────────────────────────────────┘
```

## 3. 設計詳細

### 3.1 プロセス構成

| プロセス | 用途 | 最大数 | キャンセル方式 |
|----------|------|--------|----------------|
| Main Process | UI (pywebview) | 1 | - |
| Viewport Worker | viewport STFT or CWT | 1 | terminate() |
| Snap Worker | snap CWT | 1 | terminate() |

**不変条件**: 外部ワーカープロセスは常に最大2つ（Viewport用1 + Snap用1）

### 3.2 Overview プレビュー（例外）

Overview（全体プレビュー）は**メインスレッドで実行**する：

- Canvas の最下層レイヤーとして必ず必要
- STFT で高速生成（ブロッキング許容）
- ローディング画面を表示してユーザーを待たせる
- 外部プロセス化しない

### 3.3 Viewport プレビュー

- **30秒超はSTFTのみ、30秒以下はCWTのみ**  
  Overview が常に表示できる前提のため、viewport の低品質STFTは
  30秒超の広域表示に限定し、30秒以下は高品質CWTのみを生成する。
  CWTは毎回新規プロセスで実行し、完了後に破棄してメモリ解放を確実にする。

```python
class ViewportWorker:
    """Viewportプレビュー計算用ワーカー管理"""

    def __init__(self, result_callback: Callable):
        self._process: Process | None = None
        self._result_queue: Queue = Queue()
        self._monitor_thread: Thread | None = None
        self._callback = result_callback
        self._current_request_id: str | None = None

    def submit(self, request_id: str, params: ViewportParams) -> None:
        """新しい計算リクエストを投入（古いプロセスは強制終了）"""
        self._cancel_current()
        self._current_request_id = request_id
        self._process = Process(
            target=_viewport_worker_fn,
            args=(request_id, params, self._result_queue)
        )
        self._process.start()
        self._start_monitor()

    def _cancel_current(self) -> None:
        """現在実行中のプロセスを強制終了"""
        if self._process and self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=1)
            if self._process.is_alive():
                self._process.kill()  # terminateで死なない場合
```

### 3.4 Snap 処理

```python
class SnapWorker:
    """Snapスナップ処理用ワーカー管理"""

    def __init__(self, result_callback: Callable):
        self._process: Process | None = None
        self._result_queue: Queue = Queue()
        self._monitor_thread: Thread | None = None
        self._callback = result_callback

    def submit(self, request_id: str, params: SnapParams) -> None:
        """スナップ計算リクエストを投入"""
        # Snap は基本的に1つずつ処理
        # 既存プロセスがあれば完了を待つか、必要ならキャンセル
        if self._process and self._process.is_alive():
            # 既存の処理が走っている場合の方針:
            # - ユーザー操作起点なので、基本的には待つ
            # - ただし、新しいオーディオファイルを開いた場合などはキャンセル
            pass

        self._process = Process(
            target=_snap_worker_fn,
            args=(request_id, params, self._result_queue)
        )
        self._process.start()
        self._start_monitor()
```

### 3.5 結果の受け取り

別スレッドで Queue を監視し、結果が来たらコールバックを呼ぶ：

```python
def _start_monitor(self) -> None:
    """結果監視スレッドを開始"""
    if self._monitor_thread and self._monitor_thread.is_alive():
        return

    self._monitor_thread = Thread(
        target=self._monitor_results,
        daemon=True
    )
    self._monitor_thread.start()

def _monitor_results(self) -> None:
    """Queue を監視して結果をコールバックに渡す"""
    while True:
        try:
            result = self._result_queue.get(timeout=0.1)
            if result is None:  # 終了シグナル
                break
            self._callback(result)
        except Empty:
            # プロセスが終了していたら監視終了
            if self._process and not self._process.is_alive():
                break
```

## 4. ワーカー関数

### 4.1 Viewport ワーカー

```python
def _viewport_worker_fn(
    request_id: str,
    params: ViewportParams,
    result_queue: Queue
) -> None:
    """別プロセスで実行されるviewport計算"""
    try:
        if params.use_stft:
            # STFT計算（高速、30秒超の広域用）
            stft_result = make_spectrogram_stft(
                audio=params.audio,
                sample_rate=params.sample_rate,
                settings=params.settings,
                time_start=params.time_start,
                time_end=params.time_end,
                freq_min=params.freq_min,
                freq_max=params.freq_max,
                width=params.width,
                height=params.height,
            )
            result_queue.put({
                'type': 'viewport',
                'request_id': request_id,
                'quality': 'low',
                'preview': stft_result,
                'final': True,
            })
            return

        # CWT計算（高品質、30秒以下の詳細用）
        cwt_result = make_spectrogram(
            audio=params.audio,
            sample_rate=params.sample_rate,
            settings=params.settings,
            time_start=params.time_start,
            time_end=params.time_end,
            freq_min=params.freq_min,
            freq_max=params.freq_max,
            width=params.width,
            height=params.height,
        )
        result_queue.put({
            'type': 'viewport',
            'request_id': request_id,
            'quality': 'high',
            'preview': cwt_result,
            'final': True,
        })

    except Exception as e:
        result_queue.put({
            'type': 'viewport',
            'request_id': request_id,
            'error': str(e),
        })
```

### 4.2 Snap ワーカー

```python
def _snap_worker_fn(
    request_id: str,
    params: SnapParams,
    result_queue: Queue
) -> None:
    """別プロセスで実行されるsnap計算"""
    try:
        snapped_points = snap_trace(
            audio=params.audio,
            sample_rate=params.sample_rate,
            settings=params.settings,
            trace=params.trace,
            amp_reference=params.amp_reference,
        )
        result_queue.put({
            'type': 'snap',
            'request_id': request_id,
            'points': snapped_points,
        })

    except Exception as e:
        result_queue.put({
            'type': 'snap',
            'request_id': request_id,
            'error': str(e),
        })
```

## 5. データ転送

### 5.1 プロセス間通信

- **方式**: `multiprocessing.Queue` + pickle
- **コピー**: 発生する（許容）
- **転送データ**:
  - 入力: audio (numpy.ndarray), settings, params
  - 出力: preview (SpectrogramPreview), points (list[PartialPoint])

### 5.2 パラメータ用データクラス

```python
@dataclass
class ViewportParams:
    audio: np.ndarray
    sample_rate: int
    settings: AnalysisSettings
    time_start: float
    time_end: float
    freq_min: float
    freq_max: float
    width: int
    height: int
    use_stft: bool  # 30秒超ならTrue（STFTのみ）
    amp_reference: float | None = None

@dataclass
class SnapParams:
    audio: np.ndarray
    sample_rate: int
    settings: AnalysisSettings
    trace: list[tuple[float, float]]
    amp_reference: float | None = None
```

## 6. エラーハンドリング

### 6.1 タイムアウト

- **タイムアウト時間**: 5分（300秒）
- **処理**: タイムアウト時は `terminate()` でプロセスを強制終了

```python
def _check_timeout(self) -> None:
    if self._process and self._process.is_alive():
        if time.time() - self._start_time > 300:  # 5分
            self._process.terminate()
            self._callback({'type': 'error', 'message': 'Timeout'})
```

### 6.2 クラッシュ時

- ワーカープロセスがクラッシュした場合、`exitcode` をチェック
- エラーをコールバックで通知
- 次のリクエストで新しいプロセスを起動（自動復旧）

```python
def _monitor_results(self) -> None:
    while True:
        # ... Queue監視 ...

        if self._process and not self._process.is_alive():
            if self._process.exitcode != 0:
                self._callback({
                    'type': 'error',
                    'message': f'Worker crashed (exit code: {self._process.exitcode})'
                })
            break
```

## 7. 状態遷移

### 7.1 Viewport Worker

```
     submit(new_request)
           │
           ▼
┌─────────────────────┐
│   既存プロセス確認   │
└─────────────────────┘
           │
     ┌─────┴─────┐
     │           │
  存在する    存在しない
     │           │
     ▼           │
┌──────────┐     │
│terminate │     │
│  + join  │     │
└──────────┘     │
     │           │
     └─────┬─────┘
           │
           ▼
┌─────────────────────┐
│  新プロセス起動      │
│  Process.start()    │
└─────────────────────┘
           │
           ▼
┌─────────────────────┐
│  監視スレッド開始    │
└─────────────────────┘
           │
           ▼
     ┌─────┴─────┐
     │           │
 正常完了    タイムアウト/エラー
     │           │
     ▼           ▼
 callback    terminate
 (result)    + callback(error)
```

## 8. 実装計画

### Phase 1: 基盤

1. `WorkerManager` クラスの作成（Viewport/Snap共通の基底クラス）
2. `ViewportWorker` クラスの実装
3. `SnapWorker` クラスの実装
4. パラメータ用データクラスの定義

### Phase 2: 統合

5. `SomaDocument` への統合
   - `start_viewport_preview_async()` の書き換え
   - `snap_partial()` の書き換え
6. 結果のコールバック処理
   - `_emit_spectrogram_preview()` への接続
   - Partial作成処理への接続

### Phase 3: テスト・検証

7. 不変条件のテスト（最大2プロセス）
8. キャンセル処理のテスト
9. タイムアウト処理のテスト
10. 実際のアプリでの動作確認

## 9. 注意事項

### 9.1 macOS での fork 問題

macOS では `multiprocessing` のデフォルト開始方式が `spawn` に変更されている（Python 3.8+）。
`fork` を使うと、一部のライブラリ（特にGUI関連）で問題が発生する可能性がある。

```python
# 明示的に spawn を使用
import multiprocessing as mp
mp.set_start_method('spawn', force=True)
```

### 9.2 pickle 可能性

プロセス間で渡すオブジェクトは pickle 可能である必要がある：

- `numpy.ndarray`: OK
- `dataclass`: OK（単純な型のみ含む場合）
- ラムダ関数: NG
- ファイルハンドル: NG

### 9.3 デバッグ

マルチプロセスのデバッグは難しい。ログを充実させること：

```python
def _viewport_worker_fn(...):
    import logging
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    logger.debug(f"Viewport worker started: {request_id}")
    # ...
```
