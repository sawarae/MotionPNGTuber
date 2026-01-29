# カスタムモデルの使用方法

## 概要

`anime-face-detector-nomm` ライブラリを使用して、カスタムの顔検出器とランドマーク検出器を指定できます。

- **顔検出器**: YOLOv8形式の `.pt` ファイル（例: `models/your_face_model.pt`）
- **ランドマーク検出器**: HRNetV2形式の `.pth` ファイル（例: `models/xxxxx.pth`）

## 使用方法

### 1. コマンドラインからの使用 (face_track_anime_detector.py)

#### カスタムモデルを使用する場合:
```bash
python face_track_anime_detector.py \
  --video "input.mp4" \
  --out "mouth_track.npz" \
  --custom-detector-checkpoint "models/your_face_model.pt" \
  --landmark-model "models/xxxxx.pth" \
  --device cuda:0
```

#### パラメータ説明:
- `--custom-detector-checkpoint`: カスタム顔検出器のパス（.pt）。省略時は自動ダウンロード
- `--landmark-model`: カスタムランドマーク検出器のパス（.pth）。省略時は自動ダウンロード
- `--device`: GPU/CPU指定 (cuda:0, cpu, auto)
- その他のパラメータはいつもどおり使用可能

#### デフォルトモデルで実行する場合:
```bash
python face_track_anime_detector.py \
  --video "input.mp4" \
  --out "mouth_track.npz"
```

### 2. GUIからの使用 (mouth_erase_tuner_gui.py)

1. アプリを起動
2. 左パネル「検出設定」セクションで:
   - `device`: GPU/CPUを選択
   - `face ckpt`: 顔検出器のパスを入力するか、"Browse" ボタンでファイル選択
   - `landmark`: ランドマーク検出器のパスを入力するか、"Browse" ボタンでファイル選択
3. 「検出器再作成」ボタンをクリックして新しいモデルをロード

### 3. 推奨設定

```bash
python face_track_anime_detector.py \
  --video "input.mp4" \
  --out "mouth_track.npz" \
  --custom-detector-checkpoint models/your_face_model.pt \
  --landmark-model models/xxxxx.pth \
  --det-scale 0.75 \
  --pad 2.1 \
  --min-conf 0.5 \
  --smooth-cutoff 3.0
```

## 技術詳細

### API (anime-face-detector-nomm)

コード内で `anime_face_detector.create_detector()` を以下のように呼び出します:

```python
from anime_face_detector import create_detector

detector = create_detector(
    face_detector_checkpoint_path='models/your_face_model.pt',  # カスタム顔検出器（省略可）
    landmark_checkpoint_path='models/xxxxx.pth', # カスタムランドマーク（省略可）
    device='cuda:0',                                         # デバイス指定
    box_scale_factor=1.25,                                   # バウンディングボックス拡大係数
)
```

### 対応ファイル

- **face_track_anime_detector.py**:
  - `--custom-detector-checkpoint`: 顔検出器パス
  - `--landmark-model`: ランドマーク検出器パス

- **mouth_erase_tuner_gui.py**:
  - `face ckpt`: 顔検出器チェックポイント入力フィールド
  - `landmark`: ランドマーク検出器チェックポイント入力フィールド
  - 各フィールドにファイルブラウザ機能

## 注意事項

- 顔検出器は YOLOv8 形式の `.pt` ファイルを使用
- ランドマーク検出器は HRNetV2 形式の `.pth` ファイルを使用
- GPU メモリが不足する場合は `--det-scale` を小さくしてみてください
- カスタムモデルロード時は初回のみ時間がかかります

## トラブルシューティング

### モデルロードエラー
```
[warn] detector init failed on cuda:0: ...
```
- CPUでの実行を試す: `--device cpu`
- モデルパスが正しいか確認
- メモリ不足: `--det-scale` を減らす

### 検出精度が低い
- `--min-conf` を下げてみる
- `--det-scale` を上げる (1.0 = 元のサイズ)
- `--smooth-cutoff` で平滑化を調整

### GUI でチェックポイント反映されない
- 「検出器再作成」ボタンをクリック
- ターミナルのログメッセージを確認
