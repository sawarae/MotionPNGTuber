# カスタムモデル (FacesV1.pt) の使用方法

## 概要

`models/FacesV1.pt` をカスタム顔検出器として使用できるようにアップデートしました。

## 使用方法

### 1. コマンドラインからの使用 (face_track_anime_detector.py)

#### FacesV1.pt を使用する場合:
```bash
python face_track_anime_detector.py \
  --video "input.mp4" \
  --out "mouth_track.npz" \
  --model yolov8 \
  --custom-detector-checkpoint "models/FacesV1.pt" \
  --landmark-model hrnetv2 \
  --device cuda:0
```

#### パラメータ説明:
- `--model` (yolov3 | yolov8): ベースの検出モデル選択。FacesV1.ptを使う場合は `yolov8` を推奨
- `--custom-detector-checkpoint`: カスタムモデルのパス (例: models/FacesV1.pt)
- `--landmark-model` (hrnetv2 | mobilenetv2): ランドマーク検出モデル。`hrnetv2` がより正確です
- `--device`: GPU/CPU指定 (cuda:0, cpu など)
- その他のパラメータはいつもどおり使用可能

#### デフォルトモデル (yolov3) で実行する場合:
```bash
python face_track_anime_detector.py \
  --video "input.mp4" \
  --out "mouth_track.npz"
```

### 2. GUIからの使用 (mouth_erase_tuner_gui.py)

1. アプリを起動
2. 左パネル「検出設定」セクションで:
   - `device`: GPU/CPUを選択
   - `checkpoint`: テキストフィールドにモデルパスを入力するか、"Browse" ボタンでファイル選択
   - 例: `models/FacesV1.pt`
3. 「検出器再作成」ボタンをクリックして新しいモデルをロード

### 3. 推奨設定

#### FacesV1.pt 使用時:
```bash
--model yolov8 \
--landmark-model hrnetv2 \
--custom-detector-checkpoint models/FacesV1.pt \
--det-scale 0.75 \
--pad 2.1 \
--min-conf 0.5 \
--smooth-cutoff 3.0
```

## 技術詳細

### API の変更

コード内で `anime_face_detector.create_detector()` を以下のように呼び出します:

```python
detector = create_detector(
    face_detector_name='yolov8',           # 'yolov3' or 'yolov8'
    landmark_model_name='hrnetv2',         # ランドマーク検出モデル
    device='cuda:0',                       # デバイス指定
    custom_detector_checkpoint_path='models/FacesV1.pt',  # カスタムモデル
    detector_framework='ultralytics'       # yolov8 使用時は 'ultralytics'
)
```

### 対応ファイル

- **face_track_anime_detector.py**: 
  - `--custom-detector-checkpoint` オプション追加
  - `--landmark-model` オプション追加
  - `create_detector()` 呼び出しをAPI形式に更新
  
- **mouth_erase_tuner_gui.py**:
  - チェックポイント入力フィールド追加
  - ファイルブラウザ機能追加
  - 同じ `create_detector()` API 更新

## 注意事項

- FacesV1.pt が正しいフォーマット (YOLOv8 .pt) であることを確認してください
- GPU メモリが不足する場合は `--det-scale` を小さくしてみてください
- ランドマーク検出の精度は `--landmark-model hrnetv2` がおすすめです
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
