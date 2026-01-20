#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gradio_app.py

MotionPNGTuber用の統合Gradioインターフェース
- スプライト抽出
- トラッキング
- 口消し動画作成
- リアルタイム音声リップシンク実行
"""

from __future__ import annotations

import os
import sys
import time
import threading
import queue
import subprocess
from pathlib import Path

import gradio as gr
import numpy as np
import cv2

# 既存のモジュールをインポート
try:
    import sounddevice as sd
except ImportError:
    sd = None

HERE = os.path.dirname(os.path.abspath(__file__))

# グローバル状態
runtime_process = None
runtime_thread = None
is_running = False
stop_flag = threading.Event()


# ========================================
# ユーティリティ関数
# ========================================

def list_audio_devices():
    """オーディオデバイスのリストを取得"""
    if sd is None:
        return ["デバイスが見つかりません（sounddeviceがインストールされていません）"]

    try:
        devices = []
        for i, d in enumerate(sd.query_devices()):
            if d.get("max_input_channels", 0) > 0:
                name = str(d.get("name", ""))[:64]
                devices.append(f"{i}: {name}")
        return devices if devices else ["入力デバイスが見つかりません"]
    except Exception as e:
        return [f"エラー: {e}"]


def list_sample_videos():
    """サンプル動画のリストを取得"""
    assets_dir = os.path.join(HERE, "assets")
    videos = []

    if os.path.isdir(assets_dir):
        for root, dirs, files in os.walk(assets_dir):
            for file in files:
                if file.endswith((".mp4", ".avi", ".mov")):
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, HERE)
                    videos.append(rel_path)

    return videos if videos else ["サンプル動画が見つかりません"]


def list_mouth_dirs():
    """利用可能なmouth画像フォルダのリストを取得"""
    mouth_dirs = []

    # プロジェクトルートのmouth_dirを確認
    root_mouth_dir = os.path.join(HERE, "mouth_dir")
    if os.path.isdir(root_mouth_dir):
        for entry in os.listdir(root_mouth_dir):
            path = os.path.join(root_mouth_dir, entry)
            if os.path.isdir(path):
                mouth_dirs.append(os.path.relpath(path, HERE))

    # assetsフォルダ内のmouthフォルダを確認
    assets_dir = os.path.join(HERE, "assets")
    if os.path.isdir(assets_dir):
        for root, dirs, files in os.walk(assets_dir):
            if "mouth" in dirs:
                mouth_path = os.path.join(root, "mouth")
                mouth_dirs.append(os.path.relpath(mouth_path, HERE))

    return mouth_dirs if mouth_dirs else ["mouth画像フォルダが見つかりません"]


def validate_inputs(video_path, mouth_dir):
    """入力の検証"""
    errors = []

    if not video_path:
        errors.append("動画ファイルを選択してください")
    elif not os.path.isfile(video_path):
        errors.append(f"動画ファイルが見つかりません: {video_path}")

    if not mouth_dir:
        errors.append("mouth画像フォルダを選択してください")
    elif not os.path.isdir(mouth_dir):
        errors.append(f"mouth画像フォルダが見つかりません: {mouth_dir}")

    return errors


# ========================================
# スプライト抽出
# ========================================

def extract_sprites(video_path, output_dir, feather_px, progress=gr.Progress()):
    """口スプライトを動画から抽出"""
    if not video_path or not os.path.isfile(video_path):
        return "エラー: 動画ファイルを選択してください"

    if not output_dir:
        # デフォルトの出力先: 動画と同じフォルダの mouth/
        video_dir = os.path.dirname(os.path.abspath(video_path))
        output_dir = os.path.join(video_dir, "mouth")

    os.makedirs(output_dir, exist_ok=True)

    progress(0, desc="スプライト抽出を開始...")

    cmd = [
        sys.executable,
        os.path.join(HERE, "mouth_sprite_extractor.py"),
        "--video", video_path,
        "--out", output_dir,
        "--feather", str(feather_px),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600  # 10分タイムアウト
        )

        progress(1.0, desc="完了")

        if result.returncode == 0:
            return f"✅ スプライト抽出が完了しました\n出力先: {output_dir}\n\n{result.stdout}"
        else:
            return f"❌ エラーが発生しました\n\n{result.stderr}\n{result.stdout}"

    except subprocess.TimeoutExpired:
        return "❌ タイムアウト: 処理に10分以上かかりました"
    except Exception as e:
        return f"❌ エラー: {str(e)}"


# ========================================
# トラッキング
# ========================================

def run_tracking(video_path, output_path, min_valid_rate, progress=gr.Progress()):
    """口のトラッキングを実行"""
    if not video_path or not os.path.isfile(video_path):
        return "エラー: 動画ファイルを選択してください"

    if not output_path:
        # デフォルトの出力先: 動画と同じフォルダの mouth_track.npz
        video_dir = os.path.dirname(os.path.abspath(video_path))
        output_path = os.path.join(video_dir, "mouth_track.npz")

    progress(0, desc="トラッキングを開始...")

    cmd = [
        sys.executable,
        os.path.join(HERE, "auto_mouth_track_v2.py"),
        "--video", video_path,
        "--out", output_path,
        "--min-valid-rate", str(min_valid_rate),
    ]

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        output_lines = []
        for line in process.stdout:
            output_lines.append(line.strip())
            # 進捗表示の更新
            if "progress:" in line.lower():
                try:
                    # 進捗パーセントを抽出
                    pct = float(line.split("%")[0].split()[-1]) / 100.0
                    progress(pct, desc=f"トラッキング中... {int(pct*100)}%")
                except:
                    pass

        process.wait()
        progress(1.0, desc="完了")

        output_text = "\n".join(output_lines)

        if process.returncode == 0:
            return f"✅ トラッキングが完了しました\n出力先: {output_path}\n\n{output_text}"
        else:
            return f"❌ エラーが発生しました\n\n{output_text}"

    except Exception as e:
        return f"❌ エラー: {str(e)}"


# ========================================
# 口消し動画作成
# ========================================

def create_mouthless_video(
    video_path,
    track_path,
    output_path,
    coverage,
    ref_sprite_path,
    progress=gr.Progress()
):
    """口消し動画を作成"""
    if not video_path or not os.path.isfile(video_path):
        return "エラー: 動画ファイルを選択してください"

    if not track_path:
        # デフォルトのトラックパス
        video_dir = os.path.dirname(os.path.abspath(video_path))
        track_path = os.path.join(video_dir, "mouth_track.npz")

    if not os.path.isfile(track_path):
        return f"エラー: トラックファイルが見つかりません: {track_path}"

    if not output_path:
        # デフォルトの出力先
        video_dir = os.path.dirname(os.path.abspath(video_path))
        basename = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join(video_dir, f"{basename}_mouthless.mp4")

    progress(0, desc="口消し動画作成を開始...")

    cmd = [
        sys.executable,
        os.path.join(HERE, "auto_erase_mouth.py"),
        "--video", video_path,
        "--track", track_path,
        "--out", output_path,
        "--coverage", str(coverage),
    ]

    if ref_sprite_path and os.path.isfile(ref_sprite_path):
        cmd.extend(["--ref-sprite", ref_sprite_path])

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        output_lines = []
        for line in process.stdout:
            output_lines.append(line.strip())
            # 進捗表示の更新
            if "frame" in line.lower() or "progress" in line.lower():
                try:
                    # フレーム番号から進捗を推定
                    if "/" in line:
                        parts = line.split("/")
                        current = int(parts[0].split()[-1])
                        total = int(parts[1].split()[0])
                        pct = current / max(1, total)
                        progress(pct, desc=f"口消し中... {int(pct*100)}%")
                except:
                    pass

        process.wait()
        progress(1.0, desc="完了")

        output_text = "\n".join(output_lines)

        if process.returncode == 0:
            return f"✅ 口消し動画が完了しました\n出力先: {output_path}\n\n{output_text}"
        else:
            return f"❌ エラーが発生しました\n\n{output_text}"

    except Exception as e:
        return f"❌ エラー: {str(e)}"


# ========================================
# リアルタイム実行
# ========================================

def start_runtime(
    video_path,
    mouth_dir,
    audio_device,
    emotion_auto,
    emotion_preset,
    preview_scale,
    render_fps
):
    """リアルタイム実行を開始"""
    global runtime_process, is_running, stop_flag

    # 検証
    errors = validate_inputs(video_path, mouth_dir)
    if errors:
        return "\n".join(["エラー:"] + errors)

    if is_running:
        return "既に実行中です"

    # オーディオデバイスインデックスを抽出
    device_idx = None
    if audio_device and audio_device.strip():
        try:
            device_idx = int(audio_device.split(":")[0])
        except:
            device_idx = None

    # コマンドライン引数を構築
    cmd = [
        sys.executable,
        os.path.join(HERE, "loop_lipsync_runtime_patched_emotion_auto.py"),
        "--loop-video", video_path,
        "--mouth-dir", mouth_dir,
        "--preview-scale", str(preview_scale),
        "--render-fps", str(render_fps),
    ]

    if device_idx is not None:
        cmd.extend(["--device", str(device_idx)])

    if emotion_auto:
        cmd.append("--emotion-auto")
        preset_map = {
            "安定（配信向け）": "stable",
            "標準": "standard",
            "キビキビ（ゲーム向け）": "snappy"
        }
        preset = preset_map.get(emotion_preset, "standard")
        cmd.extend(["--emotion-preset", preset])

    try:
        # プロセスを起動
        stop_flag.clear()
        runtime_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        is_running = True

        return f"✅ 実行を開始しました\n動画: {video_path}\nmouth: {mouth_dir}\n\n💡 OpenCVウィンドウでプレビューが表示されます（'q'キーで終了）"
    except Exception as e:
        is_running = False
        return f"❌ エラー: 実行の開始に失敗しました\n{str(e)}"


def stop_runtime():
    """リアルタイム実行を停止"""
    global runtime_process, is_running, stop_flag

    if not is_running or runtime_process is None:
        return "実行中のプロセスがありません"

    stop_flag.set()

    try:
        runtime_process.terminate()
        runtime_process.wait(timeout=5)
        is_running = False
        runtime_process = None
        return "✅ 実行を停止しました"
    except subprocess.TimeoutExpired:
        runtime_process.kill()
        runtime_process.wait()
        is_running = False
        runtime_process = None
        return "✅ 実行を強制終了しました"
    except Exception as e:
        return f"❌ エラー: 停止に失敗しました\n{str(e)}"


def get_status():
    """現在の実行状態を取得"""
    global is_running, runtime_process

    if is_running and runtime_process is not None:
        poll = runtime_process.poll()
        if poll is None:
            return "⚡ 実行中"
        else:
            is_running = False
            runtime_process = None
            return f"⏹️ 停止（終了コード: {poll}）"
    else:
        return "⏹️ 停止"


# ========================================
# Gradioインターフェース構築
# ========================================

with gr.Blocks(title="MotionPNGTuber Studio", theme=gr.themes.Soft()) as app:
    gr.Markdown("""
    # 🎭 MotionPNGTuber Studio

    リアルタイム音声リップシンクシステムの統合インターフェース
    """)

    with gr.Tabs():
        # ========================================
        # タブ1: スプライト抽出
        # ========================================
        with gr.Tab("1️⃣ スプライト抽出"):
            gr.Markdown("""
            ### 📸 口スプライト抽出

            動画から5種類の口画像（open, closed, half, e, u）を自動抽出します。
            """)

            with gr.Row():
                with gr.Column():
                    sprite_video = gr.File(
                        label="動画ファイル",
                        file_types=[".mp4", ".avi", ".mov"],
                        type="filepath"
                    )

                    sprite_output_dir = gr.Textbox(
                        label="出力フォルダ（空欄で自動設定）",
                        placeholder="例: assets/assets01/mouth"
                    )

                    sprite_feather = gr.Slider(
                        label="フェザー（ぼかし幅）",
                        minimum=0,
                        maximum=30,
                        value=15,
                        step=1
                    )

                    sprite_extract_btn = gr.Button("🚀 スプライト抽出開始", variant="primary", size="lg")

                with gr.Column():
                    sprite_output = gr.Textbox(
                        label="出力ログ",
                        lines=15,
                        interactive=False
                    )

            sprite_extract_btn.click(
                fn=extract_sprites,
                inputs=[sprite_video, sprite_output_dir, sprite_feather],
                outputs=[sprite_output]
            )

        # ========================================
        # タブ2: トラッキング
        # ========================================
        with gr.Tab("2️⃣ トラッキング"):
            gr.Markdown("""
            ### 🎯 口のトラッキング

            動画内の口の位置を自動追跡してトラックファイル（.npz）を作成します。
            """)

            with gr.Row():
                with gr.Column():
                    track_video = gr.File(
                        label="動画ファイル",
                        file_types=[".mp4", ".avi", ".mov"],
                        type="filepath"
                    )

                    track_output = gr.Textbox(
                        label="出力ファイル（空欄で自動設定）",
                        placeholder="例: assets/assets01/mouth_track.npz"
                    )

                    track_min_valid = gr.Slider(
                        label="最小有効率（品質閾値）",
                        minimum=0.5,
                        maximum=1.0,
                        value=0.85,
                        step=0.05
                    )

                    track_btn = gr.Button("🚀 トラッキング開始", variant="primary", size="lg")

                with gr.Column():
                    track_output_log = gr.Textbox(
                        label="出力ログ",
                        lines=15,
                        interactive=False
                    )

            track_btn.click(
                fn=run_tracking,
                inputs=[track_video, track_output, track_min_valid],
                outputs=[track_output_log]
            )

        # ========================================
        # タブ3: 口消し動画作成
        # ========================================
        with gr.Tab("3️⃣ 口消し動画"):
            gr.Markdown("""
            ### 🎨 口消し動画作成

            トラックファイルを使用して口を消した動画を作成します。
            """)

            with gr.Row():
                with gr.Column():
                    erase_video = gr.File(
                        label="元動画ファイル",
                        file_types=[".mp4", ".avi", ".mov"],
                        type="filepath"
                    )

                    erase_track = gr.Textbox(
                        label="トラックファイル（空欄で自動検索）",
                        placeholder="例: assets/assets01/mouth_track.npz"
                    )

                    erase_output = gr.Textbox(
                        label="出力ファイル（空欄で自動設定）",
                        placeholder="例: assets/assets01/loop_mouthless.mp4"
                    )

                    erase_coverage = gr.Slider(
                        label="口消し強度（0.6〜1.0）",
                        minimum=0.6,
                        maximum=1.0,
                        value=0.85,
                        step=0.05
                    )

                    erase_ref_sprite = gr.File(
                        label="参照スプライト（オプション）",
                        file_types=[".png"],
                        type="filepath"
                    )

                    erase_btn = gr.Button("🚀 口消し動画作成開始", variant="primary", size="lg")

                with gr.Column():
                    erase_output_log = gr.Textbox(
                        label="出力ログ",
                        lines=15,
                        interactive=False
                    )

            erase_btn.click(
                fn=create_mouthless_video,
                inputs=[erase_video, erase_track, erase_output, erase_coverage, erase_ref_sprite],
                outputs=[erase_output_log]
            )

        # ========================================
        # タブ4: リアルタイム実行
        # ========================================
        with gr.Tab("4️⃣ リアルタイム実行"):
            gr.Markdown("""
            ### 🎤 リアルタイム音声リップシンク

            マイク入力に合わせてキャラクターの口を動かします。
            """)

            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("### 📹 動画設定")

                    runtime_video = gr.File(
                        label="動画ファイルをアップロード",
                        file_types=[".mp4", ".avi", ".mov"],
                        type="filepath"
                    )

                    sample_videos = list_sample_videos()
                    runtime_video_dropdown = gr.Dropdown(
                        label="またはサンプル動画を選択",
                        choices=sample_videos,
                        interactive=True
                    )

                    runtime_video_path = gr.Textbox(
                        label="選択された動画パス",
                        interactive=False,
                        value=""
                    )

                    gr.Markdown("### 👄 口画像設定")

                    mouth_dirs = list_mouth_dirs()
                    runtime_mouth_dir = gr.Dropdown(
                        label="mouth画像フォルダを選択",
                        choices=mouth_dirs,
                        interactive=True
                    )

                    runtime_mouth_path = gr.Textbox(
                        label="選択されたmouthフォルダパス",
                        interactive=False,
                        value=""
                    )

                with gr.Column(scale=1):
                    gr.Markdown("### ⚙️ パラメータ設定")

                    audio_devices = list_audio_devices()
                    runtime_audio_device = gr.Dropdown(
                        label="オーディオ入力デバイス",
                        choices=audio_devices,
                        value=audio_devices[0] if audio_devices else None,
                        interactive=True
                    )

                    runtime_emotion_auto = gr.Checkbox(
                        label="感情自動判定を有効にする",
                        value=True
                    )

                    runtime_emotion_preset = gr.Radio(
                        label="感情プリセット",
                        choices=["安定（配信向け）", "標準", "キビキビ（ゲーム向け）"],
                        value="標準",
                        interactive=True
                    )

                    with gr.Accordion("詳細設定", open=False):
                        runtime_preview_scale = gr.Slider(
                            label="プレビュー表示スケール",
                            minimum=0.1,
                            maximum=1.0,
                            value=0.5,
                            step=0.1
                        )

                        runtime_render_fps = gr.Slider(
                            label="レンダリングFPS",
                            minimum=15,
                            maximum=60,
                            value=30,
                            step=5
                        )

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 🎮 実行制御")

                    runtime_status = gr.Textbox(
                        label="状態",
                        value=get_status(),
                        interactive=False
                    )

                    with gr.Row():
                        runtime_start_btn = gr.Button("▶️ 実行開始", variant="primary", size="lg")
                        runtime_stop_btn = gr.Button("⏹️ 停止", variant="stop", size="lg")
                        runtime_refresh_btn = gr.Button("🔄 状態更新", size="sm")

                    runtime_output = gr.Textbox(
                        label="出力ログ",
                        lines=10,
                        interactive=False
                    )

            # イベントハンドラ
            def update_video_path(file):
                if file:
                    return file
                return ""

            def update_video_from_dropdown(choice):
                if choice and choice != "サンプル動画が見つかりません":
                    full_path = os.path.join(HERE, choice)
                    return full_path
                return ""

            def update_mouth_dir(choice):
                if choice and choice != "mouth画像フォルダが見つかりません":
                    full_path = os.path.join(HERE, choice)
                    return full_path
                return ""

            runtime_video.change(
                fn=update_video_path,
                inputs=[runtime_video],
                outputs=[runtime_video_path]
            )

            runtime_video_dropdown.change(
                fn=update_video_from_dropdown,
                inputs=[runtime_video_dropdown],
                outputs=[runtime_video_path]
            )

            runtime_mouth_dir.change(
                fn=update_mouth_dir,
                inputs=[runtime_mouth_dir],
                outputs=[runtime_mouth_path]
            )

            runtime_start_btn.click(
                fn=start_runtime,
                inputs=[
                    runtime_video_path,
                    runtime_mouth_path,
                    runtime_audio_device,
                    runtime_emotion_auto,
                    runtime_emotion_preset,
                    runtime_preview_scale,
                    runtime_render_fps
                ],
                outputs=[runtime_output]
            )

            runtime_stop_btn.click(
                fn=stop_runtime,
                outputs=[runtime_output]
            )

            runtime_refresh_btn.click(
                fn=get_status,
                outputs=[runtime_status]
            )

    # ========================================
    # ヘルプセクション
    # ========================================
    gr.Markdown("""
    ---

    ## 📝 ワークフロー

    ### 初めて使う場合の推奨手順

    1. **スプライト抽出** 👉 動画から5種類の口画像を抽出
    2. **トラッキング** 👉 動画内の口の位置を自動追跡
    3. **口消し動画** 👉 口を消した動画を作成（リップシンクのベース）
    4. **リアルタイム実行** 👉 マイク入力でリアルタイム口パク！

    ### ⚠️ 注意事項

    - 各ステップの出力ファイルは自動的に適切な場所に保存されます
    - リアルタイム実行時のプレビューは別ウィンドウ（OpenCV）で表示されます
    - 処理には時間がかかる場合があります（特にトラッキングと口消し）
    - 初回実行時はモデルのダウンロードが発生する場合があります

    ### 💡 ヒント

    - スプライト抽出の「フェザー」を大きくすると境界がより自然になります
    - トラッキングの「最小有効率」を下げると、トラッキングが成功しやすくなります（品質は下がる可能性あり）
    - 口消し強度を調整して、最適な結果を見つけてください
    """)


def main():
    """アプリケーションのエントリーポイント"""
    print("🎭 MotionPNGTuber Studio を起動中...")
    print(f"📁 作業ディレクトリ: {HERE}")

    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True
    )


if __name__ == "__main__":
    main()
