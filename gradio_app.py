#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gradio_app.py

MotionPNGTuber用のGradioインターフェース
- 動画とmouth画像フォルダを選択
- リアルタイム音声リップシンク実行
- 各種パラメータ設定
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

        return f"実行を開始しました\n動画: {video_path}\nmouth: {mouth_dir}"
    except Exception as e:
        is_running = False
        return f"エラー: 実行の開始に失敗しました\n{str(e)}"


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
        return "実行を停止しました"
    except subprocess.TimeoutExpired:
        runtime_process.kill()
        runtime_process.wait()
        is_running = False
        runtime_process = None
        return "実行を強制終了しました"
    except Exception as e:
        return f"エラー: 停止に失敗しました\n{str(e)}"


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


# Gradioインターフェース構築
with gr.Blocks(title="MotionPNGTuber", theme=gr.themes.Soft()) as app:
    gr.Markdown("""
    # 🎭 MotionPNGTuber - Gradioインターフェース

    リアルタイム音声リップシンクシステム
    """)

    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("### 📹 動画設定")

            # 動画ファイル選択
            video_file = gr.File(
                label="動画ファイルをアップロード",
                file_types=[".mp4", ".avi", ".mov"],
                type="filepath"
            )

            sample_videos = list_sample_videos()
            video_dropdown = gr.Dropdown(
                label="またはサンプル動画を選択",
                choices=sample_videos,
                interactive=True
            )

            video_path_display = gr.Textbox(
                label="選択された動画パス",
                interactive=False,
                value=""
            )

            # mouth画像フォルダ選択
            gr.Markdown("### 👄 口画像設定")

            mouth_dirs = list_mouth_dirs()
            mouth_dir_dropdown = gr.Dropdown(
                label="mouth画像フォルダを選択",
                choices=mouth_dirs,
                interactive=True
            )

            mouth_dir_display = gr.Textbox(
                label="選択されたmouthフォルダパス",
                interactive=False,
                value=""
            )

        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ パラメータ設定")

            # オーディオデバイス選択
            audio_devices = list_audio_devices()
            audio_device = gr.Dropdown(
                label="オーディオ入力デバイス",
                choices=audio_devices,
                value=audio_devices[0] if audio_devices else None,
                interactive=True
            )

            # 感情自動判定
            emotion_auto = gr.Checkbox(
                label="感情自動判定を有効にする",
                value=True
            )

            emotion_preset = gr.Radio(
                label="感情プリセット",
                choices=["安定（配信向け）", "標準", "キビキビ（ゲーム向け）"],
                value="標準",
                interactive=True
            )

            # その他のパラメータ
            with gr.Accordion("詳細設定", open=False):
                preview_scale = gr.Slider(
                    label="プレビュー表示スケール",
                    minimum=0.1,
                    maximum=1.0,
                    value=0.5,
                    step=0.1
                )

                render_fps = gr.Slider(
                    label="レンダリングFPS",
                    minimum=15,
                    maximum=60,
                    value=30,
                    step=5
                )

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 🎮 実行制御")

            status_text = gr.Textbox(
                label="状態",
                value=get_status(),
                interactive=False
            )

            with gr.Row():
                start_btn = gr.Button("▶️ 実行開始", variant="primary", size="lg")
                stop_btn = gr.Button("⏹️ 停止", variant="stop", size="lg")
                refresh_status_btn = gr.Button("🔄 状態更新", size="sm")

            output_text = gr.Textbox(
                label="出力ログ",
                lines=10,
                interactive=False
            )

    gr.Markdown("""
    ### 📝 使い方

    1. **動画を選択**: 動画ファイルをアップロードするか、サンプル動画から選択
    2. **mouth画像を選択**: 口の表情画像が入ったフォルダを選択
    3. **パラメータ設定**: オーディオデバイスや感情設定を調整
    4. **実行開始**: ▶️ボタンを押してリアルタイム実行開始
    5. **OpenCVウィンドウ**: 別ウィンドウでプレビューが表示されます（'q'キーで終了）

    ### ⚠️ 注意事項

    - プレビューはOpenCVの別ウィンドウで表示されます
    - 終了するには停止ボタンを押すか、OpenCVウィンドウで'q'キーを押してください
    - 初回実行時はモデルのロードに時間がかかる場合があります
    """)

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

    video_file.change(
        fn=update_video_path,
        inputs=[video_file],
        outputs=[video_path_display]
    )

    video_dropdown.change(
        fn=update_video_from_dropdown,
        inputs=[video_dropdown],
        outputs=[video_path_display]
    )

    mouth_dir_dropdown.change(
        fn=update_mouth_dir,
        inputs=[mouth_dir_dropdown],
        outputs=[mouth_dir_display]
    )

    start_btn.click(
        fn=start_runtime,
        inputs=[
            video_path_display,
            mouth_dir_display,
            audio_device,
            emotion_auto,
            emotion_preset,
            preview_scale,
            render_fps
        ],
        outputs=[output_text]
    )

    stop_btn.click(
        fn=stop_runtime,
        outputs=[output_text]
    )

    refresh_status_btn.click(
        fn=get_status,
        outputs=[status_text]
    )


def main():
    """アプリケーションのエントリーポイント"""
    print("🎭 MotionPNGTuber Gradioインターフェースを起動中...")
    print(f"📁 作業ディレクトリ: {HERE}")

    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True
    )


if __name__ == "__main__":
    main()
