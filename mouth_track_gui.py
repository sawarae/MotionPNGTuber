#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mouth_track_gui.py  (One-Click HQ)

最高品質・設定最小の「一気通貫」GUI
- 動画を選ぶ
- 解析 (auto_mouth_track_v2.py)  ※自動修復 + early-stop
- 解析後にキャリブ画面を自動表示 (calibrate_mouth_track.py)
- キャリブが終わったら口消しを自動生成 (auto_erase_mouth.py)
- 最後に口消し動画を自動プレビュー

ユーザーが触る設定:
- 口消し強さ (coverage)

注意:
- サブプロセス出力の文字化け/Unicode問題を避けるため、UTF-8環境変数を付与します。
"""

from __future__ import annotations

import os
import sys
import json
import shutil
from pathlib import Path
import queue
import threading
import subprocess
import signal
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox


HERE = os.path.dirname(os.path.abspath(__file__))
LAST_SESSION_FILE = os.path.join(HERE, ".mouth_track_last_session.json")


# --- smoothing presets (GUI) ---
SMOOTHING_PRESETS: dict[str, float | None] = {
    "Auto（今のまま）": None,  # pass nothing -> keep current default behavior
    "ゆっくり（1.5）": 1.5,
    "普通（3.0）": 3.0,
    "高速（6.0）": 6.0,
    "追従最優先（0）": 0.0,  # disable smoothing
}
SMOOTHING_LABELS = list(SMOOTHING_PRESETS.keys())




# --- emotion preset (GUI / runtime) ---
EMOTION_PRESETS: dict[str, str] = {
    "安定（配信向け）": "stable",
    "標準": "standard",
    "キビキビ（ゲーム向け）": "snappy",
}
EMOTION_PRESET_LABELS = list(EMOTION_PRESETS.keys())

SOFT_STOP_GRACE_SEC = 3.0
STOP_BTN_TEXT_DEFAULT = "中断（現在の処理が終わったら停止）"
STOP_BTN_TEXT_SOFT = "停止予約中（もう一度で強制停止）"
MAX_LOG_LINES = 200  # ログ表示の上限行数

# ---------- helpers ----------
def _script_contains(path: str, needles: list[str]) -> bool:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            s = f.read()
        return all(n in s for n in needles)
    except Exception:
        return False


def _try_list_input_devices() -> list[tuple[int, str]]:
    """
    Returns list of (index, display_name) for input devices.
    Uses sounddevice if available.
    """
    try:
        import sounddevice as sd  # type: ignore
        devices = []
        for i, d in enumerate(sd.query_devices()):
            if d.get("max_input_channels", 0) > 0:
                name = str(d.get("name", ""))[:64]
                ch = int(d.get("max_input_channels", 0))
                sr = int(float(d.get("default_samplerate", 0)) or 0)
                devices.append((i, f"{i}: {name}  (ch={ch}, sr={sr})"))
        return devices
    except Exception:
        return []


def _ensure_backend_sanity(base_dir: str) -> tuple[bool, str]:
    """
    Prevent the common "file got swapped/overwritten" situation.
    """
    track_py = os.path.join(base_dir, "auto_mouth_track_v2.py")
    erase_py = os.path.join(base_dir, "auto_erase_mouth.py")

    if not os.path.isfile(track_py):
        return False, "auto_mouth_track_v2.py が見つかりません。"
    if not os.path.isfile(erase_py):
        return False, "auto_erase_mouth.py が見つかりません。"

    # Track script should mention pad/det-scale/min-conf.
    if not _script_contains(track_py, ["--pad", "--det-scale", "--min-conf"]):
        return (
            False,
            "auto_mouth_track_v2.py が追跡用スクリプトではないようです（--pad 等が見つかりません）。\n"
            "ファイルが入れ替わっていないか確認してください。",
        )

    # Erase script should mention --track / --coverage.
    if not _script_contains(erase_py, ["--track", "--coverage"]):
        return (
            False,
            "auto_erase_mouth.py が口消し用スクリプトではないようです（--track/--coverage が見つかりません）。\n"
            "ファイルが入れ替わっていないか確認してください。",
        )

    return True, ""


def guess_mouth_dir(video_path: str) -> str:
    base_dir = os.path.dirname(os.path.abspath(video_path))
    cand = os.path.join(base_dir, "mouth")
    if os.path.isdir(cand):
        return cand
    cand = os.path.join(HERE, "mouth")
    if os.path.isdir(cand):
        return cand
    return ""


def best_open_sprite(mouth_dir: str) -> str:
    if not mouth_dir or not os.path.isdir(mouth_dir):
        return ""
    p = os.path.join(mouth_dir, "open.png")
    if os.path.isfile(p):
        return p
    try:
        for name in os.listdir(mouth_dir):
            if name.lower() == "open.png":
                p2 = os.path.join(mouth_dir, name)
                if os.path.isfile(p2):
                    return p2
    except Exception:
        pass
    return ""

def _safe_bool(v, default: bool = False) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return default
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "y", "on"):
        return True
    if s in ("0", "false", "no", "n", "off"):
        return False
    return default


def _safe_int(
    v, default: int, min_v: int | None = None, max_v: int | None = None
) -> int:
    try:
        if isinstance(v, str):
            v = v.split(":", 1)[0].strip()
        iv = int(float(v)) if isinstance(v, str) else int(v)
    except Exception:
        return default
    if min_v is not None:
        iv = max(min_v, iv)
    if max_v is not None:
        iv = min(max_v, iv)
    return iv


def _safe_float(
    v, default: float, min_v: float | None = None, max_v: float | None = None
) -> float:
    try:
        fv = float(v)
    except Exception:
        return default
    if min_v is not None:
        fv = max(min_v, fv)
    if max_v is not None:
        fv = min(max_v, fv)
    return fv


_EMOTION_DIR_NAMES = {"default", "neutral", "happy", "angry", "sad", "excited"}


def _is_emotion_level_mouth_root(mouth_root: str) -> bool:
    """Heuristic: mouth_root is already a character directory (no character layer),
    if it contains open.png directly OR contains multiple emotion-named subfolders."""
    if not mouth_root or not os.path.isdir(mouth_root):
        return False
    if os.path.isfile(os.path.join(mouth_root, "open.png")):
        return True
    try:
        subs = [d for d in os.listdir(mouth_root) if os.path.isdir(os.path.join(mouth_root, d))]
        low = {d.lower() for d in subs}
        return len(low & _EMOTION_DIR_NAMES) >= 2
    except Exception:
        return False


def list_character_dirs(mouth_root: str) -> list[str]:
    """Return character folder candidates under mouth_root.
    If mouth_root looks like an emotion-level folder, return []."""
    if not mouth_root or not os.path.isdir(mouth_root):
        return []
    if _is_emotion_level_mouth_root(mouth_root):
        return []
    try:
        subs = [d for d in os.listdir(mouth_root) if os.path.isdir(os.path.join(mouth_root, d))]
        # Exclude emotion folder names just in case
        chars = [d for d in subs if d.lower() not in _EMOTION_DIR_NAMES]
        chars.sort(key=lambda x: x.lower())
        return chars
    except Exception:
        return []


def resolve_character_dir(mouth_root: str, character: str) -> str:
    """Resolve mouth directory passed to runtime / used for sprite search.
    If character is valid, use mouth_root/character, else use mouth_root."""
    if not mouth_root:
        return ""
    if character:
        cand = os.path.join(mouth_root, character)
        if os.path.isdir(cand):
            return cand
    return mouth_root


def best_open_sprite_for_character(mouth_root: str, character: str) -> str:
    """Find open.png for calibration.
    Priority:
      1) <mouth_dir>/open.png (backward compat)
      2) <mouth_dir>/(Default|neutral|...)/open.png
      3) first found in immediate subfolders
    where <mouth_dir> is mouth_root or mouth_root/character.
    """
    base = resolve_character_dir(mouth_root, character)
    if not base or not os.path.isdir(base):
        return ""

    # 1) direct
    p = os.path.join(base, "open.png")
    if os.path.isfile(p):
        return p
    try:
        for name in os.listdir(base):
            if name.lower() == "open.png":
                p2 = os.path.join(base, name)
                if os.path.isfile(p2):
                    return p2
    except Exception:
        pass

    # 2) preferred emotion folders
    preferred = ["Default", "default", "neutral", "Neutral", "Normal", "normal"]
    for em in preferred:
        d = os.path.join(base, em)
        if not os.path.isdir(d):
            continue
        p = os.path.join(d, "open.png")
        if os.path.isfile(p):
            return p
        try:
            for name in os.listdir(d):
                if name.lower() == "open.png":
                    p2 = os.path.join(d, name)
                    if os.path.isfile(p2):
                        return p2
        except Exception:
            pass

    # 3) any immediate subfolder
    try:
        for sub in os.listdir(base):
            d = os.path.join(base, sub)
            if not os.path.isdir(d):
                continue
            p = os.path.join(d, "open.png")
            if os.path.isfile(p):
                return p
            for name in os.listdir(d):
                if name.lower() == "open.png":
                    p2 = os.path.join(d, name)
                    if os.path.isfile(p2):
                        return p2
    except Exception:
        pass

    return ""



def load_session() -> dict:
    try:
        with open(LAST_SESSION_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_session(d: dict) -> None:
    """Persist session data.

    NOTE: We merge with the existing session so GUI actions that save a single
    field (e.g., audio device) do not wipe other settings.
    """
    try:
        cur = load_session()
        if not isinstance(cur, dict):
            cur = {}
        cur.update(d)
        with open(LAST_SESSION_FILE, "w", encoding="utf-8") as f:
            json.dump(cur, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


# ---------- GUI ----------
class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Mouth Track One-Click (HQ)")
        self.geometry("840x560")

        self.log_q: "queue.Queue[str]" = queue.Queue()
        self.error_q: "queue.Queue[tuple[str, str]]" = queue.Queue()  # (title, msg)
        self.ui_q: "queue.Queue[callable]" = queue.Queue()  # UI更新用キュー
        self.worker_thread: threading.Thread | None = None
        self.stop_flag = threading.Event()
        self.active_proc: subprocess.Popen | None = None
        self.stop_mode = "none"  # none / soft / force
        self.soft_requested_at: float | None = None
        self._soft_warn_job: str | None = None

        sess = load_session()
        # GUIでは元動画を表示したいが、runtimeは背景としてmouthlessを使いたいので
        # sessionには video(=背景用) と source_video(=元動画) を分けて保存する
        self.video_var = tk.StringVar(value=str(sess.get("source_video", sess.get("video", "")) or ""))
        self.mouth_dir_var = tk.StringVar(value=str(sess.get("mouth_dir", "")) or "")

        # --- character / emotion-auto (runtime) ---
        self.character_var = tk.StringVar(value=str(sess.get("character", "")))

        _ep = str(sess.get("emotion_preset", "標準"))
        if _ep not in EMOTION_PRESETS:
            _ep = "標準"
        self.emotion_preset_var = tk.StringVar(value=_ep)

        self.emotion_hud_var = tk.BooleanVar(value=_safe_bool(sess.get("emotion_hud", True), default=True))
        self.coverage_var = tk.DoubleVar(value=_safe_float(sess.get("coverage", 0.60), 0.60, min_v=0.40, max_v=0.90))
        self.pad_var = tk.DoubleVar(value=_safe_float(sess.get("pad", 2.10), 2.10, min_v=1.00, max_v=3.00))

        # Custom model checkpoints
        self.face_checkpoint_var = tk.StringVar(value=str(sess.get("face_checkpoint", "")))
        self.landmark_checkpoint_var = tk.StringVar(value=str(sess.get("landmark_checkpoint", "")))

        # erase shading preset (GUI only): plane=ON, none=OFF
        _esh = sess.get("erase_shading", sess.get("shading", "plane"))
        _esh_str = str(_esh).lower()
        self.erase_shading_var = tk.BooleanVar(value=(_esh_str != "none"))

        # tracking smoothing preset (GUI only)
        _smooth = sess.get("smoothing", "Auto（今のまま）")
        if _smooth not in SMOOTHING_PRESETS:
            _smooth = "Auto（今のまま）"
        self.smoothing_menu_var = tk.StringVar(value=_smooth)

        # runtime用：オーディオ入力デバイス
        self.audio_device_var = tk.IntVar(value=_safe_int(sess.get("audio_device", 31), 31, min_v=0))
        self.audio_device_menu_var = tk.StringVar(value="")

        # Progress (step-level)
        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_text_var = tk.StringVar(value="待機中")
        self._progress_total = 1

        self._build_ui()

        self._refresh_characters(init=True)
        # Refresh character list when mouth root changes
        self._char_refresh_job = None
        self.mouth_dir_var.trace_add("write", lambda *_: self._schedule_refresh_characters())
        self._refresh_audio_devices(init=True)

        if self.video_var.get() and not self.mouth_dir_var.get():
            self._autofill_mouth_dir()
        self._refresh_characters(init=True)

        self.after(100, self._poll_logs)
        self.after(100, self._check_error_queue)

    # ----- UI -----
    def _build_ui(self) -> None:
        pad = 10
        frm = ttk.Frame(self)
        frm.pack(fill="both", expand=True, padx=pad, pady=pad)

        # Video row
        row1 = ttk.Frame(frm)
        row1.pack(fill="x", pady=(0, 8))
        ttk.Label(row1, text="動画").pack(side="left")
        ttk.Entry(row1, textvariable=self.video_var).pack(side="left", fill="x", expand=True, padx=8)
        ttk.Button(row1, text="選択…", command=self.on_pick_video).pack(side="left")

        # Mouth dir row
        row2 = ttk.Frame(frm)
        row2.pack(fill="x", pady=(0, 8))
        ttk.Label(row2, text="mouthフォルダ").pack(side="left")
        ttk.Entry(row2, textvariable=self.mouth_dir_var).pack(side="left", fill="x", expand=True, padx=8)
        ttk.Button(row2, text="選択…", command=self.on_pick_mouth_dir).pack(side="left")


        # Character row (mouth_dir/<Character>/...)
        row2a = ttk.Frame(frm)
        row2a.pack(fill="x", pady=(0, 8))
        ttk.Label(row2a, text="キャラクター").pack(side="left")
        self.cmb_character = ttk.Combobox(row2a, textvariable=self.character_var, state="readonly")
        self.cmb_character.pack(side="left", fill="x", expand=True, padx=8)
        ttk.Button(row2a, text="再読込", command=self._refresh_characters).pack(side="left")

        def _on_char_select(_evt=None):
            save_session({"character": self.character_var.get()})

        self.cmb_character.bind("<<ComboboxSelected>>", _on_char_select)

        # Pad slider (tracking)
        row2b = ttk.Frame(frm)
        row2b.pack(fill="x", pady=(0, 8))
        ttk.Label(row2b, text="pad（追跡余白）").pack(side="left")
        pad_scale = ttk.Scale(row2b, from_=1.00, to=3.00, variable=self.pad_var, orient="horizontal")
        pad_scale.pack(side="left", fill="x", expand=True, padx=8)
        self.pad_label = ttk.Label(row2b, text=f"{self.pad_var.get():.2f}")
        self.pad_label.pack(side="left")
        self.pad_var.trace_add("write", lambda *_: self.pad_label.config(text=f"{self.pad_var.get():.2f}"))
        pad_scale.bind("<ButtonRelease-1>", lambda _evt=None: save_session({"pad": float(self.pad_var.get())}))

        # Custom model settings (collapsible)
        model_frame = ttk.LabelFrame(frm, text="検出モデル設定（上級）", padding=5)
        model_frame.pack(fill="x", pady=(0, 8))

        face_row = ttk.Frame(model_frame)
        face_row.pack(fill="x", pady=2)
        ttk.Label(face_row, text="face ckpt", width=10).pack(side="left")
        ttk.Entry(face_row, textvariable=self.face_checkpoint_var, width=30).pack(side="left", fill="x", expand=True, padx=2)
        ttk.Button(face_row, text="Browse", command=self._browse_face_checkpoint).pack(side="left")

        landmark_row = ttk.Frame(model_frame)
        landmark_row.pack(fill="x", pady=2)
        ttk.Label(landmark_row, text="landmark", width=10).pack(side="left")
        ttk.Entry(landmark_row, textvariable=self.landmark_checkpoint_var, width=30).pack(side="left", fill="x", expand=True, padx=2)
        ttk.Button(landmark_row, text="Browse", command=self._browse_landmark_checkpoint).pack(side="left")

        # Coverage slider
        row3 = ttk.Frame(frm)
        row3.pack(fill="x", pady=(0, 8))
        ttk.Label(row3, text="口消し強さ").pack(side="left")
        ttk.Scale(row3, from_=0.40, to=0.90, variable=self.coverage_var, orient="horizontal").pack(
            side="left", fill="x", expand=True, padx=8
        )
        self.cov_label = ttk.Label(row3, text=f"{self.coverage_var.get():.2f}")
        self.cov_label.pack(side="left")
        self.coverage_var.trace_add("write", lambda *_: self.cov_label.config(text=f"{self.coverage_var.get():.2f}"))


        # Erase shading toggle (plane/none) - keeps UX simple
        row3a = ttk.Frame(frm)
        row3a.pack(fill="x", pady=(0, 8))
        ttk.Label(row3a, text="影なじませ（口消し）").pack(side="left")
        ttk.Checkbutton(
            row3a,
            text="有効（plane）",
            variable=self.erase_shading_var,
            command=lambda: save_session({"erase_shading": "plane" if self.erase_shading_var.get() else "none"}),
        ).pack(side="left", padx=8)
        ttk.Label(row3a, text="OFFで顎の黒にじみを軽減").pack(side="left")

        # Smoothing preset (tracking)
        row3b = ttk.Frame(frm)
        row3b.pack(fill="x", pady=(0, 8))
        ttk.Label(row3b, text="スムージング（トラック）").pack(side="left")
        self.cmb_smooth = ttk.Combobox(
            row3b,
            textvariable=self.smoothing_menu_var,
            state="readonly",
            values=SMOOTHING_LABELS,
        )
        self.cmb_smooth.pack(side="left", fill="x", expand=True, padx=8)
        self.cmb_smooth.bind(
            "<<ComboboxSelected>>",
            lambda _evt=None: save_session({"smoothing": self.smoothing_menu_var.get()}),
        )
        # Audio device row (runtime)
        row4 = ttk.Frame(frm)
        row4.pack(fill="x", pady=(0, 10))
        ttk.Label(row4, text="オーディオ入力デバイス（ライブ用）").pack(side="left")
        self.cmb_audio = ttk.Combobox(row4, textvariable=self.audio_device_menu_var, state="readonly")
        self.cmb_audio.pack(side="left", fill="x", expand=True, padx=8)
        ttk.Button(row4, text="再読込", command=self._refresh_audio_devices).pack(side="left")


        # Emotion auto (runtime) - always AUTO, user only picks preset and HUD
        row4b = ttk.Frame(frm)
        row4b.pack(fill="x", pady=(0, 10))
        ttk.Label(row4b, text="感情オート（音声）").pack(side="left")

        self.cmb_emotion_preset = ttk.Combobox(
            row4b,
            textvariable=self.emotion_preset_var,
            state="readonly",
            values=EMOTION_PRESET_LABELS,
        )
        self.cmb_emotion_preset.pack(side="left", fill="x", expand=True, padx=8)
        self.cmb_emotion_preset.bind(
            "<<ComboboxSelected>>",
            lambda _evt=None: save_session({"emotion_preset": self.emotion_preset_var.get()}),
        )

        ttk.Checkbutton(
            row4b,
            text="HUD（😊表示）",
            variable=self.emotion_hud_var,
            command=lambda: save_session({"emotion_hud": bool(self.emotion_hud_var.get())}),
        ).pack(side="left", padx=8)

        # Buttons (workflow)
        row_btn = ttk.Frame(frm)
        row_btn.pack(fill="x", pady=(0, 10))

        self.btn_track_calib = ttk.Button(row_btn, text="① 解析→キャリブ", command=self.on_track_and_calib)
        self.btn_track_calib.pack(side="left")

        self.btn_calib_only = ttk.Button(row_btn, text="キャリブのみ（やり直し）", command=self.on_calib_only)
        self.btn_calib_only.pack(side="left", padx=8)

        self.btn_erase = ttk.Button(row_btn, text="② 口消し動画生成", command=self.on_erase_mouthless)
        self.btn_erase.pack(side="left")

        self.btn_erase_range = ttk.Button(row_btn, text="口消し範囲プレビュー", command=self.on_preview_erase_range)
        self.btn_erase_range.pack(side="left", padx=8)

        self.btn_live = ttk.Button(row_btn, text="③ ライブ実行", command=self.on_live_run)
        self.btn_live.pack(side="left", padx=8)

        self.btn_stop = ttk.Button(
            row_btn, text=STOP_BTN_TEXT_DEFAULT, command=self.on_stop, state="disabled"
        )
        self.btn_stop.pack(side="right")

        # Progress
        prog = ttk.Frame(frm)
        prog.pack(fill="x", pady=(0, 6))
        ttk.Label(prog, text="進捗").pack(side="left")
        ttk.Label(prog, textvariable=self.progress_text_var).pack(side="left", padx=8)
        self.progress = ttk.Progressbar(
            prog, variable=self.progress_var, maximum=1.0, mode="determinate"
        )
        self.progress.pack(side="left", fill="x", expand=True, padx=8)

        # Log
        log_header = ttk.Frame(frm)
        log_header.pack(fill="x")
        ttk.Label(log_header, text="ログ").pack(side="left", anchor="w")
        ttk.Button(log_header, text="ログクリア", command=self._clear_log).pack(side="right")

        self.txt = tk.Text(frm, height=22, wrap="word")
        self.txt.pack(fill="both", expand=True)
        self.txt.configure(state="disabled")

    # ----- logging (thread-safe) -----
    def log(self, s: str) -> None:
        self.log_q.put(s)

    def _poll_logs(self) -> None:
        # ログキュー処理
        try:
            while True:
                s = self.log_q.get_nowait()
                # Remove null bytes from log text
                s = s.replace("\x00", "")
                self.txt.configure(state="normal")
                self.txt.insert("end", s + "\n")
                # 上限チェック
                line_count = int(self.txt.index("end-1c").split(".")[0])
                if line_count > MAX_LOG_LINES:
                    excess = line_count - MAX_LOG_LINES
                    self.txt.delete("1.0", f"{excess + 1}.0")
                self.txt.see("end")
                self.txt.configure(state="disabled")
        except queue.Empty:
            pass
        # UI更新キュー処理
        try:
            while True:
                func = self.ui_q.get_nowait()
                func()
        except queue.Empty:
            pass
        self.after(100, self._poll_logs)

    def _clear_log(self) -> None:
        """ログをクリア（キューも空にする）"""
        # キューをドレイン
        try:
            while True:
                self.log_q.get_nowait()
        except queue.Empty:
            pass
        # Textをクリア
        self.txt.configure(state="normal")
        self.txt.delete("1.0", "end")
        self.txt.configure(state="disabled")

    # ----- misc helpers -----
    def _autofill_mouth_dir(self) -> None:
        v = self.video_var.get().strip()
        if not v:
            return
        md = guess_mouth_dir(v)
        if md:
            self.mouth_dir_var.set(md)

    def _refresh_audio_devices(self, init: bool = False) -> None:
        devices = _try_list_input_devices()
        if not devices:
            if init:
                self.audio_device_menu_var.set(f"{self.audio_device_var.get()}: (未取得)")
            return

        self._audio_items = devices  # optional stash
        values = [disp for _, disp in devices]
        self.cmb_audio["values"] = values

        cur = int(self.audio_device_var.get())
        sel = None
        for idx, disp in devices:
            if idx == cur:
                sel = disp
                break
        if sel is None:
            idx0, disp0 = devices[0]
            self.audio_device_var.set(idx0)
            sel = disp0
        self.audio_device_menu_var.set(sel)

        def _on_select(_evt=None):
            s = self.audio_device_menu_var.get()
            try:
                n = int(s.split(":", 1)[0].strip())
                self.audio_device_var.set(n)
                save_session({"audio_device": int(n)})
            except Exception:
                pass

        self.cmb_audio.bind("<<ComboboxSelected>>", _on_select)
        if init:
            _on_select()


    def _schedule_refresh_characters(self) -> None:
        """Debounced refresh for character list."""
        try:
            if getattr(self, "_char_refresh_job", None):
                self.after_cancel(self._char_refresh_job)  # type: ignore[arg-type]
        except Exception:
            pass
        self._char_refresh_job = self.after(150, self._refresh_characters)

    def _refresh_characters(self, init: bool = False) -> None:
        """Populate character combobox from mouth_dir root."""
        mouth_root = self.mouth_dir_var.get().strip()

        # If mouth_root is already emotion-level (no character layer), character selection is not needed.
        if _is_emotion_level_mouth_root(mouth_root):
            try:
                self.cmb_character.configure(state="disabled")
                self.cmb_character["values"] = ["(不要：直下が感情フォルダ)"]
            except Exception:
                pass
            self.character_var.set("")
            return

        chars = list_character_dirs(mouth_root)
        if not chars:
            # Keep enabled but show placeholder.
            try:
                self.cmb_character.configure(state="readonly")
                self.cmb_character["values"] = ["(なし)"]
            except Exception:
                pass
            self.character_var.set("")
            return

        try:
            self.cmb_character.configure(state="readonly")
            self.cmb_character["values"] = chars
        except Exception:
            pass

        cur = (self.character_var.get() or "").strip()
        if cur not in chars:
            # Auto-select when there is only one character
            if len(chars) == 1:
                self.character_var.set(chars[0])
                save_session({"character": chars[0]})
            elif init:
                self.character_var.set("")

    def _emotion_preset_key(self) -> str:
        return EMOTION_PRESETS.get(self.emotion_preset_var.get(), "standard")

    def _resolve_character_for_action(self) -> str | None:
        """Return effective character name for current mouth_root.
        - ""   : no character layer (mouth_root is emotion-level)
        - name  : selected / auto-selected character
        - None  : error (multiple candidates but not selected)
        """
        mouth_root = self.mouth_dir_var.get().strip()
        if _is_emotion_level_mouth_root(mouth_root):
            return ""

        chars = list_character_dirs(mouth_root)
        if not chars:
            return ""

        cur = (self.character_var.get() or "").strip()
        if cur in chars:
            return cur

        if len(chars) == 1:
            self.character_var.set(chars[0])
            save_session({"character": chars[0]})
            return chars[0]

        self._show_error("エラー", "キャラクターを選択してください（mouth_dir直下のフォルダから選びます）。")
        return None

    def _runtime_supports(self, runtime_py: str, flags: list[str]) -> bool:
        return _script_contains(runtime_py, flags)

    def _warn_soft_stop(self) -> None:
        if self.stop_mode != "soft":
            return
        self.log("[gui] 停止予約中: 終了待機中。必要ならもう一度で強制停止してください。")

    def _set_stop_mode(self, mode: str) -> None:
        def _apply():
            if self._soft_warn_job:
                try:
                    self.after_cancel(self._soft_warn_job)
                except Exception:
                    pass
                self._soft_warn_job = None

            self.stop_mode = mode
            if mode == "soft":
                self.stop_flag.set()
                self.soft_requested_at = time.monotonic()
                self.btn_stop.configure(text=STOP_BTN_TEXT_SOFT)
                self._soft_warn_job = self.after(
                    int(SOFT_STOP_GRACE_SEC * 1000),
                    self._warn_soft_stop,
                )
            elif mode == "force":
                self.stop_flag.set()
                self.soft_requested_at = None
                self.btn_stop.configure(text=STOP_BTN_TEXT_SOFT)
            else:
                self.stop_flag.clear()
                self.soft_requested_at = None
                self.btn_stop.configure(text=STOP_BTN_TEXT_DEFAULT)

        self.after(0, _apply)

    def _request_soft_stop(self, p: subprocess.Popen) -> bool:
        try:
            if sys.platform.startswith("win"):
                os.kill(p.pid, signal.CTRL_BREAK_EVENT)
            else:
                os.killpg(os.getpgid(p.pid), signal.SIGINT)
            return True
        except Exception:
            return False

    def _set_running(self, running: bool) -> None:
        st = "disabled" if running else "normal"
        def _apply():
            try:
                self.btn_track_calib.configure(state=st)
                self.btn_calib_only.configure(state=st)
                self.btn_erase.configure(state=st)
                self.btn_erase_range.configure(state=st)
                self.btn_live.configure(state=st)
                self.btn_stop.configure(state=("normal" if running else "disabled"))
                if not running:
                    self._set_stop_mode("none")
                    self._progress_reset()
            except Exception as e:
                print(f"[warn] _set_running error: {e}")
        # スレッドセーフにキューイング
        self.ui_q.put(_apply)

    def _progress_reset(self) -> None:
        def _apply():
            self.progress.configure(mode="determinate", maximum=1.0)
            self.progress_var.set(0.0)
            self.progress_text_var.set("待機中")
        self.after(0, _apply)

    def _progress_begin(self, total_steps: int, text: str) -> None:
        def _apply():
            self._progress_total = max(1, int(total_steps))
            self.progress.configure(mode="determinate", maximum=self._progress_total)
            self.progress_var.set(0.0)
            self.progress_text_var.set(text)
        self.after(0, _apply)

    def _progress_step(self, step: int, text: str) -> None:
        def _apply():
            self.progress.configure(mode="determinate", maximum=self._progress_total)
            val = min(max(0, int(step)), int(self._progress_total))
            self.progress_var.set(val)
            self.progress_text_var.set(text)
        self.after(0, _apply)

    def _show_error(self, title: str, msg: str) -> None:
        """メインスレッドからもworkerスレッドからも安全に呼び出せる"""
        self.error_q.put((title, msg))

    def _check_error_queue(self) -> None:
        """メインスレッドのイベントループから定期的に呼ばれる"""
        try:
            while True:
                title, msg = self.error_q.get_nowait()
                print(f"error: {title} {msg}")
                messagebox.showerror(title, msg)
        except queue.Empty:
            pass
        # 100ms後に再度チェック
        self.after(100, self._check_error_queue)

    # ----- file pickers -----
    def on_pick_video(self) -> None:
        if sys.platform == "darwin":  # Mac
            p = filedialog.askopenfilename(title="動画を選択")
        else:  # Windows/Linux
            p = filedialog.askopenfilename(
                title="動画を選択",
                filetypes=[("Video", "*.mp4;*.mov;*.mkv;*.avi;*.webm;*.m4v"), ("All", "*.*")],
            )
        if not p:
            return
        self.video_var.set(p)
        self._autofill_mouth_dir()
        # 選択直後は video=source_video として保存（まだmouthless未生成のため）
        save_session({
            "video": self.video_var.get(),
            "source_video": self.video_var.get(),
            "mouth_dir": self.mouth_dir_var.get(),
            "coverage": float(self.coverage_var.get()),
            "pad": float(self.pad_var.get()),
            "audio_device": int(self.audio_device_var.get()),
            "character": self.character_var.get(),
            "emotion_preset": self.emotion_preset_var.get(),
            "emotion_hud": bool(self.emotion_hud_var.get()),
        })

    def on_pick_mouth_dir(self) -> None:
        d = filedialog.askdirectory(title="mouthフォルダを選択")
        if not d:
            return
        self.mouth_dir_var.set(d)
        self._refresh_characters(init=True)
        save_session({
            "video": self.video_var.get(),
            "source_video": self.video_var.get(),
            "mouth_dir": self.mouth_dir_var.get(),
            "coverage": float(self.coverage_var.get()),
            "pad": float(self.pad_var.get()),
            "audio_device": int(self.audio_device_var.get()),
            "character": self.character_var.get(),
            "emotion_preset": self.emotion_preset_var.get(),
            "emotion_hud": bool(self.emotion_hud_var.get()),
        })

    def on_stop(self) -> None:
        if self.stop_mode == "none":
            self.log("[gui] stop requested. will stop after current step.")
            self._set_stop_mode("soft")
            return
        if self.stop_mode == "soft":
            self.log("[gui] force stop requested. terminating active process.")
            self._set_stop_mode("force")
            if self.active_proc and (self.active_proc.poll() is None):
                self._terminate_proc_tree(self.active_proc)

    def _terminate_proc_tree(self, p: subprocess.Popen) -> None:
        """子プロセス（可能ならプロセスツリー）を強制終了"""
        try:
            if sys.platform.startswith("win"):
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(p.pid)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            else:
                # プロセスグループごと kill（_run_cmd_stream で setsid している前提）
                try:
                    os.killpg(os.getpgid(p.pid), signal.SIGKILL)
                except Exception:
                    p.kill()
        except Exception:
            try:
                p.kill()
            except Exception:
                pass

    def _browse_face_checkpoint(self):
        """顔検出器チェックポイント選択"""
        path = filedialog.askopenfilename(
            title="Select face checkpoint",
            filetypes=[("PyTorch model", "*.pt"), ("All files", "*.*")],
            initialdir="models"
        )
        if path:
            self.face_checkpoint_var.set(path)
            save_session({"face_checkpoint": path})

    def _browse_landmark_checkpoint(self):
        """ランドマーク検出器チェックポイント選択"""
        path = filedialog.askopenfilename(
            title="Select landmark checkpoint",
            filetypes=[("PyTorch model", "*.pth"), ("All files", "*.*")],
            initialdir="models"
        )
        if path:
            self.landmark_checkpoint_var.set(path)
            save_session({"landmark_checkpoint": path})

    # ----- subprocess runner -----
    def _run_cmd_stream(
        self,
        cmd: list[str],
        cwd: str | None = None,
        *,
        allow_soft_interrupt: bool = False,
    ) -> int:
        """
        Run command and stream stdout/stderr to GUI log.
        """
        self.log("[cmd] " + " ".join(cmd))

        env = os.environ.copy()
        env.setdefault("PYTHONUTF8", "1")
        env.setdefault("PYTHONIOENCODING", "utf-8")

        popen_kw = {}
        if sys.platform.startswith("win"):
            popen_kw["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        else:
            popen_kw["preexec_fn"] = os.setsid

        try:
            p = subprocess.Popen(
                cmd,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                **popen_kw,
            )
        except Exception as e:
            self.log(f"[error] failed to start: {e}")
            return 999

        self.active_proc = p

        assert p.stdout is not None
        OUT = queue.Queue()
        SENTINEL = object()

        def _reader():
            try:
                for line in p.stdout:
                    OUT.put(line)
            finally:
                OUT.put(SENTINEL)

        threading.Thread(target=_reader, daemon=True).start()

        terminated = False
        soft_signaled = False
        while True:
            if self.stop_mode == "force" and not terminated:
                terminated = True
                self._terminate_proc_tree(p)

            if (
                allow_soft_interrupt
                and (not soft_signaled)
                and (self.stop_mode == "soft")
            ):
                soft_signaled = True
                self._request_soft_stop(p)

            try:
                item = OUT.get(timeout=0.1)
            except queue.Empty:
                item = None

            if item is SENTINEL:
                break
            if isinstance(item, str):
                self.log(item.rstrip("\n"))

            if p.poll() is not None and OUT.empty():
                break

        rc = p.wait()
        if self.active_proc is p:
            self.active_proc = None
        return rc

    # ----- preview -----
    def _open_video_preview(self, video_path: str) -> None:
        # Try OpenCV playback first (if available)
        try:
            import cv2  # type: ignore
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                win = "preview (q/ESC=close, space=pause)"
                paused = False
                while True:
                    if not paused:
                        ok, frame = cap.read()
                        if not ok:
                            break
                    cv2.imshow(win, frame)
                    k = cv2.waitKey(15) & 0xFF
                    if k in (ord("q"), 27):
                        break
                    if k == ord(" "):
                        paused = not paused
                cap.release()
                cv2.destroyWindow(win)
                return
        except Exception:
            pass

        # Fallback to OS open
        try:
            if sys.platform.startswith("win"):
                os.startfile(video_path)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", video_path])
            else:
                subprocess.Popen(["xdg-open", video_path])
        except Exception as e:
            self.log(f"[warn] cannot open preview automatically: {e}")
            self.log(f"[info] output: {video_path}")

    def _export_browser_assets(self, mouthless_mp4: str, calib_npz: str) -> None:
        if not os.path.isfile(mouthless_mp4):
            self.log("[warn] ブラウザ用出力: 口消し動画が見つかりません。")
            return
        if not os.path.isfile(calib_npz):
            self.log("[warn] ブラウザ用出力: mouth_track_calibrated.npz がありません。")
            return

        fps = None
        try:
            import numpy as np  # type: ignore
            with np.load(calib_npz, allow_pickle=False) as npz:
                if "fps" in npz:
                    fps = float(npz["fps"])
        except Exception as e:
            self.log(f"[warn] ブラウザ用出力: fps取得に失敗しました: {e}")

        if not fps or fps <= 0:
            self.log("[warn] ブラウザ用出力: fpsが不明のためCFR変換をスキップします。")
            fps = None

        out_dir = os.path.dirname(os.path.abspath(mouthless_mp4))

        try:
            from convert_npz_to_json import convert_npz_to_json  # type: ignore
            convert_npz_to_json(Path(calib_npz), Path(out_dir))
            self.log(f"[info] ブラウザ用JSON出力: {os.path.join(out_dir, 'mouth_track.json')}")
        except Exception as e:
            self.log(f"[warn] ブラウザ用JSON出力に失敗しました: {e}")

        if not fps:
            return

        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            self.log("[warn] ブラウザ用出力: ffmpegが見つからないためH.264変換をスキップします。")
            return

        h264_mp4 = os.path.splitext(mouthless_mp4)[0] + "_h264.mp4"
        cmd = [
            ffmpeg,
            "-y",
            "-i",
            mouthless_mp4,
            "-vf",
            f"fps={fps}",
            "-r",
            f"{fps}",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            h264_mp4,
        ]
        self.log("[cmd] " + " ".join(cmd))
        rc = self._run_cmd_stream(cmd, cwd=HERE)
        if rc != 0 or (not os.path.isfile(h264_mp4)):
            self.log(f"[warn] ブラウザ用H.264変換に失敗しました (rc={rc})")
        else:
            self.log(f"[info] ブラウザ用H.264出力: {h264_mp4}")

    # ----- workflow buttons -----
    def _start_worker(self, target) -> None:
        if self.worker_thread and self.worker_thread.is_alive():
            return
        self.stop_flag.clear()
        self._set_stop_mode("none")
        self._set_running(True)
        def runner():
            try:
                target()
            finally:
                # ワーカーが何で終わっても UI を戻す
                self._set_running(False)
        self.worker_thread = threading.Thread(target=runner, daemon=True)
        self.worker_thread.start()

    def on_track_and_calib(self) -> None:
        def _worker():
            try:
                base_dir = os.path.dirname(os.path.abspath(__file__))
                ok, msg = _ensure_backend_sanity(base_dir)
                if not ok:
                    self._show_error("エラー", msg)
                    return

                video = self.video_var.get().strip()
                mouth_dir = self.mouth_dir_var.get().strip()
                if not video:
                    self._show_error("エラー", "動画を選択してください。")
                    return
                if not mouth_dir:
                    self._autofill_mouth_dir()
                    mouth_dir = self.mouth_dir_var.get().strip()

                char = self._resolve_character_for_action()
                if char is None:
                    return

                open_sprite = best_open_sprite_for_character(mouth_dir, char)
                if not open_sprite:
                    self._show_error("エラー", "mouthフォルダ（キャラ/Default 等）に open.png が見つかりません（キャリブ用）")
                    return

                out_dir = os.path.dirname(os.path.abspath(video))
                track_npz = os.path.join(out_dir, "mouth_track.npz")
                calib_npz = os.path.join(out_dir, "mouth_track_calibrated.npz")

                self._progress_begin(2, "解析/キャリブ準備中…")

                save_session({
                    "video": video,
                    "source_video": video,
                    "mouth_dir": mouth_dir,
                    "coverage": float(self.coverage_var.get()),
                    "pad": float(self.pad_var.get()),
                    "audio_device": int(self.audio_device_var.get()),
                })

                self.log("\n=== [1/2] 解析（自動修復つき・最高品質） ===")
                self._progress_step(1, "解析中… (1/2)")
                cmd = [
                    sys.executable, os.path.join(base_dir, "auto_mouth_track_v2.py"),
                    "--video", video,
                    "--out", track_npz,
                    "--pad", f"{float(self.pad_var.get()):.2f}",
                    "--stride", "1",
                    "--det-scale", "1.0",
                    "--min-conf", "0.5",
                    "--early-stop",
                    "--max-tries", "4",
                ]
                # Add custom model paths if specified
                _face_ckpt = self.face_checkpoint_var.get().strip()
                _landmark_ckpt = self.landmark_checkpoint_var.get().strip()
                if _face_ckpt:
                    cmd += ["--face-checkpoint", _face_ckpt]
                if _landmark_ckpt:
                    cmd += ["--landmark-checkpoint", _landmark_ckpt]
                # Apply smoothing preset from GUI (Auto = pass nothing)
                _cutoff = SMOOTHING_PRESETS.get(self.smoothing_menu_var.get())
                if _cutoff is not None:
                    cmd += ["--smooth-cutoff", str(_cutoff)]
                save_session({"smoothing": self.smoothing_menu_var.get()})

                self.log("[cmd] " + " ".join(cmd))
                rc = self._run_cmd_stream(
                    cmd,
                    cwd=base_dir,
                    allow_soft_interrupt=True,
                )
                if rc != 0 or (not os.path.isfile(track_npz)):
                    self._show_error("失敗", f"解析に失敗しました (rc={rc})")
                    return

                self._progress_step(1, "解析完了 (1/2)")
                if self.stop_mode != "none":
                    self.log("[info] 停止予約のため、キャリブ以降をスキップします。")
                    self._progress_step(1, "解析完了 (1/2) - 停止予約")
                    return

                self.log("\n=== [2/2] キャリブレーション（画面を閉じると完了） ===")
                self._progress_step(2, "キャリブ中… (2/2)")
                cmd = [
                    sys.executable, os.path.join(base_dir, "calibrate_mouth_track.py"),
                    "--video", video,
                    "--track", track_npz,
                    "--sprite", open_sprite,
                    "--out", calib_npz,
                ]
                self.log("[cmd] " + " ".join(cmd))
                rc = self._run_cmd_stream(
                    cmd,
                    cwd=base_dir,
                    allow_soft_interrupt=True,
                )
                if rc != 0 or (not os.path.isfile(calib_npz)):
                    self._show_error("失敗", f"キャリブに失敗しました (rc={rc})")
                    return

                save_session({
                    "track": track_npz,
                    "track_calibrated": calib_npz,
                })
                self._progress_step(2, "キャリブ完了 (2/2)")
                self.log("\n完了（次は『② 口消し動画生成』）")
            except Exception as e:
                self._show_error("エラー", str(e))
            finally:
                self._set_running(False)
        self._start_worker(_worker)

    def on_calib_only(self) -> None:
        def _worker():
            try:
                base_dir = os.path.dirname(os.path.abspath(__file__))
                video = self.video_var.get().strip()
                mouth_dir = self.mouth_dir_var.get().strip()
                if not video:
                    self._show_error("エラー", "動画を選択してください。")
                    return
                char = self._resolve_character_for_action()
                if char is None:
                    return

                open_sprite = best_open_sprite_for_character(mouth_dir, char)
                if not open_sprite:
                    self._show_error("エラー", "mouthフォルダ（キャラ/Default 等）に open.png が見つかりません（キャリブ用）")
                    return

                out_dir = os.path.dirname(os.path.abspath(video))
                track_npz = os.path.join(out_dir, "mouth_track.npz")
                calib_npz = os.path.join(out_dir, "mouth_track_calibrated.npz")
                if not os.path.isfile(track_npz):
                    self._show_error("エラー", "mouth_track.npz がありません。先に『① 解析→キャリブ』を実行してください。")
                    return

                # 再キャリブ時は既存のキャリブ済みファイルがあればそれを使う（位置を維持）
                input_track = calib_npz if os.path.isfile(calib_npz) else track_npz

                self.log("\n=== キャリブレーション（やり直し） ===")
                self._progress_begin(1, "キャリブ準備中…")
                self._progress_step(1, "キャリブ中… (1/1)")
                if input_track == calib_npz:
                    self.log("[info] 既存のキャリブ済みトラックを使用（位置を維持）")
                cmd = [
                    sys.executable, os.path.join(base_dir, "calibrate_mouth_track.py"),
                    "--video", video,
                    "--track", input_track,
                    "--sprite", open_sprite,
                    "--out", calib_npz,
                ]
                self.log("[cmd] " + " ".join(cmd))
                rc = self._run_cmd_stream(
                    cmd,
                    cwd=base_dir,
                    allow_soft_interrupt=True,
                )
                if rc != 0 or (not os.path.isfile(calib_npz)):
                    self._show_error("失敗", f"キャリブに失敗しました (rc={rc})")
                    return

                save_session({"track": track_npz, "track_path": track_npz, "track_calibrated": calib_npz, "track_calibrated_path": calib_npz, "calib": calib_npz})
                self._progress_step(1, "キャリブ完了 (1/1)")
                self.log("\n完了")
            except Exception as e:
                self._show_error("エラー", str(e))
            finally:
                self._set_running(False)
        self._start_worker(_worker)

    def on_erase_mouthless(self) -> None:
        def _worker():
            try:
                base_dir = os.path.dirname(os.path.abspath(__file__))
                video = self.video_var.get().strip()
                mouth_dir = self.mouth_dir_var.get().strip()
                if not video:
                    self._show_error("エラー", "動画を選択してください。")
                    return

                out_dir = os.path.dirname(os.path.abspath(video))
                track_npz = os.path.join(out_dir, "mouth_track.npz")
                calib_npz = os.path.join(out_dir, "mouth_track_calibrated.npz")
                if not os.path.isfile(track_npz):
                    self._show_error("エラー", "mouth_track.npz がありません。先に『① 解析→キャリブ』を実行してください。")
                    return
                if not os.path.isfile(calib_npz):
                    self._show_error("エラー", "mouth_track_calibrated.npz がありません。キャリブを完了してください。")
                    return

                name = os.path.splitext(os.path.basename(video))[0]
                mouthless_mp4 = os.path.join(out_dir, f"{name}_mouthless.mp4")

                cov = float(self.coverage_var.get())
                covs = [max(0.40, min(0.90, cov + x)) for x in (0.0, 0.10, 0.20)]
                covs = sorted(set(round(x, 2) for x in covs))
                cov_arg = ",".join(f"{x:.2f}" for x in covs)

                self.log("\n=== 口消し動画生成（自動候補->自動選別） ===")
                self._progress_begin(1, "口消し準備中…")
                self._progress_step(1, "口消し生成中… (1/1)")
                cmd = [
                    sys.executable, os.path.join(base_dir, "auto_erase_mouth.py"),
                    "--video", video,
                    "--track", track_npz,
                    "--out", mouthless_mp4,
                    "--coverage", cov_arg,
                    "--try-strict",
                    "--keep-audio",
                    "--shading", ("plane" if self.erase_shading_var.get() else "none"),
                ]
                self.log("[cmd] " + " ".join(cmd))
                rc = self._run_cmd_stream(cmd, cwd=base_dir)
                if rc != 0 or (not os.path.isfile(mouthless_mp4)):
                    self._show_error("失敗", f"口消し動画生成に失敗しました (rc={rc})")
                    return

                self._progress_step(1, "口消し完了 (1/1)")
                # runtime背景をmouthlessに更新
                save_session({
                    "video": mouthless_mp4,      # runtime背景
                    "source_video": video,       # GUI表示
                    "mouth_dir": mouth_dir,
                    "track": track_npz,
                    "track_calibrated": calib_npz,
                    "coverage": float(self.coverage_var.get()),
                    "pad": float(self.pad_var.get()),
                    "audio_device": int(self.audio_device_var.get()),
                })

                if self.stop_mode != "none":
                    self.log("[info] 停止予約のため、ブラウザ用出力とプレビューをスキップします。")
                    return

                self.log("\n=== ブラウザ用データ出力 ===")
                self._export_browser_assets(mouthless_mp4, calib_npz)

                self.log("\nプレビューを起動します…")
                self._open_video_preview(mouthless_mp4)
                self.log("\n完了（次は『③ ライブ実行』）")
            except Exception as e:
                self._show_error("エラー", str(e))
            finally:
                self._set_running(False)
        self._start_worker(_worker)

    def on_preview_erase_range(self) -> None:
        """口消しのマスク範囲（inner/ring）を元フレーム上に重ねて確認するプレビュー。
        - 赤: 実際に消す中心マスク（feather込み）
        - 黄: ring（陰影推定に使う外周）

        NOTE: OpenCVのGUI関数はメインスレッドで実行する必要があるため、
        別プロセスとして起動する。
        """
        video = self.video_var.get().strip()
        if not video:
            self._show_error("エラー", "動画を選択してください。")
            return

        out_dir = os.path.dirname(os.path.abspath(video))
        track_npz = os.path.join(out_dir, "mouth_track.npz")
        if not os.path.isfile(track_npz):
            self._show_error("エラー", "mouth_track.npz がありません。先に『① 解析→キャリブ』を実行してください。")
            return

        base_dir = os.path.dirname(os.path.abspath(__file__))
        preview_script = os.path.join(base_dir, "preview_erase_range.py")
        if not os.path.isfile(preview_script):
            self._show_error("エラー", "preview_erase_range.py が見つかりません。")
            return

        cov = float(self.coverage_var.get())

        # 別プロセスとして起動（GUIをブロックしない）
        cmd = [
            sys.executable, preview_script,
            "--video", video,
            "--track", track_npz,
            "--coverage", f"{cov:.2f}",
        ]
        self.log("[cmd] " + " ".join(cmd))
        try:
            subprocess.Popen(cmd, cwd=base_dir)
        except Exception as e:
            self._show_error("エラー", f"プレビュー起動に失敗しました: {e}")

    def on_live_run(self) -> None:
        def _worker():
            try:
                base_dir = os.path.dirname(os.path.abspath(__file__))
                sess = load_session()

                # 背景は session["video"]（mouthlessが入ってる想定）を優先
                loop_video = sess.get("video") or self.video_var.get().strip()
                if not loop_video or (not os.path.isfile(loop_video)):
                    self._show_error("エラー", "背景動画が見つかりません（先に口消し動画生成を推奨）")
                    return

                video_src = self.video_var.get().strip()
                if not video_src:
                    self._show_error("エラー", "元動画が未選択です")
                    return
                out_dir = os.path.dirname(os.path.abspath(video_src))
                track_npz = os.path.join(out_dir, "mouth_track.npz")
                calib_npz = os.path.join(out_dir, "mouth_track_calibrated.npz")

                mouth_root = self.mouth_dir_var.get().strip()
                if not mouth_root:
                    self._show_error("エラー", "mouthフォルダを選択してください。")
                    return

                char = self._resolve_character_for_action()
                if char is None:
                    return
                mouth_dir = resolve_character_dir(mouth_root, char)

                device_idx = int(self.audio_device_var.get())
                save_session({
                    "audio_device": device_idx,
                    "character": char,
                    "emotion_preset": self.emotion_preset_var.get(),
                    "emotion_hud": bool(self.emotion_hud_var.get()),
                })

                # Prefer emotion-auto runtime if present
                runtime_py = os.path.join(base_dir, "loop_lipsync_runtime_patched_emotion_auto.py")
                if not os.path.isfile(runtime_py):
                    runtime_py = os.path.join(base_dir, "loop_lipsync_runtime_patched.py")

                self.log("\n=== ライブ実行（qで終了） ===")
                cmd = [
                    sys.executable, runtime_py,
                    "--no-auto-last-session",
                    "--loop-video", loop_video,
                    "--mouth-dir", mouth_dir,
                    "--track", track_npz,
                    "--track-calibrated", calib_npz,
                    "--device", str(device_idx),
                ]

                # Disable manual emotion GUI (AUTO-only)
                if self._runtime_supports(runtime_py, ["--no-emotion-gui"]):
                    cmd.append("--no-emotion-gui")

                # Emotion auto (if supported by runtime)
                if self._runtime_supports(runtime_py, ["--emotion-auto"]):
                    cmd.append("--emotion-auto")
                    if self._runtime_supports(runtime_py, ["--emotion-preset"]):
                        cmd += ["--emotion-preset", self._emotion_preset_key()]
                    if self._runtime_supports(runtime_py, ["--emotion-hud", "--no-emotion-hud"]):
                        cmd.append("--emotion-hud" if bool(self.emotion_hud_var.get()) else "--no-emotion-hud")
                    
                    # Match the CLI behavior you tested (hidden knobs, only if runtime supports them)
                    if self._runtime_supports(runtime_py, ["--emotion-silence-db"]):
                        cmd += ["--emotion-silence-db", "-65"]
                    
                    if self._runtime_supports(runtime_py, ["--emotion-min-conf"]):
                        cmd += ["--emotion-min-conf", "0.12"]
                    
                    if self._runtime_supports(runtime_py, ["--emotion-hud-font"]):
                        cmd += ["--emotion-hud-font", "28"]
                    
                    if self._runtime_supports(runtime_py, ["--emotion-hud-alpha"]):
                        cmd += ["--emotion-hud-alpha", "0.92"]
                else:
                    self.log("[warn] runtime が感情オートに未対応のため、従来モードで実行します。")
                self.log("[cmd] " + " ".join(cmd))
                self._progress_begin(1, "ライブ準備中…")
                self._progress_step(1, "ライブ実行中…")
                rc = self._run_cmd_stream(cmd, cwd=base_dir, allow_soft_interrupt=True)
                self.log(f"\n[live] finished rc={rc}")
                self._progress_step(1, "ライブ終了")
            except Exception as e:
                self._show_error("エラー", str(e))
            finally:
                self._set_running(False)
        self._start_worker(_worker)


def main() -> None:
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
