#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
preview_erase_range.py

口消し範囲プレビュー用の独立スクリプト。
OpenCVのGUI関数はメインスレッドで実行する必要があるため、
GUIアプリから別プロセスとして呼び出す。

Usage:
    python preview_erase_range.py --video VIDEO --track TRACK_NPZ --coverage COV
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="Erase range preview")
    parser.add_argument("--video", required=True, help="Video file path")
    parser.add_argument("--track", required=True, help="Track npz file path")
    parser.add_argument("--coverage", type=float, default=0.6, help="Coverage value (0.4-0.9)")
    args = parser.parse_args()

    try:
        import cv2
        import numpy as np
    except ImportError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    video = args.video
    track_npz = args.track
    cov = max(0.0, min(1.0, args.coverage))

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        print(f"Error: Cannot open video: {video}", file=sys.stderr)
        sys.exit(1)

    try:
        vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        if vid_w <= 0 or vid_h <= 0:
            print("Error: Cannot get video size", file=sys.stderr)
            sys.exit(1)

        # ---- load track (mouth_track.npz) ----
        npz = np.load(track_npz, allow_pickle=False)
        if "quad" not in npz:
            print("Error: 'quad' not found in track npz", file=sys.stderr)
            sys.exit(1)
        quads = np.asarray(npz["quad"], dtype=np.float32)
        if quads.ndim != 3 or quads.shape[1:] != (4, 2):
            print("Error: Invalid quad shape (need (N,4,2))", file=sys.stderr)
            sys.exit(1)
        N = int(quads.shape[0])
        valid = np.asarray(npz["valid"], dtype=bool) if "valid" in npz else np.ones((N,), dtype=bool)

        # scale to current video size (if track stored original w/h)
        src_w = int(npz["w"]) if "w" in npz else vid_w
        src_h = int(npz["h"]) if "h" in npz else vid_h
        sx = float(vid_w) / float(max(1, src_w))
        sy = float(vid_h) / float(max(1, src_h))
        quads = quads.copy()
        quads[..., 0] *= sx
        quads[..., 1] *= sy

        # hold-fill for invalid frames
        filled = quads.copy()
        idxs = np.where(valid)[0]
        if len(idxs) > 0:
            last = int(idxs[0])
            for i in range(N):
                if valid[i]:
                    last = i
                else:
                    filled[i] = filled[last]
            first = int(idxs[0])
            for i in range(first):
                filled[i] = filled[first]
        else:
            print("Error: All frames are invalid", file=sys.stderr)
            sys.exit(1)

        n_out = min(total_frames if total_frames > 0 else N, N)

        # ---- decide normalized patch size ----
        def ensure_even_ge2(n: int) -> int:
            n = int(n)
            if n < 2:
                return 2
            return n if (n % 2 == 0) else (n - 1)

        qsz = filled[:n_out]
        ws = np.linalg.norm(qsz[:, 1, :] - qsz[:, 0, :], axis=1)
        hs = np.linalg.norm(qsz[:, 3, :] - qsz[:, 0, :], axis=1)
        ratio = float(np.median(ws / np.maximum(1e-6, hs)))
        p95w = float(np.percentile(ws, 95))
        oversample = 1.2
        norm_w = ensure_even_ge2(max(96, int(round(p95w * oversample))))
        ratio_c = max(0.25, min(4.0, ratio))
        norm_h = ensure_even_ge2(max(64, int(round(norm_w / ratio_c))))

        # ---- build masks in normalized space from current coverage ----
        mask_scale_x = 0.55 + 0.25 * cov
        mask_scale_y = 0.48 + 0.20 * cov
        ring_px = int(round(16 + 10 * cov))
        dilate_px = int(round(8 + 8 * cov))
        feather_px = int(round(18 + 10 * cov))
        top_clip_frac = float(0.84 - 0.06 * cov)
        center_y_off = int(round(norm_h * (0.05 + 0.01 * cov)))

        def make_mouth_mask(w: int, h: int, rx: int, ry: int, *, center_y_offset_px: int = 0, top_clip_frac: float = 0.82):
            mask = np.zeros((h, w), dtype=np.uint8)
            cx, cy = w // 2, h // 2 + int(center_y_offset_px)
            rx2 = int(max(1, min(int(rx), w // 2 - 1)))
            ry2 = int(max(1, min(int(ry), h // 2 - 1)))
            cv2.ellipse(mask, (cx, cy), (rx2, ry2), 0.0, 0.0, 360.0, 255, -1)
            clip_y = int(round(h * (1.0 - float(top_clip_frac))))
            clip_y = int(np.clip(clip_y, 0, h))
            if clip_y > 0:
                mask[:clip_y, :] = 0
            return mask

        def feather_mask(mask_u8: np.ndarray, dilate_px: int, feather_px: int) -> np.ndarray:
            m = mask_u8.copy()
            if dilate_px > 0:
                k = 2 * int(dilate_px) + 1
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
                m = cv2.dilate(m, kernel, iterations=1)
            if feather_px > 0:
                k = 2 * int(feather_px) + 1
                m = cv2.GaussianBlur(m, (k, k), sigmaX=0)
            return (m.astype(np.float32) / 255.0).clip(0.0, 1.0)

        rx = int((norm_w * mask_scale_x) * 0.5)
        ry = int((norm_h * mask_scale_y) * 0.5)
        inner_u8 = make_mouth_mask(norm_w, norm_h, rx=rx, ry=ry, center_y_offset_px=center_y_off, top_clip_frac=top_clip_frac)
        outer_u8 = make_mouth_mask(norm_w, norm_h, rx=rx + ring_px, ry=ry + ring_px, center_y_offset_px=center_y_off, top_clip_frac=top_clip_frac)
        ring_u8 = cv2.subtract(outer_u8, inner_u8)

        inner_f = feather_mask(inner_u8, dilate_px=dilate_px, feather_px=feather_px)
        ring_f = (ring_u8.astype(np.float32) / 255.0).clip(0.0, 1.0)

        # ---- interactive preview ----
        win = "erase range preview (q/ESC=close, space=play/pause, a/d=step, [ ]=+-10)"
        paused = True
        idx = 0

        cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

        src_pts = np.array(
            [[0, 0], [norm_w - 1, 0], [norm_w - 1, norm_h - 1], [0, norm_h - 1]],
            dtype=np.float32,
        )

        # colors (BGR)
        red = np.zeros((vid_h, vid_w, 3), dtype=np.uint8)
        red[:, :, 2] = 255
        yellow = np.zeros((vid_h, vid_w, 3), dtype=np.uint8)
        yellow[:, :, 1] = 255
        yellow[:, :, 2] = 255

        while True:
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            q = filled[idx].astype(np.float32).reshape(4, 2)

            # warp masks into full-frame space
            M = cv2.getPerspectiveTransform(src_pts, q)
            m_inner = cv2.warpPerspective(inner_f, M, (vid_w, vid_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            m_ring = cv2.warpPerspective(ring_f, M, (vid_w, vid_h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

            out = frame.copy()

            # overlay: inner (red), ring (yellow)
            a_inner = 0.45
            a_ring = 0.25
            out = (out.astype(np.float32) * (1.0 - a_inner * m_inner[..., None]) + red.astype(np.float32) * (a_inner * m_inner[..., None])).astype(np.uint8)
            out = (out.astype(np.float32) * (1.0 - a_ring * m_ring[..., None]) + yellow.astype(np.float32) * (a_ring * m_ring[..., None])).astype(np.uint8)

            # quad outline
            pts = q.reshape(-1, 1, 2).astype(np.int32)
            cv2.polylines(out, [pts], True, (0, 255, 0), 1, cv2.LINE_AA)

            # info text
            info = f"frame {idx+1}/{n_out}  cov={cov:.2f}  (red=erase, yellow=ring)"
            cv2.putText(out, info, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(out, info, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            if not bool(valid[idx]):
                cv2.putText(out, "INVALID (filled)", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

            cv2.imshow(win, out)

            delay = max(1, int(round(1000.0 / max(1.0, fps)))) if not paused else 15
            k = cv2.waitKey(delay)
            k8 = k & 0xFF

            if k8 in (ord("q"), 27):
                break
            # Check if window was closed
            try:
                if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break
            if k8 == ord(" "):
                paused = not paused
                continue
            if k8 == ord("a"):
                idx = max(0, idx - 1)
                paused = True
                continue
            if k8 == ord("d"):
                idx = min(n_out - 1, idx + 1)
                paused = True
                continue
            if k8 == ord("["):
                idx = max(0, idx - 10)
                paused = True
                continue
            if k8 == ord("]"):
                idx = min(n_out - 1, idx + 10)
                paused = True
                continue

            if not paused:
                idx += 1
                if idx >= n_out:
                    break

        cv2.destroyWindow(win)

    finally:
        try:
            cap.release()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


if __name__ == "__main__":
    main()
