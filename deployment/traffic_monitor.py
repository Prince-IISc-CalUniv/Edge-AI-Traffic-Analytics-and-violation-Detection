#!/usr/bin/env python3
"""
Unified multi-pipeline traffic monitoring inference engine for Raspberry Pi 5.
Runs vehicle tracking, helmet/triple-riding detection, and pothole detection
simultaneously from a single camera feed.
"""

import argparse
import time
import csv
import json
import math
from collections import defaultdict, deque
from pathlib import Path
from datetime import datetime

import numpy as np
import cv2
from ultralytics import YOLO

DEFAULT_SPEED_LIMIT_KMH = 60.0
DEFAULT_IMG_SIZE        = 640
DEFAULT_OUTPUT_DIR      = 'outputs'
HELMET_FRAME_INTERVAL_S = 1.0

IDX_MOTORCYCLE = 0
IDX_PERSON     = 1
IDX_HELMET     = 2
IDX_NO_HELMET  = 3

CATEGORY_GROUPS = {
    'two_wheeler':   {'Two-wheeler', 'Bicycle', 'Bike', 'Motorcycle'},
    'three_wheeler': {'Three-wheeler', 'Rickshaw', 'Auto-rickshaw'},
    'car':           {'Hatchback', 'Sedan', 'SUV', 'MUV', 'Car'},
    'heavy':         {'Bus', 'Truck', 'Mini-bus', 'Tempo-traveller', 'LCV'},
    'van':           {'Van'},
}

DEFAULT_CALIB = {
    'image_pts':  [[300, 400], [980, 400], [1100, 700], [180, 700]],
    'world_pts':  [[0, 0], [10, 0], [10, 30], [0, 30]],
    'roi_polygon': None,
}


def box_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (a1 + a2 - inter + 1e-6)


def shrink_box(box, factor=0.7):
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    hw = (x2 - x1) * factor / 2.0
    hh = (y2 - y1) * factor / 2.0
    return [cx - hw, cy - hh, cx + hw, cy + hh]


def run_helmet_inference(helmet_model, frame, conf=0.25, imgsz=640):
    results = helmet_model(frame, imgsz=imgsz, half=False, conf=conf,
                           verbose=False, task='detect')
    annotated = results[0].plot()

    total_bikes = helmet_violations = triple_violations = safe_bikes = 0

    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            continue
        boxes = r.boxes.xyxy.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy().astype(int)

        motorcycles = boxes[classes == IDX_MOTORCYCLE]
        helmets = boxes[classes == IDX_HELMET]
        no_helmets = boxes[classes == IDX_NO_HELMET]

        total_bikes = len(motorcycles)

        for moto in motorcycles:
            moto_s = shrink_box(moto, factor=0.7)
            helm_on = [h for h in helmets if box_iou(moto_s, h) > 0.1]
            no_helm_on = [n for n in no_helmets if box_iou(moto_s, n) > 0.1]

            estimated_riders = len(helm_on) + len(no_helm_on)
            is_triple = estimated_riders > 2
            has_violation = len(no_helm_on) > 0

            if is_triple: triple_violations += 1
            if has_violation: helmet_violations += 1
            if not is_triple and not has_violation: safe_bikes += 1

            x1, y1, x2, y2 = moto.astype(int)
            color = (0, 0, 255) if (has_violation or is_triple) else (0, 255, 0)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)

            tags = []
            if is_triple: tags.append('TRIPLE!')
            if has_violation: tags.append('NO HELMET!')
            else: tags.append('SAFE')

            label = f'R:{estimated_riders} | {" | ".join(tags)}'
            cv2.putText(annotated, label, (x1, max(y1 - 15, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

    summary = {
        'total_bikes': total_bikes,
        'helmet_violations': helmet_violations,
        'triple_violations': triple_violations,
        'safe_bikes': safe_bikes,
    }
    return annotated, summary


class SpeedEstimator:
    def __init__(self, calib, fps):
        self.H, _ = cv2.findHomography(
            np.array(calib['image_pts'], dtype=np.float32),
            np.array(calib['world_pts'], dtype=np.float32),
        )
        self.fps = fps
        self.tracks = defaultdict(lambda: deque(maxlen=10))

    def img_to_world(self, x, y):
        pt = np.array([[x, y]], dtype=np.float32).reshape(-1, 1, 2)
        wp = cv2.perspectiveTransform(pt, self.H)[0, 0]
        return float(wp[0]), float(wp[1])

    def update(self, track_id, cx, cy, t):
        wx, wy = self.img_to_world(cx, cy)
        self.tracks[track_id].append((t, wx, wy))

    def speed_kmh(self, track_id):
        hist = self.tracks[track_id]
        if len(hist) < 3: return None
        t0, x0, y0 = hist[0]
        t1, x1, y1 = hist[-1]
        dt = t1 - t0
        if dt <= 0: return None
        d = math.hypot(x1 - x0, y1 - y0)
        return (d / dt) * 3.6


def classify_congestion(num_in_roi, avg_speed_kmh, roi_area_m2):
    if avg_speed_kmh is None or num_in_roi == 0:
        return 'A', 'Free flow'
    density = num_in_roi / max(roi_area_m2, 1.0) * 1000.0
    if   avg_speed_kmh > 50 and density <  5: return 'A', 'Free flow'
    elif avg_speed_kmh > 40 and density < 10: return 'B', 'Reasonable free flow'
    elif avg_speed_kmh > 30 and density < 18: return 'C', 'Stable flow'
    elif avg_speed_kmh > 20 and density < 26: return 'D', 'Approaching unstable'
    elif avg_speed_kmh > 10 and density < 40: return 'E', 'Unstable / at capacity'
    else:                                      return 'F', 'Forced flow / jam'


def save_speed_distribution(all_speeds, speed_limit, out_path):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        if not all_speeds:
            print("No speed data collected.")
            return

        speeds = np.array(all_speeds)
        fig, ax = plt.subplots(figsize=(10, 5))
        max_spd = max(float(speeds.max()), speed_limit + 20)
        bins = np.arange(0, max_spd + 5, 5)
        counts, edges = np.histogram(speeds, bins=bins)

        for lo, hi, cnt in zip(edges[:-1], edges[1:], counts):
            color = '#e74c3c' if lo >= speed_limit else '#2ecc71'
            ax.bar(lo, cnt, width=(hi - lo) * 0.9, align='edge', color=color, edgecolor='white')

        ax.axvline(speed_limit, color='#e67e22', linewidth=2, linestyle='--')
        pct_over = 100.0 * np.sum(speeds > speed_limit) / len(speeds)
        ax.text(0.98, 0.96, f'Overspeeding: {pct_over:.1f}%', transform=ax.transAxes,
                ha='right', va='top', fontsize=10, color='#e74c3c')

        ax.set_xlabel('Speed (km/h)')
        ax.set_ylabel('Number of Vehicles')
        ax.set_title('Vehicle Speed Distribution')
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Speed distribution plot saved: {out_path}")
    except ImportError:
        print("matplotlib not installed, skipping plot.")


def open_source(src):
    if src == 'picam':
        from picamera2 import Picamera2
        picam = Picamera2()
        cfg = picam.create_video_configuration(main={'size': (1280, 720), 'format': 'RGB888'})
        picam.configure(cfg)
        picam.start()
        return 'picam', picam, 30.0
    else:
        cap = cv2.VideoCapture(int(src) if src.isdigit() else src)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        return 'cv2', cap, fps


def read_frame(kind, handle):
    if kind == 'picam':
        frame = handle.capture_array()
        return True, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return handle.read()


def release_source(kind, handle):
    if kind == 'cv2':
        handle.release()
    elif kind == 'picam':
        handle.stop()


def class_to_group(name):
    for g, members in CATEGORY_GROUPS.items():
        if name in members:
            return g
    return 'other'


def put_hud_text(frame, text, y, font_scale=0.65):
    cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,0), 4, cv2.LINE_AA)
    cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), 1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser(description='Edge AI Traffic Monitor - Raspberry Pi 5')
    ap.add_argument('--model', default=None, help='Path to main traffic model (auto-detected)')
    ap.add_argument('--helmet-model', default='helmet_ncnn_model', help='Path to helmet NCNN model folder')
    ap.add_argument('--pothole-model', default='pothole', help='Path to pothole detection model')
    ap.add_argument('--source', default='picam', help='picam | 0 | video.mp4')
    ap.add_argument('--imgsz', type=int, default=DEFAULT_IMG_SIZE)
    ap.add_argument('--conf', type=float, default=0.05)
    ap.add_argument('--iou', type=float, default=0.5)
    ap.add_argument('--speed-limit', type=float, default=DEFAULT_SPEED_LIMIT_KMH)
    ap.add_argument('--calib', default='calibration.json')
    ap.add_argument('--no-display', action='store_true')
    ap.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR)
    ap.add_argument('--run-tag', default=None)

    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.run_tag or datetime.now().strftime('%Y%m%d_%H%M%S')
    stats_path = out_dir / f'stats_{tag}.csv'
    video_path = out_dir / f'annotated_{tag}.mp4'
    plot_path  = out_dir / f'speed_{tag}.png'

    print('='*80)
    print('EDGE AI TRAFFIC MONITOR - Raspberry Pi 5')
    print('='*80)

    print("\nLOADING MODELS")
    print("="*60)

    traffic_model_path = None
    traffic_candidates = [
        Path("best_ncnn_model"),
        Path("best_ncnn_model/best_ncnn_model"),
        Path("models/best_ncnn_model"),
        Path.cwd() / "best_ncnn_model",
    ]
    for p in traffic_candidates:
        if p.exists():
            traffic_model_path = p
            print(f"Main traffic model found: {traffic_model_path}")
            break

    if not traffic_model_path:
        print("Main traffic model 'best_ncnn_model' not found.")
        print("Please ensure 'best_ncnn_model' folder exists in the current directory.")
        return

    try:
        model = YOLO(str(traffic_model_path))
        print("Main traffic model loaded successfully")
    except Exception as e:
        print(f"Failed to load main traffic model: {e}")
        return

    helmet_model = None
    helmet_candidates = [
        Path(args.helmet_model),
        Path("helmet_ncnn_model"),
        Path("helmet_ncnn_model/helmet_ncnn_model"),
        Path("models/helmet_ncnn_model"),
    ]
    helmet_path = None
    for p in helmet_candidates:
        if p.exists():
            helmet_path = p
            break

    if helmet_path:
        try:
            helmet_model = YOLO(str(helmet_path))
            print(f"Helmet model loaded: {helmet_path}")
        except Exception as e:
            print(f"Failed to load helmet model: {e}")
    else:
        print("Helmet model not found, helmet and triple riding detection disabled")

    pothole_model = None
    pothole_candidates = [
        Path(args.pothole_model),
        Path("pothole"),
        Path("pothole/pothole"),
        Path("models/pothole"),
    ]
    for p in pothole_candidates:
        if p.exists():
            try:
                pothole_model = YOLO(str(p))
                print(f"Pothole model loaded: {p}")
                break
            except Exception as e:
                print(f"Failed to load pothole model: {e}")
                break
    if not pothole_model:
        print("Pothole model not loaded (can be integrated later)")

    calib = DEFAULT_CALIB
    if Path(args.calib).exists():
        calib = json.loads(Path(args.calib).read_text())
        print("Calibration loaded")
    else:
        print("Using default calibration")

    kind, handle, fps = open_source(args.source)
    print(f"Source: {args.source} | FPS: {fps:.1f}")

    roi_pts = np.array(calib.get('roi_polygon') or calib['image_pts'], dtype=np.int32)
    wpts = np.array(calib['world_pts'])
    roi_area_m2 = (wpts[:,0].max() - wpts[:,0].min()) * (wpts[:,1].max() - wpts[:,1].min())

    speed_est = SpeedEstimator(calib, fps)

    seen_ids = set()
    class_counts = defaultdict(int)
    overspeed_ids = set()
    overspeed_class = defaultdict(int)
    group_counts = defaultdict(int)
    vehicle_max_speeds = {}

    total_helmet_bikes = total_helmet_violations = total_triple_violations = total_safe_bikes = 0
    last_helmet_t = -HELMET_FRAME_INTERVAL_S

    with open(stats_path, 'w', newline='') as stats_csv:
        csvw = csv.writer(stats_csv)
        csvw.writerow(['frame', 't_s', 'vehicles_in_roi', 'avg_speed_kmh', 'los',
                        'overspeeding', 'total_unique', 'helmet_bikes',
                        'helmet_violations', 'triple_violations'])

        WINDOW_NAME = 'Edge AI Traffic Monitor'
        if not args.no_display:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        writer = None
        frame_idx = 0
        t_start = time.time()
        fps_smoother = deque(maxlen=30)
        last_t = time.time()

        print("Starting main loop... Press 'q' to quit.")

        while True:
            ok, frame = read_frame(kind, handle)
            if not ok or frame is None:
                print("End of stream. Looping video for demo...")
                if kind == 'cv2':
                    handle.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            t_now = time.time() - t_start
            frame_idx += 1

            results = model.track(frame, imgsz=args.imgsz, conf=args.conf,
                                  iou=args.iou, persist=True,
                                  tracker='bytetrack.yaml', verbose=False)
            r = results[0]

            in_roi_count = 0
            current_speeds = []

            if r.boxes is not None and r.boxes.id is not None:
                ids = r.boxes.id.cpu().numpy().astype(int)
                cls = r.boxes.cls.cpu().numpy().astype(int)
                xyxy = r.boxes.xyxy.cpu().numpy()

                for box, c, tid in zip(xyxy, cls, ids):
                    x1, y1, x2, y2 = box
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    cname = model.names[int(c)]

                    if tid not in seen_ids:
                        seen_ids.add(tid)
                        class_counts[cname] += 1
                        group_counts[class_to_group(cname)] += 1

                    in_roi = cv2.pointPolygonTest(roi_pts, (float(cx), float(cy)), False) >= 0
                    if in_roi:
                        in_roi_count += 1
                        speed_est.update(int(tid), cx, cy, t_now)
                        s = speed_est.speed_kmh(int(tid))
                        if s is not None:
                            current_speeds.append(s)
                            vehicle_max_speeds[int(tid)] = max(vehicle_max_speeds.get(int(tid), 0), s)
                            if s > args.speed_limit:
                                overspeed_ids.add(int(tid))
                                overspeed_class[cname] += 1

                    color = (0, 0, 255) if int(tid) in overspeed_ids else (0, 255, 0)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 4)

                    lbl = f'{cname} #{tid}'
                    if (spd := speed_est.speed_kmh(int(tid))) is not None:
                        lbl += f' {int(spd)}km/h'
                    cv2.putText(frame, lbl, (int(x1), max(int(y1)-10, 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            if helmet_model and (t_now - last_helmet_t >= HELMET_FRAME_INTERVAL_S):
                last_helmet_t = t_now
                _, h_sum = run_helmet_inference(helmet_model, frame.copy())
                total_helmet_bikes += h_sum['total_bikes']
                total_helmet_violations += h_sum['helmet_violations']
                total_triple_violations += h_sum['triple_violations']
                total_safe_bikes += h_sum['safe_bikes']

            avg_speed = np.mean(current_speeds) if current_speeds else None
            avg_speed_display = avg_speed if avg_speed is not None else 0.0

            los, los_desc = classify_congestion(in_roi_count, avg_speed, roi_area_m2)

            fps_smoother.append(1.0 / max(time.time() - last_t, 1e-6))
            last_t = time.time()
            cur_fps = np.mean(fps_smoother)

            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], 60), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

            hud_lines = [
                f"Edge AI Traffic Monitor | FPS: {cur_fps:.1f} | Frame: {frame_idx}",
                f"Congestion: LOS {los} ({los_desc}) | Vehicles in ROI: {in_roi_count}",
                f"Avg Speed: {avg_speed_display:.1f} km/h | Overspeeding: {len(overspeed_ids)}",
                f"Total Vehicles: {len(seen_ids)} | Helmet Violations: {total_helmet_violations}"
            ]
            for i, line in enumerate(hud_lines):
                put_hud_text(frame, line, 25 + i * 28, font_scale=0.7)

            if not args.no_display:
                cv2.imshow(WINDOW_NAME, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            writer.write(frame)

            csvw.writerow([frame_idx, round(t_now, 2), in_roi_count,
                          round(avg_speed, 2) if avg_speed is not None else '',
                          los, len(overspeed_ids), len(seen_ids),
                          total_helmet_bikes, total_helmet_violations, total_triple_violations])

    elapsed = time.time() - t_start
    print("\n" + "="*80)
    print("FINAL ANALYSIS REPORT")
    print("="*80)
    print(f"Runtime: {elapsed:.1f}s | Processed Frames: {frame_idx}")
    print(f"Total Unique Vehicles: {len(seen_ids)}")
    print(f"Overspeeding: {len(overspeed_ids)}")
    print(f"Helmet Violations: {total_helmet_violations} | Triple Riding: {total_triple_violations}")

    save_speed_distribution(list(vehicle_max_speeds.values()), args.speed_limit, str(plot_path))

    if writer: writer.release()
    release_source(kind, handle)
    cv2.destroyAllWindows()

    print(f"\nOutputs saved in: {out_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
