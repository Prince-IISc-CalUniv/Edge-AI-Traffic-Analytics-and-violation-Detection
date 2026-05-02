#!/usr/bin/env python3
"""
Interactive homography calibrator for traffic speed estimation.
Maps pixel coordinates to real-world meters using a 4-point perspective transform.
"""

import argparse
import json
import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', required=True, help='Reference frame')
    ap.add_argument('--width', type=float, required=True,
                    help='Real-world width (across road) in meters')
    ap.add_argument('--length', type=float, required=True,
                    help='Real-world length (along traffic) in meters')
    ap.add_argument('--out', default='calibration.json')
    args = ap.parse_args()

    img = cv2.imread(args.image)
    assert img is not None, f"Cannot read {args.image}"

    pts = []
    clone = img.copy()

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append([x, y])
            cv2.circle(clone, (x, y), 8, (0, 0, 255), -1)
            cv2.putText(clone, str(len(pts)), (x+15, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
            cv2.imshow('calibrate', clone)

    print("\n=== HOMOGRAPHY CALIBRATION ===")
    print("Click 4 points on the ROAD SURFACE:")
    print("1. Top-Left     (far, left)")
    print("2. Top-Right    (far, right)")
    print("3. Bottom-Right (near, right)")
    print("4. Bottom-Left  (near, left)")
    print("Press 'q' to quit\n")

    cv2.namedWindow('calibrate', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('calibrate', on_click)

    while True:
        cv2.imshow('calibrate', clone)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            print("Calibration cancelled.")
            return
        if len(pts) == 4:
            break

    cv2.destroyAllWindows()

    W, L = args.width, args.length
    world_pts = [[0, 0], [W, 0], [W, L], [0, L]]

    calib = {
        'image_pts': pts,
        'world_pts': world_pts,
        'roi_polygon': pts,
        'width_m': W,
        'length_m': L,
        'calibrated_on': args.image
    }

    with open(args.out, 'w') as f:
        json.dump(calib, f, indent=2)

    print(f"\nCalibration saved to {args.out}")
    print(f"   Width:  {W} m | Length: {L} m")
    print("   Image points:", pts)


if __name__ == '__main__':
    main()
