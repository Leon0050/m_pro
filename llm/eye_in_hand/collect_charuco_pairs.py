#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# need numpy, pyrealsense2, opencv-python

import argparse
import json
import os
import time
from datetime import datetime

import numpy as np
import cv2


def make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, dtype=float).reshape(3)
    return T


def rvec_tvec_to_T(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=float).reshape(3, 1))
    t = np.asarray(tvec, dtype=float).reshape(3)
    return make_T(R, t)


def T_to_list(T: np.ndarray):
    return T.astype(float).tolist()


def parse_T_from_string(s: str) -> np.ndarray:
    """
    Parse a 4x4 matrix from a string of 16 numbers (row-major),
    e.g. "1 0 0 0  0 1 0 0  0 0 1 0  0 0 0 1"
    """
    nums = [float(x) for x in s.replace(",", " ").split()]
    if len(nums) != 16:
        raise ValueError("Expected 16 numbers for a 4x4 matrix.")
    T = np.array(nums, dtype=float).reshape(4, 4)
    return T


def get_aruco_dict(dict_name: str):
    # Common options: DICT_4X4_50, DICT_5X5_100, DICT_6X6_250, DICT_7X7_1000 ...
    if not hasattr(cv2.aruco, dict_name):
        raise ValueError(f"Unknown aruco dictionary: {dict_name}")
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))


def build_charuco_board(squares_x: int, squares_y: int, square_len: float, marker_len: float, aruco_dict):
    # OpenCV API compatibility
    if hasattr(cv2.aruco, "CharucoBoard_create"):
        board = cv2.aruco.CharucoBoard_create(
            squares_x, squares_y, square_len, marker_len, aruco_dict
        )
    else:
        # Newer OpenCV sometimes uses CharucoBoard(...) constructor
        board = cv2.aruco.CharucoBoard(
            (squares_x, squares_y), square_len, marker_len, aruco_dict
        )
    return board


def get_detector_params():
    # OpenCV API compatibility
    if hasattr(cv2.aruco, "DetectorParameters_create"):
        return cv2.aruco.DetectorParameters_create()
    return cv2.aruco.DetectorParameters()


def detect_charuco_pose(
    img_bgr: np.ndarray,
    aruco_dict,
    charuco_board,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
):
    """
    Returns:
      ok (bool),
      T_target_to_cam (4x4) if ok else None,
      debug_img (BGR) with overlays,
      num_charuco_corners (int)
    """
    debug = img_bgr.copy()
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    params = get_detector_params()

    # Detect ArUco markers
    corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=params)

    if ids is not None and len(ids) > 0:
        cv2.aruco.drawDetectedMarkers(debug, corners, ids)

        # Interpolate Charuco corners
        retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            corners, ids, gray, charuco_board, cameraMatrix=camera_matrix, distCoeffs=dist_coeffs
        )

        num = 0 if charuco_ids is None else len(charuco_ids)

        if charuco_corners is not None and charuco_ids is not None and num >= 6:
            # Draw detected charuco corners
            cv2.aruco.drawDetectedCornersCharuco(debug, charuco_corners, charuco_ids)

            # Estimate pose of the Charuco board (target->camera)
            ok, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                charuco_corners, charuco_ids, charuco_board, camera_matrix, dist_coeffs, None, None
            )

            if ok:
                cv2.drawFrameAxes(debug, camera_matrix, dist_coeffs, rvec, tvec, 0.05)
                T_tc = rvec_tvec_to_T(rvec, tvec)  # target->camera
                return True, T_tc, debug, num

        # Not enough charuco corners or pose failed
        cv2.putText(debug, f"Charuco corners: {num} (need >= 6)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return False, None, debug, num

    cv2.putText(debug, "No ArUco markers detected", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    return False, None, debug, 0


def load_intrinsics_from_file(path: str):
    """
    Accepts JSON:
      { "fx":..., "fy":..., "cx":..., "cy":..., "dist": [k1,k2,p1,p2,k3] }
    dist can be length 5 or 8; if missing, assume zeros.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    fx, fy, cx, cy = data["fx"], data["fy"], data["cx"], data["cy"]
    dist = data.get("dist", [0, 0, 0, 0, 0])
    camera_matrix = np.array([[fx, 0, cx],
                              [0, fy, cy],
                              [0,  0,  1]], dtype=float)
    dist_coeffs = np.array(dist, dtype=float).reshape(-1, 1)
    return camera_matrix, dist_coeffs


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def append_jsonl(path: str, obj: dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="captures/pose_pairs.jsonl", help="Output JSONL file")
    ap.add_argument("--save-images", action="store_true", help="Also save debug images for each accepted sample")
    ap.add_argument("--img-dir", default="captures/images", help="Directory to save debug images")

    # Charuco config
    ap.add_argument("--dict", default="DICT_4X4_50", help="Aruco dictionary name")
    ap.add_argument("--squares-x", type=int, default=5)
    ap.add_argument("--squares-y", type=int, default=7)
    ap.add_argument("--square-len", type=float, default=0.04, help="Square length in meters")
    ap.add_argument("--marker-len", type=float, default=0.02, help="Marker length in meters")

    # Camera intrinsics
    ap.add_argument("--intrinsics", default="", help="Path to intrinsics JSON (fx,fy,cx,cy,dist). If empty and using RealSense, we will read intrinsics from camera.")
    ap.add_argument("--dist-zeros", action="store_true", help="Force distCoeffs=0 (useful for quick testing)")

    # Input source
    ap.add_argument("--png", default="", help="If set, run in offline mode on a single PNG/JPG instead of RealSense.")
    ap.add_argument("--realsense", action="store_true", help="Use RealSense live stream (default if --png not set)")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)

    # Robot pose input method
    ap.add_argument("--robot-pose", default="manual",
                    choices=["manual", "identity"],
                    help="How to obtain gripper->base 4x4: manual paste, or identity (dry-run).")

    args = ap.parse_args()

    ensure_dir(os.path.dirname(args.out) or ".")
    if args.save_images:
        ensure_dir(args.img_dir)

    aruco_dict = get_aruco_dict(args.dict)
    charuco_board = build_charuco_board(args.squares_x, args.squares_y, args.square_len, args.marker_len, aruco_dict)

    # Intrinsics
    camera_matrix = None
    dist_coeffs = None

    if args.intrinsics:
        camera_matrix, dist_coeffs = load_intrinsics_from_file(args.intrinsics)

    if args.dist_zeros:
        dist_coeffs = np.zeros((5, 1), dtype=float)

    def get_robot_T_gb_manual() -> np.ndarray:
        print("\nPaste gripper->base 4x4 (16 numbers row-major), or empty for IDENTITY:")
        line = input("> ").strip()
        if not line:
            return np.eye(4, dtype=float)
        return parse_T_from_string(line)

    def get_robot_T_gb_identity() -> np.ndarray:
        return np.eye(4, dtype=float)

    get_robot_T_gb = get_robot_T_gb_manual if args.robot_pose == "manual" else get_robot_T_gb_identity

    # ===== Offline PNG mode =====
    if args.png:
        img = cv2.imread(args.png)
        if img is None:
            raise RuntimeError(f"Failed to read image: {args.png}")

        if camera_matrix is None:
            raise RuntimeError("Offline PNG mode requires --intrinsics JSON (fx,fy,cx,cy,dist).")

        if dist_coeffs is None:
            dist_coeffs = np.zeros((5, 1), dtype=float)

        ok, T_tc, debug, num = detect_charuco_pose(img, aruco_dict, charuco_board, camera_matrix, dist_coeffs)
        cv2.imshow("Charuco Debug", debug)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        if not ok:
            print("Charuco pose NOT detected. No output saved.")
            return

        T_gb = get_robot_T_gb()

        sample = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "source": "offline_png",
            "image_path": args.png,
            "charuco_corners": int(num),
            "T_target_to_camera": T_to_list(T_tc),
            "T_gripper_to_base": T_to_list(T_gb),
        }
        append_jsonl(args.out, sample)
        print(f"Saved 1 sample to: {args.out}")
        return

    # ===== Live RealSense mode =====
    if not args.realsense:
        args.realsense = True

    try:
        import pyrealsense2 as rs
    except Exception as e:
        raise RuntimeError(
            "pyrealsense2 not installed. Either install it in your venv, "
            "or use --png offline mode.\n"
            f"Original error: {e}"
        )

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)

    profile = pipeline.start(config)

    # If intrinsics not provided, read from RealSense
    if camera_matrix is None:
        color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = color_stream.get_intrinsics()
        camera_matrix = np.array([[intr.fx, 0, intr.ppx],
                                  [0, intr.fy, intr.ppy],
                                  [0, 0, 1]], dtype=float)
        # RealSense provides 5 distortion coeffs for Brown-Conrady typically
        dist = np.array(intr.coeffs[:5], dtype=float).reshape(5, 1)
        dist_coeffs = dist

    if dist_coeffs is None:
        dist_coeffs = np.zeros((5, 1), dtype=float)

    print("\nLive mode controls:")
    print("  - Look at the window.")
    print("  - When Charuco is detected and the arm is STOPPED, press 's' to SAVE one sample.")
    print("  - Press 'q' to quit.\n")

    sample_idx = 0
    out_file = args.out

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color = frames.get_color_frame()
            if not color:
                continue

            img = np.asanyarray(color.get_data())  # BGR
            ok, T_tc, debug, num = detect_charuco_pose(img, aruco_dict, charuco_board, camera_matrix, dist_coeffs)

            status = "OK" if ok else "NO"
            cv2.putText(debug, f"Charuco: {status}  corners={num}", (10, debug.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if ok else (0, 0, 255), 2)

            cv2.imshow("Charuco Live", debug)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord("s"):
                if not ok:
                    print("Not saved: Charuco pose not detected.")
                    continue

                # Get robot pose for this sample
                T_gb = get_robot_T_gb()

                sample_idx += 1
                sample = {
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "source": "realsense_live",
                    "sample_idx": sample_idx,
                    "charuco_corners": int(num),
                    "T_target_to_camera": T_to_list(T_tc),
                    "T_gripper_to_base": T_to_list(T_gb),
                }
                append_jsonl(out_file, sample)
                print(f"Saved sample #{sample_idx} to {out_file}")

                if args.save_images:
                    img_path = os.path.join(args.img_dir, f"sample_{sample_idx:04d}.png")
                    cv2.imwrite(img_path, debug)
                    print(f"Saved debug image: {img_path}")

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
