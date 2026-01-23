#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import numpy as np
import cv2


def list_to_T(m):
    return np.array(m, dtype=float)


def make_T(R, t):
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, dtype=float).reshape(3)
    return T


def inv_T(T):
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=float)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Input JSONL pose pairs file")
    ap.add_argument("--method", default="TSAI",
                    choices=["TSAI", "PARK", "HORAUD", "DANIILIDIS"],
                    help="Hand-eye method")
    ap.add_argument("--out", default="captures/handeye_result.json", help="Output result JSON")
    ap.add_argument("--output", default="cam2gripper", choices=["cam2gripper", "gripper2cam"],
                    help="Which transform to output")
    args = ap.parse_args()

    method_map = {
        "TSAI": cv2.CALIB_HAND_EYE_TSAI,
        "PARK": cv2.CALIB_HAND_EYE_PARK,
        "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
        "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
    }
    method = method_map[args.method]

    R_gripper2base = []
    t_gripper2base = []
    R_target2cam = []
    t_target2cam = []

    n = 0
    with open(args.inp, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            T_gb = list_to_T(obj["T_gripper_to_base"])       # gripper -> base (absolute pose)
            T_tc = list_to_T(obj["T_target_to_camera"])      # target  -> camera (absolute pose)

            R_gripper2base.append(T_gb[:3, :3])
            t_gripper2base.append(T_gb[:3, 3])

            R_target2cam.append(T_tc[:3, :3])
            t_target2cam.append(T_tc[:3, 3])

            n += 1

    if n < 8:
        raise RuntimeError(f"Need more samples. Got {n}, recommend >= 10–20.")

    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        R_gripper2base, t_gripper2base,
        R_target2cam, t_target2cam,
        method=method
    )

    T_cam2gripper = make_T(R_cam2gripper, t_cam2gripper)

    if args.output == "gripper2cam":
        T_out = inv_T(T_cam2gripper)
    else:
        T_out = T_cam2gripper

    result = {
        "input_file": args.inp,
        "method": args.method,
        "num_samples": n,
        "output": args.output,
        "T": T_out.tolist()
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("\n=== Hand-eye result ===")
    print(f"Samples: {n}")
    print(f"Method:  {args.method}")
    print(f"Output:  {args.output}")
    print("T (4x4):")
    print(np.array(result["T"]))
    print(f"\nSaved to: {args.out}")


if __name__ == "__main__":
    main()
