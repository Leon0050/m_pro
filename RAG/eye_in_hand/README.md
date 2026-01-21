# Hand–Eye Calibration & Coordinate Conversion Pipeline

## 1. Overview

This project implements a **minimal and practical eye-in-hand calibration pipeline** for a robotic manipulation system.

- The camera is rigidly mounted on the robot end-effector (TCP).
- A Charuco board is used as the visual reference target.
- The final goal is to compute object poses in the robot **base frame** for grasping.

The system is intentionally designed to be **simple, modular, and SDK-agnostic**.

---

## 2. Coordinate Frames

The following coordinate frames are used throughout the system:

- **Base (B)**: Robot base frame
- **TCP / End-Effector (E)**: Robot tool frame
- **Camera (C)**: Camera frame (mounted on TCP)
- **Target (T)**: Charuco board frame

Two transformation matrices are fundamental:

- **T<sub>C→E</sub>**: Camera → TCP  
  - Obtained once via hand–eye calibration  
  - Fixed after calibration

- **T<sub>E→B</sub>**: TCP → Base  
  - Provided in real time by the robot controller / SDK  
  - Changes with robot motion

During operation, object poses are transformed as:


---

## 3. Required Inputs

### 3.1 Vision Inputs
- RGB image from the camera
- Pixel coordinates of detected features (e.g. Charuco corners)
- Depth value (mm) for the target
- Camera intrinsics:
  - fx, fy, cx, cy

### 3.2 Robot Inputs
- Current TCP pose relative to base (**T<sub>E→B</sub>**)
  - Read directly from robot SDK (no calibration required)

### 3.3 TCP → Base Transformation (Robot Pose)

The transformation from TCP to Base (**T_E→B**) is **not calibrated** and **not estimated** in this project.

It is provided **directly and in real time** by the robot controller via its SDK.

Typical sources include:
- Robot vendor SDK (e.g. UR RTDE, URScript, proprietary APIs)
- Robot controller Cartesian pose interface

The robot controller computes this transformation internally using:
- Joint encoder readings
- Forward kinematics

At each data collection step (`S` key press), the current TCP pose is queried once
and converted into a 4×4 homogeneous transformation matrix.

**This matrix is required for:**
- Data collection during hand–eye calibration
- Runtime conversion from TCP frame to Base frame

---

## 4. Workflow
### Data Collection Procedure (User Interaction)

The data collection script runs **continuously** once started.

**Keyboard controls:**

- **`S`**  
  Save one pose sample **only when**:
  - The robot is fully stopped
  - The Charuco board is successfully detected (status = `OK`)

  When `S` is pressed, the script records:
  - Target → Camera transformation (from vision)
  - TCP → Base transformation (from robot SDK)

- **`Q`**  
  Quit the data collection process.

**Important notes:**
- The script is started **once**.
- The user presses `S` repeatedly (10–20 times) at different robot poses.
- The script automatically validates Charuco detection; invalid frames are not saved.

### Step 1: Data Collection (Hand–Eye Calibration)

1. Start the data collection script.
2. Move the robot to multiple distinct poses.
3. For each pose:
   - Ensure the Charuco board is fully detected.
   - Save:
     - **T<sub>T→C</sub>** (from vision)
     - **T<sub>E→B</sub>** (from robot SDK)
4. Repeat for 10–20 poses.

### Step 2: Hand–Eye Calibration

- All collected pose pairs are used together to compute:
  - **T<sub>C→E</sub>** (Camera → TCP)

This matrix is fixed and reused in all later stages.

### Step 3: Runtime Coordinate Conversion

At runtime:

1. Detect object position in the camera frame.
2. Convert to TCP frame using **T<sub>C→E</sub>**.
3. Convert to base frame using **T<sub>E→B</sub>**.

---

## 5. Scripts

### `collect_charuco_pairs.py`
- Live camera visualization
- Automatic Charuco detection
- Saves paired transformations:
  - Target → Camera
  - TCP → Base
- Output: `pose_pairs.jsonl`

### `run_handeye_calib.py`
- Reads all pose pairs
- Computes final **T<sub>C→E</sub>**
- Output: `handeye_result.json`

### `dimension_convert.py`
- Converts 2D + depth observations into 3D poses
- Applies:
  - Camera → TCP
  - TCP → Base
- Final output: object pose in base frame

---

## 6. Output

The final system provides:

- **T<sub>C→E</sub>**: Camera to TCP (fixed, calibrated)
- **T<sub>E→B</sub>**: TCP to Base (real-time)
- **Object pose in Base frame** for robot execution

---

## 7. Notes & Common Pitfalls

- The TCP → Base transform is **not calibrated**; it must be read from the robot SDK.
- Only the Camera → TCP transform is estimated via hand–eye calibration.
- Always ensure:
  - Sufficient pose diversity during data collection
  - Clear and complete Charuco detection

Note:
Only the Camera → TCP transformation is estimated via hand–eye calibration.
The TCP → Base transformation is always obtained from the robot system itself.
