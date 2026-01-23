import numpy as np
import cv2

def make_T(R, t):
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t).reshape(3)
    return T

def inv_T(T):
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=float)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti

def random_R(min_deg=15, max_deg=60):
    axis = np.random.randn(3)
    axis /= np.linalg.norm(axis) + 1e-12
    ang = np.deg2rad(np.random.uniform(min_deg, max_deg))
    rvec = axis * ang
    R, _ = cv2.Rodrigues(rvec)
    return R

def random_T():
    R = random_R(20, 80)
    t = np.random.uniform(-0.3, 0.3, size=3)
    return make_T(R, t)

def rot_err_deg(R1, R2):
    R = R1.T @ R2
    v = (np.trace(R) - 1) / 2
    v = np.clip(v, -1.0, 1.0)
    return np.degrees(np.arccos(v))

np.random.seed(0)

# ===== 真值：X = cam->gripper（你希望算法估计回它）=====
X_true = random_T()

# ===== 真值：Y = base->target（标定板在 base 下固定不动）=====
Y = random_T()

# ===== 生成 N 组一致的“绝对位姿”采样 =====
N = 25

R_gripper2base, t_gripper2base = [], []
R_target2cam, t_target2cam = [], []

for _ in range(N):
    # 随机一个 base->gripper 的绝对位姿（模拟机械臂采样姿态）
    T_bg = random_T()  # base->gripper

    # 推导 target->cam:
    # base->gripper * gripper->cam * cam->target = base->target (=Y)
    # 其中 gripper->cam = inv(X_true)，且 cam->target = inv(target->cam)
    # 推导得到： target->cam = inv(Y) * T_bg * inv(X_true)
    T_tc = inv_T(Y) @ T_bg @ inv_T(X_true)   # target->cam

    # OpenCV 要的是 gripper->base（注意方向！）
    T_gb = inv_T(T_bg)  # gripper->base

    R_gripper2base.append(T_gb[:3, :3])
    t_gripper2base.append(T_gb[:3, 3])

    R_target2cam.append(T_tc[:3, :3])
    t_target2cam.append(T_tc[:3, 3])

# ===== 调用 OpenCV 手眼标定 =====
R_est, t_est = cv2.calibrateHandEye(
    R_gripper2base, t_gripper2base,
    R_target2cam, t_target2cam,
    # method=cv2.CALIB_HAND_EYE_TSAI
    method=cv2.CALIB_HAND_EYE_PARK
)

X_est = make_T(R_est, t_est)

print("X_true (cam->gripper):\n", X_true)
print("\nX_est  (cam->gripper):\n", X_est)

re = rot_err_deg(X_true[:3, :3], X_est[:3, :3])
te = np.linalg.norm(X_true[:3, 3] - X_est[:3, 3])
print(f"\nRotation error (deg): {re:.6f}")
print(f"Translation error (L2): {te:.6f}")
# ====== 误差对比：直接 vs 取逆 ======
def rot_err_deg(R1, R2):
    R = R1.T @ R2
    v = (np.trace(R) - 1) / 2
    v = np.clip(v, -1.0, 1.0)
    return np.degrees(np.arccos(v))

def trans_err(t1, t2):
    return np.linalg.norm(t1 - t2)

X_A = X_est
X_B = inv_T(X_est)

re_A = rot_err_deg(X_true[:3, :3], X_A[:3, :3])
te_A = trans_err(X_true[:3, 3], X_A[:3, 3])

re_B = rot_err_deg(X_true[:3, :3], X_B[:3, :3])
te_B = trans_err(X_true[:3, 3], X_B[:3, 3])

print("\nCompare solutions:")
print(f"A (as-is)   Rot(deg)={re_A:.6f}  Trans={te_A:.6f}")
print(f"B (inverse) Rot(deg)={re_B:.6f}  Trans={te_B:.6f}")
