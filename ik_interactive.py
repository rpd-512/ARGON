import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf
import yaml
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# === Load and Scale Dataset ===
df = pd.read_csv("datasets/kuka_youbot.csv")
X_pos = df[["target_pos_x", "target_pos_y", "target_pos_z"]].values

scaler = MinMaxScaler().fit(X_pos)

# === Load IK Model ===
model = tf.keras.models.load_model("ik_model.keras", compile=False)

# === Load DH Parameters ===
def load_dh(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)["dh_parameters"]

dh_params = load_dh("dh_parameters/kuka_youbot.yaml")

# === DH Transformation Matrix ===
def dh_matrix(a, alpha, d, theta):
    alpha = np.radians(alpha)
    return np.array([
        [np.cos(theta), -np.sin(theta)*np.cos(alpha),  np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
        [np.sin(theta),  np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
        [0,              np.sin(alpha),                np.cos(alpha),               d],
        [0,              0,                            0,                           1]
    ])

# === FK Using Predicted Angles ===
def compute_fk(joint_angles):
    T = np.eye(4)
    points = [T[:3, 3].copy()]
    for i, p in enumerate(dh_params):
        T = T @ dh_matrix(p["a"], p["alpha"], p["d"], joint_angles[i])
        points.append(T[:3, 3].copy())
    return np.array(points)

# === Predict Joint Angles from Target Position ===
def predict_angles(position):
    scaled = np.array([position])
    angles = model.predict(scaled, verbose=0)[0]
    return angles  # already in radians

# === Plot Setup ===
fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection='3d')
plt.subplots_adjust(left=0.25, bottom=0.3)

robot_line, = ax.plot([], [], [], '-o', lw=2, color='blue', label='Predicted Pose')
target_dot = ax.scatter([], [], [], color='green', s=50, label='Target Position')

ax.set_xlim(-500, 500)
ax.set_ylim(-500, 500)
ax.set_zlim(0, 550)
ax.set_xlabel("X (mm)")
ax.set_ylabel("Y (mm)")
ax.set_zlabel("Z (mm)")
ax.set_title("FANUC M-20iA Inverse Kinematics Visualization")
ax.legend()

# === Sliders for X, Y, Z ===
sliders = {}
slider_axes = {}
init_pos = [0, 0, 700]
labels = ['X (mm)', 'Y (mm)', 'Z (mm)']
ranges = [(-800, 800), (-800, 800), (300, 1400)]

for i, (label, (vmin, vmax)) in enumerate(zip(labels, ranges)):
    ax_slider = plt.axes([0.25, 0.2 - i * 0.05, 0.6, 0.03])
    slider = Slider(ax_slider, label, vmin, vmax, valinit=init_pos[i], valstep=10)
    sliders[label] = slider

# === Update Function ===
def update(val):
    pos = [sliders[label].val for label in labels]
    joint_angles = predict_angles(pos)
    fk_points = compute_fk(joint_angles)

    # Update line
    robot_line.set_data(fk_points[:, 0], fk_points[:, 1])
    robot_line.set_3d_properties(fk_points[:, 2])

    # Update green dot
    target_dot._offsets3d = ([pos[0]], [pos[1]], [pos[2]])

    fig.canvas.draw_idle()

# Connect sliders
for s in sliders.values():
    s.on_changed(update)

update(None)  # Initial draw
plt.show()
