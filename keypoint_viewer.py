import os
import numpy as np

ik_path = "data/selfharm/inference_keypoint.npy"
tk_path = "data/selfharm/test_keypoint.npy"

ik = np.load(ik_path)
tk = np.load(tk_path)

pose_1 = tk[0][0][0]
pose_2 = ik[0][0][0]
pair = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 17), (12, 17), (9, 7), (7, 5), (10, 8), (8, 6), (5, 19), (6, 19), (1, 0), (3, 1), (2, 0), (4, 2), (0, 19), (17, 18), (18, 19)]

import matplotlib.pyplot as plt

# Extract x, y coordinates (ignoring confidence values)
pose_1_xy = pose_1[:, :2]  # Shape: (20, 2)
pose_2_xy = pose_2[:, :2]  # Shape: (20, 2)

# Create visualization
plt.figure(figsize=(10, 8))
plt.scatter(pose_1_xy[:, 0], pose_1_xy[:, 1], c='red', s=50, label='pose_1 (test)', alpha=0.7)
plt.scatter(pose_2_xy[:, 0], pose_2_xy[:, 1], c='blue', s=50, label='pose_2 (inference)', alpha=0.7)

# Draw skeleton connections for pose_1
for joint1, joint2 in pair:
    plt.plot([pose_1_xy[joint1, 0], pose_1_xy[joint2, 0]], 
             [pose_1_xy[joint1, 1], pose_1_xy[joint2, 1]], 'r-', alpha=0.6, linewidth=2)

# Draw skeleton connections for pose_2
for joint1, joint2 in pair:
    plt.plot([pose_2_xy[joint1, 0], pose_2_xy[joint2, 0]], 
             [pose_2_xy[joint1, 1], pose_2_xy[joint2, 1]], 'b-', alpha=0.6, linewidth=2)

# Add joint numbers for each point
for i in range(20):
    plt.annotate(str(i), (pose_1_xy[i, 0], pose_1_xy[i, 1]), xytext=(5, 5), 
                textcoords='offset points', fontsize=8, color='red')
    plt.annotate(str(i), (pose_2_xy[i, 0], pose_2_xy[i, 1]), xytext=(5, 5), 
                textcoords='offset points', fontsize=8, color='blue')

plt.xlabel('X coordinate')
plt.ylabel('Y coordinate')
plt.title('Pose Comparison: Test vs Inference')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.savefig('pose_comparison.png', dpi=300, bbox_inches='tight')