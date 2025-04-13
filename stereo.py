import cv2
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from typing import Tuple

class StereoProcessor:
    def __init__(self):
        self.focal_length = 1350  # in pixels
        self.baseline = 0.07      # in meters

    def match_color_balance(self, reference_img: np.ndarray, target_img: np.ndarray) -> np.ndarray:
        ref_channels = cv2.split(reference_img)
        target_channels = cv2.split(target_img)
        ref_means = [np.mean(ch) for ch in ref_channels]
        target_means = [np.mean(ch) for ch in target_channels]

        balanced_channels = []
        for ref_mean, target_mean, channel in zip(ref_means, target_means, target_channels):
            scale = ref_mean / target_mean if target_mean != 0 else 1
            balanced = np.clip(channel * scale, 0, 255).astype(np.uint8)
            balanced_channels.append(balanced)

        return cv2.merge(balanced_channels)

    def match_lighting(self, reference_img: np.ndarray, target_img: np.ndarray) -> np.ndarray:
        ref_gray = cv2.cvtColor(reference_img, cv2.COLOR_BGR2GRAY)
        target_gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)

        ref_mean, ref_std = np.mean(ref_gray), np.std(ref_gray)
        target_mean, target_std = np.mean(target_gray), np.std(target_gray)

        adjusted = target_img.astype(np.float32)
        adjusted = adjusted - target_mean + ref_mean

        if target_std != 0:
            adjusted *= (ref_std / target_std)

        return np.clip(adjusted, 0, 255).astype(np.uint8)

    def setup_stereo_matcher(self, is_left: bool = True) -> cv2.StereoSGBM:
        params = {
            'minDisparity': 0 if is_left else -144,
            'numDisparities': 144,
            'blockSize': 5,
            'P1': 8 * 3 * 5**3,
            'P2': 32 * 3 * 5**3,
            'disp12MaxDiff': 2,
            'uniquenessRatio': 5,
            'speckleWindowSize': 200,
            'speckleRange': 2,
            'mode': cv2.STEREO_SGBM_MODE_SGBM_3WAY
        }
        return cv2.StereoSGBM_create(**params)

    def process_frame_pair(self, left_frame: np.ndarray, right_frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        right_color_balanced = self.match_color_balance(left_frame, right_frame)
        right_final = self.match_lighting(left_frame, right_color_balanced)

        left_gray = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_final, cv2.COLOR_BGR2GRAY)

        stereo_left = self.setup_stereo_matcher(True)
        stereo_right = self.setup_stereo_matcher(False)

        disp_left = stereo_left.compute(left_gray, right_gray)
        disp_right = stereo_right.compute(right_gray, left_gray)

        wls_filter = cv2.ximgproc.createDisparityWLSFilter(stereo_left)
        wls_filter.setLambda(4700)
        wls_filter.setSigmaColor(1.2)
        filtered_disp = wls_filter.filter(disp_left, left_frame, None, disp_right)

        depth = np.zeros_like(filtered_disp, dtype=np.float32)
        valid_mask = filtered_disp > 0
        depth[valid_mask] = (self.focal_length * self.baseline) / filtered_disp[valid_mask] * 10  # mm to meters

        return filtered_disp, depth

def load_frames(left_path: str, right_path: str, frame_num: int = 99) -> Tuple[np.ndarray, np.ndarray]:
    cap_left = cv2.VideoCapture(left_path)
    cap_right = cv2.VideoCapture(right_path)

    try:
        if not all([cap_left.isOpened(), cap_right.isOpened()]):
            raise ValueError("Failed to open video files")

        frames_left = int(cap_left.get(cv2.CAP_PROP_FRAME_COUNT))
        frames_right = int(cap_right.get(cv2.CAP_PROP_FRAME_COUNT))
        if min(frames_left, frames_right) < frame_num + 1:
            raise ValueError(f"Videos must have at least {frame_num + 1} frames")

        cap_left.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        cap_right.set(cv2.CAP_PROP_POS_FRAMES, frame_num)

        ret_left, frame_left = cap_left.read()
        ret_right, frame_right = cap_right.read()
        if not (ret_left and ret_right):
            raise ValueError(f"Failed to read frame {frame_num + 1}")

        return frame_left, frame_right

    finally:
        cap_left.release()
        cap_right.release()

def visualize_disparity(disparity: np.ndarray, depth: np.ndarray):
    disp_visual = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(disp_visual, cmap='jet')
    ax.set_title('Disparity Map (WLS Filtered)')
    ax.axis('off')

    depth_text = ax.text(0.5, -0.1, 'Depth: N/A', ha='center', va='top', transform=ax.transAxes, fontsize=12)

    def update_depth(event):
        if event.inaxes == ax:
            x, y = int(event.xdata), int(event.ydata)
            if 0 <= x < depth.shape[1] and 0 <= y < depth.shape[0]:
                depth_val = depth[y, x]
                depth_text.set_text(f'Depth: {depth_val:.2f} meter' if depth_val > 0 else 'Depth: N/A')
            else:
                depth_text.set_text('Depth: N/A')
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect('motion_notify_event', update_depth)
    plt.tight_layout()
    plt.show()

def show_point_cloud(depth_map: np.ndarray, color_image: np.ndarray, fx: float, fy: float, cx: float, cy: float):
    h, w = depth_map.shape
    mask = depth_map > 0
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    x3 = (x[mask] - cx) * depth_map[mask] / fx
    y3 = (y[mask] - cy) * depth_map[mask] / fy
    z3 = depth_map[mask]

    points = np.stack((x3, y3, z3), axis=-1)
    colors = color_image[mask].astype(np.float32) / 255.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([pcd])

def main():
    processor = StereoProcessor()
    try:
        left_frame, right_frame = load_frames("video/left.avi", "video/right.avi")
        disparity, depth = processor.process_frame_pair(left_frame, right_frame)
        visualize_disparity(disparity, depth)

        fx = processor.focal_length
        fy = processor.focal_length
        cx = left_frame.shape[1] // 2
        cy = left_frame.shape[0] // 2

        show_point_cloud(depth, left_frame, fx, fy, cx, cy)
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()