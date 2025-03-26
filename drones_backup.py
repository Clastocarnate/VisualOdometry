import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from constants import RANSAC_RESIDUAL_THRES, RANSAC_MAX_TRIALS
from skimage.transform import FundamentalMatrixTransform
from skimage.measure import ransac

def extract_and_match_features(img1, img2):
    """
    Extract corners using Shi-Tomasi and track them using optical flow.
    """
    if img1 is None or img2 is None:
        print("Error: One or both input images are None")
        return None, None, None

    # Convert to grayscale before feature detection
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Parameters for Shi-Tomasi corner detection
    feature_params = dict(maxCorners=1000,
                          qualityLevel=0.01,
                          minDistance=7,
                          blockSize=7)

    # Detect corners in the first image
    keypoints1 = cv2.goodFeaturesToTrack(gray1, mask=None, **feature_params)

    if keypoints1 is None or len(keypoints1) < 10:
        print("Not enough corners detected in first image")
        return None, None, None

    # Calculate optical flow to track points in the second image
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    p1 = keypoints1.astype(np.float32)
    p2, status, err = cv2.calcOpticalFlowPyrLK(gray1, gray2, p1, None, **lk_params)

    # Select good points (where tracking was successful)
    good_old = p1[status == 1]
    good_new = p2[status == 1]

    if len(good_new) < 10:
        print(f"Not enough good tracked points: {len(good_new)}")
        return None, None, None

    # Convert points to keypoint objects
    keypoints1 = [cv2.KeyPoint(x[0], x[1], 10) for x in good_old]
    keypoints2 = [cv2.KeyPoint(x[0], x[1], 10) for x in good_new]

    # Create matches
    matches = [cv2.DMatch(i, i, 0) for i in range(len(good_new))]

    print(f"Corners detected: {len(keypoints1)}, Tracked points: {len(keypoints2)}")
    print(f"Matches: {len(matches)}")

    return keypoints1, keypoints2, matches

def draw_keypoints(img, keypoints):
    """
    Draw keypoints on the image using cv2.circle
    """
    img_with_keypoints = img.copy()
    for kp in keypoints:
        x, y = kp.pt
        cv2.circle(img_with_keypoints, (int(x), int(y)), 3, (0, 255, 0), 1)
    return img_with_keypoints

def get_pose(keypoints1, keypoints2, matches, K):
    """
    Estimates the pose between two images using the matched keypoints.
    """
    points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
    points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])

    focal_length = K[0, 0]
    principal_point = (K[0, 2], K[1, 2])
    essential_matrix, _ = cv2.findEssentialMat(points1, points2, focal=focal_length, pp=principal_point)

    R = np.eye(3)
    t = np.zeros((3, 1))
    _, R, t, _ = cv2.recoverPose(essential_matrix, points1, points2, K)

    print(f"Rotation Matrix:\n{R}")
    print(f"Translation Vector:\n{t}")

    return R, t

def main():
    # Camera intrinsic parameters
    K = np.array([[203.1076720832533, 0.0, 325.5842274588375],
                  [0.0, 204.8079121262164, 246.67564927792367],
                  [0.0, 0.0, 1.0]])

    # Open video file
    cap = cv2.VideoCapture('test_countryroad.mp4')
    if not cap.isOpened():
        print("Error: Could not open video file")
        return

    # Initialize pose variables
    global_R = np.eye(3)
    global_t = np.zeros((3, 1))
    trajectory = [global_t.flatten()]

    # Set up live plotting
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title("Visual Odometry Trajectory (2D) ")
    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Z Position (m)")
    ax.grid(True)
    ax.set_aspect('equal')
    line, = ax.plot([], [], marker='o', linestyle='-')
    
    # Hardcoded rotation of 135° clockwise
    # The rotation matrix for a clockwise rotation by theta is:
    # [ cos(theta)   sin(theta)]
    # [-sin(theta)   cos(theta)]
    angle_deg = 110
    theta = np.deg2rad(angle_deg)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
    # Read first frame
    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Could not read first frame")
        return

    while cap.isOpened():
        ret, curr_frame = cap.read()
        if not ret:
            break

        # Match features between consecutive frames
        kp1, kp2, matches = extract_and_match_features(prev_frame, curr_frame)
        
        if kp1 is not None and matches is not None:
            R, t = get_pose(kp1, kp2, matches, K)
            # Update the global pose
            global_t = global_t + global_R @ t
            global_R = global_R @ R
            trajectory.append(global_t.flatten())

            # Rotate the trajectory points by 135° clockwise
            # For each point, x = pos[0] and z = pos[2]
            x_rot = [cos_theta * pos[0] + sin_theta * pos[2] for pos in trajectory]
            z_rot = [-sin_theta * pos[0] + cos_theta * pos[2] for pos in trajectory]
            line.set_data(x_rot, z_rot)
            ax.relim()
            ax.autoscale_view()
            plt.draw()
            plt.pause(0.001)

            # Draw keypoints on current frame
            img_with_keypoints = draw_keypoints(curr_frame, kp2)
            cv2.imshow('Frame with Keypoints', img_with_keypoints)
        else:
            cv2.imshow('Frame with Keypoints', curr_frame)

        prev_frame = curr_frame.copy()
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()
