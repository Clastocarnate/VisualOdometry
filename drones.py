import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def extract_and_match_features(img1, img2):
    """
    Extract corners using Shi-Tomasi, compute BRIEF descriptors, and match using BFMatcher.
    """
    if img1 is None or img2 is None:
        print("Error: One or both input images are None")
        return None, None, None

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    feature_params = dict(maxCorners=1000,
                         qualityLevel=0.01,
                         minDistance=7,
                         blockSize=7)

    corners1 = cv2.goodFeaturesToTrack(gray1, mask=None, **feature_params)
    corners2 = cv2.goodFeaturesToTrack(gray2, mask=None, **feature_params)

    if corners1 is None or corners2 is None or len(corners1) < 10 or len(corners2) < 10:
        print("Not enough corners detected in one or both images")
        return None, None, None

    keypoints1 = [cv2.KeyPoint(x[0][0], x[0][1], 10) for x in corners1]
    keypoints2 = [cv2.KeyPoint(x[0][0], x[0][1], 10) for x in corners2]

    brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
    keypoints1, descriptors1 = brief.compute(gray1, keypoints1)
    keypoints2, descriptors2 = brief.compute(gray2, keypoints2)

    if descriptors1 is None or descriptors2 is None or len(descriptors1) < 10 or len(descriptors2) < 10:
        print("Not enough descriptors computed")
        return None, None, None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)
    good_matches = matches[:int(len(matches) * 0.75)]

    if len(good_matches) < 10:
        print(f"Not enough good matches: {len(good_matches)}")
        return None, None, None

    print(f"Keypoints1: {len(keypoints1)}, Keypoints2: {len(keypoints2)}")
    print(f"Total Matches: {len(good_matches)}")

    return keypoints1, keypoints2, good_matches

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
    Estimates the pose between two images using the matched keypoints with RANSAC,
    and performs SVD on the essential matrix.
    """
    points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
    points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])

    focal_length = K[0, 0]
    principal_point = (K[0, 2], K[1, 2])
    
    # Use RANSAC with findEssentialMat for outlier removal
    essential_matrix, mask = cv2.findEssentialMat(points1, points2, 
                                                 focal=focal_length, 
                                                 pp=principal_point, 
                                                 method=cv2.RANSAC, 
                                                 prob=0.999, 
                                                 threshold=1.0)

    # Filter matches based on RANSAC inliers
    inlier_matches = [matches[i] for i in range(len(matches)) if mask[i]]

    if len(inlier_matches) < 10:
        print(f"Not enough inlier matches after RANSAC: {len(inlier_matches)}")
        return None, None

    # Recompute points with inlier matches
    points1 = np.float32([keypoints1[m.queryIdx].pt for m in inlier_matches])
    points2 = np.float32([keypoints2[m.trainIdx].pt for m in inlier_matches])

    # Perform SVD on the essential matrix
    U, S, Vt = np.linalg.svd(essential_matrix)
    V = Vt.T  # Transpose Vt to get V (right singular vectors)
    print(f"SVD V matrix:\n{S}")

    # Recover pose using the inlier matches
    R = np.eye(3)
    t = np.zeros((3, 1))
    _, R, t, _ = cv2.recoverPose(essential_matrix, points1, points2, K)

    # Normalize translation vector to unit length
    t_norm = np.linalg.norm(t)
    if t_norm > 0:
        t = t / t_norm

    print(f"Rotation Matrix:\n{R}")
    print(f"Translation Vector:\n{t}")

    return R, t

def main():
    # Camera parameters
    K = np.array([[1283.1076720832533, 0.0, 325.5842274588375],
                  [0.0, 1284.8079121262164, 246.67564927792367],
                  [0.0, 0.0, 1.0]])

    # Open video file
    cap = cv2.VideoCapture('vid2_vo.mp4')
    if not cap.isOpened():
        print("Error: Could not open video file")
        return

    # Initialize pose variables
    global_R = np.eye(3)
    global_t = np.zeros((3, 1))
    trajectory = [global_t.flatten()]

    # Set up live 2D plotting
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title("Visual Odometry Trajectory (2D)")
    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Z Position (m)")
    ax.grid(True)
    ax.set_aspect('equal')
    line, = ax.plot([], [], marker='o', linestyle='-')
    
    # Read first frame and crop
    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Could not read first frame")
        return
    
    h, w = prev_frame.shape[:2]
    crop_top = 100
    crop_bottom = h - 100
    prev_frame = prev_frame[crop_top:crop_bottom, :]

    while cap.isOpened():
        ret, curr_frame = cap.read()
        if not ret:
            break

        curr_frame = curr_frame[crop_top:crop_bottom, :]

        kp1, kp2, matches = extract_and_match_features(prev_frame, curr_frame)
        
        if kp1 is not None and matches is not None:
            R, t = get_pose(kp1, kp2, matches, K)
            if R is not None and t is not None:
                global_t = global_t + global_R @ t
                global_R = global_R @ R
                trajectory.append(global_t.flatten())

                # Update 2D trajectory plot (x-z plane)
                x = np.array([pos[0] for pos in trajectory])
                z = np.array([pos[2] for pos in trajectory])
                line.set_data(x, z)

                # Dynamically adjust axis limits with padding
                padding = 1.0
                if len(x) > 1:
                    ax.set_xlim(min(x) - padding, max(x) + padding)
                    ax.set_ylim(min(z) - padding, max(z) + padding)

                plt.draw()
                plt.pause(0.001)

                img_with_keypoints = draw_keypoints(curr_frame, kp2)
                cv2.imshow('Frame with Keypoints', img_with_keypoints)
            else:
                cv2.imshow('Frame with Keypoints', curr_frame)
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