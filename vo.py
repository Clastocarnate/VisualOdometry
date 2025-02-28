import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def extract_and_match_features(image_path1, image_path2):
    """
    Extract features using SIFT, match using FLANN, and apply RANSAC to remove outliers.

    Parameters:
        image_path1: Path to the first image.
        image_path2: Path to the second image.

    Returns:
        None (Displays matched features with outliers removed using RANSAC)
    """
    # Load images
    img1 = cv2.imread(image_path1, cv2.IMREAD_COLOR)
    img2 = cv2.imread(image_path2, cv2.IMREAD_COLOR)

    if img1 is None or img2 is None:
        print(f"Error loading images: {image_path1}, {image_path2}")
        return

    # Convert images to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Initialize SIFT detector
    orb = cv2.ORB_create()

    # Detect and compute features
    keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)

    if descriptors1 is None or descriptors2 is None:
        print("No descriptors found in one or both images.")
        return

    # Use FLANN-based matcher with KDTree algorithm
    index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Match descriptors using KNN
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    # Apply Lowe's ratio test
    good_matches = []
    for m_n in matches:
        if len(m_n) != 2:
            continue
        m,n = m_n
        if m.distance < 0.8 * n.distance:
            good_matches.append(m)


    # Extract matched keypoints
    if len(good_matches) > 10:  # Minimum matches required for RANSAC
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Apply RANSAC to filter outliers
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # Keep only inlier matches
        inlier_matches = [good_matches[i] for i in range(len(mask)) if mask[i]]

        print(f"Keypoints1: {len(keypoints1)}, Keypoints2: {len(keypoints2)}")
        print(f"Total Matches: {len(good_matches)}, Inliers after RANSAC: {len(inlier_matches)}")

        # Draw inlier matches
        match_img = cv2.drawMatches(img1, keypoints1, img2, keypoints2, inlier_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        # Show results using OpenCV
        cv2.imshow('Matched Features (RANSAC Filtered)', match_img)
        cv2.waitKey(1)
        return keypoints1, keypoints2, good_matches
    else:
        print(f"Not enough good matches for RANSAC: {len(good_matches)}")

# Path to images folder

def get_pose(keypoints1, keypoints2, matches, K):
    """
    Estimates the pose between two images using the matched keypoints.

    Parameters:
        keypoints1, keypoints2: Keypoints from the two images.
        matches: Matches between the keypoints.
        K: Intrinsic camera matrix.

    Returns:
        R, t: Rotation and translation matrices.
    """
    # Convert keypoints to Point2f format
    points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
    points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])

    # Compute the Essential matrix
    focal_length = K[0, 0]
    principal_point = (K[0, 2], K[1, 2])
    essential_matrix, _ = cv2.findEssentialMat(points1, points2, focal=focal_length, pp=principal_point)

    # Recover pose from the Essential matrix
    _, R, t, _ = cv2.recoverPose(essential_matrix, points1, points2, K)

    print(f"Essential Matrix:\n{essential_matrix}")
    print(f"Rotation Matrix:\n{R}")
    print(f"Translation Vector:\n{t}")

    return R, t



def main():
# Match features between consecutive images
    with open('KITTI_sequence_1/calib.txt', 'r') as f:
                params = np.fromstring(f.readline(), dtype=np.float64, sep=' ')
                P = np.reshape(params, (3, 4))
                K = P[0:3, 0:3]

    path = 'KITTI_sequence_1/image_l'
    images = sorted(os.listdir(path))

    global_R = np.eye(3)
    global_t = np.zeros((3, 1))
    trajectory = [global_t.flatten()]

    for i in range(1, len(images)):
        kp1, kp2, matches = extract_and_match_features(os.path.join(path, images[i-1]),
                                                       os.path.join(path, images[i]))
        if kp1 is None:
            continue  # Skip if feature extraction failed
        R, t = get_pose(kp1, kp2, matches, K)
        # Update the global pose:
        # The new pose is given by: T_i = T_(i-1) * T_rel
        global_t = global_t + global_R @ t
        global_R = global_R @ R
        trajectory.append(global_t.flatten())

    cv2.destroyAllWindows()

    # Convert trajectory list to a NumPy array for plotting
    trajectory = np.array(trajectory)
    # For a 2D plot, we take the x and z coordinates. Adjust indices if needed.
    x = trajectory[:, 0]
    z = trajectory[:, 2]

    plt.figure(figsize=(8, 6))
    plt.plot(x, z, marker='o', linestyle='-')
    plt.title("Visual Odometry Trajectory (2D)")
    plt.xlabel("X Position (m)")
    plt.ylabel("Z Position (m)")
    plt.grid(True)
    plt.axis('equal')
    plt.show()

if __name__ == "__main__":
    main()