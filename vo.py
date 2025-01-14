import cv2
import numpy as np


def find_feature_matches(img_1, img_2):
    """
    Finds feature matches between two images using ORB detector and BFMatcher.

    Parameters:
        img_1: The first input image.
        img_2: The second input image.

    Returns:
        keypoints_1, keypoints_2: Keypoints detected in the two images.
        good_matches: Filtered matches between the two images.
    """
    # Initialize ORB detector
    orb = cv2.ORB_create()

    # Detect ORB keypoints and descriptors
    keypoints_1, descriptors_1 = orb.detectAndCompute(img_1, None)
    keypoints_2, descriptors_2 = orb.detectAndCompute(img_2, None)

    # Match descriptors using Hamming distance and BFMatcher
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = matcher.match(descriptors_1, descriptors_2)

    # Filter matches based on distance
    min_dist = min(match.distance for match in matches)
    max_dist = max(match.distance for match in matches)

    print(f"-- Max dist: {max_dist}")
    print(f"-- Min dist: {min_dist}")

    # Keep matches with distance < 2 * min_dist or an empirical threshold
    good_matches = [m for m in matches if m.distance <= max(2 * min_dist, 30.0)]

    return keypoints_1, keypoints_2, good_matches


def pose_estimation_2d2d(keypoints_1, keypoints_2, matches, K):
    """
    Estimates the pose between two images using the matched keypoints.

    Parameters:
        keypoints_1, keypoints_2: Keypoints from the two images.
        matches: Matches between the keypoints.
        K: Intrinsic camera matrix.

    Returns:
        R, t: Rotation and translation matrices.
    """
    # Convert keypoints to Point2f format
    points1 = np.float32([keypoints_1[m.queryIdx].pt for m in matches])
    points2 = np.float32([keypoints_2[m.trainIdx].pt for m in matches])

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


def pixel2cam(p, K):
    """
    Converts pixel coordinates to normalized camera coordinates.

    Parameters:
        p: Pixel coordinates (x, y).
        K: Intrinsic camera matrix.

    Returns:
        Normalized camera coordinates.
    """
    x = (p[0] - K[0, 2]) / K[0, 0]
    y = (p[1] - K[1, 2]) / K[1, 1]
    return np.array([x, y])


def main():
    # Load images
    img_1 = cv2.imread("1.png", cv2.IMREAD_COLOR)
    img_2 = cv2.imread("2.png", cv2.IMREAD_COLOR)

    if img_1 is None or img_2 is None:
        print("Error loading images.")
        return

    # Find feature matches
    keypoints_1, keypoints_2, matches = find_feature_matches(img_1, img_2)
    print(f"Number of good matches: {len(matches)}")

    # Camera intrinsic matrix (TUM dataset)
    K = np.array([
        [520.9, 0, 325.1],
        [0, 521.0, 249.7],
        [0, 0, 1]
    ])

    # Estimate pose
    R, t = pose_estimation_2d2d(keypoints_1, keypoints_2, matches, K)

    # Verify the epipolar constraint
    t_x = np.array([
        [0, -t[2, 0], t[1, 0]],
        [t[2, 0], 0, -t[0, 0]],
        [-t[1, 0], t[0, 0], 0]
    ])
    for match in matches:
        pt1 = pixel2cam(keypoints_1[match.queryIdx].pt, K)
        pt2 = pixel2cam(keypoints_2[match.trainIdx].pt, K)
        y1 = np.array([pt1[0], pt1[1], 1])
        y2 = np.array([pt2[0], pt2[1], 1])
        epipolar_constraint = y2.T @ t_x @ R @ y1
        print(f"Epipolar constraint: {epipolar_constraint}")


if __name__ == "__main__":
    main()
