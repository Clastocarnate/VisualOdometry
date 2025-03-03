# import os
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.optimize import least_squares

# def extract_and_match_features(image_path1, image_path2):
#     img1 = cv2.imread(image_path1, cv2.IMREAD_COLOR)
#     img2 = cv2.imread(image_path2, cv2.IMREAD_COLOR)

#     if img1 is None or img2 is None:
#         print(f"Error loading images: {image_path1}, {image_path2}")
#         return

#     # Convert images to grayscale
#     gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
#     gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

#     # Initialize SIFT detector
#     orb = cv2.ORB_create()

#     # Detect and compute features
#     keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
#     keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)

#     if descriptors1 is None or descriptors2 is None:
#         print("No descriptors found in one or both images.")
#         return

#     # Use FLANN-based matcher with KDTree algorithm
#     index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)
#     search_params = dict(checks=50)
#     flann = cv2.FlannBasedMatcher(index_params, search_params)

#     # Match descriptors using KNN
#     matches = flann.knnMatch(descriptors1, descriptors2, k=2)

#     # Apply Lowe's ratio test
#     good_matches = []
#     for m_n in matches:
#         if len(m_n) != 2:
#             continue
#         m,n = m_n
#         if m.distance < 0.8 * n.distance:
#             good_matches.append(m)


#     # Extract matched keypoints
#     if len(good_matches) > 10:  # Minimum matches required for RANSAC
#         src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
#         dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

#         # Apply RANSAC to filter outliers
#         H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

#         # Keep only inlier matches
#         inlier_matches = [good_matches[i] for i in range(len(mask)) if mask[i]]

#         print(f"Keypoints1: {len(keypoints1)}, Keypoints2: {len(keypoints2)}")
#         print(f"Total Matches: {len(good_matches)}, Inliers after RANSAC: {len(inlier_matches)}")

#         # Draw inlier matches
#         match_img = cv2.drawMatches(img1, keypoints1, img2, keypoints2, inlier_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

#         # Show results using OpenCV
#         cv2.imshow('Matched Features (RANSAC Filtered)', match_img)
#         cv2.waitKey(1)
#         return keypoints1, keypoints2, good_matches
#     else:
#         print(f"Not enough good matches for RANSAC: {len(good_matches)}")


# def get_pose(keypoints1, keypoints2, matches, K):
#     points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
#     points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])

#     # Compute the Essential matrix
#     focal_length = K[0, 0]
#     principal_point = (K[0, 2], K[1, 2])
#     essential_matrix, _ = cv2.findEssentialMat(points1, points2, focal=focal_length, pp=principal_point)

#     # Recover pose from the Essential matrix
#     _, R, t, _ = cv2.recoverPose(essential_matrix, points1, points2, K)

#     print(f"Essential Matrix:\n{essential_matrix}")
#     print(f"Rotation Matrix:\n{R}")
#     print(f"Translation Vector:\n{t}")

#     return R, t


# def project_points(points_3d, R, t, K):
#     """Project 3D points to 2D image plane."""
#     R_t = np.hstack((R, t))
#     points_3d_h = np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))
#     points_cam = R_t @ points_3d_h.T
#     points_2d = K @ points_cam
#     points_2d = points_2d[:2] / points_2d[2]
#     return points_2d.T

# def reprojection_error(params, n_poses, n_points, observations, K):
#     """Calculate reprojection error for bundle adjustment."""
#     # Unpack parameters
#     poses_end = n_poses * 6  # Each pose has 6 parameters (3 for rotation, 3 for translation)
#     pose_params = params[:poses_end].reshape(n_poses, 6)
#     points_3d = params[poses_end:].reshape(n_points, 3)
    
#     error = []
#     for i, obs in enumerate(observations):
#         frame_idx, point_idx, measurement = obs
#         # Convert rotation vector to matrix
#         R, _ = cv2.Rodrigues(pose_params[frame_idx, :3])
#         t = pose_params[frame_idx, 3:].reshape(3, 1)
        
#         # Project 3D point to 2D
#         projected = project_points(points_3d[point_idx:point_idx+1], R, t, K)
#         error.extend(projected[0] - measurement)
    
#     return np.array(error)

# def local_bundle_adjustment(poses, points_3d, observations, K, window_size=5):
#     """Perform local bundle adjustment on a sliding window."""
#     n_poses_total = len(poses)
    
#     refined_poses = []
#     refined_points = points_3d.copy()
    
#     for start_idx in range(0, n_poses_total - window_size + 1):
#         end_idx = start_idx + window_size
        
#         # Extract local window data
#         local_poses = poses[start_idx:end_idx]
#         local_obs = [obs for obs in observations if start_idx <= obs[0] < end_idx]
        
#         # Get unique point indices in this window
#         point_indices = sorted(set(obs[1] for obs in local_obs))
#         point_map = {old_idx: new_idx for new_idx, old_idx in enumerate(point_indices)}
        
#         # Remap observations to local indices
#         local_obs_remapped = [(obs[0] - start_idx, point_map[obs[1]], obs[2]) 
#                             for obs in local_obs]
#         local_points = refined_points[point_indices]
        
#         # Prepare optimization parameters
#         pose_params = []
#         for R, t in local_poses:
#             rvec, _ = cv2.Rodrigues(R)
#             pose_params.extend(rvec.flatten())
#             pose_params.extend(t.flatten())
        
#         initial_params = np.hstack((pose_params, local_points.flatten()))
        
#         # Run optimization
#         result = least_squares(
#             reprojection_error,
#             initial_params,
#             args=(window_size, len(point_indices), local_obs_remapped, K),
#             verbose=0
#         )
        
#         # Extract optimized parameters
#         optimized_params = result.x
#         poses_end = window_size * 6
#         opt_poses = optimized_params[:poses_end].reshape(window_size, 6)
#         opt_points = optimized_params[poses_end:].reshape(len(point_indices), 3)
        
#         # Update global data
#         if start_idx == 0:
#             refined_poses.extend([(cv2.Rodrigues(p[:3])[0], p[3:].reshape(3, 1)) 
#                                 for p in opt_poses])
#         else:
#             refined_poses.append((cv2.Rodrigues(opt_poses[-1][:3])[0], 
#                                 opt_poses[-1][3:].reshape(3, 1)))
        
#         for old_idx, new_idx in point_map.items():
#             refined_points[old_idx] = opt_points[new_idx]
    
#     # Add remaining poses if any
#     if len(refined_poses) < n_poses_total:
#         refined_poses.extend(poses[len(refined_poses):])
    
#     return refined_poses, refined_points

# def main():
#     # Load camera parameters
#     with open('KITTI_sequence_1/calib.txt', 'r') as f:
#         params = np.fromstring(f.readline(), dtype=np.float64, sep=' ')
#         P = np.reshape(params, (3, 4))
#         K = P[0:3, 0:3]

#     path = 'KITTI_sequence_1/image_l'
#     images = sorted(os.listdir(path))

#     poses = [(np.eye(3), np.zeros((3, 1)))]  # List of (R, t) tuples
#     points_3d = []
#     observations = []  # List of (frame_idx, point_idx, [x, y])
#     point_counter = 0

#     for i in range(1, len(images)):
#         # Feature matching
#         kp1, kp2, matches = extract_and_match_features(
#             os.path.join(path, images[i-1]),
#             os.path.join(path, images[i])
#         )
#         if kp1 is None:
#             continue

#         # Initial pose estimation
#         R, t = get_pose(kp1, kp2, matches, K)
#         poses.append((R, t))

#         # Triangulate points
#         points1 = np.float32([kp1[m.queryIdx].pt for m in matches])
#         points2 = np.float32([kp2[m.trainIdx].pt for m in matches])
        
#         # Convert to homogeneous coordinates
#         points1_h = cv2.convertPointsToHomogeneous(points1)[:, 0, :]
#         points2_h = cv2.convertPointsToHomogeneous(points2)[:, 0, :]
        
#         # Projection matrices
#         P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
#         P2 = K @ np.hstack((R, t))
        
#         points_4d = cv2.triangulatePoints(P1, P2, points1.T, points2.T)
#         points_3d_new = (points_4d[:3] / points_4d[3]).T
        
#         # Add new points and observations
#         for j, pt in enumerate(points_3d_new):
#             points_3d.append(pt)
#             observations.append((i-1, point_counter, points1[j]))
#             observations.append((i, point_counter, points2[j]))
#             point_counter += 1

#         # Apply bundle adjustment every 5 frames
#         if i % 5 == 0 and i >= 5:
#             poses, points_3d = local_bundle_adjustment(
#                 poses, np.array(points_3d), observations, K, window_size=5
#             )

#     cv2.destroyAllWindows()

#     # Compute trajectory from optimized poses
#     trajectory = [np.zeros(3)]
#     global_R = np.eye(3)
#     for R, t in poses[1:]:
#         global_t = trajectory[-1].reshape(3, 1) + global_R @ t
#         global_R = global_R @ R
#         trajectory.append(global_t.flatten())

#     # Plot trajectory
#     trajectory = np.array(trajectory)
#     x = trajectory[:, 0]
#     z = trajectory[:, 2]

#     plt.figure(figsize=(8, 6))
#     plt.plot(x, z, marker='o', linestyle='-')
#     plt.title("Visual Odometry Trajectory with Bundle Adjustment (2D)")
#     plt.xlabel("X Position (m)")
#     plt.ylabel("Z Position (m)")
#     plt.grid(True)
#     plt.axis('equal')
#     plt.show()

# if __name__ == "__main__":
#     main()

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares

def extract_and_match_features(image_path1, image_path2):
    """
    Extract features using ORB, match using FLANN, and apply RANSAC to remove outliers.
    """
    img1 = cv2.imread(image_path1, cv2.IMREAD_COLOR)
    img2 = cv2.imread(image_path2, cv2.IMREAD_COLOR)

    if img1 is None or img2 is None:
        print(f"Error loading images: {image_path1}, {image_path2}")
        return None, None, None

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create()
    keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)

    if descriptors1 is None or descriptors2 is None:
        print("No descriptors found in one or both images.")
        return None, None, None

    index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    good_matches = []
    for m_n in matches:
        if len(m_n) != 2:
            continue
        m, n = m_n
        if m.distance < 0.8 * n.distance:
            good_matches.append(m)

    if len(good_matches) > 10:
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        inlier_matches = [good_matches[i] for i in range(len(mask)) if mask[i]]
        print(f"Keypoints1: {len(keypoints1)}, Keypoints2: {len(keypoints2)}")
        print(f"Total Matches: {len(good_matches)}, Inliers after RANSAC: {len(inlier_matches)}")
        return keypoints1, keypoints2, inlier_matches
    else:
        print(f"Not enough good matches for RANSAC: {len(good_matches)}")
        return None, None, None

def get_initial_pose(keypoints1, keypoints2, matches, K):
    points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
    points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])
    focal_length = K[0, 0]
    principal_point = (K[0, 2], K[1, 2])
    E, _ = cv2.findEssentialMat(points1, points2, focal=focal_length, pp=principal_point)
    _, R, t, _ = cv2.recoverPose(E, points1, points2, K)
    return R, t, points1, points2

def triangulate_points(R, t, points1, points2, K):
    """
    Triangulate 3D points from two views.
    """
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K @ np.hstack((R, t))
    points4d = cv2.triangulatePoints(P1, P2, points1.T, points2.T)
    points3d = points4d[:3] / points4d[3]
    return points3d.T

def project_points(points3d, R, t, K):
    """
    Project 3D points onto 2D image plane.
    """
    R_t = np.hstack((R, t))
    points2d_h = K @ R_t @ np.hstack((points3d, np.ones((points3d.shape[0], 1)))).T
    points2d = points2d_h[:2] / points2d_h[2]
    return points2d.T

def bundle_adjustment_cost(params, n_poses, n_points, observations, K):
    """
    Compute reprojection error for bundle adjustment.
    """
    poses = params[:n_poses * 6]  # 6 params per pose (3 for rotation, 3 for translation)
    points3d = params[n_poses * 6:].reshape((n_points, 3))

    residuals = []
    for i in range(n_poses):
        pose_params = poses[i * 6:(i + 1) * 6]
        R = cv2.Rodrigues(pose_params[:3])[0]
        t = pose_params[3:].reshape(3, 1)
        for obs in observations[i]:
            point_idx, point2d = obs
            proj_point = project_points(points3d[point_idx].reshape(1, -1), R, t, K)
            residuals.append(proj_point.flatten() - point2d)
    
    return np.concatenate(residuals)

def main():
    # Load calibration data
    with open('KITTI_sequence_1/calib.txt', 'r') as f:
        params = np.fromstring(f.readline(), dtype=np.float64, sep=' ')
        P = np.reshape(params, (3, 4))
        K = P[0:3, 0:3]

    path = 'KITTI_sequence_1/image_l'
    images = sorted(os.listdir(path))

    # Initialize data structures
    poses = [np.hstack((np.eye(3), np.zeros((3, 1))))]  # List of [R|t]
    points3d = []  # List of 3D points
    observations = [[]]  # List of lists: observations[i] contains (point_idx, point2d) for pose i
    point_index_map = {}  # Map from (frame, keypoint_idx) to global point index
    trajectory = [np.zeros(3)]

    window_size = 5  # Number of frames to include in bundle adjustment

    for i in range(1, len(images)):
        kp1, kp2, matches = extract_and_match_features(os.path.join(path, images[i-1]),
                                                       os.path.join(path, images[i]))
        if kp1 is None:
            continue

        R, t, points1, points2 = get_initial_pose(kp1, kp2, matches, K)
        new_pose = np.hstack((R, t))
        poses.append(new_pose)

        # Triangulate new points
        points3d_new = triangulate_points(R, t, points1, points2, K)
        
        # Add new points and observations
        for j, (p1, p2) in enumerate(zip(points1, points2)):
            key = (i-1, matches[j].queryIdx)
            if key not in point_index_map:
                point_index_map[key] = len(points3d)
                points3d.append(points3d_new[j])
            point_idx = point_index_map[key]
            observations[i-1].append((point_idx, p1))  # Observation in previous frame
            observations.append([(point_idx, p2)])     # Observation in current frame

        # Update trajectory
        global_t = trajectory[-1] + poses[-2][:3, :3] @ t.flatten()
        trajectory.append(global_t)

        # Perform bundle adjustment every 'window_size' frames
        if i % window_size == 0 and i > 1:
            print(f"Performing bundle adjustment at frame {i}...")
            n_poses = min(window_size + 1, len(poses))  # Include first pose
            start_idx = max(0, len(poses) - n_poses)
            poses_window = poses[start_idx:]
            obs_window = observations[start_idx:]
            
            # Prepare parameters for optimization
            pose_params = []
            for pose in poses_window:
                rvec = cv2.Rodrigues(pose[:3, :3])[0].flatten()
                tvec = pose[:3, 3]
                pose_params.extend(np.concatenate((rvec, tvec)))
            initial_params = np.concatenate((pose_params, np.array(points3d).flatten()))

            # Run bundle adjustment
            result = least_squares(
                bundle_adjustment_cost,
                initial_params,
                args=(n_poses, len(points3d), obs_window, K),
                verbose=1
            )

            # Update poses and points
            optimized_params = result.x
            for j in range(n_poses):
                pose_vec = optimized_params[j * 6:(j + 1) * 6]
                R = cv2.Rodrigues(pose_vec[:3])[0]
                t = pose_vec[3:].reshape(3, 1)
                poses[start_idx + j] = np.hstack((R, t))
            points3d = optimized_params[n_poses * 6:].reshape(-1, 3).tolist()

            # Update trajectory based on optimized poses
            trajectory[start_idx:] = []
            current_t = trajectory[start_idx - 1] if start_idx > 0 else np.zeros(3)
            for pose in poses[start_idx:]:
                current_t = current_t + pose[:3, :3] @ pose[:3, 3]
                trajectory.append(current_t)

    cv2.destroyAllWindows()

    # Plot trajectory
    trajectory = np.array(trajectory)
    x = trajectory[:, 0]
    z = trajectory[:, 2]
    plt.figure(figsize=(8, 6))
    plt.plot(x, z, marker='o', linestyle='-')
    plt.title("Visual Odometry Trajectory with Bundle Adjustment (2D)")
    plt.xlabel("X Position (m)")
    plt.ylabel("Z Position (m)")
    plt.grid(True)
    plt.axis('equal')
    plt.show()

if __name__ == "__main__":
    main()