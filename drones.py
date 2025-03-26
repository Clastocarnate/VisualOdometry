#!/usr/bin/env python3
import os
import cv2
import numpy as np
import pypangolin as pangolin
import OpenGL.GL as gl
import time
from multiprocessing import Process, Queue
from constants import RANSAC_RESIDUAL_THRES, RANSAC_MAX_TRIALS

# -------------------------
# Utility Functions
# -------------------------
def project_feature_to_3d(kp, K, depth=1.0, R=np.eye(3), t=np.zeros((3,1))):
    """
    Projects a 2D keypoint into 3D using the pinhole camera model.
    Assumes a fixed depth.
    """
    u, v = kp.pt
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    Xc = (u - cx) * depth / fx
    Yc = (v - cy) * depth / fy
    Zc = depth
    p_cam = np.array([Xc, Yc, Zc]).reshape(3,1)
    p_vo = R @ p_cam + t
    return p_vo.flatten()

def extract_and_match_features(img1, img2):
    """
    Extract corners using Shi-Tomasi and track them using optical flow.
    """
    if img1 is None or img2 is None:
        print("Error: One or both input images is None")
        return None, None, None

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    feature_params = dict(maxCorners=1000,
                          qualityLevel=0.01,
                          minDistance=7,
                          blockSize=7)
    keypoints1 = cv2.goodFeaturesToTrack(gray1, mask=None, **feature_params)
    if keypoints1 is None or len(keypoints1) < 10:
        print("Not enough corners detected in first image")
        return None, None, None

    lk_params = dict(winSize=(15,15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    p1 = keypoints1.astype(np.float32)
    p2, status, err = cv2.calcOpticalFlowPyrLK(gray1, gray2, p1, None, **lk_params)

    good_old = p1[status==1]
    good_new = p2[status==1]
    if len(good_new) < 10:
        print(f"Not enough good tracked points: {len(good_new)}")
        return None, None, None

    keypoints1 = [cv2.KeyPoint(x[0], x[1], 10) for x in good_old]
    keypoints2 = [cv2.KeyPoint(x[0], x[1], 10) for x in good_new]
    matches = [cv2.DMatch(i, i, 0) for i in range(len(good_new))]
    print(f"Corners detected: {len(keypoints1)}, Tracked points: {len(keypoints2)}")
    print(f"Matches: {len(matches)}")
    return keypoints1, keypoints2, matches

def get_pose(keypoints1, keypoints2, matches, K):
    """
    Estimate the pose between two images using matched keypoints.
    """
    points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
    points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])
    focal_length = K[0,0]
    principal_point = (K[0,2], K[1,2])
    essential_matrix, _ = cv2.findEssentialMat(points1, points2,
                                               focal=focal_length,
                                               pp=principal_point)
    R = np.eye(3)
    t = np.zeros((3,1))
    _, R, t, _ = cv2.recoverPose(essential_matrix, points1, points2, K)
    print(f"Rotation Matrix:\n{R}")
    print(f"Translation Vector:\n{t}")
    return R, t

def draw_camera(gl, pose_matrix, size=1.0):
    """
    Draw a simple camera representation in OpenGL.
    Assumes pose_matrix is 4x4.
    """
    gl.glPushMatrix()
    gl.glMultMatrixf(pose_matrix.T)  # Transpose for OpenGL column-major order

    # Draw camera frustum
    gl.glColor3f(0.8, 0.8, 0.8)  # Light gray
    gl.glBegin(gl.GL_LINES)
    gl.glVertex3f(0,0,0)
    gl.glVertex3f(-size/2, -size/2, size)
    gl.glVertex3f(0,0,0)
    gl.glVertex3f(size/2, -size/2, size)
    gl.glVertex3f(0,0,0)
    gl.glVertex3f(size/2, size/2, size)
    gl.glVertex3f(0,0,0)
    gl.glVertex3f(-size/2, size/2, size)
    # Closing the frustum rectangle
    gl.glVertex3f(-size/2, -size/2, size)
    gl.glVertex3f(size/2, -size/2, size)
    gl.glVertex3f(size/2, -size/2, size)
    gl.glVertex3f(size/2, size/2, size)
    gl.glVertex3f(size/2, size/2, size)
    gl.glVertex3f(-size/2, size/2, size)
    gl.glVertex3f(-size/2, size/2, size)
    gl.glVertex3f(-size/2, -size/2, size)
    gl.glEnd()
    gl.glPopMatrix()

# -------------------------
# Display Process (Multiprocessing)
# -------------------------
def display_process(q):
    """
    This process creates a Pangolin window and continuously refreshes the display.
    It reads state from the multiprocessing queue.
    State is a tuple: (trajectory, camera_poses, all_features_3d)
    """
    # Setup Pangolin window and OpenGL state
    pangolin.CreateWindowAndBind('Visual Odometry', 1024, 768)
    gl.glEnable(gl.GL_DEPTH_TEST)
    proj = pangolin.ProjectionMatrix(1024,768,500,500,512,389,0.1,1000)
    look_view = pangolin.ModelViewLookAt(0, -0.5, -3, 0, 0, 0, pangolin.AxisY)
    s_cam = pangolin.OpenGlRenderState(proj, look_view)
    handler = pangolin.Handler3D(s_cam)
    d_cam = pangolin.CreateDisplay()
    d_cam.SetBounds(pangolin.Attach(0.0), pangolin.Attach(1.0),
                    pangolin.Attach(0.0), pangolin.Attach(1.0),
                    -1024.0/768.0)
    d_cam.SetHandler(handler)

    # Default state in case queue is empty
    state = (np.array([]), [], [])
    while True:
        # Get the most recent state from the queue (non-blocking)
        while not q.empty():
            state = q.get()
        # Clear buffers and set up the view
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        d_cam.Activate(s_cam)
        traj, cam_poses, all_feats = state

        # Draw trajectory as a red line strip
        if traj.size > 0:
            gl.glColor3f(1.0, 0.0, 0.0)
            gl.glBegin(gl.GL_LINE_STRIP)
            for point in traj:
                gl.glVertex3f(*point)
            gl.glEnd()

        # Draw cameras (using draw_camera)
        for pose in cam_poses:
            draw_camera(gl, pose, size=0.5)

        # Draw features as green points
        gl.glPointSize(3.0)
        gl.glColor3f(0.0, 1.0, 0.0)
        gl.glBegin(gl.GL_POINTS)
        for frame_feats in all_feats:
            for feature in frame_feats:
                gl.glVertex3f(*feature)
        gl.glEnd()

        pangolin.FinishFrame()
        time.sleep(0.01)

# -------------------------
# Main Process: Frame Processing and State Update
# -------------------------
def main():
    # Camera intrinsic parameters
    K = np.array([[203.1076720832533, 0.0, 325.5842274588375],
                  [0.0, 204.8079121262164, 246.67564927792367],
                  [0.0, 0.0, 1.0]])
    cap = cv2.VideoCapture('/Users/ayushupadhyay/Documents/SLAM/VisualOdometry/test_countryroad.mp4')
    if not cap.isOpened():
        print("Error: Could not open video file")
        return

    # Initialize global pose variables
    global_R = np.eye(3)
    global_t = np.zeros((3,1))
    trajectory = [global_t.flatten()]  # List of 3D translation vectors
    camera_poses = []  # List of 4x4 camera pose matrices
    all_features_3d = []  # List of arrays (each frame's 3D features)

    # Create a multiprocessing Queue and start the display process
    q = Queue()
    disp_proc = Process(target=display_process, args=(q,))
    disp_proc.daemon = True
    disp_proc.start()

    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Could not read first frame")
        return

    while cap.isOpened():
        ret, curr_frame = cap.read()
        if not ret:
            break

        # Optionally resize frame if needed (here we assume same size)
        # Process the frame pair using feature extraction and pose estimation
        kp1, kp2, matches = extract_and_match_features(prev_frame, curr_frame)
        if kp1 is not None and matches is not None:
            # Estimate relative pose from previous to current frame
            R, t = get_pose(kp1, kp2, matches, K)
            # Update global pose: compose with previous global_R and global_t
            global_t = global_t + global_R @ t
            global_R = global_R @ R
            trajectory.append(global_t.flatten())

            # Form the full 4x4 camera pose for display
            pose_matrix = np.eye(4)
            pose_matrix[:3, :3] = global_R
            pose_matrix[:3, 3] = global_t.flatten()
            camera_poses.append(pose_matrix)

            # Project current keypoints to 3D (assume fixed depth=1.0)
            features_3d = [project_feature_to_3d(kp, K, depth=1.0, R=global_R, t=global_t)
                           for kp in kp2]
            features_3d = np.array(features_3d)
            all_features_3d.append(features_3d)

        # Update display state by putting current state in the Queue.
        # Convert trajectory list to a NumPy array.
        q.put((np.array(trajectory), camera_poses, all_features_3d))

        prev_frame = curr_frame.copy()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    disp_proc.terminate()

if __name__ == "__main__":
    main()
