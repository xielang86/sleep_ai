from dataclasses import dataclass

@dataclass
class PoseFeature:
  head_angle: float = -1
  head_angle_from_face: float = -1
  shoulder_hip_angle: float = -1
  face_knee_angle: float = -1

  pitch_angle: float = -1
  yaw_angle: float = -1
  roll_angle: float = -1

  left_ear_vis: float = 0
  right_ear_vis: float = 0
  left_knee_vis: float = 0
  right_knee_vis: float = 0
  left_hip_vis: float = 0
  right_hip_vis: float = 0

  left_wrist_vis: float = 0
  right_wrist_vis: float = 0
  left_thumb_vis: float = 0
  left_index_vis: float = 0 
  right_thumb_vis: float = 0
  right_index_vis: float = 0
  left_elbow_vis: float = 0 
  right_elbow_vis: float = 0

  left_ear_knee_y_dist: float = -1
  right_ear_knee_y_dist: float = -1 

  left_hip_knee_y_dist: float = -1
  left_eye_shoulder_y_dist: float = -1
  left_shoulder_hip_y_dist: float = -1

  right_hip_knee_y_dist: float = -1
  right_eye_shoulder_y_dist: float = -1
  right_shoulder_hip_y_dist: float = -1

  left_dist_wrist_hip: float = -1 
  left_dist_wrist_elbow : float = -1
  left_dist_thumb_elbow: float = -1
  left_dist_wrist_shoulder: float = -1
  left_dist_wrist_nose: float = -1
  
  left_wrist_dist_right_elbow: float = -1
  right_wrist_dist_left_elbow: float = -1

  left_dist_wrist_knee: float = -1
  right_dist_wrist_knee: float = -1

  left_index_angle: float = -1
  right_index_angle: float = -1
  left_hand_angle: float = -1
  right_hand_angle: float = -1

  left_thumb_body_dist: float = -1
  right_thumb_body_dist: float = -1

  left_arm_angle: float = -1
  right_arm_angle: float = -1

  left_hand_above: float = -1
  left_hand_above_dist: float = -1

  right_hand_above: float = -1
  right_hand_above_dist: float = -1