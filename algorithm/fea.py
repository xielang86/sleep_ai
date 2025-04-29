from dataclasses import dataclass
from common.logger import CreateCustomLogger

@dataclass
class PoseFeature:
  head_angle: float = -1000
  head_angle_from_face: float = -1000
  shoulder_hip_angle: float = -1000
  face_knee_angle: float = -1000

  pitch_angle: float = -1000
  yaw_angle: float = -1000
  roll_angle: float = -1000

  left_ear_vis: float = 0
  right_ear_vis: float = 0
  left_eye_vis : float = 0
  right_eye_vis : float = 0

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

  left_ear_knee_y_dist: float = 0
  right_ear_knee_y_dist: float = 0
  eye_y_dist: float = 0
  

  left_hip_knee_y_dist: float = 0
  left_eye_shoulder_y_dist: float = 0
  left_shoulder_hip_y_dist: float = 0

  right_hip_knee_y_dist: float = 0
  right_eye_shoulder_y_dist: float = 0
  right_shoulder_hip_y_dist: float = 0

  left_dist_wrist_hip: float = 0
  left_dist_wrist_elbow : float = 0
  left_dist_thumb_elbow: float = 0
  left_dist_wrist_shoulder: float = 0
  left_dist_wrist_nose: float = 0
  left_dist_ear_knee: float = 0
  left_dist_shoulder_knee: float = 0

  right_dist_wrist_hip: float = 0
  right_dist_wrist_elbow : float = 0
  right_dist_thumb_elbow: float = 0
  right_dist_wrist_shoulder: float = 0
  right_dist_wrist_nose: float = 0
  right_dist_ear_knee: float = 0
  right_dist_shoulder_knee: float = 0
  
  left_wrist_dist_right_elbow: float = 0
  right_wrist_dist_left_elbow: float = 0

  left_dist_wrist_knee: float = 0
  right_dist_wrist_knee: float = 0

  left_index_angle: float = 0
  right_index_angle: float = 0
  left_hand_angle: float = 0
  right_hand_angle: float = 0

  left_thumb_body_dist: float = 0
  right_thumb_body_dist: float = 0

  left_arm_angle: float = 0
  right_arm_angle: float = 0
  
  left_arm_body_angle: float = 0
  right_arm_body_angle: float = 0

  left_hand_above: float = 0
  left_hand_above_dist: float = 0

  right_hand_above: float = 0
  right_hand_above_dist: float = 0

  def info(self):
    log_str = "all fea values:\n"
    a = 0
    for field, value in self.__dict__.items():
      a +=1
      if a >= 5:
        a = 0
        log_str += "\n"
      log_str += (f"\t{field}:\t{value}")
    return log_str