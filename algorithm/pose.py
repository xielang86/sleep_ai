import math
import mediapipe as mp
import cv2
from typing import Literal
from enum import Enum
from dataclasses import dataclass
from algorithm.util import distance

class PoseType(Enum):
  LieFlat = 1
  HalfLie = 2
  LieSide = 3
  LieFaceDown = 4
  SitDown = 5
  Stand = 6
  Other = 16
  
@dataclass
class PoseResult:
  pose_type: PoseType
  pose_prob: float

  leftEyeClosed: bool
  rightEyeClosed: bool
  eyeCloseProb: float

"""
NOSE：鼻子的位置。这个关键点对于定位面部方向以及整体头部位置很有用。
LEFT_EYE_INNER：左眼的内眼角。在分析眼部动作或者面部表情时可以用到这个关键点。
LEFT_EYE：左眼的中心位置。与其他眼部关键点一起可以用于判断眼睛的状态，如眨眼、注视方向等。
LEFT_EYE_OUTER：左眼的外眼角。
RIGHT_EYE_INNER：右眼的内眼角。
RIGHT_EYE：右眼的中心位置。
RIGHT_EYE_OUTER：右眼的外眼角。
LEFT_EAR：左耳的位置。可以辅助判断头部的倾斜方向等信息。
RIGHT_EAR：右耳的位置。
MOUTH_LEFT：嘴巴的左角。在分析嘴部动作和表情（如微笑、说话）时有重要作用。
MOUTH_RIGHT：嘴巴的右角。
LEFT_SHOULDER：左肩的位置。这是人体上半身姿势分析的关键节点，用于判断肩部的位置、倾斜程度等。
RIGHT_SHOULDER：右肩的位置。
LEFT_ELBOW：左肘的位置。可以结合肩部和手腕关键点来分析手臂的弯曲程度和姿态。
RIGHT_ELBOW：右肘的位置。
LEFT_WRIST：左手腕的位置。对于手部动作分析以及姿态估计非常重要，比如判断手是举起还是放下。
RIGHT_WRIST：右手腕的位置。
LEFT_PINKY：左手小指的位置。在精细的手部动作分析（如手势识别）中有作用。
RIGHT_PINKY：右手小指的位置。
LEFT_INDEX：左手食指的位置。
RIGHT_INDEX：右手食指的位置。
LEFT_THUMB：左手拇指的位置。
RIGHT_THUMB：右手拇指的位置。
LEFT_HIP：左髋的位置。对于判断人体下半身的姿势（如站立、坐、躺等）非常关键。
RIGHT_HIP：右髋的位置。
LEFT_KNEE：左膝的位置。结合髋部和脚踝关键点可以分析腿部的弯曲程度和姿势。
RIGHT_KNEE：右膝的位置。
LEFT_ANKLE：左脚踝的位置。用于判断脚部的位置和姿态。
RIGHT_ANKLE：右脚踝的位置。
LEFT_HEEL：左脚跟的位置。在分析脚部动作（如走路、跑步时脚跟的抬起和放下）时有用。
RIGHT_HEEL：右脚跟的位置。
LEFT_FOOT_INDEX：左脚食指（脚趾）的位置。在分析脚部细节动作（如踮脚等）时可以用到。
RIGHT_FOOT_INDEX：右脚食指（脚趾）的位置
"""

import cv2
import mediapipe as mp
import numpy as np

# thread unsafe
class PoseDetector:
  def __init__(self):
    self.mp_pose = mp.solutions.pose
    self.pose = self.mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5,
                                 min_tracking_confidence=0.5)
    self.mp_face_mesh = mp.solutions.face_mesh


  def DetectEyeOpen(self, image, face_landmarks):
    # 眼睛关键点的索引（根据 MediaPipe 的标准）
    eye_prob = 0.5
    left_eye_landmarks = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
    right_eye_landmarks = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    left_eye_points = []
    right_eye_points = []
    for idx in left_eye_landmarks:
      pt = face_landmarks.landmark[idx]
      x, y = int(pt.x * image.shape[1]), int(pt.y * image.shape[0])
      left_eye_points.append((x, y))
    for idx in right_eye_landmarks:
      pt = face_landmarks.landmark[idx]
      x, y = int(pt.x * image.shape[1]), int(pt.y * image.shape[0])
      right_eye_points.append((x, y))

    # 计算眼睛的纵横比（EAR）来判断是否闭眼
    def eye_aspect_ratio(eye):
      A = np.linalg.norm(np.array(eye[1]) - np.array(eye[5]))
      B = np.linalg.norm(np.array(eye[2]) - np.array(eye[4]))
      C = np.linalg.norm(np.array(eye[0]) - np.array(eye[3]))
      ear = (A + B) / (2.0 * C)
      return ear

    def CheckEyeBall(image, eye_points):
      eye_roi = image[min([y for x, y in eye_points]):max([y for x, y in eye_points]),
                    min([x for x, y in eye_points]):max([x for x, y in eye_points])]
      # 计算眼睛区域的平均灰度值
      average_gray = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY).mean()
      eyeball_threshold = 85  # 可以根据实际情况调整
      print(average_gray)
      if average_gray < eyeball_threshold:
        return True
      else:
        return False

    left_ear = eye_aspect_ratio(left_eye_points)
    right_ear = eye_aspect_ratio(right_eye_points)
    left_eyeball_exists = CheckEyeBall(image, left_eye_points)
    right_eyeball_exists = CheckEyeBall(image, right_eye_points)
    ear_threshold = 0.2  # 可以根据实际情况调整这个阈值
    left_eye_closed = not left_eyeball_exists
    right_eye_closed = not right_eyeball_exists
    return left_eye_closed, right_eye_closed,eye_prob

  def CalcHeadAngle(self, image, face_landmarks)->float:
    left_eye_corner = face_landmarks.landmark[133]
    left_mouth_corner = face_landmarks.landmark[61]
    # 获取关键点的像素坐标
    left_eye_corner_x, left_eye_corner_y = int(left_eye_corner.x * image.shape[1]), int(left_eye_corner.y * image.shape[0])
    left_mouth_corner_x, left_mouth_corner_y = int(left_mouth_corner.x * image.shape[1]), int(left_mouth_corner.y * image.shape[0])
    # 计算向量
    dx = left_eye_corner_x - left_mouth_corner_x
    dy = left_eye_corner_y - left_mouth_corner_y
    # 计算角度（弧度）
    angle_rad = math.atan2(dy, dx)
    # 转换为角度
    angle_deg = math.degrees(angle_rad)
    return angle_deg

  def CalcHeadAngle2(self, pos_landmarks)->float:
    left_eye= pos_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_EYE.value]
    left_mouth= pos_landmarks.landmark[self.mp_pose.PoseLandmark.MOUTH_LEFT.value]
    right_eye = pos_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_EYE.value]
    right_mouth= pos_landmarks.landmark[self.mp_pose.PoseLandmark.MOUTH_RIGHT.value]

    dx = left_eye.x - left_mouth.x
    dy = left_eye.y - left_mouth.y
    # 计算角度（弧度）
    angle_rad = math.atan2(dy, dx)
    # 转换为角度
    angle_deg = math.degrees(angle_rad)

    print(f"calc head angle2, dx={dx}, dv={dy}, rad={angle_rad} angle_deg={angle_deg}")
    return angle_deg
    
  def CalcBodyAngle(self, image, landmark)->float:
    left_shoulder_landmark = landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    left_hip_landmark = landmark[self.mp_pose.PoseLandmark.LEFT_HIP.value]
    left_shoulder_x, left_shoulder_y = left_shoulder_landmark.x * image.shape[1], left_shoulder_landmark.y * image.shape[0]
    left_hip_x, left_hip_y = left_hip_landmark.x * image.shape[1], left_hip_landmark.y * image.shape[0]
    # 计算上半身向后仰的角度（弧度制）
    dx_body = left_shoulder_x - left_hip_x
    dy_body = left_shoulder_y - left_hip_y
    body_angle_rad = math.atan2(dy_body, dx_body)
    body_angle_deg = math.degrees(body_angle_rad)
    return body_angle_deg

  def DetectPoseByRule(self, landmarks, head_angle, body_angle) -> PoseType:
    nose = landmarks.landmark[self.mp_pose.PoseLandmark.NOSE.value]

    left_eye= landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_EYE.value]
    left_mouth= landmarks.landmark[self.mp_pose.PoseLandmark.MOUTH_LEFT.value]
    left_hip = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP.value]
    left_knee = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
    left_ankle = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
    left_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    left_ear = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_EAR.value]

    right_eye = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_EYE.value]
    right_mouth= landmarks.landmark[self.mp_pose.PoseLandmark.MOUTH_RIGHT.value]
    left_hip = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP.value]
    right_hip = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
    right_knee = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_KNEE.value]
    right_ankle = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]
    right_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    right_ear = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_EAR.value]

    mouth_vis = max(left_mouth.visibility, right_mouth.visibility)
    knee_vis = max(left_knee.visibility, right_knee.visibility)

    # first judge sleep pose, xxxlie
    print(f"mouth_prob={mouth_vis}, knee_vis={knee_vis}")
    print(f"eyey={left_eye.y}, nosey={nose.y}, eary={left_ear.y}, mouthy={left_mouth.y}, hip={left_hip.y}, knee={left_knee.y}, ankle={left_ankle.y}")
    print(f"delta nose and mouth, mout-nose={left_mouth.y-nose.y}, ear-nose={left_ear.y - nose.y}, mouth-ear={left_mouth.y - left_ear.y}")

    pose_prob = 0.5
    pose_type = PoseType.SitDown

    if (head_angle < -150 ) or \
    (left_ear.visibility and left_knee.visibility and left_ear.y > left_knee.y) or \
    (right_ear.visibility and right_knee.visibility and right_ear.y > right_knee.y):
      pose_type = PoseType.LieFlat
    elif (head_angle < -100 and body_angle < -88) or (body_angle < -100 and head_angle < -90):
      pose_type = PoseType.HalfLie 
    elif left_knee.visibility > 0.5 and (left_knee.y - left_hip.y) > (left_shoulder.y - left_eye.y):
      pose_type = PoseType.Stand
    # 这里简单假设侧躺的情况（可根据实际情况精确调整）
    # elif abs(left_hip.y - left_knee.y) > 0.2 or abs(right_hip.y - right_knee.y) > 0.2:
    #     pose_type = PoseType.LieSide
    return pose_type,pose_prob
         
  def Detect(self, image) -> PoseResult:
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pose_results = self.pose.process(image_rgb)

    if pose_results is None or pose_results.pose_landmarks is None:
      print("mediapipe detect none body")
      return PoseResult(PoseType.HalfLie, 0.1, True, True, 0.1)
    landmarks = pose_results.pose_landmarks
    
    body_angle = self.CalcBodyAngle(image, landmarks.landmark)
    # detect eye closed
    face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    results = face_mesh.process(image)
    leftEyeClosed, rightEyeClosed = True,True
    eyeCloseProb = 0.5
    head_angle = self.CalcHeadAngle2(landmarks)
    print(f"body angle={body_angle}, head angle2={head_angle}")
    if results.multi_face_landmarks:
      face_landmarks = results.multi_face_landmarks[0]
      leftEyeClosed, rightEyeClosed, eyeCloseProb = self.DetectEyeOpen(image, face_landmarks)
      
      head_angle = min(head_angle, self.CalcHeadAngle(image, face_landmarks))
      print(f"head angle={head_angle}")

    pose_type,pose_prob = self.DetectPoseByRule(landmarks, head_angle, body_angle)
    return PoseResult(pose_type, pose_prob, leftEyeClosed, rightEyeClosed, eyeCloseProb)