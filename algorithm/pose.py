import json,logging
import math
import mediapipe as mp
import cv2
from enum import Enum
from dataclasses import asdict, dataclass
from common.util import *
import numpy as np
import logging

# 创建自定义日志记录器
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# 创建控制台处理器
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# 创建文件处理器
file_handler = logging.FileHandler('pose.log')
file_handler.setLevel(logging.DEBUG)

# 创建日志格式
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# 将处理器添加到日志记录器
logger.addHandler(console_handler)
logger.addHandler(file_handler)

class Pose(Enum):
  def __str__(self):
    return self.name

class BodyPose(Pose):
  LieFlat = 1
  HalfLie = 2
  LieSide = 3
  LieFaceDown = 4
  SitDown = 5
  Stand = 6
  Other = 16
  
class HeadPose(Pose):
  Up = 1
  Bow = 2

class HandPose(Pose):
  OnThigh = 1 
  OnAbdomen = 2
  BodySide = 3
  UpwardToSky = 4
  PalmDown = 5
  OnChest = 6
  LiftOn = 7

class EyePose(Pose):
  Open = 1
  Closed = 2

class MouthPose(Pose):
  Open = 1
  Closed = 2

class FootPose(Pose):
  UpLift = 1
  OnLoad = 2
@dataclass
class PoseResult:
  body: BodyPose = BodyPose.SitDown
  body_prob: float = 0

  head: HeadPose = HeadPose.Up
  head_prob:  float = 0

  left_hand: HandPose = HandPose.BodySide
  left_hand_prob = 0
  right_hand: HandPose = HandPose.BodySide
  right_hand_prob = 0
  
  left_eye: EyePose = EyePose.Closed
  left_eye_prob : float = 0
  right_eye: EyePose = EyePose.Closed
  right_eye_prob: float = 0

  mouth : MouthPose = MouthPose.Closed
  mouth_prob : float = 0

  foot: FootPose = FootPose.OnLoad
  foot_prob : float = 0

  def __str__(self):
    return json.dumps(asdict(self), indent=4, default=str)

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
# thread unsafe
class PoseDetector:
  def __init__(self):
    self.mp_pose = mp.solutions.pose
    self.pose = self.mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.2,
                                 min_tracking_confidence=0.2)
    self.face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

  def DetectEyePose(self, image, face_landmarks):
    # 眼睛关键点的索引（根据 MediaPipe 的标准）
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
      eyeball_norm = 100  # 可以根据实际情况调整
      logger.info(f'eyeball gray={average_gray} threshold={eyeball_threshold}')

      if average_gray < eyeball_threshold:
        return True,(eyeball_norm - average_gray) / eyeball_norm
      else:
        return False,average_gray / eyeball_norm

    # left_ear = eye_aspect_ratio(left_eye_points)
    # right_ear = eye_aspect_ratio(right_eye_points)
    left_eyeball_exists,left_eye_prob = CheckEyeBall(image, left_eye_points)
    right_eyeball_exists,right_eye_prob = CheckEyeBall(image, right_eye_points)
    # ear_threshold = 0.2  # 可以根据实际情况调整这个阈值
    
    left_eye_pose = EyePose.Closed
    right_eye_pose = EyePose.Closed
    if (left_eyeball_exists):
      left_eye_pose = EyePose.Open
    if (right_eyeball_exists):
      right_eye_pose = EyePose.Open

    return left_eye_pose, left_eye_prob, right_eye_pose,right_eye_prob

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

    # print(f"calc head angle2, dx={dx}, dv={dy}, rad={angle_rad} angle_deg={angle_deg}")
    return angle_deg
    
  def CalcBodyAngle(self, image, landmark)->float:
    left_shoulder_landmark = landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    left_hip_landmark = landmark[self.mp_pose.PoseLandmark.LEFT_HIP.value]
    left_shoulder_x, left_shoulder_y = left_shoulder_landmark.x * image.shape[1], left_shoulder_landmark.y * image.shape[0]
    left_hip_x, left_hip_y = left_hip_landmark.x * image.shape[1], left_hip_landmark.y * image.shape[0]
    # 计算上半身向后仰的角度（弧度制）
    logger.debug(f"left shoulder = {left_shoulder_x} left_hip_x={left_hip_x},leftshoudy={left_shoulder_y},lefthipy={left_hip_y}")
    dx_body = left_shoulder_x - left_hip_x
    dy_body = left_shoulder_y - left_hip_y
    body_angle_rad = math.atan2(dy_body, dx_body)
    body_angle_deg = math.degrees(body_angle_rad)
    return body_angle_deg

  def DetectXDirection(self, landmarks):
    # means camera on leftside
    return 1
     
  def DetectPoseByRule(self, landmarks, head_angle, body_angle):
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
    logger.debug(f"mouth_prob={mouth_vis}, knee_vis={knee_vis}")
    logger.debug(f"eyey={left_eye.y}, nosey={nose.y}, eary={left_ear.y}, mouthy={left_mouth.y}, hip={left_hip.y}, knee={left_knee.y}, ankle={left_ankle.y}")
    logger.debug(f"mout-nose={left_mouth.y-nose.y}, ear-nose={left_ear.y - nose.y}, mouth-ear={left_mouth.y - left_ear.y}")

    body_prob = 0.5
    body_pose = BodyPose.SitDown

    if (body_angle > -15 ) or \
    (left_ear.visibility and left_knee.visibility and left_ear.y > left_knee.y) or \
    (right_ear.visibility and right_knee.visibility and right_ear.y > right_knee.y):
      body_pose = BodyPose.LieFlat
    elif (head_angle > -80 and body_angle > -75) or (body_angle > -90 and head_angle > -70):
      body_pose = BodyPose.HalfLie 
    elif left_knee.visibility > 0.5 and (left_knee.y - left_hip.y) > (left_shoulder.y - left_eye.y):
      body_pose = BodyPose.Stand
    # 这里简单假设侧躺的情况（可根据实际情况精确调整）
    # elif abs(left_hip.y - left_knee.y) > 0.2 or abs(right_hip.y - right_knee.y) > 0.2:
    #     pose_type = PoseType.LieSide
    return body_pose,body_prob

  def DetectHandPose(self, image, landmarks):
    image_height, image_width, _ = image.shape
    left_hand_pose = HandPose.BodySide
    left_hand_prob = 0.5
    right_hand_pose = HandPose.BodySide
    right_hand_prob = 0.5
    landmark = landmarks.landmark
    # 获取左手和右手关键点坐标
    left_wrist = (int(landmark[self.mp_pose.PoseLandmark.LEFT_WRIST].x * image_width),
                  int(landmark[self.mp_pose.PoseLandmark.LEFT_WRIST].y * image_height))
    right_wrist = (int(landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST].x * image_width),
                  int(landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST].y * image_height))

    # print(show_file_and_line(sys._getframe()))
    # 获取肩部、腹部、胸口关键点坐标
    left_shoulder = (int(landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER].x * image_width),
                    int(landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y * image_height))
    right_shoulder = (int(landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].x * image_width),
                      int(landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y * image_height))
    mid_shoulder_x = (left_shoulder[0] + right_shoulder[0]) // 2
    mid_shoulder_y = (left_shoulder[1] + right_shoulder[1]) // 2

    hip_x = (int(landmark[self.mp_pose.PoseLandmark.LEFT_HIP].x * image_width) +
             int(landmark[self.mp_pose.PoseLandmark.RIGHT_HIP].x * image_width)) // 2
    hip_y = (int(landmark[self.mp_pose.PoseLandmark.LEFT_HIP].y * image_height) +
             int(landmark[self.mp_pose.PoseLandmark.RIGHT_HIP].y * image_height)) // 2

    mid_abdomen_x = (mid_shoulder_x + hip_x) // 2
    mid_abdomen_y = (mid_shoulder_y + hip_y) // 2

    # 定义判断范围的阈值
    threshold_x = 30
    threshold_y = 8
    # 判断左手位置
    try:
      if abs(left_wrist[0] - left_shoulder[0]) < threshold_x and left_wrist[1] > left_shoulder[1]:
        left_hand_pose = HandPose.BodySide
      elif abs(left_wrist[0] - mid_abdomen_x) < threshold_x and abs(left_wrist[1] - mid_abdomen_y) < threshold_y:
        left_hand_pose = HandPose.OnAbdomen
      elif abs(left_wrist[0] - mid_shoulder_x) < threshold_x and abs(left_wrist[1] - mid_shoulder_y) < threshold_y:
        left_hand_pose = HandPose.OnChest
      elif abs(left_wrist[0] - left_shoulder[0]) > threshold_x and abs(left_wrist[1] - left_shoulder[1]) < threshold_y:
        print(f"left_wristx={left_wrist}, left_shoulderx={left_shoulder}")
        left_hand_pose = HandPose.BodySide
    except Exception as e:
      print(e)
      raise(e)

        # 判断右手位置
    if abs(right_wrist[0] - right_shoulder[0]) < threshold_x and right_wrist[1] > right_shoulder[1]:
        right_hand_pose = HandPose.BodySide
    elif abs(right_wrist[0] - mid_abdomen_x) < threshold_x and abs(right_wrist[1] - mid_abdomen_y) < threshold_y:
        right_hand_pose = HandPose.OnAbdomen
    elif abs(right_wrist[0] - mid_shoulder_x) < threshold_x and abs(right_wrist[1] - mid_shoulder_y) < threshold_y:
        right_hand_pose = HandPose.OnChest
    elif abs(right_wrist[0] - right_shoulder[0]) > threshold_x and abs(right_wrist[1] - right_shoulder[1]) < threshold_y:
        print(f"right_wristx={right_wrist}, right_shoulderx={right_shoulder}")
        right_hand_pose = HandPose.LiftOn

    return left_hand_pose,left_hand_prob,right_hand_pose, right_hand_prob

  def DetectMouthPose(self, image, face_landmarks):
    # 获取上下嘴唇中心点的索引
    upper_lip_center_index = 0
    lower_lip_center_index = 17
    # 获取上下嘴唇中心点的坐标
    upper_lip_center = face_landmarks.landmark[upper_lip_center_index]
    lower_lip_center = face_landmarks.landmark[lower_lip_center_index]

    # 获取图像的高度和宽度
    image_height, image_width, _ = image.shape
    # 将归一化的坐标转换为实际像素坐标
    upper_lip_center_x = int(upper_lip_center.x * image_width)
    upper_lip_center_y = int(upper_lip_center.y * image_height)
    lower_lip_center_x = int(lower_lip_center.x * image_width)
    lower_lip_center_y = int(lower_lip_center.y * image_height)
    # 计算上下嘴唇中心点之间的垂直距离
    lip_distance = abs(upper_lip_center_y - lower_lip_center_y)
    # 设置张嘴的阈值
    threshold = 10
    if lip_distance > threshold:
      return MouthPose.Open,0.5 + min(0.5, lip_distance / threshold - 1)
    return MouthPose.Closed, 1 - 1.0 * lip_distance / threshold

  def Detect(self, image) -> PoseResult:
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_result = self.pose.process(image_rgb)
    pose_result = PoseResult()

    if mp_result is None or mp_result.pose_landmarks is None:
      print("mediapipe detect none body")
      return pose_result

    landmarks = mp_result.pose_landmarks
    
    body_angle = self.CalcBodyAngle(image, landmarks.landmark)
    # detect eye closed
    face_results = self.face_mesh.process(image)

    head_angle = self.CalcHeadAngle2(landmarks)
    logger.info(f"body angle={body_angle}, head angle2={head_angle}")
    if face_results.multi_face_landmarks and len(face_results.multi_face_landmarks) > 0:
      logger.info("detect face")
      face_landmarks = face_results.multi_face_landmarks[0]
      # eye
      pose_result.left_eye, pose_result.left_eye_prob, pose_result.right_eye, pose_result.right_eye_prob = self.DetectEyePose(image, face_landmarks)
      
      head_angle = min(head_angle, self.CalcHeadAngle(image, face_landmarks))
      logger.info(f"head angle={head_angle}")

      #mouth 
      pose_result.mouth,pose_result.mouth_prob = self.DetectMouthPose(image, face_landmarks)
      logger.info(f"mouth={pose_result.mouth}")
    # body

    pose_result.body, pose_result.body_prob = self.DetectPoseByRule(landmarks, head_angle, body_angle)
    # head
    pose_result.head = HeadPose.Bow
    pose_result.head_prob = 0.5
    if (head_angle < -90):
      pose_result.head = HeadPose.Up 
      pose_result.head_prob = max(0 - head_angle, 150) / 150.0

    # hand
    pose_result.left_hand,pose_result.left_hand_prob,pose_result.right_hand,pose_result.right_hand_prob = self.DetectHandPose(image, landmarks)
    logger.info(f"hand={pose_result.left_hand}")

    # foot
    pose_result.foot = FootPose.OnLoad
    pose_result.foot_prob = 0.5

    sys.stdout.flush()
    return pose_result