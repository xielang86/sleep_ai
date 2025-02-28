import logging
import math
import mediapipe as mp
import cv2
from common.util import *
from common.logger import CreateCustomLogger
from .hand_detector import HandDetector
from .face_detector import FaceDetector
from .pose import BodyPose,HeadPose,PoseResult,FootPose

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
class PoseDetector:
  logger = CreateCustomLogger("pose.log", __name__, logging.DEBUG)
  def __init__(self):
    self.mp_pose = mp.solutions.pose
    self.pose = self.mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.05,
                                 min_tracking_confidence=0.1)
    self.face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.05, min_tracking_confidence=0.1)
    self.face_detector = FaceDetector()
    self.hand_detector = HandDetector()

  def CalcHeadAngle(self, pos_landmarks, image)->float:
    left_eye= pos_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_EYE.value]
    left_mouth= pos_landmarks.landmark[self.mp_pose.PoseLandmark.MOUTH_LEFT.value]
    right_eye = pos_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_EYE.value]
    right_mouth= pos_landmarks.landmark[self.mp_pose.PoseLandmark.MOUTH_RIGHT.value]

    PoseDetector.logger.debug(f"left eye= {left_eye} left_mouth={left_mouth},right_eye={right_eye},right_mouth={right_mouth}")

    # 获取关键点的像素坐标
    left_eye_x, left_eye_y= int(left_eye.x * image.shape[1]), int(left_eye.y * image.shape[0])
    left_mouth_x, left_mouth_y = int(left_mouth.x * image.shape[1]), int(left_mouth.y * image.shape[0])
    dx = left_eye_x - left_mouth_x
    dy = left_eye_y - left_mouth_y
    if(right_mouth.visibility > 0.5):
      right_eye_x, right_eye_y= int(right_eye.x * image.shape[1]), int(right_eye.y * image.shape[0])
      right_mouth_x, right_mouth_y = int(right_mouth.x * image.shape[1]), int(right_mouth.y * image.shape[0])
      dx = right_eye_x - right_mouth_x
      dy = right_eye_y - right_mouth_y

    # 计算角度
    angle_rad = math.atan2(dy, dx)
    # 转换为角度
    angle_deg = math.degrees(angle_rad)
    PoseDetector.logger.debug(f"dy= {dy} dx={dx},angle={angle_rad},angle_deg={angle_deg}")

    return angle_deg
    
  def CalcBodyAngle(self, image, landmark)->float: 
    left_shoulder_landmark = landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    left_hip_landmark = landmark[self.mp_pose.PoseLandmark.LEFT_HIP.value]
    left_shoulder_x, left_shoulder_y = left_shoulder_landmark.x * image.shape[1], left_shoulder_landmark.y * image.shape[0]
    left_hip_x, left_hip_y = left_hip_landmark.x * image.shape[1], left_hip_landmark.y * image.shape[0]
    # 计算上半身向后仰的角度（弧度制）
    PoseDetector.logger.debug(f"left shoulder = {left_shoulder_x} left_hip_x={left_hip_x},leftshoudy={left_shoulder_y},lefthipy={left_hip_y}")

    right_shoulder_landmark = landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    right_hip_landmark = landmark[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
    right_shoulder_x, right_shoulder_y = right_shoulder_landmark.x * image.shape[1], right_shoulder_landmark.y * image.shape[0]
    right_hip_x, right_hip_y = right_hip_landmark.x * image.shape[1], right_hip_landmark.y * image.shape[0]
    PoseDetector.logger.debug(f"right shoulderx = {right_shoulder_x} right_hip_x={right_hip_x},rightshoudy={right_shoulder_y},righthipy={right_hip_y}")

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
    left_hip_vis = left_hip.visibility
    right_hip_vis = right_hip.visibility
    # first judge sleep pose, xxxlie
    PoseDetector.logger.debug(f"mouth_vis={mouth_vis}, knee_vis={knee_vis}, left_hip_vis={left_hip_vis},right_hip_vis={right_hip_vis}")
    PoseDetector.logger.debug(f"eyey={left_eye.y}, nosey={nose.y}, eary={left_ear.y}, mouthy={left_mouth.y}, hip={left_hip.y}, knee={left_knee.y}, ankle={left_ankle.y}")
    PoseDetector.logger.debug(f"mout-nose={left_mouth.y-nose.y}, ear-nose={left_ear.y - nose.y}, mouth-ear={left_mouth.y - left_ear.y}")

    body_prob = 0.5
    body_pose = BodyPose.HalfLie
    if body_angle < 0 and ((body_angle > -10 or body_angle < -170) or \
    ((body_angle < -165 or body_angle > -15) and ((left_ear.visibility > 0.5 and left_knee.visibility > 0.5 and left_ear.y > left_knee.y) or \
    (right_ear.visibility > 0.5 and right_knee.visibility > 0.5 and right_ear.y > right_knee.y)))):
      body_pose = BodyPose.LieFlat
    elif abs(body_angle) > 85 and abs(body_angle) < 95 and left_knee.visibility > 0.5 and left_hip.visibility > 0.5 \
      and (left_knee.y - left_hip.y) > (left_shoulder.y - left_eye.y) and (left_hip.y - left_shoulder.y) > (left_shoulder.y - left_eye.y):
      body_pose = BodyPose.Stand
    elif abs(body_angle) > 85 and abs(body_angle) < 95 or \
      (abs(body_angle) > 41 and abs(body_angle) < 113 and abs(head_angle) > 78 and abs(head_angle) < 102) or \
      (abs(body_angle) > 75 and abs(body_angle) < 105 and abs(head_angle) > 63 and abs(head_angle) < 117):
      body_pose = BodyPose.SitDown
    elif (head_angle < 0 and head_angle > -80 and body_angle > -75) or (body_angle > -90 and head_angle > -60 and head_angle< 0) or (body_angle < -108 and head_angle < -96):
      body_pose = BodyPose.HalfLie 
    
    # 这里简单假设侧躺的情况（可根据实际情况精确调整）
    # elif abs(left_hip.y - left_knee.y) > 0.2 or abs(right_hip.y - right_knee.y) > 0.2:
    #     pose_type = PoseType.LieSide
    return body_pose,body_prob

  def Detect(self, message_id, image) -> PoseResult:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_result = self.pose.process(image)
    pose_result = PoseResult()

    PoseDetector.logger.info(f"message_id={message_id}")
    if mp_result is None or mp_result.pose_landmarks is None:
      print("mediapipe detect none body")
      return pose_result

    landmarks = mp_result.pose_landmarks
    
    body_angle = self.CalcBodyAngle(image, landmarks.landmark)
    # detect eye closed
    face_results = self.face_mesh.process(image)

    head_angle = self.CalcHeadAngle(landmarks, image)
    PoseDetector.logger.info(f"body angle={body_angle}, head angle2={head_angle}")
    if face_results.multi_face_landmarks and len(face_results.multi_face_landmarks) > 0:
      face_landmarks = face_results.multi_face_landmarks[0]
      # eye
      pose_result.left_eye, pose_result.left_eye_prob, pose_result.right_eye, pose_result.right_eye_prob = self.face_detector.DetectEyePose(image, face_landmarks)
      # TODO(xl): would norm to 0-180 in future, only use to judge body pose
      head_angle = min(head_angle, self.face_detector.CalcHeadAngle(image, face_landmarks))
      PoseDetector.logger.info(f"detect face, then head angle={head_angle}")
       
      # head and face
      face_angle = self.face_detector.CalFaceAngle(image, face_landmarks)
      PoseDetector.logger.info(f"face angle={face_angle}")
      # head_angle = min(head_angle, -face_angle[0])
      pose_result.head = HeadPose.Bow
      
      pose_result.head_prob = 0.5
      if (head_angle < -90):
        pose_result.head = HeadPose.Up 
        pose_result.head_prob = max(0 - head_angle, 150) / 150.0

      #mouth 
      pose_result.mouth,pose_result.mouth_prob = self.face_detector.DetectMouthCloseOpen(image, face_landmarks)
      PoseDetector.logger.info(f"mouth={pose_result.mouth}")

    # body
    pose_result.body, pose_result.body_prob = self.DetectPoseByRule(landmarks, head_angle, body_angle)


    # hand
    pose_result.left_hand,pose_result.left_hand_prob,pose_result.right_hand,pose_result.right_hand_prob = self.hand_detector.DetectHandPose(message_id, image, landmarks, head_angle)
    PoseDetector.logger.info(f"hand={pose_result.left_hand},{pose_result.right_hand}")

    # foot
    pose_result.foot = FootPose.OnLoad
    pose_result.foot_prob = 0.5

    sys.stdout.flush()
    return pose_result