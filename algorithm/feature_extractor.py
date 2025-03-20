from dataclasses import dataclass
from .pose_detector import PoseDetector
from .hand_detector import HandDetector
import logging
import mediapipe as mp
import cv2
from common.util import *
from common.logger import CreateCustomLogger
from .hand_detector import HandDetector
from .face_detector import FaceDetector

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

class FeatureExtractor:
  logger = CreateCustomLogger("feature.log", __name__, logging.DEBUG)
  def __init__(self):
    self.mp_pose = mp.solutions.pose
    self.pose = self.mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.03,
                                 min_tracking_confidence=0.01)

    self.pose = self.mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.03, min_tracking_confidence=0.01)
    self.face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.05, min_tracking_confidence=0.1)

    self.face_detector = FaceDetector()
    self.hand_detector = HandDetector()
    self.pose_detector = PoseDetector(self.pose, self.face_mesh)

  def GetAllPart(self, landmark, iw, ih):
    self.left_eye = (int(landmark[self.mp_pose.PoseLandmark.LEFT_EYE].x * iw), int(landmark[self.mp_pose.PoseLandmark.LEFT_EYE].y))
    self.right_eye = (int(landmark[self.mp_pose.PoseLandmark.RIGHT_EYE].x * iw), int(landmark[self.mp_pose.PoseLandmark.RIGHT_EYE].y * ih))
    self.nose = (int(landmark[self.mp_pose.PoseLandmark.NOSE].x * iw), int(landmark[self.mp_pose.PoseLandmark.NOSE].y * ih))
    self.left_mouth = (int(landmark[self.mp_pose.PoseLandmark.MOUTH_LEFT].x * iw), int(landmark[self.mp_pose.PoseLandmark.MOUTH_LEFT].y * ih))
    self.right_mouth = (int(landmark[self.mp_pose.PoseLandmark.MOUTH_RIGHT].x * iw), int(landmark[self.mp_pose.PoseLandmark.MOUTH_RIGHT].y * ih))
    self.left_ear = (int(landmark[self.mp_pose.PoseLandmark.LEFT_EAR].x * iw), int(landmark[self.mp_pose.PoseLandmark.LEFT_EAR].y * ih))
    self.right_ear = (int(landmark[self.mp_pose.PoseLandmark.RIGHT_EAR].x * iw), int(landmark[self.mp_pose.PoseLandmark.RIGHT_EAR].y * ih))

    self.left_hip = (int(landmark[self.mp_pose.PoseLandmark.LEFT_HIP].x * iw), int(landmark[self.mp_pose.PoseLandmark.LEFT_HIP].y * ih))
    self.right_hip = (int(landmark[self.mp_pose.PoseLandmark.RIGHT_HIP].x * iw), int(landmark[self.mp_pose.PoseLandmark.RIGHT_HIP].y * ih))
    self.left_knee = (int(landmark[self.mp_pose.PoseLandmark.LEFT_KNEE].x * iw), int(landmark[self.mp_pose.PoseLandmark.LEFT_KNEE].y * ih))
    self.right_knee = (int(landmark[self.mp_pose.PoseLandmark.RIGHT_KNEE].x * iw), int(landmark[self.mp_pose.PoseLandmark.RIGHT_KNEE].y * ih))
    self.left_ankle = (int(landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE].x * iw), int(landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE].y * ih))
    self.right_ankle = (int(landmark[self.mp_pose.PoseLandmark.RIGHT_ANKLE].x * iw), int(landmark[self.mp_pose.PoseLandmark.RIGHT_ANKLE].y * ih))

    self.left_shoulder = (int(landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER].x * iw), int(landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y * ih))
    self.right_shoulder = (int(landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].x * iw), int(landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y * ih))
    self.left_wrist = (int(landmark[self.mp_pose.PoseLandmark.LEFT_WRIST].x * iw),
                  int(landmark[self.mp_pose.PoseLandmark.LEFT_WRIST].y * ih))
    self.right_wrist = (int(landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST].x * iw),
                  int(landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST].y * ih))
    self.left_elbow = (int(landmark[self.mp_pose.PoseLandmark.LEFT_ELBOW].x * iw),
                  int(landmark[self.mp_pose.PoseLandmark.LEFT_ELBOW].y * ih))
    self.right_elbow = (int(landmark[self.mp_pose.PoseLandmark.RIGHT_ELBOW].x * iw),
                  int(landmark[self.mp_pose.PoseLandmark.RIGHT_ELBOW].y * ih))
    self.left_thumb = (int(landmark[self.mp_pose.PoseLandmark.LEFT_THUMB].x * iw),
                  int(landmark[self.mp_pose.PoseLandmark.LEFT_THUMB].y * ih))
    self.right_thumb = (int(landmark[self.mp_pose.PoseLandmark.RIGHT_THUMB].x * iw),
                  int(landmark[self.mp_pose.PoseLandmark.RIGHT_THUMB].y * ih))
    self.left_index = (int(landmark[self.mp_pose.PoseLandmark.LEFT_INDEX].x * iw),
                  int(landmark[self.mp_pose.PoseLandmark.LEFT_INDEX].y * ih))
    self.right_index = (int(landmark[self.mp_pose.PoseLandmark.RIGHT_INDEX].x * iw),
                  int(landmark[self.mp_pose.PoseLandmark.RIGHT_INDEX].y * ih))

    self.left_wrist_vis = landmark[self.mp_pose.PoseLandmark.LEFT_WRIST].visibility
    self.right_wrist_vis = landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST].visibility
    self.left_thumb_vis = landmark[self.mp_pose.PoseLandmark.LEFT_THUMB].visibility
    self.left_index_vis = landmark[self.mp_pose.PoseLandmark.LEFT_INDEX].visibility
    self.right_thumb_vis = landmark[self.mp_pose.PoseLandmark.RIGHT_THUMB].visibility
    self.right_index_vis = landmark[self.mp_pose.PoseLandmark.RIGHT_INDEX].visibility
    self.left_elbow_vis = landmark[self.mp_pose.PoseLandmark.LEFT_ELBOW].visibility
    self.right_elbow_vis = landmark[self.mp_pose.PoseLandmark.RIGHT_ELBOW].visibility
    self.nose_vis = landmark[self.mp_pose.PoseLandmark.NOSE].visibility

    self.left_ear_vis = landmark[self.mp_pose.PoseLandmark.LEFT_EAR].visibility
    self.right_ear_vis =landmark[self.mp_pose.PoseLandmark.RIGHT_EAR].visibility
    self.left_knee_vis = landmark[self.mp_pose.PoseLandmark.LEFT_KNEE].visibility
    self.right_knee_vis = landmark[self.mp_pose.PoseLandmark.RIGHT_KNEE].visibility
    self.left_hip_vis = landmark[self.mp_pose.PoseLandmark.LEFT_HIP].visibility
    self.right_hip_vis = landmark[self.mp_pose.PoseLandmark.RIGHT_HIP].visibility

  def CalcVis(self, fea):
    fea.left_ear_vis = self.left_ear_vis
    fea.right_ear_vis = self.right_ear_vis
    fea.left_knee_vis = self.left_knee_vis
    fea.right_knee_vis = self.right_knee_vis
    fea.left_hip_vis = self.left_hip_vis
    fea.right_hip_vis = self.right_hip_vis

    fea.left_wrist_vis = self.left_wrist_vis 
    fea.right_wrist_vis = self.right_wrist_vis
    fea.left_thumb_vis  = self.left_thumb_vis 
    fea.left_index_vis  = self.left_index_vis 
    fea.right_thumb_vis = self.right_thumb_vis
    fea.right_index_vis = self.right_index_vis
    fea.left_elbow_vis  = self.left_elbow_vis 
    fea.right_elbow_vis = self.right_elbow_vis

  def CalcDist(self, landmark, ih, iw, fea):
    fea.left_ear_knee_y_dist = ih * (self.left_knee[1] - self.left_ear[1])
    fea.right_ear_knee_y_dist = ih * (self.right_knee[1] - self.right_ear[1])

    fea.left_hip_knee_y_dist = ih * (self.left_knee[1] - self.left_hip[1])
    fea.left_eye_shoulder_y_dist = ih * (self.left_shoulder[1] - self.left_eye[1])
    fea.left_shoulder_hip_y_dist = ih * (self.left_hip[1] - self.left_shoulder[1])

    fea.right_hip_knee_y_dist = ih * (self.right_knee[1] - self.right_hip[1])
    fea.right_eye_shoulder_y_dist = ih * (self.right_shoulder[1] - self.right_eye[1])
    fea.right_shoulder_hip_y_dist = ih * (self.right_hip[1] - self.right_shoulder[1])

    shoulder = ((self.left_shoulder[0] + self.right_shoulder[0]) // 2, (self.left_shoulder[1] + self.right_shoulder[1]) // 2)

    hip = ((self.left_hip[0] + self.right_hip[0]) // 2, (self.left_hip[1] + self.right_hip[1]) // 2)


    abdomen = ((shoulder[0] + hip[0]) // 2,  (shoulder[1] + hip[1])//2)

    fea.left_dist_wrist_hip = distance_pair(hip, self.left_wrist)
    fea.left_dist_wrist_elbow = distance_pair(self.left_wrist, self.left_elbow)
    fea.left_dist_thumb_elbow = distance_pair(self.left_thumb, self.left_elbow)
    fea.left_dist_wrist_shoulder = distance_pair(self.left_wrist, self.left_shoulder)
    fea.left_dist_wrist_nose = distance_pair(self.left_wrist, self.nose)
    
    fea.left_wrist_dist_right_elbow = distance_pair(self.left_wrist, self.right_elbow)
    fea.right_wrist_dist_left_elbow = distance_pair(self.right_wrist, self.left_elbow)

    fea.left_dist_wrist_knee = distance_pair(self.left_wrist, self.left_knee)
    fea.right_dist_wrist_knee = distance_pair(self.right_wrist, self.right_knee)

    fea.left_index_angle = HandDetector.CalcHandAngle(self.left_index, self.left_elbow) 
    fea.right_index_angle = HandDetector.CalcHandAngle(self.right_index, self.right_elbow)
    fea.left_hand_angle = HandDetector.CalcHandAngle(self.left_thumb, self.left_elbow)
    fea.right_hand_angle = HandDetector.CalcHandAngle(self.right_thumb, self.right_elbow)

    fea.left_thumb_body_dist,fea.right_thumb_body_dist = self.hand_detector.CalcThumbBodyDist(landmark,ih, iw)

    # calc the angle of arm
    fea.left_arm_angle = self.hand_detector.CalcArmAngle(self.left_wrist, self.left_elbow, self.left_shoulder)
    fea.right_arm_angle = self.hand_detector.CalcArmAngle(self.right_wrist, self.right_elbow, self.right_shoulder)

    fea.left_hand_above, fea.left_hand_above_dist = self.hand_detector.CalcHandFaceKneeLineDistance(self.left_thumb, self.nose, self.left_knee)

    fea.right_hand_above, fea.right_hand_above_dist = self.hand_detector.CalcHandFaceKneeLineDistance(self.right_thumb, self.nose, self.right_knee)

  # image: byte seq, return feature
  def Extract(self, image):
    # extract body pose fea
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_result = self.pose.process(image)

    # FeatureExtractor.logger.info(f"message_id={message_id}")
    if mp_result is None or mp_result.pose_landmarks is None:
      print("mediapipe detect none body")
      return None

    landmarks = mp_result.pose_landmarks

    ih, iw, _ = image.shape
    self.GetAllPart(landmarks.landmark, iw, ih)
    fea = PoseFeature()

    self.CalcVis(fea)

    fea.shoulder_hip_angle = self.pose_detector.CalcBodyAngle(landmarks.landmark, ih, iw)
    fea.face_knee_angle = self.pose_detector.CalFaceKneeAngle(landmarks.landmark, ih, iw)
    fea.head_angle = self.pose_detector.CalcHeadAngle(landmarks, ih, iw)

    # detect face related fea
    face_results = self.face_mesh.process(image)
    if face_results.multi_face_landmarks and len(face_results.multi_face_landmarks) > 0:
      face_landmarks = face_results.multi_face_landmarks[0]

      fea.head_angle_from_face = self.face_detector.CalcHeadAngle(face_landmarks, ih, iw)

      fea.pitch_angle,fea.yaw_angle,fea.roll_angle = self.face_detector.CalFaceAngle(face_landmarks, ih, iw)

  # calc distance
    self.CalcDist(landmarks.landmark, ih, iw, fea)

    return fea