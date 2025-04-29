import logging
import numpy as np
import mediapipe as mp
import cv2
from common.util import *
from common.logger import CreateCustomLogger
from .hand_detector import HandDetector
from .face_detector import FaceDetector
from .body_detector import BodyDetector
from .feature_extractor import FeatureExtractor
from .pose import BodyPose,HeadPose,PoseResult,FootPose,FaceDirection,HandPose

class PoseDetector:
  logger = CreateCustomLogger("pose.log", __name__, logging.DEBUG)
  def __init__(self, pose, face_mesh):
    self.mp_pose = mp.solutions.pose
    self.pose = pose
    self.face_mesh = face_mesh

    self.face_detector = FaceDetector()
    self.hand_detector = HandDetector()
    self.body_detector = BodyDetector()
    self.fea_extractor = FeatureExtractor()

    self.left_hand_fea = ["left_dist_wrist_hip", "left_dist_wrist_elbow ", "left_dist_thumb_elbow",
      "left_dist_wrist_shoulder", "left_dist_wrist_nose", "left_wrist_dist_right_elbow", "left_dist_wrist_knee",
      "left_index_angle", "left_hand_angle", "left_arm_angle", 
      "left_thumb_body_dist", "left_hand_above","left_hand_above_dist"]

    self.right_hand_fea = ["right_dist_wrist_hip", "right_dist_wrist_elbow ", "right_dist_thumb_elbow",
      "right_dist_wrist_shoulder", "right_dist_wrist_nose", "right_wrist_dist_right_elbow", "right_dist_wrist_knee",
      "right_index_angle", "right_hand_angle", "right_arm_angle", 
      "right_thumb_body_dist", "right_hand_above","right_hand_above_dist"]

  def DetectByOldRule(self, landmarks, face_results, fea, ih, iw) -> PoseResult:
    pose_result = PoseResult()
    # head_angle = self.body_detector.CalcHeadAngle(landmarks, ih, iw)
    # head_angle = fea.head_angle
    if face_results.multi_face_landmarks and len(face_results.multi_face_landmarks) > 0:
      face_landmarks = face_results.multi_face_landmarks[0]
      # eye
      pose_result.left_eye, pose_result.left_eye_prob, pose_result.right_eye, pose_result.right_eye_prob = self.face_detector.DetectEyePose(face_landmarks, ih, iw)
      # TODO(xl): would norm to 0-180 in future, only use to judge body pose
      # head_angle = min(head_angle, self.face_detector.CalcHeadAngle(face_landmarks, ih, iw))
      # head_angle = min(head_angle, -fea.pitch_angle)
      # head_angle = fea.head_angle
      PoseDetector.logger.info(f"detect face, then head angle={fea.head_angle}")
       
      # head and face
      # face_angle = self.face_detector.CalFaceAngle(face_landmarks, ih, iw)

      PoseDetector.logger.info(f"face angle={fea.pitch_angle},{fea.yaw_angle},{fea.roll_angle}")
      pose_result.head = HeadPose.Bow
      if (fea.yaw_angle < 30):
        pose_result.face_direction = FaceDirection.TowardToCamera
      
      pose_result.head_prob = 0.5
      if (fea.head_angle < -90):
        pose_result.head = HeadPose.Up 
        pose_result.head_prob = max(0 - fea.head_angle, 150) / 150.0

      #mouth 
      pose_result.mouth,pose_result.mouth_prob = self.face_detector.DetectMouthCloseOpen(face_landmarks, ih, iw)
      PoseDetector.logger.info(f"mouth={pose_result.mouth}")

    # body_angle = self.body_detector.CalcBodyAngle(landmarks.landmark, ih, iw)
    # body_angle2 = self.body_detector.CalFaceKneeAngle(landmarks.landmark, ih, iw)

    PoseDetector.logger.info(f"body angle={fea.shoulder_hip_angle}, body angle2 = {fea.face_knee_angle}, head angle={fea.head_angle}")
    # body
    # pose_result.body, pose_result.body_prob = self.body_detector.DetectPoseByRule(landmarks, head_angle, body_angle, body_angle2)
    pose_result.body, pose_result.body_prob = self.body_detector.DetectPoseByRule(landmarks, fea)
    # hand
    pose_result.left_hand,pose_result.left_hand_prob,pose_result.right_hand,pose_result.right_hand_prob = \
      self.hand_detector.DetectHandPose(self.message_id, landmarks, fea, pose_result.body, ih, iw)
    PoseDetector.logger.info(f"hand={pose_result.left_hand},{pose_result.right_hand}")

    # foot
    pose_result.foot = FootPose.OnLoad
    pose_result.foot_prob = 0.5

    sys.stdout.flush()
    return pose_result

  def CalcHandDiff(self, ground_fea, target_fea):
    # calc each relative change, acc all the change
    left_ground_vec = ChangeToVector(ground_fea, self.left_hand_fea)
    left_target_vec = ChangeToVector(target_fea, self.left_hand_fea)
    left_diff_vec = CalculateRelativeDiff(left_ground_vec, left_target_vec)

    right_ground_vec = ChangeToVector(ground_fea, self.right_hand_fea)
    right_target_vec = ChangeToVector(target_fea, self.right_hand_fea)
    right_diff_vec = CalculateRelativeDiff(right_ground_vec, right_target_vec)
    
    return np.linalg.norm(left_diff_vec),np.linalg.norm(right_diff_vec)

  def ModifyByInitPose(self, sleep_fea, fea, pose_result):
    left_hand_diff, right_hand_diff = self.CalcHandDiff(sleep_fea, fea)
    if left_hand_diff < 0.1 and pose_result.left_hand == HandPose.LiftOn:
      pose_result.left_hand = HandPose.BodySide
    if right_hand_diff < 0.1 and pose_result.right_hand == HandPose.LiftOn:
      pose_result.right_hand = HandPose.BodySide

  def get_mp_result(self):
    return self.mp_result

  def Detect(self, message_id, image, sleep_fea=None) -> PoseResult:
    self.message_id = message_id
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ih, iw, _ = image.shape
    self.mp_result = self.pose.process(image)
    pose_result = PoseResult()

    PoseDetector.logger.info(f"message_id={self.message_id}")
    if self.mp_result is None or self.mp_result.pose_landmarks is None:
      print("mediapipe detect none body")
      return pose_result

    landmarks = self.mp_result.pose_landmarks
    # detect eye closed
    face_results = self.face_mesh.process(image)
    fea = self.fea_extractor.Extract(landmarks, face_results, ih, iw) 

    pose_result = self.DetectByOldRule(landmarks, face_results, fea, ih, iw)
    
    if (sleep_fea):
      self.ModifyByInitPose(sleep_fea,  fea, pose_result)

    return pose_result

  def DetectBodyByFeaRule(self, fea):
    pose_result = PoseResult()
    return pose_result
  