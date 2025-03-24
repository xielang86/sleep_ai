import logging
import math
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

  def DetectByOldRule(self, landmarks, face_results, ih, iw) -> PoseResult:
    pose_result = PoseResult()
    head_angle = self.body_detector.CalcHeadAngle(landmarks, ih, iw)
    if face_results.multi_face_landmarks and len(face_results.multi_face_landmarks) > 0:
      face_landmarks = face_results.multi_face_landmarks[0]
      # eye
      pose_result.left_eye, pose_result.left_eye_prob, pose_result.right_eye, pose_result.right_eye_prob = self.face_detector.DetectEyePose(face_landmarks, ih, iw)
      # TODO(xl): would norm to 0-180 in future, only use to judge body pose
      head_angle = min(head_angle, self.face_detector.CalcHeadAngle(face_landmarks, ih, iw))
      PoseDetector.logger.info(f"detect face, then head angle={head_angle}")
       
      # head and face
      face_angle = self.face_detector.CalFaceAngle(face_landmarks, ih, iw)
      PoseDetector.logger.info(f"face angle={face_angle}")
      head_angle = min(head_angle, -face_angle[0])
      pose_result.head = HeadPose.Bow
      if (face_angle[1] < 30):
        pose_result.face_direction = FaceDirection.TowardToCamera
      
      pose_result.head_prob = 0.5
      if (head_angle < -90):
        pose_result.head = HeadPose.Up 
        pose_result.head_prob = max(0 - head_angle, 150) / 150.0

      #mouth 
      pose_result.mouth,pose_result.mouth_prob = self.face_detector.DetectMouthCloseOpen(face_landmarks, ih, iw)
      PoseDetector.logger.info(f"mouth={pose_result.mouth}")

    body_angle = self.body_detector.CalcBodyAngle(landmarks.landmark, ih, iw)

    body_angle2 = self.body_detector.CalFaceKneeAngle(landmarks.landmark, ih, iw)

    PoseDetector.logger.info(f"body angle={body_angle}, body angle2 = {body_angle2}, head angle2={head_angle}")
    # body
    pose_result.body, pose_result.body_prob = self.body_detector.DetectPoseByRule(landmarks, head_angle, body_angle, body_angle2)

    # hand
    pose_result.left_hand,pose_result.left_hand_prob,pose_result.right_hand,pose_result.right_hand_prob = \
      self.hand_detector.DetectHandPose(self.message_id, landmarks, head_angle, body_angle, ih, iw)
    PoseDetector.logger.info(f"hand={pose_result.left_hand},{pose_result.right_hand}")

    # foot
    pose_result.foot = FootPose.OnLoad
    pose_result.foot_prob = 0.5

    sys.stdout.flush()
    return pose_result

  def CalcHandSim(ground_fea, target_fea):
    sim = 0
    return sim

  def CalcHandSim(ground_fea, target_fea):
    sim = 0
    return sim

  def ModifyByInitPose(self, sleep_fea, fea, pose_result):
    sit_sim = self.CalcBodySim(sleep_fea, fea)
    left_hand_sim, right_hand_sim = self.CalcHandSim(sleep_fea, fea)
    if sit_sim > 0.9 and pose_result.body == BodyPose.SitDown:
      pose_result.body = BodyPose.HalfLie
    if left_hand_sim > 0.9 and pose_result.left_hand == HandPose.LiftOn:
      pose_result.left_hand = HandPose.BodySide
    if right_hand_sim > 0.9 and pose_result.right_hand == HandPose.LiftOn:
      pose_result.right_hand = HandPose.BodySide

  def Detect(self, message_id, image, sleep_fea=None) -> PoseResult:
    self.message_id = message_id
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ih, iw, _ = image.shape
    mp_result = self.pose.process(image)
    pose_result = PoseResult()

    PoseDetector.logger.info(f"message_id={self.message_id}")
    if mp_result is None or mp_result.pose_landmarks is None:
      print("mediapipe detect none body")
      return pose_result

    landmarks = mp_result.pose_landmarks
    # detect eye closed
    face_results = self.face_mesh.process(image)

    pose_result = self.DetectByOldRule(landmarks, face_results, ih, iw)
    if (sleep_fea):
      fea = self.fea_extractor.Extract(landmarks, face_results, ih, iw) 
      self.ModifyByInitPose(sleep_fea,  fea, pose_result)

    return pose_result

  def DetectBodyByFeaRule(self, fea):
    pose_result = PoseResult()
    return pose_result
  