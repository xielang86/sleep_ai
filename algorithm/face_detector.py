import math,logging

import numpy as np
from .pose import EyePose,MouthPose
from common.logger import CreateCustomLogger
from common.util import *

class FaceDetector:
  left_eye_landmarks = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
  right_eye_landmarks = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
  logger = CreateCustomLogger("face.log", __name__, logging.DEBUG)

  def __init__(self):
    return  

  def DetectEyePose(self, face_landmarks, ih, iw):
    def CalcEyeCloseOpen(ratio):
      eye_pose = EyePose.Closed
      prob = 0
      if ratio < 0.304:
        prob = min (1 - (ratio - 0.1), 1)
        eye_pose = EyePose.Closed
      else:
        eye_pose = EyePose.Open
        prob = 3*(ratio - 0.1)

      return eye_pose, prob

    def CalEyeOpenRatio(ih, iw, eye_top, eye_bottom, eye_inner, eye_outer)->float:
      eye_top_point = (int(eye_top.x * iw), int(eye_top.y * ih))
      eye_bottom_point = (int(eye_bottom.x * iw), int(eye_bottom.y * ih))
      inner_point = (int(eye_inner.x * iw), int(eye_inner.y * ih))
      outer_point = (int(eye_outer.x * iw), int(eye_outer.y * ih))
      major_dist = np.linalg.norm(np.array(inner_point) - np.array(outer_point))
      minor_dist = np.linalg.norm(np.array(eye_top_point) - np.array(eye_bottom_point))

      return minor_dist / major_dist

    left_eye_top = face_landmarks.landmark[159]
    left_eye_bottom = face_landmarks.landmark[145]
    left_inner_corner = face_landmarks.landmark[133]
    left_outer_corner = face_landmarks.landmark[33]
    left_ratio = CalEyeOpenRatio(ih, iw, left_eye_top, left_eye_bottom, left_inner_corner, left_outer_corner)
    left_eye_pose, left_prob = CalcEyeCloseOpen(left_ratio)

    # 获取右眼上下眼睑中间点位
    right_eye_top = face_landmarks.landmark[386]
    right_eye_bottom = face_landmarks.landmark[374]
    right_inner_corner = face_landmarks.landmark[362]
    right_outer_corner = face_landmarks.landmark[263]

    right_ratio = CalEyeOpenRatio(ih, iw, right_eye_top, right_eye_bottom, right_inner_corner, right_outer_corner)
    right_eye_pose, right_prob = CalcEyeCloseOpen(right_ratio)

    FaceDetector.logger.info(f"left_eye_ratio={left_ratio} right_eye_ratio={right_ratio}")

    return left_eye_pose,left_prob,right_eye_pose,right_prob

  def CalFaceAngle(self, face_landmarks, ih, iw):
    mid_brow = face_landmarks.landmark[168]
    mid_brow_point = (int(mid_brow.x * iw), int(mid_brow.y * ih))
    # 下巴
    chin = face_landmarks.landmark[152]
    chin_point = (int(chin.x * iw), int(chin.y * ih))
    # 左耳
    left_ear = face_landmarks.landmark[234]
    left_ear_point = (int(left_ear.x * iw), int(left_ear.y * ih))
    # 右耳
    right_ear = face_landmarks.landmark[454]
    right_ear_point = (int(right_ear.x * iw), int(right_ear.y * ih))
    # 左眼眼角
    left_eye_corner = face_landmarks.landmark[33]
    left_eye_corner_point = (int(left_eye_corner.x * iw), int(left_eye_corner.y * ih))
    # 右眼眼角
    right_eye_corner = face_landmarks.landmark[263]
    right_eye_corner_point = (int(right_eye_corner.x * iw), int(right_eye_corner.y * ih))

    # 计算俯仰角, 90 是正中间
    pitch_angle = np.arctan2(chin_point[1] - mid_brow_point[1], chin_point[0] - mid_brow_point[0]) * 180 / np.pi

    # 计算偏航角, 90 脸平行于摄像头
    yaw_angle = np.arctan2(right_ear_point[0] - left_ear_point[0], right_ear_point[1] - left_ear_point[1]) * 180 / np.pi

    # 计算翻滚角 , 0是头不歪
    roll_angle = np.arctan2(right_eye_corner_point[1] - left_eye_corner_point[1], right_eye_corner_point[0] - left_eye_corner_point[0]) * 180 / np.pi
    return (pitch_angle, yaw_angle, roll_angle)

  def CalcHeadAngle(self, face_landmarks, ih, iw)->float:
    def CalcByEyeMouth(ih, iw, eye_outer_corner, mouth_corner):
      # 获取关键点的像素坐标
      eye_corner_x, eye_corner_y = int(eye_outer_corner.x * iw), int(eye_outer_corner.y * ih)
      mouth_corner_x, mouth_corner_y = int(mouth_corner.x * iw), int(mouth_corner.y * ih)
      # 计算向量
      dx = eye_corner_x - mouth_corner_x
      dy = eye_corner_y - mouth_corner_y
      # 计算角度（弧度）
      angle_rad = math.atan2(dy, dx)
      # 转换为角度
      angle_deg = math.degrees(angle_rad)
      return angle_deg
      
    left_eye_corner = face_landmarks.landmark[133]
    left_mouth_corner = face_landmarks.landmark[61]
    right_eye_corner = face_landmarks.landmark[263]
    right_mouth_corner = face_landmarks.landmark[308]

    left_angle = CalcByEyeMouth(ih, iw, left_eye_corner, left_mouth_corner)
    right_angle = CalcByEyeMouth(ih, iw, right_eye_corner, right_mouth_corner)

    if left_eye_corner.visibility > 0.5:
      return left_angle
    else:
      return right_angle

  def DetectMouthCloseOpen(self, face_landmarks, ih, iw):
    # 获取上下嘴唇中心点的坐标
    upper_lip_center = face_landmarks.landmark[0]
    lower_lip_center = face_landmarks.landmark[17]
    left_mouth_corner = face_landmarks.landmark[61]
    right_mouth_corner = face_landmarks.landmark[308]

    # 将归一化的坐标转换为实际像素坐标
    upper_lip_center_point = (int(upper_lip_center.x * iw), int(upper_lip_center.y * ih))
    lower_lip_center_point = (int(lower_lip_center.x * iw), int(lower_lip_center.y * ih))
    left_mouth_corner_point = (int(left_mouth_corner.x * iw), int(left_mouth_corner.y * ih))
    right_mouth_corner_point = (int(right_mouth_corner.x * iw), int(right_mouth_corner.y * ih))

    major_dist = np.linalg.norm(np.array(left_mouth_corner_point) - np.array(right_mouth_corner_point))
    minor_dist = np.linalg.norm(np.array(upper_lip_center_point) - np.array(lower_lip_center_point))
    ratio = minor_dist / major_dist
    FaceDetector.logger.info(f"mouth ratio={ratio}")

    mouth_pose = MouthPose.Closed
    prob = 0
    if ratio < 0.66:
      mouth_pose = MouthPose.Closed
      prob = min(1.0, 1 - (ratio - 0.2))
    else:
      mouth_pose = MouthPose.Open
      prob = ratio

    return mouth_pose, prob