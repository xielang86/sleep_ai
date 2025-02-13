import logging

import numpy as np
import mediapipe as mp

from common.util import distance,distance_pair
from common.logger import CreateCustomLogger
from .pose import HandPose

class HandDetector:
  logger = CreateCustomLogger("hand.log", __name__, logging.DEBUG)

  def __init__(self):
    self.mp_pose = mp.solutions.pose
    return  
  
  def DetectHandPose(self, image, landmarks):
    image_height, image_width, _ = image.shape
    left_hand_pose = HandPose.BodySide
    left_hand_prob = 0.5
    right_hand_pose = HandPose.BodySide
    right_hand_prob = 0.5
    landmark = landmarks.landmark
    left_wrist_vis = landmark[self.mp_pose.PoseLandmark.LEFT_WRIST].visibility
    right_wrist_vis = landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST].visibility
    left_thumb_vis = landmark[self.mp_pose.PoseLandmark.LEFT_THUMB].visibility
    right_thumb_vis = landmark[self.mp_pose.PoseLandmark.RIGHT_THUMB].visibility
    HandDetector.logger.debug(f"left_wrist_vis = {left_wrist_vis}, right_wrist_vis={right_wrist_vis}, left_thumb_vs={left_thumb_vis}, right_thumb_vs={right_thumb_vis}")
    # 获取左手和右手关键点坐标
    left_wrist = (int(landmark[self.mp_pose.PoseLandmark.LEFT_WRIST].x * image_width),
                  int(landmark[self.mp_pose.PoseLandmark.LEFT_WRIST].y * image_height))
    right_wrist = (int(landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST].x * image_width),
                  int(landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST].y * image_height))

    left_elbow = (int(landmark[self.mp_pose.PoseLandmark.LEFT_ELBOW].x * image_width),
                  int(landmark[self.mp_pose.PoseLandmark.LEFT_ELBOW].y * image_height))
    right_elbow = (int(landmark[self.mp_pose.PoseLandmark.RIGHT_ELBOW].x * image_width),
                  int(landmark[self.mp_pose.PoseLandmark.RIGHT_ELBOW].y * image_height))

    left_thumb = (int(landmark[self.mp_pose.PoseLandmark.LEFT_THUMB].x * image_width),
                  int(landmark[self.mp_pose.PoseLandmark.LEFT_THUMB].y * image_height))
    right_thumb = (int(landmark[self.mp_pose.PoseLandmark.RIGHT_THUMB].x * image_width),
                  int(landmark[self.mp_pose.PoseLandmark.RIGHT_THUMB].y * image_height))

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
    hip_pair = (hip_x, hip_y)
    dist_wrist_hip = distance_pair(hip_pair, left_wrist)
    dist_wrist_elbow = distance_pair(left_wrist, left_elbow)
    dist_thumb_elbow = distance_pair(left_thumb, left_elbow)
    dist_wrist_shoulder = distance_pair(left_wrist, left_shoulder)
    # 定义判断范围的阈值
    threshold_x = 30
    threshold_y = 8
    # 判断左手位置
    HandDetector.logger.debug(f"left_wrist={left_wrist}, left_shoulder={left_shoulder}, left_ebow={left_elbow}, wrist_elbow={dist_wrist_elbow},wrist_hip={dist_wrist_hip},wrist_shoulder={dist_wrist_shoulder}")
    HandDetector.logger.debug(f"left_thumb={left_thumb}, right_thumb={right_thumb}, thumb_elbow={dist_thumb_elbow}")
    if abs(left_wrist[0] - left_shoulder[0]) < threshold_x and left_wrist[1] > left_shoulder[1]:
      left_hand_pose = HandPose.BodySide
    elif abs(left_wrist[0] - mid_abdomen_x) < threshold_x and abs(left_wrist[1] - mid_abdomen_y) < threshold_y:
      left_hand_pose = HandPose.OnAbdomen
    elif abs(left_wrist[0] - mid_shoulder_x) < threshold_x and abs(left_wrist[1] - mid_shoulder_y) < threshold_y:
      left_hand_pose = HandPose.OnChest
    elif left_wrist_vis > 0.2 and left_wrist[1] - left_shoulder[1] < 5 and 2 * dist_wrist_elbow < dist_wrist_hip:
      left_hand_pose = HandPose.LiftOn

        # 判断右手位置
    right_dist_wrist_hip = distance_pair(hip_pair, right_wrist)
    right_dist_wrist_elbow = distance_pair(left_wrist, right_elbow)
    if abs(right_wrist[0] - right_shoulder[0]) < threshold_x and right_wrist[1] > right_shoulder[1]:
      right_hand_pose = HandPose.BodySide
    elif abs(right_wrist[0] - mid_abdomen_x) < threshold_x and abs(right_wrist[1] - mid_abdomen_y) < threshold_y:
      right_hand_pose = HandPose.OnAbdomen
    elif abs(right_wrist[0] - mid_shoulder_x) < threshold_x and abs(right_wrist[1] - mid_shoulder_y) < threshold_y:
      right_hand_pose = HandPose.OnChest
    elif right_wrist_vis > 0.2 and right_wrist[1] - right_shoulder[1] < 5 and 2 * right_dist_wrist_elbow < right_dist_wrist_hip:
      right_hand_pose = HandPose.LiftOn

    return left_hand_pose,left_hand_prob,right_hand_pose, right_hand_prob