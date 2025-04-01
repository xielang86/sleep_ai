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

  def CalcArmAngle(self, wrist, elbow, shoulder):
    vector1 = np.array(shoulder) - np.array(elbow)
    vector2 = np.array(wrist) - np.array(elbow)
    # 计算夹角
    dot_product = np.dot(vector1, vector2)
    norm_vector1 = np.linalg.norm(vector1)
    norm_vector2 = np.linalg.norm(vector2)
    cos_angle = dot_product / (norm_vector1 * norm_vector2)
    angle = np.arccos(cos_angle)
    angle_degrees = np.degrees(angle)

    return angle_degrees

  def CalcHandFaceKneeLineDistance(self, hand, nose, knee):
    if knee[0] == nose[0]:
      return False,0

    slope = (knee[1] - nose[1]) / (knee[0] - nose[0])
    intercept = nose[1] - slope * nose[0]

    # 计算大拇指到直线的垂直距离
    distance = abs(hand[1] - (slope * hand[0] + intercept)) / np.sqrt(1 + slope ** 2)

    # 判断大拇指是否在直线上方
    line_y = slope * hand[0] + intercept
    is_above = hand[1] < line_y
    return is_above, distance

  def CalcThumbBodyDist(self, landmark, ih, iw):
    left_shoulder = [landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                     landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y,
                     1]

    right_shoulder = [landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                      landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y,
                      2]
    left_hip = [landmark[self.mp_pose.PoseLandmark.LEFT_HIP].x,
                landmark[self.mp_pose.PoseLandmark.LEFT_HIP].y,
                1]
    right_hip = [landmark[self.mp_pose.PoseLandmark.RIGHT_HIP].x,
                 landmark[self.mp_pose.PoseLandmark.RIGHT_HIP].y,
                 2]
    # 计算身体平面的法向量
    v1 = np.array(right_shoulder) - np.array(left_shoulder)
    v2 = np.array(left_hip) - np.array(right_hip)
    normal = np.cross(v1, v2)
    d = -np.dot(normal, left_shoulder)

    left_thumb = [int(landmark[self.mp_pose.PoseLandmark.LEFT_THUMB].x ),
                  int(landmark[self.mp_pose.PoseLandmark.LEFT_THUMB].y),
                  1]
    right_thumb = [int(landmark[self.mp_pose.PoseLandmark.RIGHT_THUMB].x ),
                  int(landmark[self.mp_pose.PoseLandmark.RIGHT_THUMB].y),
                  2]

    left_thumb_distance = abs(np.dot(normal, left_thumb) + d) / np.linalg.norm(normal) * max(iw, ih)
    right_thumb_distance = abs(np.dot(normal, right_thumb) + d) / np.linalg.norm(normal) * max(iw, ih)

    return left_thumb_distance,right_thumb_distance

  def CalcHandAngle(thumb, wrist):
    angle = np.arctan2(-thumb[1] + wrist[1], abs(thumb[0] - wrist[0])) * 180 / np.pi
    return angle

  def DetectHandPose(self, message_id, landmarks, head_angle, body_angle, image_height, image_width):
    left_hand_pose = HandPose.BodySide
    left_hand_prob = 0.5
    right_hand_pose = HandPose.BodySide
    right_hand_prob = 0.5
    landmark = landmarks.landmark
    left_wrist_vis = landmark[self.mp_pose.PoseLandmark.LEFT_WRIST].visibility
    right_wrist_vis = landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST].visibility
    left_thumb_vis = landmark[self.mp_pose.PoseLandmark.LEFT_THUMB].visibility
    left_index_vis = landmark[self.mp_pose.PoseLandmark.LEFT_INDEX].visibility
    right_thumb_vis = landmark[self.mp_pose.PoseLandmark.RIGHT_THUMB].visibility
    right_index_vis = landmark[self.mp_pose.PoseLandmark.RIGHT_INDEX].visibility
    left_elbow_vis = landmark[self.mp_pose.PoseLandmark.LEFT_ELBOW].visibility
    right_elbow_vis = landmark[self.mp_pose.PoseLandmark.RIGHT_ELBOW].visibility
    nose_vis = landmark[self.mp_pose.PoseLandmark.NOSE].visibility
    knee_vis = max(landmark[self.mp_pose.PoseLandmark.LEFT_KNEE].visibility, landmark[self.mp_pose.PoseLandmark.RIGHT_KNEE].visibility)
    HandDetector.logger.info(f"message_id{message_id}")
    HandDetector.logger.debug(f"left_index_vis={left_index_vis},left_wrist_vis={left_wrist_vis},right_wrist_vis={right_wrist_vis}, left_thumb_vs={left_thumb_vis}, right_thumb_vs={right_thumb_vis}, left_elbow_vis={left_elbow_vis} right_elbow_vis={right_elbow_vis},right_index_vis={right_index_vis}, nose={nose_vis}")
    nose = (int(landmark[self.mp_pose.PoseLandmark.NOSE].x * image_width),
               int(landmark[self.mp_pose.PoseLandmark.NOSE].y * image_height))
    # 获取左手和右手int(landmark[self.mp_pose.PoseLandmark.LEFT_WRIST].y * image_height))关键点坐标
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

    left_index = (int(landmark[self.mp_pose.PoseLandmark.LEFT_INDEX].x * image_width),
                  int(landmark[self.mp_pose.PoseLandmark.LEFT_INDEX].y * image_height))
    right_index = (int(landmark[self.mp_pose.PoseLandmark.RIGHT_INDEX].x * image_width),
                  int(landmark[self.mp_pose.PoseLandmark.RIGHT_INDEX].y * image_height))

    left_knee = (int(landmark[self.mp_pose.PoseLandmark.LEFT_KNEE].x * image_width),
                  int(landmark[self.mp_pose.PoseLandmark.LEFT_KNEE].y * image_height))

    right_knee = (int(landmark[self.mp_pose.PoseLandmark.RIGHT_KNEE].x * image_width),
                  int(landmark[self.mp_pose.PoseLandmark.LEFT_KNEE].y * image_height))

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
    left_dist_wrist_hip = distance_pair(hip_pair, left_wrist)
    left_dist_wrist_elbow = distance_pair(left_wrist, left_elbow)
    left_dist_thumb_elbow = distance_pair(left_thumb, left_elbow)
    left_dist_wrist_shoulder = distance_pair(left_wrist, left_shoulder)
    left_dist_wrist_nose = distance_pair(left_wrist, nose)
    
    left_wrist_dist_right_elbow = distance_pair(left_wrist, right_elbow)
    right_wrist_dist_left_elbow = distance_pair(right_wrist, left_elbow)

    left_dist_wrist_knee = distance_pair(left_wrist, left_knee)
    right_dist_wrist_knee = distance_pair(right_wrist, right_knee)


    # cal hand angle
    # left_hand_angle = CalcHandAngle(left_thumb, left_wrist)
    # right_hand_angle = CalcHandAngle(right_thumb, right_wrist)

    left_index_angle = HandDetector.CalcHandAngle(left_index, left_elbow) 
    right_index_angle = HandDetector.CalcHandAngle(right_index, right_elbow)
    left_hand_angle = HandDetector.CalcHandAngle(left_thumb, left_elbow)
    right_hand_angle = HandDetector.CalcHandAngle(right_thumb, right_elbow)

    left_thumb_body_dist,right_thumb_body_dist = self.CalcThumbBodyDist(landmark, image_height, image_width)

    # calc the angle of arm
    left_arm_angle = self.CalcArmAngle(left_wrist, left_elbow, left_shoulder)
    right_arm_angle = self.CalcArmAngle(right_wrist, right_elbow, right_shoulder)

    vis_thres = 0.22

    left_hand_above = False
    left_hand_above_dist = 0
    if left_wrist_vis > vis_thres and knee_vis > vis_thres + 0.1:
      left_hand_above, left_hand_above_dist = self.CalcHandFaceKneeLineDistance(left_thumb, nose, left_knee)

    right_hand_above = False
    right_hand_above_dist = 0
    if right_wrist_vis > vis_thres and knee_vis > vis_thres + 0.1:
      right_hand_above, right_hand_above_dist = self.CalcHandFaceKneeLineDistance(right_thumb, nose, right_knee)

    # 定义判断范围的阈值
    threshold_x = 30
    threshold_y = 8
     
    # 判断左手位置
    HandDetector.logger.debug(f"left_wrist={left_wrist}, left_shoulder={left_shoulder}, left_ebow={left_elbow}, wrist_elbow={left_dist_wrist_elbow},wrist_hip={left_dist_wrist_hip},wrist_shoulder={left_dist_wrist_shoulder}")
    HandDetector.logger.debug(f"left_thumb={left_thumb}, left_thumb_elbow={left_dist_thumb_elbow}, left_wrist_nose={left_dist_wrist_nose} ,left_hand_angle={left_hand_angle}, left_index_angle={left_index_angle}")
    HandDetector.logger.debug(f"left_knee={left_knee}, right_knee={right_knee}, left_wrist_knee={left_dist_wrist_knee} ,right_wrist_knee={right_dist_wrist_knee}")
    HandDetector.logger.debug(f"left_thumb_body_dist={left_thumb_body_dist}, right_thumb_body_dist={right_thumb_body_dist}")
    HandDetector.logger.debug(f"body_angle={body_angle},head_angle={head_angle},left_arm_angle={left_arm_angle}, left_above={left_hand_above}, left_above_dist={left_hand_above_dist},")
    restrict_sit = abs(body_angle) > 80 and abs(body_angle) < 100 and abs(head_angle) > 80 and abs(head_angle) < 100
    head_sit = abs(head_angle) < 110 and abs(head_angle) > 75
    left_vis = (left_wrist_vis > vis_thres or left_index_vis > vis_thres or left_thumb_vis > vis_thres) and left_elbow_vis > vis_thres
    HandDetector.logger.debug(f"leftvis={left_vis}")
    if abs(left_wrist[0] - left_shoulder[0]) < threshold_x and left_wrist[1] > left_shoulder[1]:
      left_hand_pose = HandPose.BodySide
    elif left_vis and abs(head_angle) < 115 and \
      ((left_hand_above and (abs(left_arm_angle) < 100 or abs(left_hand_angle) > 80 and abs(left_hand_angle) < 100)) or \
      (abs(left_arm_angle) < 109 and head_sit and (left_hand_above or restrict_sit)) or \
        (abs(left_arm_angle) < 65) and head_sit and (left_hand_above or restrict_sit or 23 * left_hand_above_dist < left_dist_wrist_elbow)):
      HandDetector.logger.debug("lifton1")
      left_hand_pose = HandPose.LiftOn
    elif left_vis and left_thumb[1] < left_elbow[1] and \
       ((abs(left_arm_angle) > 75 and abs(left_arm_angle) < 105 and abs(left_hand_angle - 90 ) < 10) or \
      (left_hand_above and abs(left_arm_angle) < 109.7 and (abs(left_hand_angle) < 41 and abs(left_hand_angle) > 38.9 or (abs(left_hand_angle - 90) < 51 and left_hand_above_dist * 2.5 > left_dist_wrist_elbow)))):
      HandDetector.logger.debug("lifton2")
      left_hand_pose = HandPose.LiftOn
    # elif left_dist_wrist_elbow*3.5 > left_dist_wrist_shoulder and left_elbow_vis > 0.5 and left_wrist_vis > 0.15 and left_thumb[1] < left_elbow[1] and \
    elif  ((left_index_vis > 0.1 and abs(left_arm_angle) < 119 and abs(left_hand_angle - 90) < 4 and abs(left_index_angle - 90) < 4) or \
      (left_hand_above and abs(left_arm_angle) < 145 and abs(left_hand_angle - 90) < 33 and abs(left_index_angle) < 36)):
      HandDetector.logger.debug("lifton3")
      left_hand_pose = HandPose.LiftOn
    elif left_vis and abs(left_wrist[0] - mid_abdomen_x) < threshold_x and abs(left_wrist[1] - mid_abdomen_y) < threshold_y:
      left_hand_pose = HandPose.OnAbdomen
    elif left_vis and left_hand_angle > 39 and left_hand_angle < 60 and right_elbow_vis and left_wrist_dist_right_elbow < left_dist_wrist_elbow:
      left_hand_pose = HandPose.OnChest

        # 判断右手位置
    right_vis = (right_wrist_vis > vis_thres or right_index_vis > vis_thres or right_thumb_vis > vis_thres) and right_elbow_vis > vis_thres
    right_dist_wrist_hip = distance_pair(hip_pair, right_wrist)
    right_dist_wrist_elbow = distance_pair(right_wrist, right_elbow)
    right_dist_wrist_nose = distance_pair(right_wrist, nose)
    HandDetector.logger.debug(f"right_wrist={right_wrist}, right_shoulder={right_shoulder}, right_elbow={right_elbow}")
    HandDetector.logger.debug(f"right_thumb={right_thumb}, right_dist_wrist_hip={right_dist_wrist_hip},right_dist_wrist_elbow={right_dist_wrist_elbow},right_wrist_nose={right_dist_wrist_nose},right_angle={right_hand_angle}")
    HandDetector.logger.debug(f"right_index_angle={right_index_angle}, right_arm_angle={right_arm_angle},  right_above={right_hand_above}, right_above_dist{right_hand_above_dist}")
    HandDetector.logger.debug(f"left_wrist_right_elbow={left_wrist_dist_right_elbow}, right_wrist_left_elbow={right_wrist_dist_left_elbow}")
    if abs(right_wrist[0] - right_shoulder[0]) < threshold_x and right_wrist[1] > right_shoulder[1]:
      right_hand_pose = HandPose.BodySide
    elif right_wrist_vis > vis_thres and right_thumb_vis > vis_thres and \
      ((right_hand_above and (abs(right_arm_angle) > 54.2 and abs(right_arm_angle) < 100 or abs(right_hand_angle) > 80 and abs(right_hand_angle) < 100)) or \
      (abs(right_arm_angle) < 109 and head_sit and (right_hand_above or restrict_sit))):
      HandDetector.logger.debug("rightlifton1")
      right_hand_pose = HandPose.LiftOn
    elif right_wrist_vis > vis_thres and right_thumb_vis > vis_thres and right_thumb[1] < right_elbow[1] and \
      right_dist_wrist_nose < right_dist_wrist_elbow * 3.2 and right_dist_wrist_hip > right_dist_wrist_elbow and right_hand_angle > 24 and \
       (abs(right_hand_angle - 90 ) < 15 or \
      (abs(right_hand_angle - 90) < 51 and right_thumb_body_dist > 2* right_dist_wrist_elbow and \
       (right_dist_wrist_elbow * 1.2 < right_dist_wrist_hip or right_dist_wrist_hip > right_dist_wrist_nose))):
      HandDetector.logger.debug("rightlifton2")
      right_hand_pose = HandPose.LiftOn
    elif abs(right_hand_angle) > 10 and abs(right_arm_angle) > 20 and right_dist_wrist_nose * 1.2 < right_dist_wrist_elbow and right_dist_wrist_hip > 3 * right_dist_wrist_elbow and right_hand_above_dist < right_dist_wrist_nose:
      HandDetector.logger.debug("rightlifton3")
      right_hand_pose = HandPose.LiftOn
    elif right_wrist_vis > vis_thres and abs(right_wrist[0] - mid_abdomen_x) < threshold_x and abs(right_wrist[1] - mid_abdomen_y) < threshold_y:
      right_hand_pose = HandPose.OnAbdomen
    elif right_wrist_vis > vis_thres and right_hand_angle > 39 and right_hand_angle < 60 and right_wrist_dist_left_elbow < right_dist_wrist_elbow:
      right_hand_pose = HandPose.OnChest

    return left_hand_pose,left_hand_prob,right_hand_pose, right_hand_prob