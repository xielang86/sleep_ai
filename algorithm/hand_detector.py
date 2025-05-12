import logging

import numpy as np
import mediapipe as mp

from common.util import distance,distance_pair,NormAngle
from common.logger import CreateCustomLogger
from .pose import HandPose,BodyPose

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

  def DetectHandPose(self, message_id, landmarks, fea, body_pose, image_height, image_width):
    head_angle = fea.head_angle
    if fea.head_angle_from_face > -180:
      head_angle = min(head_angle, fea.head_angle_from_face)

    body_angle = fea.shoulder_hip_angle

    left_hand_pose = HandPose.BodySide
    left_hand_prob = 0.5
    right_hand_pose = HandPose.BodySide
    right_hand_prob = 0.5


    left_wrist_vis = fea.left_wrist_vis
    right_wrist_vis = fea.right_wrist_vis
    left_thumb_vis = fea.left_thumb_vis
    left_index_vis = fea.left_index_vis
    right_thumb_vis = fea.right_thumb_vis
    right_index_vis = fea.right_index_vis
    left_elbow_vis = fea.left_elbow_vis
    right_elbow_vis = fea.right_elbow_vis

    HandDetector.logger.info(f"message_id{message_id}")

    right_wrist_dist_left_elbow = fea.right_wrist_dist_left_elbow
    left_dist_wrist_elbow = fea.left_dist_wrist_elbow
    left_wrist_dist_right_elbow = fea.left_wrist_dist_right_elbow
    right_wrist_dist_left_elbow = fea.right_wrist_dist_left_elbow
    # cal hand angle
    # left_hand_angle = CalcHandAngle(left_thumb, left_wrist)
    # right_hand_angle = CalcHandAngle(right_thumb, right_wrist)

    left_index_angle = fea.left_index_angle 
    right_index_angle = fea.right_index_angle
    left_hand_angle = fea.left_hand_angle
    right_hand_angle = fea.right_hand_angle

    # left_thumb_body_dist,right_thumb_body_dist = self.CalcThumbBodyDist(landmark, image_height, image_width)
    left_thumb_body_dist= fea.left_thumb_body_dist
    right_thumb_body_dist= fea.right_thumb_body_dist

    # calc the angle of arm
    left_arm_angle = fea.left_arm_angle
    right_arm_angle = fea.right_arm_angle

    vis_thres = 0.22

    left_hand_above = fea.left_hand_above
    left_hand_above_dist = fea.left_hand_above_dist
    right_hand_above = fea.right_hand_above
    right_hand_above_dist = fea.right_hand_above_dist
     
    # 判断左手位置
    HandDetector.logger.debug(f"left_thumb_body_dist={left_thumb_body_dist}, right_thumb_body_dist={right_thumb_body_dist}")
    HandDetector.logger.debug(f"body_angle={body_angle},head_angle={head_angle},left_arm_angle={left_arm_angle}, left_above={left_hand_above}, left_above_dist={left_hand_above_dist},")
    # restrict_sit = abs(body_angle) > 80 and abs(body_angle) < 100 and abs(head_angle) > 80 and abs(head_angle) < 100
    left_vis = (left_wrist_vis > vis_thres or left_index_vis > vis_thres or left_thumb_vis > vis_thres) and left_elbow_vis > vis_thres
    HandDetector.logger.debug(f"leftvis={left_vis}")
    norm_head_angle = NormAngle(fea.head_angle)

    if left_vis and norm_head_angle > 74 and \
      ((left_hand_above and (abs(left_arm_angle) < 100 or abs(left_hand_angle) > 80 and abs(left_hand_angle) < 100)) or \
      (abs(left_arm_angle) < 109 and body_pose == BodyPose.SitDown and (left_hand_above )) or \
        (abs(left_arm_angle) < 65) and body_pose == BodyPose.SitDown and (left_hand_above or 23 * left_hand_above_dist < left_dist_wrist_elbow)):
      HandDetector.logger.debug("lifton1")
      left_hand_pose = HandPose.LiftOn
    elif left_vis and \
       ((abs(left_arm_angle) > 75 and abs(left_arm_angle) < 105 and abs(left_hand_angle - 90 ) < 10) or \
      (left_hand_above and abs(left_arm_angle) < 109.7 and (abs(left_hand_angle) < 41 and abs(left_hand_angle) > 38.9 or (abs(left_hand_angle - 90) < 51 and left_hand_above_dist * 2.5 > left_dist_wrist_elbow)))):
      HandDetector.logger.debug("lifton2")
      left_hand_pose = HandPose.LiftOn
    # elif left_dist_wrist_elbow*3.5 > left_dist_wrist_shoulder and left_elbow_vis > 0.5 and left_wrist_vis > 0.15 and left_thumb[1] < left_elbow[1] and \
    elif  ((left_index_vis > 0.1 and abs(left_arm_angle) < 119 and abs(left_hand_angle - 90) < 4 and abs(left_index_angle - 90) < 4) or \
      (left_hand_above and abs(left_arm_angle) < 145 and abs(left_hand_angle - 90) < 33 and abs(left_index_angle) < 36) or \
      (left_hand_above and abs(left_index_angle -90) < 15 and abs(left_hand_angle -90) < 15 and abs(left_arm_angle) < 10)):
      HandDetector.logger.debug("lifton3")
      left_hand_pose = HandPose.LiftOn
    elif left_vis and left_thumb_body_dist < 0.5 * left_dist_wrist_elbow:
      left_hand_pose = HandPose.OnAbdomen
    elif left_vis and left_hand_angle > 39 and left_hand_angle < 60 and right_elbow_vis and left_wrist_dist_right_elbow < left_dist_wrist_elbow:
      left_hand_pose = HandPose.OnChest

        # 判断右手位置
    right_vis = (right_wrist_vis > vis_thres or right_index_vis > vis_thres or right_thumb_vis > vis_thres) and right_elbow_vis > vis_thres
    right_dist_wrist_hip = fea.right_dist_wrist_hip
    right_dist_wrist_elbow = fea.right_dist_wrist_elbow
    right_dist_wrist_nose = fea.right_dist_wrist_nose
    HandDetector.logger.debug(f"right_index_angle={right_index_angle}, right_arm_angle={right_arm_angle},  right_above={right_hand_above}, right_above_dist{right_hand_above_dist}")
    HandDetector.logger.debug(f"left_wrist_right_elbow={left_wrist_dist_right_elbow}, right_wrist_left_elbow={right_wrist_dist_left_elbow}")

    if right_wrist_vis > vis_thres and right_thumb_vis > vis_thres and \
      ((right_hand_above and (abs(right_hand_angle) > 19 and abs(right_arm_angle) > 54.2 and abs(right_arm_angle) < 100 or abs(right_hand_angle) > 80 and abs(right_hand_angle) < 100)) or \
      (abs(right_arm_angle) < 109 and body_pose == BodyPose.SitDown and (right_hand_above))):
      HandDetector.logger.debug("rightlifton1")
      right_hand_pose = HandPose.LiftOn
    elif right_wrist_vis > vis_thres and right_thumb_vis > vis_thres and \
      right_dist_wrist_nose < right_dist_wrist_elbow * 3.2 and right_dist_wrist_hip > right_dist_wrist_elbow and right_hand_angle > 24 and \
       (abs(right_hand_angle - 90 ) < 15 or \
      (abs(right_hand_angle - 90) < 51 and right_thumb_body_dist > 2* right_dist_wrist_elbow and \
       (right_dist_wrist_elbow * 1.2 < right_dist_wrist_hip or right_dist_wrist_hip > right_dist_wrist_nose))):
      HandDetector.logger.debug("rightlifton2")
      right_hand_pose = HandPose.LiftOn
    elif abs(right_hand_angle) > 10 and abs(right_arm_angle) > 20 and right_dist_wrist_nose * 1.2 < right_dist_wrist_elbow and right_dist_wrist_hip > 3 * right_dist_wrist_elbow and right_hand_above_dist < right_dist_wrist_nose:
      HandDetector.logger.debug("rightlifton3")
      right_hand_pose = HandPose.LiftOn
    elif right_vis and right_thumb_body_dist < 0.5 * right_dist_wrist_elbow:
      right_hand_pose = HandPose.OnAbdomen
    elif right_wrist_vis > vis_thres and right_hand_angle > 39 and right_hand_angle < 60 and right_wrist_dist_left_elbow < right_dist_wrist_elbow:
      right_hand_pose = HandPose.OnChest

    return left_hand_pose,left_hand_prob,right_hand_pose, right_hand_prob