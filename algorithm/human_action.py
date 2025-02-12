from collections import deque
import logging
from enum import Enum
from dataclasses import dataclass
from common.util import *
from common.logger import CreateCustomLogger

class HumanAction(Enum):
  Talking = 0
  Walking = 1
  ArmMoving = 2
  MotionLess = 8

  def __str__(self):
    return self.name

@dataclass
class ActionResult:
  face_action: HumanAction = HumanAction.MotionLess
  face_prob: float = 0

  body_action: HumanAction = HumanAction.MotionLess
  body_prob : float = 0

  arm_action: HumanAction = HumanAction.MotionLess
  arm_prob : float = 0

# thread unsafe
class HumanActionDetector:
  logger = CreateCustomLogger("action.log", __name__, logging.DEBUG)
  def __init__(self):
    self.recent_poses = deque(maxlen=8) # 2s for 4 pose , 4s

  def AddPose(self, cur_poses):
    for pose in cur_poses:
      self.recent_poses.append(pose)

  def DetectFaceAction(self):
    mouth_action = [0, 0, 0] 
    result = HumanAction.MotionLess
    prob = 0
    for pose in self.recent_poses:
      if mouth_action[pose.mouth.value] == 0:
        mouth_action[pose.mouth.value] = 1

    s = 0  
    for a in mouth_action:
      s += a
    if s > 1:
      result = HumanAction.Talking
      HumanActionDetector.logger.info(f"mouth action={result}")
      prob = 0.5

    return result,prob

  def DetectBodyAction(self):
    result = HumanAction.MotionLess
    prob = 0
    return result,prob
  
  def DetectArmAction(self):
    result = HumanAction.MotionLess
    prob = 0
    return result,prob
    
  def DetectAction(self, new_poses)->ActionResult:
    result = ActionResult()
    self.AddPose(new_poses)
    
    result.face_action,result.face_prob = self.DetectFaceAction()
    result.body_action,result.body_prob = self.DetectBodyAction()
    result.arm_action,result.arm_prob = self.DetectArmAction()
    return result