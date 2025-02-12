import json
from enum import Enum
from dataclasses import asdict, dataclass
from common.util import *

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
  body: BodyPose = BodyPose.HalfLie
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