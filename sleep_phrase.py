import json,cv2
from common.util import *
from algorithm import pose,human_action,pose_detector
from dataclasses import dataclass, asdict, field
from enum import Enum
from common.cache import *
from common.util import *
import mediapipe as mp

# thread safe
class SleepType(Enum):
  Awake = 0
  HalfSleep = 1
  LightSleep = 2
  DeepSleep = 3

  def __str__(self):
    return self.name

@dataclass
class SleepResult:
  # sleep_type: SleepType = field(
  #   metadata=config(encoder=lambda x: x.name, decoder=lambda x: SleepType[x])
  #   )
  sleep_type : SleepType = SleepType.Awake
  sleep_prob: float = 0
  duration : int = 0
  timestamp: int = 0 # seconds

  # pose_info: pose.PoseResult = pose.PoseResult()
  pose_info: pose.PoseResult = field(default_factory = pose.PoseResult)
  recent_action: human_action.ActionResult = field(default_factory = human_action.ActionResult)
  def __str__(self):
    return json.dumps(asdict(self), indent=4, default=str)
  
# thread safe
class SleepPhraseDetector:
  def __init__(self, num):
    self.mp_pose = mp.solutions.pose

    self.pose_detectors = [pose_detector.PoseDetector( \
      self.mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.03, min_tracking_confidence=0.01), \
      mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.05, min_tracking_confidence=0.1) \
    ) for _ in range(num)]

    self.action_detectors = [human_action.HumanActionDetector() for _ in range(num)]
    self.user_cache = ThreadSafeCache(1024*1024, 3600)

  def VoteForCurrentPhrase(self, action_result, pose_results) -> SleepType:
    if action_result.body_action != human_action.HumanAction.MotionLess or \
      action_result.arm_action != human_action.HumanAction.MotionLess:
      return SleepType.Awake

    if pose_results == None or len(pose_results) == 0:
      print("detect none pose for vote sleeptype")
      return SleepType.HalfSleep

    if pose_results[0].left_eye == pose.EyePose.Open or pose_results[0].right_eye == pose.EyePose.Open:
      return SleepType.Awake
    
    if pose_results[0].left_hand == pose.HandPose.LiftOn or pose_results[0].right_hand == pose.HandPose.LiftOn:
      return SleepType.Awake

    pose_to_freq = [0] * pose.BodyPose.Other.value

    for r in pose_results:
      pose_to_freq[r.body.value] += r.body_prob
    max_pose = pose.BodyPose.SitDown.value
    max_freq = pose_to_freq[max_pose]

    i = 1 
    for i in range(1, len(pose_to_freq)):
      freq = pose_to_freq[i]
      if freq > max_freq:
        max_freq = freq
        max_pose = i

    if (max_pose == pose.BodyPose.HalfLie.value or max_pose == pose.BodyPose.LieFlat.value):
      if max_freq < 480:
        return SleepType.Awake
      elif max_freq < 1024:
        return SleepType.HalfSleep
      elif max_freq < 2048:
        return SleepType.LightSleep
      else:
        return SleepType.DeepSleep
    return SleepType.Awake

  def SaveForDebug(self, timestamp, image):
      filename="%d.jpg" % timestamp
      print("save %s" % filename)
      cv2.imwrite(filename, image)

  def DetectSleepPhrase(self, uid, session_id, images, audio) -> int: 
    if audio != None:
      print("detect for audio emotion")
      # do the emotion detect
    timestamp = int(time.time()) / 1000
    # self.SaveForDebug(timestamp, images[0])

    if images == None or len(images) == 0:
      print(f"empty image for uid={uid}")
      return None
    # do detect
    detector_index = int(timestamp) % len(self.pose_detectors)
    pose_detector = self.pose_detectors[detector_index]
    action_detector = self.action_detectors[detector_index]

    # detect poses
    pose_results = []
    for image in images:
      pose_result = pose_detector.Detect(session_id, image)
      pose_results.append(pose_result)


    # detect action 
    # action_result = action_detector.DetectAction(pose_results)
    action_result = human_action.ActionResult()
    result = self.VoteForCurrentPhrase(action_result, pose_results)
    result_list = self.user_cache.get(uid)
    if result_list != None and len(result_list) > 0:
      if result_list[-1].session_id != session_id:
        # clear
        result_list = [result]
        # self.user_cache.set(uid, result_list)
      else:
        result_list.append(result)
    sleep_result = SleepResult()
    sleep_result.sleep_type = result
    sleep_result.sleep_prob = 0.5
    sleep_result.timestamp = timestamp
    sleep_result.pose_info = pose_results[-1]
    sleep_result.recent_action = action_result
    print(f"pose result={pose_results}")
    print(f"sleep result={sleep_result}")
    return sleep_result
