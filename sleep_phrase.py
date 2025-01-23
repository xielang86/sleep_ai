import json
from common.util import *
from algorithm import pose,human_action
from dataclasses import dataclass, asdict
from enum import Enum
from common.cache import *
from common.util import *
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

  pose_info: pose.PoseResult = pose.PoseResult()
  recent_action: human_action.ActionResult = human_action.ActionResult()
  def __str__(self):
    return json.dumps(asdict(self), indent=4, default=str)
  
# thread safe
class SleepPhraseDetector:
  def __init__(self, num):
    self.pose_detectors = [pose.PoseDetector() for _ in range(num)]
    self.action_detectors = [human_action.HumanActionDetector for _ in range(num)]
    self.user_cache = ThreadSafeCache(1024*1024, 3600)

  def VoteForCurrentPhrase(self, action_result, pose_results) -> SleepType:
    if action_result.body_action != human_action.HumanAction.MotionLess or \
      action_result.arm_action != human_action.HumanAction.MotionLess:
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

    if (max_pose == pose.BodyPose.HalfLie or max_pose == pose.BodyPose.LieFlat):
      if max_freq < 100:
        return SleepType.LightSleep
      else:
        return SleepType.DeepSleep
    return SleepType.Awake

  def DetectSleepPhrase(self, uid, session_id, images, audio) -> int: 
    if audio != None:
      print("detect for audio emotion")
      # do the emotion detect
    timestamp = int(time.time()) / 1000

    if images == None or len(images) == 0:
      print(f"empty image for uid={uid}")
      return None
    # do detect
    # detector_index = uid % len(self.detectors)
    detector_index = 0  
    pose_detector = self.pose_detectors[detector_index]
    action_detector = self.action_detectors[detector_index]

    # detect poses
    pose_results = []
    for image in images:
      pose_result = pose_detector.Detect(image)
      pose_results.append(pose_result)

    # detect action 
    # action_result = action_detector.DetectAction(pose_results)
    action_result = human_action.ActionResult()

    print(show_file_and_line(sys._getframe()))
    result = self.VoteForCurrentPhrase(action_result, pose_results)
    print(show_file_and_line(sys._getframe()))
    result_list = self.user_cache.get(uid)
    if result_list != None and len(result_list) > 0:
      if result_list[-1].session_id != session_id:
        # clear
        result_list = [result]
        self.user_cache.set(uid, result_list)
      else:
        result_list.append(result)
    sleep_result = SleepResult()
    sleep_result.sleep_type = result
    sleep_result.sleep_prob = 0.5
    sleep_result.timestamp = timestamp
    sleep_result.pose_info = pose_results[0]
    sleep_result.recent_action = action_result
    print(f"result={result}")
    print(f"sleep result={sleep_result}")
    return sleep_result
