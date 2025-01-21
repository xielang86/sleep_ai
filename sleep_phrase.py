from algorithm import pose
from dataclasses import dataclass, asdict
from enum import Enum
from common.cache import *
import json
# thread safe
class SleepType(Enum):
  Awake = 0
  HalfSleep = 1
  LightSleep = 2
  DeepSleep = 3
  
@dataclass
class SleepResult:
  sleep_type: SleepType
  sleep_prob: float
  duration : int
  timestamp: int  # seconds

  pose_info : pose.PoseResult
  session_id: int
  def __str__(self):
    return json.dumps(asdict(self), indent=4)

class SleepPhraseDetector:
  def __init__(self, num):
    self.detectors = [pose.PoseDetector() for _ in range(num)]
    self.user_cache = ThreadSafeCache(1024*1024, 3600)

  def VoteForCurrentPhrase(self, pose_results) -> SleepType:
    pose_to_freq = [0] * pose.PoseType.Other.value
    for r in pose_results:
      pose_to_freq[r.pose_type.value] += r.pose_prob
    max_pose = pose.PoseType.SitDown.value
    max_freq = pose_to_freq[max_pose]

    i = 1 
    for i in range(1, len(pose_to_freq)):
      freq = pose_to_freq[i]
      if freq > max_freq:
        max_freq = freq
        max_pose = i

    if (max_pose == pose.PoseType.HalfLie or max_pose == pose.PoseType.LieFlat):
      if max_freq < 100:
        return SleepType.LightSleep
      else:
        return SleepType.DeepSleep
    else:
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
    pose_detector = self.detectors[detector_index]
    pose_results = []
    for image in images:
      pose_results.append(pose_detector.Detect(image))
     
    result = self.VoteForCurrentPhrase(pose_results)

    result_list = self.user_cache.get(uid)
    if result_list != None and len(result_list) > 0:
      if result_list[-1].session_id != session_id:
        # clear
        result_list = [result]
        self.user_cache.set(uid, result)
      else:
        result_list.append(result)
    sleep_result = SleepResult(result, 0.5, 1, timestamp, pose_results[0], session_id)
    return sleep_result
