from algorithm import pose
from dataclasses import dataclass, asdict, field
from dataclasses_json import config, dataclass_json
from enum import Enum
from common.cache import *
# thread safe
class SleepType(Enum):
  Awake = 0
  HalfSleep = 1
  LightSleep = 2
  DeepSleep = 3
  
@dataclass
@dataclass_json
class SleepResult:
  sleep_type: SleepType = field(
    metadata=config(encoder=lambda x: x.name, decoder=lambda x: SleepType[x])
    )
  # sleep_type : SleepType = SleepType.Awake
  sleep_prob: float = 0
  duration : int = 0
  timestamp: int = 0 # seconds

  pose_info : pose.PoseResult = field(default_factory=lambda: pose.PoseType()) 
  # pose_info : str = "pose info"
  session_id: int = 0

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
      pose_result = pose_detector.Detect(image)
      pose_results.append(pose_result)
     
    result = self.VoteForCurrentPhrase(pose_results)
    result_list = self.user_cache.get(uid)
    if result_list != None and len(result_list) > 0:
      if result_list[-1].session_id != session_id:
        # clear
        result_list = [result]
        self.user_cache.set(uid, result_list)
      else:
        result_list.append(result)
    sleep_result = SleepResult()
    print("here")
    sleep_result.sleep_type = result
    sleep_result.sleep_prob = 0.5
    sleep_result.timestamp = timestamp
    sleep_result.pose_info = pose_results[0].to_json()
    sleep_result.session_id = session_id
    print(f"result={result}")
    print(f"sleep result={sleep_result}")
    print(sleep_result.to_json())
    return sleep_result
