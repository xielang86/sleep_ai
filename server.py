from concurrent.futures import ThreadPoolExecutor
from sleep_phrase import SleepPhraseDetector,SleepResult,SleepType

import zerorpc
import base64
import cv2
import numpy as np
from contextlib import closing

""" request construct

"""
class RpcServer:
  def __init__(self):
    self.executor = ThreadPoolExecutor(max_workers=5)
    self.sleep_phrase_detector = SleepPhraseDetector(16)

  def DetectSleepPhrase(self, request):
      future = self.executor.submit(self._process_request, request)
      return future.result()

  def _process_request(self, request):
    uid = request.get('uid')
    session_id = request.get('conversationid')
    message_id = request.get('messageid')
    data = request.get("data")

    if data is None:
      print(f"empty data in request for request={request}")
      return

    image_data = data.get('images')
    audio_base64 = data.get('audio')

    images_base64 = image_data.get("data") 
    if images_base64 == None:
      print(f"empty image in request for request={request}")
      return

    images = []
    # 解析 Base64 编码的图像数据
    for image_base64 in images_base64:
      image_bytes = base64.b64decode(image_base64)
      image_np = np.frombuffer(image_bytes, dtype=np.uint8)
      image_decoded = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
      images.append(image_decoded)

    # 解析 Base64 编码的音频数据
    audio_bytes = None
    if audio_base64 != None:
      audio_bytes = base64.b64decode(audio_base64)

    # 调用 posedetector 类的方法进行处理
    result = self.sleep_phrase_detector.DetectSleepPhrase(uid, session_id, images, audio_bytes)
    sleep_status = "Awake"
    if result != None:
      sleep_status = result.sleep_type.name

    # 构建统一的响应结构
    response = {
      "status_code": 200,  # 状态码，200 表示成功，其他可以表示不同的状态
      "user_id": uid,
      "conversation_id": session_id,
      "message_id": message_id,
      "message": "Pose detected successfully",
      "sleep_status": sleep_status,
      "data": str(result)
    }
    return response

def main():
  with closing(zerorpc.Server(RpcServer())) as s:
    s.bind("tcp://0.0.0.0:4242")
    print("ZeroRPC Server started on port 4242")
    s.run()


if __name__ == "__main__":
  main()