import base64
import numpy as np
import zerorpc,sys,cv2


def main():
  c = zerorpc.Client()
  # c.connect("tcp://121.43.54.25:4242")
  c.connect("tcp://192.168.0.116:4242")
  # c.connect("tcp://127.0.0.1:4242")
  # c.connect("tcp://114.55.90.104:4242")

  request = {
      "uid": "user123",
      "messageid": "msg456",
      "conversationid": "conv789",
      "data": {
          "images":
              {
                  "format": "jpg",
                  "data": []
              }
      }
  }

  # read image data
  image_paths = sys.argv[1].split(",")
  for image_path in image_paths: 
    image = cv2.imread(image_path)
    if image is None:
      print(f"Could not read the image at {image_path}")
      continue
    _, buffer = cv2.imencode('.jpg', image)
    image_bytes = buffer.tobytes()
    # 将字节数据进行 Base64 编码
    base64_string = base64.b64encode(image_bytes).decode('utf-8')

    # 将 Base64 编码的字符串解码为字节数据
    image_bytes_decoded = base64.b64decode(base64_string)
    # 将字节数据转换为 numpy 数组
    image_np = np.frombuffer(image_bytes_decoded, dtype=np.uint8)
    # 将字节数据解码为图像
    image_decoded = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    # 将 BGR 图像转换为 RGB 图像
    rgb_image = cv2.cvtColor(image_decoded, cv2.COLOR_BGR2RGB)
    print(f"rgb image= {len(rgb_image)}")
    request.get("data").get("images").get("data").append(base64_string)
  # 构建请求数据
  # print(request)
  response = c.DetectSleepPhrase(request)
  print(response)

if __name__ == "__main__":
  main()