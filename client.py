import base64
import zerorpc,sys,cv2


def main():
  c = zerorpc.Client()
  c.connect("tcp://127.0.0.1:4242")

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
    # 将 BGR 图像转换为 RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    base64_string = base64.b64encode(rgb_image).decode("utf-8")
    print(f"add image{len(base64_string)}")
    request.get("data").get("images").get("data").append(base64_string)
  # 构建请求数据
  print(request)
  response = c.DetectSleepPhrase(request)
  print(response)

if __name__ == "__main__":
  main()