import requests
import json,os,sys
from typing import Dict, Any, Optional, Union
from urllib.parse import urlparse, unquote
from pathlib import Path

image_base_url = "http://114.55.90.104:9001/upload"
def send_trace_request(
  url: str = "http://114.55.90.104:9001/api/v1/trace/",
  trace_key: str = "gQewyXpQRTG",
  page: int = 1,
  page_size: int = 20,
  start_time: str = "2025-03-12T19:35",
  end_time: str = "2025-03-14T17:35"
) -> Dict[str, Any]:
  """
  发送追踪数据请求（模拟浏览器行为）
  
  参数:
      url: 接口地址
      trace_key: 追踪标识
      page: 页码
      page_size: 每页数量
      start_time: 开始时间 (格式: YYYY-MM-DDTHH:mm)
      end_time: 结束时间 (格式: YYYY-MM-DDTHH:mm)
  
  返回:
      接口响应JSON数据
  """
  headers = {
    "Accept": "*/*",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8,en-US;q=0.7,zh-TW;q=0.6",
    "Content-Type": "application/json",
    "Origin": "http://114.55.90.104:9001",
    "Proxy-Connection": "keep-alive",
    "Referer": f"http://114.55.90.104:9001/trace?key={trace_key}",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36"
  }
  
  payload = {
    "key": trace_key,
    "page": page,
    "page_size": page_size,
    "time_range": "custom",
    "start_time": start_time,
    "end_time": end_time
  }
  
  try:
    response = requests.post(
      url,
      headers=headers,
      json=payload,  # 自动处理JSON序列化和Content-Type
      timeout=5  # 5秒超时控制
    )
    response.raise_for_status()  # 处理4xx/5xx错误
    return response.json()
  except requests.exceptions.RequestException as e:
    print(f"请求失败: {str(e)}")
    return {"error": str(e)}
  
def download_image(
  url: str,
  save_dir: str = "./data3",
  filename: Optional[str] = None,
  timeout: int = 10,
  headers: dict = None,
  verify_ssl: bool = True
) -> Union[str, None]:
  """
  智能下载图片到本地目录
  
  参数:
      url: 图片URL
      save_dir: 保存目录（默认: 当前目录下的downloads）
      filename: 自定义文件名（留空则自动生成）
      timeout: 超时时间（秒）
      headers: 自定义请求头（默认包含浏览器UA）
      verify_ssl: 是否验证SSL证书（默认True）
  
  返回:
      保存路径（成功）/ None（失败）
  """
  # 初始化目录
  Path(save_dir).mkdir(parents=True, exist_ok=True)
  
  # 生成默认请求头
  default_headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36"
  }
  if headers:
    default_headers.update(headers)
  
  try:
    # 发送请求（跟随重定向，流式下载）
    with requests.get(url, headers=default_headers, stream=True, 
                     timeout=timeout, verify=verify_ssl) as response:
      response.raise_for_status()  # 检查HTTP错误状态码
      
      if not filename:
        filename = _get_suggested_filename(response, url)
      
      # 生成完整路径
      save_path = os.path.join(save_dir, filename)
      
      # 分块写入（支持大文件）
      with open(save_path, "wb") as f:
        total_size = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        chunk_size = 8192
        
        for chunk in response.iter_content(chunk_size=chunk_size):
          f.write(chunk)
          downloaded += len(chunk)
          _print_progress(downloaded, total_size)  # 可选进度显示
    
    print(f"\n✅ 下载完成：{save_path}")
    return save_path
  
  except requests.exceptions.RequestException as e:
    print(f"❌ 下载失败（请求错误）: {str(e)}")
  except IOError as e:
    print(f"❌ 保存失败（文件错误）: {str(e)}")
  except Exception as e:
    print(f"❌ 发生未知错误: {str(e)}")
  
  return None

def _get_suggested_filename(response, url) -> str:
  """智能生成文件名（优先使用Content-Disposition，其次URL路径，最后哈希）"""
  # 从响应头获取文件名
  cd = response.headers.get("Content-Disposition")
  if cd and "filename=" in cd:
    filename = unquote(cd.split("filename=")[1].strip('"'))
    return filename
  
  # 从URL路径获取文件名
  path = urlparse(url).path
  if path:
    filename = os.path.basename(path)
    if filename:
      return filename if "." in filename else f"image_{hash(url)}.jpg"  # 补全扩展名
  
  # 兜底方案：根据Content-Type生成
  content_type = response.headers.get("Content-Type", "")
  ext = content_type.split("/")[-1] if "image/" in content_type else "jpg"
  return f"image_{hash(url)}.{ext}"


def _print_progress(downloaded: int, total: int) -> None:
  """可选的进度条显示（控制台输出）"""
  if total == 0:
    return
  percent = downloaded / total * 100
  bar = "█" * int(percent // 2) + " " * (50 - int(percent // 2))
  print(f"\r⏳ 下载中: |{bar}| {percent:.1f}% ({downloaded}/{total} bytes)", end="")


def ExtractData(result):
  err_body_results = []
  err_hand_results = []
  body_num = 0
  hand_num = 0

  for item in result["items"]:
    trace_tree = item["trace_tree"]
    sleep_req = trace_tree["sleep_req"]

    if sleep_req is None or not sleep_req.get("data"):
      sys.stderr.write("miss image, drop case")
      continue

    sleep_resp = trace_tree["sleep_api_rsp"]
    
    if sleep_resp is None or sleep_resp.get("data") is None:
      continue
    
    pose_info = sleep_resp["data"]["pose_info"]

    body_judge = pose_info["body"] == "SitDown"

    hand_judge = pose_info["left_hand"] == "LiftOn" or pose_info["right_hand"] == "LiftOn"

    llm_resp = trace_tree["sleep_api_rsp_llm"]
    if llm_resp is None or llm_resp.get("sleepSignals") is None:
      continue 

    llm_result = llm_resp["sleepSignals"]

    ground_tru_hand = llm_result["handActivity"] == "active_device_use"
    ground_tru_body = llm_result["posture"] == "sitting"

    image_file = sleep_req["data"]["image_files"][-1]
    image_url = f"{image_base_url}/{image_file}"

    hand_num += 1 
    if hand_judge != ground_tru_hand:
      save_dir="./data_hand"
      download_image(url=image_url,save_dir=save_dir) 
      local_file = f"{save_dir}/{image_file}"
      err_hand_results.append({"image_path": local_file, "pose_info":pose_info})

    body_num += 1
    if body_judge != ground_tru_body:
      save_dir="./data_body"
      download_image(url=image_url, save_dir=save_dir) 
      local_file = f"{save_dir}/{image_file}"
      err_body_results.append({"image_path": local_file, "pose_info":pose_info})

  return hand_num, err_hand_results, body_num, err_body_results

if __name__ == "__main__":
  # 示例调用
  page_size = 100
  start_time = "2025-04-29T13:20"
  end_time = "2025-04-29T14:00"
  result = send_trace_request(
    trace_key="gQewyXpQRTG",
    page=1,
    page_size=page_size,
    start_time=start_time,
    end_time=end_time
  )

  cnt = result["total"] 
  print(cnt)
  all_page = cnt / page_size

  page = 1
  all_hand_num = 0
  all_body_num = 0
  all_err_hand_results = []
  all_err_body_results = []
  while page < all_page:
    print(len(result["items"]))
    hand_num,err_hand_results,body_num,err_body_results = ExtractData(result)
    all_hand_num += hand_num
    all_body_num += body_num
    all_err_hand_results += err_hand_results
    all_err_body_results += err_body_results

    page += 1
    result = send_trace_request(
      trace_key="gQewyXpQRTG",
      page=page,
      page_size=page_size,
      start_time=start_time,
      end_time=end_time
  )

  # print(json.dumps(result, indent=2, ensure_ascii=False))
  with open(sys.argv[1], 'w', encoding='utf-8') as file:
  # 使用 json.dump() 将列表写入文件
    json.dump(all_err_hand_results, file, ensure_ascii=False, indent=2)

  with open(sys.argv[2], 'w', encoding='utf-8') as file:
    json.dump(all_err_body_results, file, ensure_ascii=False, indent=2)

  print(all_hand_num)
  print(1.0 * len(all_err_hand_results) / all_hand_num)

  print(all_body_num)
  print(1.0 * len(all_err_body_results) / all_body_num)
  
