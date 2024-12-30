import base64
import json
import requests


# import openai
from openai import OpenAI

# 设置 OpenAI API 密钥
api_key = "sk-4DVePMbvdeDieqUPeQ3xHfR3oJkouK7kj22UunVloI82O3q8"


def ask_gpt4_vision(images, query):
    """
    使用 GPT-4 Vision 模型回答基于图片和文本的问题
    :param images: list，包含图片的文件路径
    :param query: str，文本 query
    :return: str，GPT-4 Vision 的回答
    """
    # 将图片转换为 base64 格式
    image_contents = []
    for image_path in images:
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")
            image_contents.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                }
            )

    # 构造消息内容
    content = [{"type": "text", "text": query}] + image_contents

    try:
        client = OpenAI(api_key=api_key, base_url="https://api.openai-proxy.org/v1")
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "你是一位多模态 AI，擅长处理图片和文本输入并回答问题。",
                },
                {"role": "user", "content": content},
            ],
            max_tokens=1000,  # 可以根据需要调整
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error calling GPT-4 Vision API: {e}")
        return str(e)


# # 示例输入
# images = ["image1.jpg", "image2.png"]  # 替换为图片的实际路径
# query = "这些图片中显示的主要内容是什么？"

# # 调用函数并打印结果
# answer = ask_gpt4_vision(images, query)
# print("GPT-4 Vision 的回答:", answer)


def upload_pdf():
    pdf_url = "https://www.manuallib.com/download//HAIER-HOUSEHOLD-REFRIGERATOR-BCD-256KT-BCD-256KS-BCD-256TGM-MANUAL-WARRANTY-CARD.PDF"
    url = "http://127.0.0.1:8001/v1/documents/upsert-document/"
    pdf_url = "http://10.112.1.133:5500/NVIDIA-2024-Annual-Report_Maximum.pdf"
    headers = {
        "Authorization": "Bearer a057CEQeolH5iebfqW8rrdU3UNnTYmAV",
    }

    # 读取本地pdf，转为base64
    # with open("/Users/lc/data/tmp/haier.pdf", "rb") as pdf_file:
    #     base64_pdf = base64.b64encode(pdf_file.read()).decode("utf-8")

    payload = {
        "name": "nv2",
        "metadata": {},
        "collection_name": "nv2",
        "url": pdf_url,
        # "base64": base64_pdf,
        "wait": False,
        "use_proxy": False,
    }

    response = requests.post(url, json=payload, timeout=10, headers=headers)

    print(response.json())


def query_pdf():
    url = "http://127.0.0.1:8001/v1/search/"
    headers = {
        "Authorization": "Bearer a057CEQeolH5iebfqW8rrdU3UNnTYmAV",
    }
    # my_query = "who is President and CEO of NVIDIA"
    my_query = "compensation earned by our NEOs during Fiscal 2024, 2023,and 2022. Fiscal 2024, 2023, and 2022 were 52-week years."
    payload = {
        "query": my_query,
        "collection_name": "nv2",
        "top_k": 3,
        # "query_filter": {
        #     "on": "document",
        #     "key": "breed",
        #     "value": "collie",
        #     "lookup": "contains",
        # },
    }
    response = requests.post(url, json=payload, timeout=10, headers=headers)

    with open("response.json", "w") as f:
        json.dump(response.json(), f)

    # print(type(response.json()["results"][0].keys()))

    """
    返回的结果这样
    {
     "query": "海尔",
     "results": [
          {
               "collection_name": "default_collection",
               "collection_id": 1,
               "collection_metadata": {},
               "document_name": "test_name",
               "document_id": 7,
               "document_metadata": {},
               "page_number": 1,
               "raw_score": 10.27880173921585,
               "normalized_score": 0.6852534492810567,
               "img_base64": "iVBORw0KGgoAAAANSUh"
          }
     ]
    }
    """

    for result in response.json()["results"]:
        img_base64 = result["img_base64"]
        img_name = f"{result['document_name']}_{result['page_number']}.png"
        with open(img_name, "wb") as f:
            f.write(base64.b64decode(img_base64))

    images = [
        f"{result['document_name']}_{result['page_number']}.png"
        for result in response.json()["results"]
    ]
    answer = ask_gpt4_vision(images, my_query)
    print("GPT-4 Vision 的回答:", answer)


# upload_pdf()
query_pdf()
