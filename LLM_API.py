import os

import dashscope
from openai import OpenAI

# API Key 从环境变量读取，请勿在代码中硬编码：
#   PowerShell:  $env:DASHSCOPE_API_KEY="你的DashScope API Key"
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

dashscope.api_key = DASHSCOPE_API_KEY


def get_response():
    if not DASHSCOPE_API_KEY:
        raise RuntimeError(
            "未配置 DASHSCOPE_API_KEY 环境变量，无法调用大模型。"
        )

    client = OpenAI(
        api_key=DASHSCOPE_API_KEY,
        base_url=DASHSCOPE_BASE_URL,
    )
    local_image_path = "1.jpg"
    completion = client.chat.completions.create(
        model="qwen-vl-max",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": local_image_path,
                    },
                    {
                        "type": "text",
                        "text": "框住图中的芬达",
                    },
                ],
            }
        ],
    )
    print(completion.model_dump_json())


if __name__ == "__main__":
    get_response()
