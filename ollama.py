import json

import requests

from property import Property


class ollama:
    ollama_url = "http://localhost:11434/api/generate"

    @classmethod
    def generate_description(cls, prompt: str):
        # 请求体
        payload = {
            "model": "yi:6b",  # 替换为你本地运行的模型名称，如 llama3, gemma, mistral 等
            "prompt": prompt,
            "stream": False  # 设置为 False 表示一次性返回完整响应
        }

        # 发起请求
        response = requests.post(ollama.ollama_url, json=payload)
        # 解析返回结果
        if response.status_code == 200:
            result = response.json()
            # 去掉模型生成的思考部分 <think>
            if "<think>" in result["response"]:
                return result["response"].split("</think>")[1]
            else:
                return result["response"]
        else:
            return ""
