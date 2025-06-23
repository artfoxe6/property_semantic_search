from gen import Property


class ollama:
    ollama_url = "http://localhost:11434/api/generate"

    @classmethod
    def generate_description(cls, property: Property):

        # 构造提示语
        prompt = f"""请根据以下房地产信息生成一段简洁、生动的中文描述，适合用于房产平台展示： 
        - 地点：{property.location}
        - 房间数：{property.bedrooms}
        - 浴室数：{property.carspaces}
        - 面积：{property_data['area']} 平方米 
        - 楼层：{property_data['floor']}
        - 朝向：{property_data['orientation']}
        - 建造年份：{property_data['year_built']}
        - 总价：{property_data['price']} 万元

        生成的描述："""

        # 请求体
        payload = {
            "model": "qwen2.5:7b",  # 替换为你本地运行的模型名称，如 llama3, gemma, mistral 等
            "prompt": prompt,
            "stream": False  # 设置为 False 表示一次性返回完整响应
        }

        # 发起请求
        response = requests.post(ollama_url, json=payload)

        # 解析返回结果
        if response.status_code == 200:
            result = response.json()
            print("生成的房源描述：\n")
            print(result["response"])
        else:
            print("请求失败，状态码：", response.status_code)
            print("返回内容：", response.text)
