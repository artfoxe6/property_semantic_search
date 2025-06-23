import requests

from property import Property


class ollama:
    ollama_url = "http://localhost:11434/api/generate"

    @classmethod
    def generate_description(cls, prop: Property):
        prompt = f"""请根据以下房屋信息生成一段自然流畅、具有吸引力的中文房源介绍，用于展示在房地产网站中，字数控制在100字到150字之间：
        - 房产类型：{prop.type}
        - 房屋面积：{prop.area}平方米
        - 卧室数：{prop.bedrooms}间
        - 浴室数：{prop.bathrooms}间
        - 车位数：{prop.carspaces}个
        - 楼层：{prop.floor}层
        - 建造年份：{prop.build_year}年
        - 装修情况：{prop.decoration}
        - 所在省市：{prop.province} {prop.city} {prop.district}
        - 房屋总价：{prop.price}万人民币
        - 离地铁距离：{prop.distance_to_metro}米
        - 离学校距离：{prop.distance_to_school}米
        - 房屋描述补充：{prop.description}
        请综合这些信息生成一段吸引购房者兴趣的介绍，尽量突出亮点。
        """
        # 请求体

        payload = {
            "model": "qwen3:0.6b",  # 替换为你本地运行的模型名称，如 llama3, gemma, mistral 等
            "prompt": prompt,
            "stream": False  # 设置为 False 表示一次性返回完整响应
        }

        # 发起请求
        response = requests.post(ollama.ollama_url, json=payload)

        # 解析返回结果
        if response.status_code == 200:
            result = response.json()
            print(result["response"])
        else:
            print("请求失败，状态码：", response.status_code)
