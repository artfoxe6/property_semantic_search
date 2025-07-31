import re

import ollama
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from pymilvus import connections, Collection, FieldSchema, DataType
import logging
from sbert import SentenceBert

"""
    启动api服务
    
    uvicorn api:app --reload --port 8010
    or 
    fastapi dev api.py --port=8010
    
    
    请求接口 curl
    
    curl --location 'http://127.0.0.1:8010/search' \
    --header 'Content-Type: application/json' \
    --data '{
      "query": "找一个武侯区的三房两卫，总价100万，靠近地铁"
    }'

"""
# ==================== 配置 ====================
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
COLLECTION_NAME = "properties"
# OLLAMA_MODEL = 'qwen3:8b'
OLLAMA_MODEL = 'deepseek-r1:1.5b'


# 假设你已经有一个 text2vector 函数（返回长度为384的list[float]）
def text2vector(text: str) -> List[float]:
    # 示例：调用 sentence-transformers 或其他 embedding 模型
    # 这里仅模拟，实际应替换为真实模型
    import random
    return [random.random() for _ in range(384)]


# 初始化 FastAPI
app = FastAPI(title="房产智能搜索接口")

# 连接 Milvus
try:
    connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
    collection = Collection(COLLECTION_NAME)
    collection.load()  # 加载到内存
except Exception as e:
    logging.error(f"Failed to connect to Milvus: {e}")
    raise


# ==================== 请求/响应模型 ====================
class SearchRequest(BaseModel):
    query: str


class PropertyResult(BaseModel):
    id: int
    bedrooms: int
    bathrooms: int
    carspaces: int
    floor: int
    area: float
    price: float
    province: str
    city: str
    district: str
    build_year: int
    list_at: str
    decoration: str
    type: str
    distance_to_metro: float
    distance_to_school: float
    description: str
    ai_comment: str


class SearchResponse(BaseModel):
    results: List[PropertyResult]


# ==================== 核心处理函数 ====================

def filter_and_comment_with_ollama(query: str, candidates: List[Dict]) -> List[Dict]:
    """
    使用 Ollama 模型：
    1. 过滤不满足 query 条件的房产
    2. 为保留的房产生成 AI 点评
    """
    filtered_with_comment = []

    for prop in candidates:
        # Step 1: 判断是否符合条件
        filter_prompt = f"""
        你是一个房产专家。请判断以下房源是否满足用户的需求。
        距离地铁1000米内认为很近
        价格和面积相差不过20%当作满足
        地区和房间数需要严格满足查询条件

        用户需求：{query}

        房源信息：
        - 户型：{prop['bedrooms']}室{prop['bathrooms']}厅{prop['carspaces']}车位
        - 面积：{prop['area']}平米
        - 价格：{prop['price']}万元
        - 楼层：{prop['floor']}
        - 城市：{prop['province']}-{prop['city']}-{prop['district']}
        - 装修：{prop['decoration']}
        - 建造年份：{prop['build_year']}
        - 地铁距离：{prop['distance_to_metro']}米
        - 学校距离：{prop['distance_to_school']}米
        - 描述：{prop['description']}

        不要解释为什么，请回答“是”或“否”，仅输出一个字。
        """

        try:
            filter_response = ollama.generate(
                model=OLLAMA_MODEL,
                prompt=filter_prompt,
                options={"temperature": 0.0}
            )
            cleaned_output = re.sub(r'<think>.*?</think>', '', filter_response['response'].strip(), flags=re.DOTALL).strip()
            is_match = cleaned_output.strip().lower() == '是'
        except Exception as e:
            print(f"Ollama filter error: {e}")
            is_match = False  # 出错时保守保留

        if not is_match:
            continue

        # Step 2: 生成 AI 点评
        comment_prompt = f"""
        你是专业房产分析师，请用中文为以下优质房源写一段100字左右的点评，突出亮点，语气亲切专业。

        户型：{prop['bedrooms']}室{prop['bathrooms']}厅，面积{prop['area']}㎡，价格{prop['price']}万元
        楼层：{prop['floor']}，建造年份：{prop['build_year']}，装修：{prop['decoration']}
        位置：{prop['province']}-{prop['city']}-{prop['district']}
        地铁距离：{prop['distance_to_metro']}米，学校距离：{prop['distance_to_school']}米
        描述：{prop['description']}

        请生成一段吸引人的点评：
        """
        try:
            comment_response = ollama.generate(
                model=OLLAMA_MODEL,
                prompt=comment_prompt,
                options={"temperature": 0.7}
            )
            cleaned_output = re.sub(r'<think>.*?</think>', '', comment_response['response'].strip(),
                                    flags=re.DOTALL).strip()
            ai_comment = cleaned_output.strip()
        except Exception as e:
            ai_comment = "AI点评生成失败。"

        prop['ai_comment'] = ai_comment
        filtered_with_comment.append(prop)

    return filtered_with_comment


# ==================== FastAPI 接口 ====================
@app.post("/search", response_model=SearchResponse)
async def search_properties(request: SearchRequest):
    try:
        sb = SentenceBert()
        # 1. 文本转向量
        query_vector = sb.text2vector(request.query)

        # 2. Milvus 搜索
        search_params = {
            "metric_type": "COSINE",  # 使用余弦相似度
            "params": {"nprobe": 10}
        }

        results = collection.search(
            data=[query_vector],
            anns_field="desc_vector",
            param=search_params,
            limit=5,
            output_fields=[
                "id", "bedrooms", "bathrooms", "carspaces", "floor", "area", "price",
                "province", "city", "district", "build_year", "list_at", "decoration",
                "type", "distance_to_metro", "distance_to_school", "description"
            ]
        )

        # 3. 提取候选结果
        candidates = []
        for res in results[0]:
            entity = res.entity
            prop = {
                "id": entity.id,
                "bedrooms": entity.bedrooms,
                "bathrooms": entity.bathrooms,
                "carspaces": entity.carspaces,
                "floor": entity.floor,
                "area": float(entity.area),
                "price": float(entity.price),
                "province": entity.province,
                "city": entity.city,
                "district": entity.district,
                "build_year": entity.build_year,
                "list_at": entity.list_at,
                "decoration": entity.decoration,
                "type": entity.type,
                "distance_to_metro": float(entity.distance_to_metro),
                "distance_to_school": float(entity.distance_to_school),
                "description": entity.description
            }
            candidates.append(prop)

        # 4. 使用 Ollama 过滤 + 生成点评
        final_results = filter_and_comment_with_ollama(request.query, candidates)

        # 5. 返回结果
        return SearchResponse(results=final_results)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 健康检查
@app.get("/health")
def health_check():
    return {"status": "OK", "milvus": "connected" if connections.has_connection("default") else "disconnected"}



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8010)