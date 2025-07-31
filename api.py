import json
import re

import ollama
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from pymilvus import connections, Collection, FieldSchema, DataType
import logging
from sbert import SentenceBert
from fastapi.middleware.cors import CORSMiddleware  # 导入 CORS 中间件

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
OLLAMA_MODEL = 'qwen3:8b'
# OLLAMA_MODEL = 'qwen3:0.6b'


# 初始化 FastAPI
app = FastAPI(title="房产智能搜索接口")
# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[  # 明确列出前端地址
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "http://localhost:5173",   # Vite 默认
        "http://127.0.0.1:5173",
        "http://localhost:63342",
        ["*"],
    ],
    allow_credentials=True,
    allow_methods=["*"],      # 允许 POST, GET, OPTIONS 等
    allow_headers=["*"],      # 允许所有请求头
)

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
    # ai_comment: str


class SearchResponse(BaseModel):
    results: List[PropertyResult]


# ==================== 核心处理函数 ====================

def filter_and_comment_with_ollama(query: str, candidates: List[Dict]) -> List[Dict]:
    """
    使用 Ollama 一次性完成：
    1. 过滤符合条件的房源
    2. 为符合条件的房源生成 AI 点评
    """
    if not candidates:
        return []

    # 构造输入房源信息
    properties_str = "\n---\n".join([
        f"ID: {prop['id']}\n"
        f"地区：{prop['district']}\n"
        f"户型：{prop['bedrooms']}室{prop['bathrooms']}卫\n"
        f"面积：{prop['area']}平米\n"
        f"价格：{prop['price']}万元\n"
        f"车位：{prop['carspaces']}车位\n"
        f"装修：{prop['decoration']}\n"
        f"建造年份：{prop['build_year']}\n"
        f"地铁距离：{prop['distance_to_metro']}米\n"
        f"学校距离：{prop['distance_to_school']}米\n"
        for prop in candidates
    ])

    system_prompt1 = """
    你是一个专业的房产分析师，请根据用户需求，逐个判断下面每个房源是否满足要求，并为所有满足条件的房源撰写一段简短中文点评（约100字）,不符合条件的房源编辑原因。

    判断标准如下：
    1. 地区（district）必须严格匹配；
    2. 房间数（bedrooms、bathrooms）必须满足；
    3. 面积和价格允许 ±20% 浮动；
    4. 地铁距离在 1000 米以内视为“近地铁”。

    输出要求：
    - 输出一个 **完整的 JSON 数组**，每个元素包含以下字段：
      - "id"：房源ID，整数
      - "include"：布尔值，表示是否符合需求
      - "comment"：字符串。当 include 为 true 时填写点评，否则填写原因
    - 必须逐条评估输入中的每个房源，不能跳过；
    - 不能省略字段；
    - 输出所有房源，include为 true和false都需要输出，必须是合法 JSON，不允许出现 Markdown、解释说明或其它文本。

    【格式示例】：
    [
      {"id": 1, "include": true, "comment": "这套房源位于核心区域，三房两卫，靠近地铁，性价比高。"},
      {"id": 2, "include": false, "comment": "查询条件的房间数和目标房源不匹配"},
      ...
    ]

    请严格遵守格式要求。
    """.strip()

    user_prompt = f"""
    {system_prompt1}
    
    
    用户需求：{query}

    以下是候选房源信息：

    {properties_str}

    请评估每套房源并输出 JSON 数组结果：
    """.strip()
    print(f"{system_prompt1}\n\n{user_prompt}")
    try:
        # response = ollama.generate(
        #     model=OLLAMA_MODEL,
        #     prompt=f"{system_prompt}\n\n{user_prompt}",
        #     options={
        #         "temperature": 0.7,
        #         "num_ctx": 8192,  # 确保上下文足够
        #     },
        #     format="json"  # 强制返回 JSON 格式（需要模型支持）
        # )
        # text = response['response'].strip()

        # 假设你的 system_prompt 和 user_prompt 已定义
        system_prompt = "You are a helpful assistant."  # Open WebUI 默认常用 system prompt
        # user_prompt = "请介绍一下你自己。"

        response = ollama.chat(
            model=OLLAMA_MODEL,  # 如 'llama3' 或 'mistral' 等
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            options={
                "temperature": 0.7,
                "num_ctx": 8192,
                "top_p": 0.9,
                "frequency_penalty": 0.0,
                # 可根据需要添加更多参数
            },
            format="json"  # 如果模型支持 JSON 输出格式（需模型训练支持）
        )

        text = response['message']['content'].strip()

        # 清理可能的 think 标签等非 JSON 内容
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:-3].strip()  # 去掉 ```json 和 ```

        text = re.sub(r'\n', '', text, flags=re.DOTALL)
        ok, parsed_results = safe_json_loads(text)
        if not ok:
            raise ValueError(f"JSON 解析失败: {parsed_results}")
        # 映射回原始数据并添加点评
        id_to_prop = {prop['id']: prop for prop in candidates}
        filtered_with_comment = []

        for item in parsed_results:
            prop_id = item.get("id")
            include = item.get("include", False)
            comment = (item.get("comment") or "").strip()

            if prop_id not in id_to_prop:
                continue
            if not include:
                continue

            prop = id_to_prop[prop_id]
            prop['ai_comment'] = comment if comment else "这是一套符合您需求的优质房源。"
            filtered_with_comment.append(prop)

        return filtered_with_comment

    except Exception as e:
        logging.error(f"Ollama 批量处理失败: {e}")
        # 备降：保守全部保留，生成基础点评
        fallback_results = []
        for prop in candidates:
            prop['ai_comment'] = "AI点评生成中..."
            fallback_results.append(prop)
        return fallback_results


def safe_json_loads(s):
    try:
        obj = json.loads(s)
        return True, obj
    except json.JSONDecodeError as e:
        return False, str(e)


# ==================== FastAPI 接口 ====================
@app.post("/search", response_model=SearchResponse)
async def search_properties(request: SearchRequest):
    try:
        sb = SentenceBert()
        # 1. 文本转向量
        query_vector = sb.text2vector(request.query)
        print(request.query)
        print(query_vector)

        # 2. Milvus 搜索
        search_params = {
            "metric_type": "COSINE",  # 使用余弦相似度
            "params": {"nprobe": 10}
        }

        results = collection.search(
            data=[query_vector],
            anns_field="desc_vector",
            param=search_params,
            limit=15,
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
        # final_results = filter_and_comment_with_ollama(request.query, candidates)
        # 组装 返回结果 List[PropertyResult]
        final_results = [PropertyResult(**prop) for prop in candidates]
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

    uvicorn.run(app, host="0.0.0.0", port=8010)
