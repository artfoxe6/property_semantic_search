import json
import re

import ollama
import redis
from fastapi import HTTPException, FastAPI
from fastapi.staticfiles import StaticFiles  # 新增：用于提供静态文件
from fastapi.responses import FileResponse  # 新增：用于返回 HTML 文件
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
      "query": "找一个武侯区的三房两卫，总价100万左右，靠近地铁"
    }'

"""
# ==================== 配置 ====================
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
COLLECTION_NAME = "properties"
# OLLAMA_MODEL = 'qwen3:8b'
OLLAMA_MODEL = 'qwen3:4b'


# 初始化 FastAPI
app = FastAPI(title="房产智能搜索接口")
app.mount("/static", StaticFiles(directory="static"), name="static")
# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[  # 明确列出前端地址
        "http://localhost:63342",
        "http://localhost:8010",
        "http://127.0.0.1:8010",
        ["*"],
    ],
    allow_credentials=True,
    allow_methods=["*"],      # 允许 POST, GET, OPTIONS 等
    allow_headers=["*"],      # 允许所有请求头
)

# ==================== 配置 Redis ====================
# 请确保本地或远程 Redis 服务正在运行
try:
    r = redis.StrictRedis(host='localhost', port=6379, db=0, decode_responses=True)
    r.ping()  # 测试连接
    REDIS_AVAILABLE = True
    logging.info("Redis 连接成功")
except Exception as e:
    logging.warning(f"Redis 连接失败: {e}")
    REDIS_AVAILABLE = False
    r = None

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
    distance_to_metro: float
    distance_to_school: float
    ai_comment: str
    image: str


class SearchResponse(BaseModel):
    results: List[PropertyResult]


# ==================== 核心处理函数 ====================
def add_comment_with_ollama(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    调用本地 Ollama 模型，批量为房源生成精彩介绍（300–500字），返回 JSON 格式。
    已在 Redis 中缓存的房源直接读取，未生成的才请求模型。
    :param candidates: 房源列表
    :return: 带 ai_comment 的房源列表
    """
    if not candidates:
        return candidates

    model_cache_key = f"ollama_comment:{OLLAMA_MODEL}"  # 按模型区分缓存

    # Step 1: 查找已缓存的房源
    uncached_candidates = []
    for prop in candidates:
        cache_key = f"{model_cache_key}:id_{prop['id']}"
        cached_comment = r.get(cache_key) if REDIS_AVAILABLE else None
        if cached_comment:
            prop['ai_comment'] = cached_comment
        else:
            uncached_candidates.append(prop)

    # 如果全部已缓存，直接返回
    if not uncached_candidates:
        return candidates

    try:
        # Step 2: 构造 JSON 输出格式的 prompt
        prompt = f"""
你是一位专业的房产文案专家。请根据以下 {len(uncached_candidates)} 套房源信息，
为每套房生成一段 300 到 500 字之间的精彩介绍文案，突出地段、户型、价格、交通、学区等优势。
语言要生动、有画面感，适合用于房产推荐页。

要求：
- 输出必须是严格的 JSON 数组格式，数组长度为 {len(uncached_candidates)}
- 每个元素是一个字符串，即 ai_comment 内容
- 不要包含任何额外说明、Markdown 符号或字段名
- 使用中文，不要使用“您”或“我们”

示例输出格式：
["这是第一套房的精彩描述...", "这是第二套房的精彩描述..."]

请开始，对应以下房源：
"""

        # 添加房源详情
        input_list = []
        for prop in uncached_candidates:
            input_list.append({
                "id": prop["id"],
                "户型": f"{prop['bedrooms']}室{prop['bathrooms']}卫",
                "面积": f"{prop['area']:.0f}㎡",
                "车位": f"{prop['carspaces']}车位",
                "总价": f"{prop['price']:.0f}万元",
                "单价": f"{(prop['price'] * 10000) / prop['area']:.0f}元/㎡",
                "楼层": f"{prop['floor']}层",
                "建造年份": f"{prop['build_year']}年",
                "装修": prop['decoration'],
                "区域": f"{prop['province']}{prop['city']}{prop['district']}",
                "地铁距离": f"约{prop['distance_to_metro']:.0f}米",
                "学校距离": f"约{prop['distance_to_school']:.0f}米",
                "上市时间": prop['list_at']
            })

        prompt += json.dumps(input_list, ensure_ascii=False, indent=2)

        # Step 3: 调用 Ollama
        response = ollama.generate(
            model=OLLAMA_MODEL,
            prompt=prompt,
            options={
                "temperature": 0.7,
                "num_ctx": 8192  # 确保上下文足够长
            }
        )

        raw_output = response['response'].strip()
        raw_output =  re.sub(r'<think>.*?</think>', '', raw_output, flags=re.DOTALL).strip()

        # Step 4: 尝试解析 JSON
        try:
            comments = json.loads(raw_output)
            if not isinstance(comments, list):
                raise ValueError("输出不是数组")
            if len(comments) != len(uncached_candidates):
                raise ValueError(f"长度不匹配：期望 {len(uncached_candidates)}，得到 {len(comments)}")
        except json.JSONDecodeError as e:
            logging.warning(f"JSON 解析失败: {e}, 原始输出: {raw_output[:200]}...")
            raise RuntimeError("模型未返回有效 JSON")

        # Step 5: 写入结果并缓存到 Redis
        for prop, comment in zip(uncached_candidates, comments):
            prop['ai_comment'] = comment
            cache_key = f"{model_cache_key}:id_{prop['id']}"
            if REDIS_AVAILABLE:
                try:
                    r.set(cache_key, comment, ex=3600 * 24 * 7)  # 缓存 7 天
                except Exception as e:
                    logging.warning(f"Redis 写入失败: {e}")

        return candidates

    except Exception as e:
        logging.warning(f"Ollama生成失败，使用默认文案: {e}")
        # 兜底：使用简单描述
        for prop in uncached_candidates:
            prop['ai_comment'] = (
                f"位于{prop['district']}的{prop['bedrooms']}室{prop['bathrooms']}卫房源，"
                f"面积约{prop['area']:.0f}㎡，总价{prop['price']:.0f}万元，"
                f"距离地铁{prop['distance_to_metro']:.0f}米，生活便利，值得考虑。"
            )
        return candidates
# ==================== FastAPI 接口 ====================

@app.get("/")
def read_root():
    return FileResponse("static/index.html")

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
                "distance_to_metro", "distance_to_school"
            ]
        )

        # 3. 提取候选结果
        candidates = []
        for i, res in enumerate(results[0]):
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
                "distance_to_metro": float(entity.distance_to_metro),
                "distance_to_school": float(entity.distance_to_school),
                "ai_comment": "",
                "image":f"/static/images/{i+1}.jpg",
            }
            candidates.append(prop)

        # 4. 使用 Ollama 生成点评
        candidates_with_comment = add_comment_with_ollama(candidates)
        # 组装 返回结果 List[PropertyResult]
        final_results = [PropertyResult(**prop) for prop in candidates_with_comment]
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
