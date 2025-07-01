import sqlite3

from property import Property

import random


def property_to_random_text(prop: Property):
    # 模板：基础信息（总是适用）
    templates = [
        f"{prop.city}{prop.district}在售一套{prop.bedrooms}室{prop.bathrooms}卫{prop.type}，{prop.area}㎡，售价{prop.price}万，{prop.decoration}，位于第{prop.floor}层。",
        f"{prop.city}{prop.district}住宅，{prop.bedrooms}房{prop.area}㎡，总价{prop.price}万，{prop.decoration}装修。"]

    # 小户型推荐（面积 <= 60）
    if prop.area <= 60:
        templates.append(
            f"{prop.city}{prop.district}刚需小户型，{prop.bedrooms}室{prop.area}㎡，总价{prop.price}万，地铁{prop.distance_to_metro}米内，适合单身或情侣。"
        )

    # 大户型推荐（面积 >= 100）
    if prop.area >= 100:
        templates.append(
            f"{prop.city}{prop.district}{prop.bedrooms}房大户型，{prop.area}㎡宽敞空间，适合大家庭，总价{prop.price}万，生活舒适。"
        )

    # 地铁房（地铁距离 <= 800 米）
    if prop.distance_to_metro and prop.distance_to_metro <= 800:
        templates.append(
            f"{prop.city}{prop.district}地铁口好房，{prop.area}㎡，距离地铁仅{prop.distance_to_metro}米，售价{prop.price}万，通勤方便。"
        )

    # 学区房（距离学校 <= 1000 米）
    if prop.distance_to_school and prop.distance_to_school <= 1000:
        templates.append(
            f"{prop.city}{prop.district}学区优选，距离学校{prop.distance_to_school}米，{prop.bedrooms}室，{prop.area}㎡，总价{prop.price}万，教育资源便利。"
        )

    # 有车位
    if prop.carspaces > 0:
        templates.append(
            f"{prop.city}{prop.district}{prop.bedrooms}房带车位，{prop.area}㎡，{prop.decoration}装修，售价{prop.price}万，停车无忧。"
        )

    # 新房（2015年及以后）
    if prop.build_year and prop.build_year >= 2015:
        templates.append(
            f"{prop.city}{prop.district}{prop.build_year}年建成住宅，{prop.bedrooms}室{prop.area}㎡，售价{prop.price}万，社区环境新颖。"
        )

    # 高性价比（价格 / 面积 < 1 万/㎡）
    if prop.area > 0 and prop.price / prop.area < 1:
        templates.append(
            f"{prop.city}{prop.district}性价比好房，{prop.area}㎡仅售{prop.price}万，{prop.bedrooms}房，居住投资两相宜。"
        )
    return random.choice(templates)


def property_to_query_texts(prop: Property):
    queries = [
        f"{prop.city}{prop.district}有哪些{prop.bedrooms}室{prop.bathrooms}卫的房子？价格大概在{prop.price}万以内。",
        f"找个{prop.area}平左右的{prop.bedrooms}房，在{prop.city}{prop.district}，装修不要太差。",
        f"{prop.city}有没有{prop.bedrooms}房，{prop.area}平米，预算{prop.price}万左右的房子？"]

    # 1. 基础搜索意图

    # 2. 小户型场景
    if prop.area <= 60:
        queries.append(f"有没有{prop.city}{prop.district}的小户型房子？60平以内，适合一个人住的。")
        queries.append(f"预算不高，找套靠近地铁的{prop.bedrooms}房小户型。")

    # 3. 大户型需求
    if prop.area >= 100:
        queries.append(f"想找一套{prop.bedrooms}房的大户型，面积100平以上，适合一家四口的。")
        queries.append(f"{prop.city}{prop.district}有没有三房大面积户型，安静，适合自住的？")

    # 4. 地铁需求
    if prop.distance_to_metro and prop.distance_to_metro <= 800:
        queries.append(f"地铁附近的房子有推荐吗？在{prop.city}{prop.district}，交通方便一点的。")
        queries.append(f"{prop.city} {prop.bedrooms}房 {prop.area}平 地铁附近")

    # 5. 学区房需求
    if prop.distance_to_school and prop.distance_to_school <= 1000:
        queries.append(f"想买套靠近学校的房子，最好在{prop.city}{prop.district}，适合孩子上学。")

    # 6. 车位需求
    if prop.carspaces > 0:
        queries.append(f"有没有带车位的{prop.bedrooms}房推荐？最好在{prop.city}{prop.district}附近。")

    # 7. 新房偏好
    if prop.build_year >= 2015:
        queries.append(f"在{prop.city}{prop.district}找个2015年以后的新房，有没有性价比高的？")

    # 8. 简洁关键词式
    if prop.type:
        queries.append(f"{prop.city}{prop.district} {prop.price}万内住宅")

    return queries

def get_negative_samples(conn, prop_id, num_random=2, num_hard=2):
    cursor = conn.cursor()

    # 获取目标房源
    cursor.execute("SELECT * FROM properties WHERE id = ?", (prop_id,))
    row = cursor.fetchone()
    if not row:
        return []

    columns = [desc[0] for desc in cursor.description]
    prop_dict = dict(zip(columns, row))

    def row_to_property(row):
        d = dict(zip(columns, row))
        return Property(**d)

    # --- 1. 静态负样本：随机挑选与当前房源无关的 ---
    cursor.execute(
        "SELECT * FROM properties WHERE id != ? ORDER BY RANDOM() LIMIT ?",
        (prop_id, num_random)
    )
    random_negatives = [row_to_property(r) for r in cursor.fetchall()]

    # --- 2. 硬负样本：部分相似但关键字段差异 ---
    cursor.execute("""
        SELECT * FROM properties
        WHERE id != ?
          AND city = ?
          AND ABS(price - ?) < 20
          AND ABS(area - ?) < 15
          AND ABS(distance_to_metro - ?) > 1000
        ORDER BY RANDOM()
        LIMIT ?
    """, (
        prop_dict['id'], prop_dict['city'],
        prop_dict['price'], prop_dict['area'],
        prop_dict['distance_to_metro'], num_hard
    ))
    hard_negatives = [row_to_property(r) for r in cursor.fetchall()]

    return random_negatives + hard_negatives


if __name__ == "__main__":
    # 打开数据库连接
    conn = sqlite3.connect("property.db")

    # 获取某个房源的负样本
    negatives = get_negative_samples(conn, prop_id=42)

    # 转为描述句子
    for neg_prop in negatives:
        text = property_to_random_text(neg_prop)
        print(text)
