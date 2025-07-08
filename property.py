import copy
import random
from datetime import datetime, timedelta
from random import randint, choice
from location import Location


# 站位图片 https://dummyimage.com/400x300/39BBf0/ffffff&text=image
class Property:
    def __init__(self, id=0, bedrooms=0, bathrooms=0, carspaces=0, floor=0, area=0, price=0, province="", city="",
                 district="", build_year=0, list_at="", decoration="", type="", distance_to_metro=0,
                 distance_to_school=0, description=""):
        # 初始化基本属性
        self.id = id
        self.bedrooms = bedrooms
        self.bathrooms = bathrooms
        self.carspaces = carspaces
        self.floor = floor
        self.area = area
        self.price = price

        # 地理位置信息
        self.province = province
        self.city = city
        self.district = district

        # 时间相关属性
        self.build_year = build_year
        self.list_at = list_at

        # 装修和类型信息
        self.decoration = decoration
        self.type = type

        # 位置便利性指标
        self.distance_to_metro = distance_to_metro
        self.distance_to_school = distance_to_school

        # 描述信息
        self.description = description

    def generate_property(self):
        self.bedrooms = randint(1, 6)

        # 浴室数一般少于或等于卧室数，最多为卧室数
        if self.bedrooms == 1:
            self.bathrooms = 1
        else:
            self.bathrooms = randint(1, min(self.bedrooms, 3))  # 3间以上浴室较为稀有

        # 车位数与卧室略有关联
        self.carspaces = randint(0, 1 if self.bedrooms <= 2 else 2)

        # 房屋面积与房间数有关，每间房间大约 20~50 平米加上公共空间
        self.area = randint(self.bedrooms * 30, self.bedrooms * 60)

        # 价格与面积略有关联，均价大约 0.8~2 万每平米
        avg_price_per_m2 = randint(8000, 20000)
        self.price = int((self.area * avg_price_per_m2) / 10000)  # 单位为“万”

        self.build_year = randint(2000, datetime.now().year)
        self.decoration = choice(["清水", "简单装修", "豪华装修"])
        self.type = choice(["住宅", "公寓", "别墅"])

        # 距地铁和学校距离：较远也不会超过 5000 米
        self.distance_to_metro = randint(50, 3000)
        self.distance_to_school = randint(50, 3000)

        self.floor = randint(1, 30 if self.type != "别墅" else 3)

        l = Location()
        self.province, self.city, self.district = l.randomLocation()

        current_date = datetime.now()
        three_years_ago = current_date - timedelta(days=3 * 365)
        random_date = three_years_ago + timedelta(days=randint(0, 1095))
        self.list_at = random_date.strftime("%Y-%m-%d")

        self.description = self.combine_description()

    def to_dict(self):
        return {
            "id": self.id,
            "bedrooms": self.bedrooms,
            "bathrooms": self.bathrooms,
            "carspaces": self.carspaces,
            "build_year": self.build_year,
            "decoration": self.decoration,
            "type": self.type,
            "distance_to_metro": self.distance_to_metro,
            "distance_to_school": self.distance_to_school,
            "floor": self.floor,
            "province": self.province,
            "city": self.city,
            "district": self.district,
            "price": self.price,
            "area": self.area,
            "description": self.description,
            "list_at": self.list_at,
        }

    def to_prompt(self) -> str:
        return f"""请根据以下房屋信息生成一段自然流畅、具有吸引力的中文房源介绍，用于展示在房地产网站中，字数控制在100字到150字之间：
                - 房产类型：{self.type}
                - 房屋面积：{self.area}平方米
                - 卧室数：{self.bedrooms}间
                - 浴室数：{self.bathrooms}间
                - 车位数：{self.carspaces}个
                - 楼层：{self.floor}层
                - 建造年份：{self.build_year}年
                - 上市时间：{self.list_at}
                - 装修情况：{self.decoration}
                - 所在省市：{self.province} {self.city} {self.district}
                - 房屋总价：{self.price}万人民币
                - 离地铁距离：{self.distance_to_metro}米
                - 离学校距离：{self.distance_to_school}米
                """

    def combine_description(self) -> str:
        return f"""房产类型：{self.type},房屋面积：{self.area}平方米,卧室数：{self.bedrooms}间,浴室数：{self.bathrooms}间,车位数：{self.carspaces}个,楼层：{self.floor}层,建造年份：{self.build_year}年,上市时间：{self.list_at},装修情况：{self.decoration},所在省市：{self.province} {self.city} {self.district},房屋总价：{self.price}万人民币,离地铁距离：{self.distance_to_metro}米,离学校距离：{self.distance_to_school}米"""

    @classmethod
    def up_round_to_5(cls, num):
        if num % 5 == 0:
            return num
        else:
            return ((num // 5) + 1) * 5

    def property_to_query_texts(self):
        # 基础搜索意图
        queries = [
            (1, f"{self.city}{self.district}有哪些{self.bedrooms}室{self.bathrooms}卫的房子？价格大概在{self.price}万以内。"),
            (2, f"找个{self.area}平左右的{self.bedrooms}房，在{self.city}{self.district}。"),
            (3, f"{self.district}有没有{self.bedrooms}房，{self.area}平米，预算{self.price}万左右的房子？")
        ]

        # 小户型场景
        if self.area <= 60:
            queries.append(
                (4, f"有没有{self.city}{self.district}的{self.get_synonyms('area', self.area)}房子？{self.area}平以内"))

        # 价格需求
        if self.price < 70:
            queries.append((5, f"预算不高，找套{self.city}{self.district}的{self.up_round_to_5(self.price)}内的房子。"))

        # 大户型需求
        if self.area >= 120:
            queries.append((6,
                            f"想找一套{self.bedrooms}房的{self.get_synonyms('area', self.area)}，面积{self.area}平以上，适合一家{self.bedrooms}口的。"))
            queries.append((7,
                            f"{self.city}{self.district}有没有{self.bedrooms}房{self.get_synonyms('area', self.area)}，安静，适合自住的？"))

        # 地铁需求
        if self.distance_to_metro and self.distance_to_metro <= 800:
            queries.append((8, f"地铁附近的房子有推荐吗？在{self.city}{self.district}，交通方便一点的。"))
            queries.append((9, f"{self.district} {self.bedrooms}房 {self.area}平 地铁附近"))

        # 学区房需求
        if self.distance_to_school and self.distance_to_school <= 1000:
            queries.append((10, f"想买套靠近学校的房子，最好在{self.city}{self.district}，适合孩子上学。"))

        # 车位需求
        if self.carspaces > 0:
            queries.append((11, f"有没有带车位的{self.bedrooms}房推荐？最好在{self.city}{self.district}附近。"))

        # 新房偏好
        if self.build_year > 2015:
            queries.append((12,
                            f"在{self.city}{self.district}找个{self.build_year}年的{self.get_synonyms('build_year', self.build_year)}"))

        return queries

    def gen_negative_property(self, group):
        if group == 1:
            return self.negative_property(district=self.district, bed=self.bedrooms, bath=self.bathrooms, price=self.price)
        elif group == 2:
            return self.negative_property(district=self.district, area=self.area, bed=self.bedrooms)
        elif group == 3:
            return self.negative_property(district=self.district, bed=self.bedrooms, area=self.area, price=self.price)
        elif group == 4:
            return self.negative_property(district=self.district, area=self.area)
        elif group == 5:
            return self.negative_property(district=self.district, price=self.price)
        elif group == 6:
            return self.negative_property(bed=self.bedrooms, area=self.area)
        elif group == 7:
            return self.negative_property(district=self.district, bed=self.bedrooms, area=self.area)
        elif group == 8:
            return self.negative_property(district=self.district, dis_m=self.distance_to_metro)
        elif group == 9:
            return self.negative_property(district=self.district, bed=self.bedrooms, dis_m=self.distance_to_metro, area=self.area)
        elif group == 10:
            return self.negative_property(district=self.district, dis_s=self.distance_to_school)
        elif group == 11:
            return self.negative_property(district=self.district, bed=self.bedrooms, car=self.carspaces)
        elif group == 12:
            return self.negative_property(district=self.district, b_y=self.build_year)
        else:
            raise ValueError("Invalid group")

    def negative_property(self, district = None, bed=None, bath=None, car=None, area=None, price=None, b_y=None, type=None, dis_m=None,
                          dis_s=None,compare=None):
        p = copy.deepcopy(self)
        diff = 0

        if bed is not None:
            p.bedrooms = bed + choice([-3, -2, -1, 1, 2, 3])
            if p.bedrooms != p.bedrooms:
                diff += 1

        if bath is not None:
            p.bathrooms = bath + choice([-2, -1, 1, 2])
            if p.bathrooms != p.bathrooms:
                diff += 1

        if car is not None:
            p.carspaces = car + choice([-2, -1, 1, 2])
            if car != car:
                diff += 1

        if area is not None:
            p.area = area + choice([randint(10, 100), randint(-50, 10)])
            if area < 10:
                p.area = randint(0, 10)
            if p.area != area:
                diff += 1

        if price is not None:
            p.price = price + choice([randint(10, 100), randint(-50, 10)])
            if price < 10:
                p.price = randint(0, 10)
            if p.price != price:
                diff += 1

        if b_y is not None:
            p.build_year = b_y + randint(-30, -5)
            if p.build_year != b_y:
                diff += 1

        if type is not None:
            p.type = random.choice([t for t in ["住宅", "公寓", "别墅"] if t != self.type])
            if p.type != type:
                diff += 1

        if dis_m is not None:
            p.distance_to_metro = randint(1500, 3000)
            if p.distance_to_metro != dis_m:
                diff += 1
        if dis_s is not None:
            p.distance_to_school = randint(1500, 3000)
            if p.distance_to_school != dis_s:
                diff += 1

        if district is not None:
            l = Location()
            p.province, p.city, p.district = l.randomLocationExclude(self.district)
        else:
            p.province, p.city, p.district = self.province, self.city, self.district

        self.description = self.combine_description()

    def get_synonyms(self, field, value):
        property_synonym_map = {
            "bedrooms": {
                "ranges": [
                    {"min": 0, "max": 1, "synonyms": ["单间", "一室一厅"]},
                    {"min": 2, "max": 2, "synonyms": ["两居室", "两房"]},
                    {"min": 3, "max": 3, "synonyms": ["三居室", "三房"]},
                    {"min": 4, "max": 100, "synonyms": ["四房以上", "大户型"]}
                ]
            },
            "bathrooms": {
                "ranges": [
                    {"min": 1, "max": 1, "synonyms": ["一卫"]},
                    {"min": 2, "max": 2, "synonyms": ["双卫", "两卫"]},
                    {"min": 3, "max": 100, "synonyms": ["三卫以上"]}
                ]
            },
            "carspaces": {
                "ranges": [
                    {"min": 0, "max": 0, "synonyms": ["无车位"]},
                    {"min": 1, "max": 1, "synonyms": ["带车位"]},
                    {"min": 2, "max": 100, "synonyms": ["双车位", "多个车位"]}
                ]
            },
            "floor": {
                "ranges": [
                    {"min": 1, "max": 3, "synonyms": ["低楼层"]},
                    {"min": 4, "max": 7, "synonyms": ["中楼层"]},
                    {"min": 8, "max": 100, "synonyms": ["高楼层"]}
                ]
            },
            "area": {
                "ranges": [
                    {"min": 0, "max": 60, "synonyms": ["小户型", "紧凑型"]},
                    {"min": 60, "max": 120, "synonyms": ["中户型", "舒适型"]},
                    {"min": 120, "max": 10000, "synonyms": ["大户型", "宽敞"]}
                ]
            },
            "price": {
                "ranges": [
                    {"min": 0, "max": 100, "synonyms": ["百万元以内", "总价低"]},
                    {"min": 100, "max": 300, "synonyms": ["总价适中", "百万元区间"]},
                    {"min": 300, "max": 10000, "synonyms": ["高端房源", "豪宅"]}
                ]
            },
            "build_year": {
                "ranges": [
                    {"min": 1900, "max": 1999, "synonyms": ["老房子"]},
                    {"min": 2000, "max": 2015, "synonyms": ["次新房"]},
                    {"min": 2016, "max": 2100, "synonyms": ["新房"]}
                ]
            },
            "distance_to_metro": {
                "ranges": [
                    {"min": 0, "max": 500, "synonyms": ["地铁房", "靠近地铁"]},
                    {"min": 500, "max": 1200, "synonyms": ["步行可达地铁", "附近有地铁"]}
                ]
            },
            "distance_to_school": {
                "ranges": [
                    {"min": 0, "max": 500, "synonyms": ["学区房", "临近学校"]},
                    {"min": 500, "max": 1500, "synonyms": ["步行可达学校"]}
                ]
            }
        }

        mapping = property_synonym_map.get(field, {})
        if "ranges" in mapping:
            for r in mapping["ranges"]:
                if r["min"] <= value <= r["max"]:
                    return choice(r["synonyms"])
        return ""
