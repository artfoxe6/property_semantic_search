from datetime import datetime, timedelta
from random import randint, choice
from location import Location

# 站位图片 https://dummyimage.com/400x300/39BBf0/ffffff&text=image
class Property:
    def __init__(self, id=0, bedrooms=0, bathrooms=0, carspaces=0, floor=0, area=0, price=0, province="", city="", district="", build_year=0, list_at="", decoration="简装", type="住宅", distance_to_metro=0, distance_to_school=0, description=""):
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

    def generate_property(self, id=0, bedrooms=0, bathrooms=0, carspaces=0, floor=0, area=0, price=0, province="", city="", district="", build_year=0, list_at="", decoration="简装", type="住宅", distance_to_metro=0, distance_to_school=0, description=""):
        self.id = id
        self.bedrooms = bedrooms
        self.bathrooms = bathrooms
        self.carspaces = carspaces
        self.floor = floor
        self.area = area
        self.price = price
        self.province = province
        self.city = city
        self.district = district
        self.build_year = build_year
        self.list_at = list_at
        self.decoration = decoration
        self.type = type
        self.distance_to_metro = distance_to_metro
        self.distance_to_school = distance_to_school
        self.description = description

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
