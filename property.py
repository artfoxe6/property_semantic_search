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
