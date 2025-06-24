from datetime import datetime, timedelta
from random import randint, choice
from location import Location


class Property:
    id = 0
    # 卧室数
    bedrooms = 0
    # 浴室数
    bathrooms = 0
    # 车位数
    carspaces = 0
    # floor
    floor = 0
    # 房屋面积
    area = 0
    # 价格 单位万
    price = 0
    # 省份
    province = ""
    # 城市
    city = ""
    # 城区
    district = ""
    # 建筑年代
    build_year = 0
    # 上市时间
    list_at = ""
    # 装修等级
    decoration = "简装"
    # 房产类型
    type = "住宅"
    # 离地铁距离
    distance_to_metro = 0
    # 离学校距离
    distance_to_school = 0
    # 房屋描述
    description = ""

    def generate_property(self):
        self.bedrooms = randint(1, 10)
        self.bathrooms = randint(1, 5)
        self.carspaces = randint(0, 2)
        self.area = randint(10, 1000)
        self.price = randint(10, 2000)
        self.build_year = randint(1900, 2025)
        self.decoration = choice(["清水", "简装", "精装"])
        self.type = choice(["住宅", "公寓", "别墅"])
        self.distance_to_metro = randint(1, 5000)
        self.distance_to_school = randint(1, 5000)
        self.floor = randint(1, 50)
        l = Location()
        self.province, self.city, self.district = l.randomLocation()
        current_date = datetime.now()
        three_years_ago = current_date - timedelta(days=3 * 365)
        random_date = three_years_ago + timedelta(days=randint(0, 1095))
        self.list_at = random_date.strftime("%Y-%m-%d")

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
