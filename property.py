from random import randint, choice
from location import Location


class Property:
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