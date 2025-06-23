from random import randint
from cnmaps import get_adm_maps

class Property:
    # 卧室数
    bedrooms = 0
    # 浴室数
    bathrooms = 0
    # 车位数
    carspaces = 0
    # 房屋面积
    building_area = 0
    # 价格 单位万
    price = 0
    # 城市
    city = ""
    # 城区
    district = ""
    # 建筑年代
    build_year = 0
    # 装修等级 0清水 1简装 3精装
    decoration = 0
    # 房产类型 1住宅,2商业
    property_type = 1
    # 离地铁距离
    distance_to_metro = 0
    # 离学校距离
    distance_to_school = 0
    # 中介描述
    description = ""

    @classmethod
    def randGen(cls):
        bedrooms = randint(1, 5)
        bathrooms = randint(1, 5)
        carspaces = randint(0, 5)
        building_area = randint(50, 500)
        price = randint(10, 1000)
        district = \
            ["武侯区", "成华区", "锦江区", "青羊区", "金牛区", "龙泉驿区", "青白江区", "双流区", "郫都区", "新都区",
             "温江区"][randint(0, 3)]
        build_year = randint(0, 50)
        decoration = randint(0, 3)
        property_type = randint(1, 2)
        distance_to_metro = randint(0, 1)
        distance_to_school = randint(0, 1)
        description = ""

