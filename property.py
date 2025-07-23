import copy
import random
from datetime import datetime, timedelta
from random import randint, choice
from location import Location


# 站位图片 https://dummyimage.com/400x300/39BBf0/ffffff&text=image
def up_round_to_10(num):
    if num % 10 == 0:
        return num
    else:
        return ((num // 10) + 1) * 10


def down_round_to_10(num):
    return (num // 10) * 10


def number_to_chinese(num):
    """
    将 0~10 的阿拉伯数字转换为中文数字（简体中文）
    :param num: 整数 (0 <= num <= 10)
    :return: 中文数字字符串
    """
    if not isinstance(num, int) or num < 0 or num > 10:
        return "输入必须是 0~10 的整数"

    chinese_numbers = ['零', '一', '二', '三', '四', '五', '六', '七', '八', '九', '十']
    return chinese_numbers[num]


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

    def random_value(self):
        self.bedrooms = randint(1, 6)

        # 浴室数一般少于或等于卧室数，最多为卧室数
        if self.bedrooms == 1:
            self.bathrooms = 1
        else:
            self.bathrooms = randint(1, 2)

        # 车位数与卧室略有关联
        self.carspaces = randint(0, 1 if self.bedrooms <= 3 else 2)

        self.type = choice(
            ["住宅", "住宅", "住宅", "住宅", "住宅", "住宅", "住宅", "住宅", "住宅", "公寓", "公寓", "公寓", "别墅"])

        # 房屋面积与房间数有关，每间房间大约 20~50 平米加上公共空间
        if type == "别墅":
            self.area = randint(100, 1000)
        else:
            self.area = randint(self.bedrooms * 20, self.bedrooms * 50)

        # 价格与面积略有关联，均价大约 0.5~3 万每平米
        avg_price_per_m2 = randint(5000, 30000)
        self.price = int((self.area * avg_price_per_m2) / 10000)  # 单位为“万”

        self.build_year = randint(2000, datetime.now().year)
        self.decoration = choice(["简单装修", "豪华装修"])

        # 距地铁和学校距离：较远也不会超过 3000 米
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
        # return f"""房产类型：{self.type},房屋面积：{self.area}平方米,卧室数：{self.bedrooms}间,浴室数：{self.bathrooms}间,车位数：{self.carspaces}个,楼层：{self.floor}层,建造年份：{self.build_year}年,上市时间：{self.list_at},装修情况：{self.decoration},所在省市：{self.province} {self.city} {self.district},房屋总价：{self.price}万人民币,离地铁距离：{self.distance_to_metro}米,离学校距离：{self.distance_to_school}米"""
        description = f"""位于{self.district} {self.bedrooms}室{self.bathrooms}卫 {self.type} {self.area}平 总价{self.price}万 {self.build_year}年建造 {self.decoration}"""
        if self.carspaces > 0:
            description += f" 带{self.carspaces}车位"
        if self.distance_to_metro < 1000:
            description += f" 靠近地铁"
        if self.distance_to_school < 1000:
            description += f" 靠近学校"
        return description

    def price_to_query_text(self):
        price = up_round_to_10(self.price)
        return choice([
            f"{price}万左右",
            f"{price}万",
            f"{price}万以内",
            f"{down_round_to_10(self.price - self.price / 10)}到{up_round_to_10(self.price + self.price / 10)}万",
        ])

    def area_to_query_text(self):
        area = up_round_to_10(self.area)
        return choice([
            f"{area}平左右",
            f"{area}平",
            f"{area}平以内",
            f"{down_round_to_10(self.area - self.area / 10)}到{up_round_to_10(self.area + self.area / 10)}平",
        ])

    def metro_to_query_text(self):
        if self.distance_to_metro <= 1000:
            return choice([
                f"靠近地铁",
                f"地铁附近",
                f"附近有地铁",
                f"邻近地铁站",
                f"步行可达地铁站",
                f"紧邻地铁站",
                f"交通便利",
                f"交通方便",
            ])
        return ""

    def school_to_query_text(self):
        if self.distance_to_school <= 1500:
            return choice([
                f"靠近学校",
                f"学校附近",
                f"附近有学校",
                f"邻近学校",
                f"步行可达学校",
                f"紧邻学校",
            ])
        return ""

    def decoration_to_query_text(self):
        if self.decoration == "豪华装修":
            return choice([
                f"豪华装修",
                f"精装豪宅",
                f"高档装修",
                f"品质装修",
            ])
        elif self.decoration == "简单装修":
            return choice([
                f"简单装修",
                f"普通装修",
                f"基础装修",
                f"常规装修",
                f"简装",
                f"基本装修",
                f"简约风格",
            ])
        return ""

    def carspace_to_query_text(self):
        if self.carspaces > 0:
            return choice([
                f"有车位",
                f"带车位",
                f"附带车位",
                f"含车位",
                f"带{number_to_chinese(self.carspaces)}个车位",
                f"含固定车位",
                f"带私家车位",
                f"带停车位",
                f"停车方便",
                f"送车位",
            ])
        return ""

    def build_year_to_query_text(self):
        queries = [
            f"{self.build_year}年左右",
            f"{self.build_year}年",
            f"{down_round_to_10(self.build_year)}到{up_round_to_10(self.build_year)}年",
        ]
        if self.build_year > 2018:
            queries.append(f"{self.build_year}年新建的")
            queries.append(f"房龄较新"),
            queries.append(f"近几年新建的"),
            queries.append(f"建成时间较短"),
            queries.append(f"建筑年代较近"),
            queries.append(f"次新房"),
            queries.append(f"建筑年份较新"),
            queries.append(f"现代新建住宅"),

        return choice(queries)

    def bedrooms_to_query_text(self):
        queries = [
            f"{number_to_chinese(self.bedrooms)}室",
            f"{number_to_chinese(self.bedrooms)}房",
            f"{number_to_chinese(self.bedrooms)}居室",
        ]
        if self.type == "公寓" and self.bedrooms == 1:
            queries.append(f"单身公寓")
        return choice(queries)

    def bathrooms_to_query_text(self):
        queries = [
            f"{number_to_chinese(self.bathrooms)}卫",
            f"{number_to_chinese(self.bathrooms)}间浴室",
            f"{number_to_chinese(self.bathrooms)}个卫生间",
        ]
        if self.bathrooms == 1:
            queries.append(f"独卫")
        if self.bathrooms == 2:
            queries.append(f"双卫")
        return choice(queries)

    def district_to_query_text(self):
        return choice([
            f"{self.district}",
            f"位于{self.district}",
            f"{self.city}{self.district}",
        ])

    def prefix_to_query_text(self):
        return choice([
            f"找", f"有哪些", f"找一个", f"有没有", f"想找", f"我要", f"是否有", f"", f"", f"", f"",
        ])

    def property_to_query_texts_v2(self):
        # 常规搜索
        queries = [
            (11,
             f"{self.district_to_query_text()}有哪些{self.bedrooms}房{self.bathrooms}卫的房子？价格{self.price_to_query_text()}。"),
            (12,
             f"{self.prefix_to_query_text()}{self.district_to_query_text()}的{self.bedrooms}房推荐，总价{self.price_to_query_text()}")
        ]
        # 豪华装修
        if self.decoration == "豪华装修":
            queries.append((21,
                            f"{self.prefix_to_query_text()}{self.area_to_query_text()}{self.decoration_to_query_text()}的房子，在{self.district_to_query_text()}。"))
            queries.append((22,
                            f"{self.prefix_to_query_text()}{self.district_to_query_text()}的{self.bedrooms}房{self.bathrooms}卫{self.decoration_to_query_text()}的房子"))
        # 地铁需求
        if self.distance_to_metro < 1000:
            queries.append((31,
                            f"{self.prefix_to_query_text()}{self.district_to_query_text()}{self.bedrooms}房{self.bathrooms}卫的房子，{self.metro_to_query_text()}"))
            queries.append((32,
                            f"{self.prefix_to_query_text()}{self.area_to_query_text()}{self.metro_to_query_text()}的房子,在 {self.district_to_query_text()}"))
            queries.append((33,
                            f"{self.prefix_to_query_text()}{self.district_to_query_text()}{self.metro_to_query_text()}的房子，价格{self.price_to_query_text()}"))

        # 学区房需求
        if self.distance_to_school < 1000:
            queries.append((41,
                            f"{self.prefix_to_query_text()}{self.school_to_query_text()}的房子，最好在{self.district_to_query_text()}，适合孩子上学。"))
            queries.append((42,
                            f"{self.prefix_to_query_text()}{self.district_to_query_text()}{self.bedrooms}房{self.bathrooms}卫的房子，{self.school_to_query_text()}"))
            queries.append((43,
                            f"{self.prefix_to_query_text()}{self.district_to_query_text()}{self.school_to_query_text()}的房子，总价{self.price_to_query_text()}"))

        # 车位需求
        if self.carspaces > 0:
            queries.append((51,
                            f"{self.prefix_to_query_text()}{self.bedrooms}房{self.carspace_to_query_text()}的推荐？最好在{self.district_to_query_text()}附近。"))
            queries.append((52,
                            f"{self.prefix_to_query_text()}{self.district_to_query_text()}{self.carspace_to_query_text()}的房子，价格{self.price_to_query_text()}"))
            queries.append((53,
                            f"{self.prefix_to_query_text()}{self.district_to_query_text()}{self.carspace_to_query_text()}的房子，面积{self.area_to_query_text()}"))
        # 新房偏好
        if self.build_year > 2018:
            queries.append((61,
                            f"在{self.district_to_query_text()}找个{self.build_year_to_query_text()}{self.area_to_query_text()}的房子"))
            queries.append((62,
                            f"{self.district_to_query_text()}有哪些{self.bedrooms}房{self.bathrooms}卫的房子？{self.build_year_to_query_text()}。"), )
            queries.append((63,
                            f"{self.prefix_to_query_text()}{self.district_to_query_text()}的{self.build_year_to_query_text()}。价格{self.price_to_query_text()}"), )

        return queries

    def property_to_query_texts(self):
        queries = []
        direction, price_text = choice(self.price_to_query_texts())
        queries.append(
            (1, direction, f"{self.district}有哪些{self.bedrooms}室{self.bathrooms}卫的{self.type}？{price_text}。"))
        queries.append((2, "", f"找个{self.area}平左右的{self.bedrooms}房，在{self.district}。"))
        direction, area_text = choice(self.area_to_query_texts())
        queries.append((3, direction, f"{self.district}有没有{self.bedrooms}房，{area_text}，{price_text}"))
        # 地铁需求
        if self.distance_to_metro and self.distance_to_metro <= 1000:
            queries.append((8, f"地铁附近的房子有推荐吗？在{self.district}，交通方便一点的。"))
            queries.append((9, f"{self.district} {self.bedrooms}房 {self.area}平 地铁附近"))

        # 学区房需求
        if self.distance_to_school and self.distance_to_school <= 1500:
            queries.append((10,
                            f"想买套{self.get_synonyms('distance_to_school', self.distance_to_school)}的房子，最好在{self.district}，适合孩子上学。"))

        # 车位需求
        if self.carspaces > 0:
            queries.append((11,
                            f"有没有{self.get_synonyms('carspaces', self.carspaces)}的{self.get_synonyms('bedrooms', self.bedrooms)}推荐？最好在{self.district}附近。"))

        # 新房偏好
        if self.build_year > 2015:
            queries.append((12,
                            f"在{self.district}找个{self.build_year}年后的{self.get_synonyms('build_year', self.build_year)}"))

        return queries

    def gen_negative_property(self, group):
        if group == 11:
            return self.negative_property_v2(["district", "bed", "bath", "price"])
        elif group == 12:
            return self.negative_property_v2(["district", "bed", "price"])

        elif group == 21:
            return self.negative_property_v2(["district", "area", "deco"])
        elif group == 22:
            return self.negative_property_v2(["district", "deco", "bed", "bath"])

        elif group == 31:
            return self.negative_property_v2(["district", "bed", "bath", "metro"])
        elif group == 32:
            return self.negative_property_v2(["district", "metro", "area"])
        elif group == 33:
            return self.negative_property_v2(["district", "metro", "price"])

        elif group == 41:
            return self.negative_property_v2(["district", "school"])
        elif group == 42:
            return self.negative_property_v2(["district", "bed", "school", "bath"])
        elif group == 43:
            return self.negative_property_v2(["district", "price", "school"])

        elif group == 51:
            return self.negative_property_v2(["district", "bed", "car"])
        elif group == 52:
            return self.negative_property_v2(["district", "car", "price"])
        elif group == 53:
            return self.negative_property_v2(["district", "car", "area"])

        elif group == 61:
            return self.negative_property_v2(["district", "build_year", "area"])
        elif group == 62:
            return self.negative_property_v2(["district", "build_year", "bed", "bath"])
        elif group == 63:
            return self.negative_property_v2(["district", "build_year", "price"])
        else:
            raise ValueError("Invalid group")

    # 构造负样本
    def negative_property_v2(self, mask=None):
        # 随机抛弃一个条件
        mask = random.sample(mask, len(mask)- choice([0,1]))  # 生成新数组
        l = Location()
        # p = copy.deepcopy(self)
        p = Property()
        p.random_value()
        # mask中可能指定4个条件用来生成负样本，但是随机取部分条件使用
        # mask = random.sample(mask, randint(1, len(mask)))
        for m in mask:
            if m == 'bed':
                p.bedrooms = choice([x for x in range(1, 7) if x != self.bedrooms])
            elif m == 'bath':
                p.bathrooms = choice([x for x in range(1, 3) if x != self.bathrooms])
            elif m == 'car':
                p.carspaces = 0
            elif m == 'deco':
                p.decoration = "简单装修"
            elif m == 'area':
                min = down_round_to_10(self.area - self.area / 10)
                max = up_round_to_10(self.area + self.area / 10)
                if min < 50:
                    p.area = choice([max, 1000])
                else:
                    p.area = choice([choice([20, min]), choice([max, 1000])])
            elif m == 'price':
                min = down_round_to_10(self.price - self.price / 10)
                max = up_round_to_10(self.price + self.price / 10)
                if min < 50:
                    p.price = choice([max, 1000])
                else:
                    p.price = choice([choice([20, min]), choice([max, 1000])])
            elif m == 'type':
                p.type = choice([x for x in ["住宅", "公寓", "别墅"] if x != self.type])
            elif m == 'metro':
                p.distance_to_metro = randint(1000, 3000)
            elif m == 'school':
                p.distance_to_school = randint(1000, 3000)
            elif m == 'build_year':
                p.build_year = randint(2000, 2018)
            elif m == 'district':
                p.district = l.randomDistrict(self.district)
        p.description = p.combine_description()
        return p.description

    def negative_property(self, district=None, bed=None, bath=None, car=None, area=None, price=None, b_y=None,
                          type=None, dis_m=None,
                          dis_s=None, compare=None):
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

        p.description = self.combine_description()

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

    # 根据 property的这些属性 [地区，bedrooms,bathrooms, 面积，总价，修建年代，装修等级，靠近学校，靠近学校，带车位] 生成 query
    # 随机抽取一个属性，生成一个query
    # 随机抽取两个属性，生成两个query
    # 随机抽取三个属性，生成三个query
    # 随机抽取四个属性，生成三个query
    # 随机抽取五个属性，生成两个query
    # 随机抽取六个属性，生成一个query
    # 每个属性设置一个抽取的权重， 地区 = 0.5, bedrooms = 0.5, bathrooms = 0.3, area = 0.3, price = 0.3, build_year = 0.1, distance_to_metro = 0.3, distance_to_school = 0.3, type = 0.3, carspaces = 0.2
    def create_property_queries(self):
        queries = []
        for i in range(1, 6):
            query = ""


if __name__ == "__main__":
    p = Property()
    p.random_value()
    print(f"{down_round_to_10(p.price - p.price / 10)}到{up_round_to_10(p.price + p.price / 10)}万")
    print(p.description)
