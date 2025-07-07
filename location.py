import json
import random


class Location:
    file = "pca.json"
    locations = {}

    def __init__(self):
        with open(self.file) as f:
            self.locations = json.load(f)

    def getProvinces(self) -> list:
        return list(self.locations.keys())

    def getCities(self, province) -> list:
        return list(self.locations[province].keys())

    def getDistricts(self, province, city) -> list:
        return self.locations[province][city]

    def randomLocation(self) -> tuple[str, str, str]:
        # p = random.choice(self.getProvinces())
        p = random.choice(["四川省"])
        # c = random.choice(self.getCities(p))
        c = random.choice(["成都市"])
        d = random.choice(self.getDistricts(p, c))
        return p, c, d

    def randomLocationExclude(self,district) -> tuple[str, str, str]:
        # p = random.choice(self.getProvinces())
        p = random.choice(["四川省"])
        # c = random.choice(self.getCities(p))
        if random.randint(1, 10) == 1:
            c = random.choice(self.getCities(p))
        else:
            c = random.choice(["成都市"])
        d = random.choice([di for di in self.getDistricts(p, c) if di != district])
        return p, c, d
