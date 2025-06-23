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
        p = random.choice(self.getProvinces())
        c = random.choice(self.getCities(p))
        d = random.choice(self.getDistricts(p, c))
        return p, c, d
