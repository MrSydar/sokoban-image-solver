import numpy as np

class Shape:
    def get_center_x(self):
        return self.center[0]

    def get_center_y(self):
        return self.center[1]

    def __init__(self, shape):
        self.shape = shape
        self.center = np.average(shape, axis=0)[0]

class Skewer:
    def sort_objects(self):
        return sorted(self.objects, key=Shape.get_center_x)

    def get_row(self):
        return self.row

    def add_object(self, obj):
        self.objects.append(obj)

    def __init__(self, obj, row):
        self.objects = [obj]
        self.row = row

class Grill:
    def get_lunch(self):
        sorted_skewers = sorted(self.skewers, key=Skewer.get_row)
        matrix = [s.sort_objects() for s in sorted_skewers]
        return matrix

    def put(self, obj):
        for s in self.skewers:
            if s.row - self.delta < obj.get_center_y() < s.row + self.delta:
                s.add_object(obj)
                return

        s = Skewer(obj, row=obj.center[1])
        self.skewers.append(s)

    def __init__(self, delta):
        self.delta = delta
        self.skewers = []
