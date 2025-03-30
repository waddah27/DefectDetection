import os
from enum import Enum, auto

class StrEnum(Enum):
    def _generate_next_value_(name,*args):
        return name

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")
MODEL_WEIGHTS = os.path.join(ROOT_DIR, 'weights')

class DataCols(StrEnum):
    x1 = auto()
    y1 = auto()
    x2 = auto()
    y2 = auto()
    conf = auto()
    cls = auto()

    def __str__(self):
        return self._name_

class ModelBackbones(StrEnum):
    yolov10s = auto() #"yolov10s.pt"

    def __str__(self):
        return self._name_+'.pt'


print(DATA_DIR)