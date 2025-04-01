import os
import torch
from enum import Enum, auto

class StrEnum(Enum):
    def _generate_next_value_(name,*args):
        return name

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")
REPORT_DIR = os.path.join(ROOT_DIR, "report")
MODEL_WEIGHTS = os.path.join(ROOT_DIR, 'weights')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class DataCols(StrEnum):
    x1 = auto()
    y1 = auto()
    x2 = auto()
    y2 = auto()
    conf = auto()
    cls = auto()
    name = auto()

    def __str__(self):
        return self._name_

class ModelBackbones(StrEnum):
    yolov10s = auto() #"yolov10s.pt"
    yolo11n = auto()

    def __str__(self):
        return self._name_+'.pt'


print(DATA_DIR)