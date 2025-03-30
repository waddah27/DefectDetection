import numpy as np
import pandas as pd
from consts import *
from ultralytics import YOLO

model_name = ModelBackbones.yolov10s
print(DataCols.cls)
print(type(model_name))
class Model:
    def __init__(self, model_name):
        # Load model
        self.model = YOLO(os.path.join(MODEL_WEIGHTS, str(model_name))) # Small version

        self._ret = None

    def inference(self, img):
        self._ret = self.model(img)

    @property
    def dataFrame(self):
        data_dict = {}
        data_dict[DataCols.x1] = list()
        data_dict[DataCols.y1] = list()
        data_dict[DataCols.x2] = list()
        data_dict[DataCols.y2] = list()
        data_dict[DataCols.cls] = list()
        data_dict[DataCols.conf] = list()
        for box in self.get_data:
            x1,y1,x2,y2,p,c = box
            data_dict[DataCols.x1].append(x1)
            data_dict[DataCols.y1].append(y1)
            data_dict[DataCols.x2].append(x2)
            data_dict[DataCols.y2].append(y2)
            data_dict[DataCols.cls].append(c)
            data_dict[DataCols.conf].append(p)
        return data_dict

    @property
    def ret(self):
        if self._ret:
            return self._ret[0]
        return None

    @property
    def get_predicted_objects_names(self):
        if self._ret is None: set()
        return {self._ret[0].names[int(cls)] for cls in self.ret.boxes.cls}

    @property
    def get_data(self):
        return list(map(lambda x: x.detach().cpu().numpy(), self.ret.boxes.data))

    @property
    def localize(self):
        # ids = list[map(lambda x:[y[-1] for y in x], self.ret.boxes.data)]
        return {k:v for k,v in zip(self.get_predicted_objects_names, self.get_data)}

    @property
    def get_report(self):
        return pd.DataFrame(self.localize)

    def show(self):
        self._ret[0].show()




if __name__=='__main__':
    model = Model(model_name)
    img_path = os.path.join(DATA_DIR, "woman-riding-bike-with-cat-basket_1316799-24317.jpg")
    model.inference(img_path)
    # print(model.ret[0].boxes.cls)
    # print(model.ret[0].names)
    print(model.get_predicted_objects_names)
    # print(model.ret[0].boxes.data)
    print(model.get_data)
    print(model.localize)
    print(model.get_report)
    print(model.dataFrame)
    model.show(img_path)
