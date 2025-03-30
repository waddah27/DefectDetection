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

    def init_dataDict(self):
        dataDict = {}
        dataDict[DataCols.x1] = list()
        dataDict[DataCols.y1] = list()
        dataDict[DataCols.x2] = list()
        dataDict[DataCols.y2] = list()
        dataDict[DataCols.cls] = list()
        dataDict[DataCols.conf] = list()
        dataDict[DataCols.name] = list()
        return dataDict
    def inference(self, img):
        self._ret = self.model(img)

    @property
    def dataFrame(self):
        dataDict = self.init_dataDict()
        for name, box in zip(self.get_cls_names, self.get_data):
            x1,y1,x2,y2,p,c = box
            dataDict[DataCols.x1].append(x1)
            dataDict[DataCols.y1].append(y1)
            dataDict[DataCols.x2].append(x2)
            dataDict[DataCols.y2].append(y2)
            dataDict[DataCols.cls].append(c)
            dataDict[DataCols.conf].append(p)
            dataDict[DataCols.name].append(name)

        return pd.DataFrame(dataDict)

    @property
    def ret(self):
        if self._ret:
            return self._ret[0]
        return None

    @property
    def get_cls_names(self):
        if self._ret is None: set()
        return {self._ret[0].names[int(cls)] for cls in self.ret.boxes.cls}

    @property
    def get_data(self):
        return list(map(lambda x: x.detach().cpu().numpy(), self.ret.boxes.data))

    @property
    def localize(self):
        # ids = list[map(lambda x:[y[-1] for y in x], self.ret.boxes.data)]
        return {k:v for k,v in zip(self.get_cls_names, self.get_data)}

    @property
    def get_report(self):
        self.dataFrame.to_csv(os.path.join(REPORT_DIR, 'report.csv'))

    def show(self):
        self._ret[0].show()




if __name__=='__main__':
    model = Model(model_name)
    img_path = os.path.join(DATA_DIR, "woman-riding-bike-with-cat-basket_1316799-24317.jpg")
    model.inference(img_path)
    # print(model.ret[0].boxes.cls)
    # print(model.ret[0].names)
    print(model.get_cls_names)
    # print(model.ret[0].boxes.data)
    print(model.get_data)
    print(model.localize)
    print(model.dataFrame)
    model.get_report
    # model.show(img_path)
