from consts import *
from ultralytics import YOLO

model_name = "yolov10s.pt"
class Model:
    def __init__(self, model_name):
        # Load model
        self.model = YOLO(os.path.join(MODEL_WEIGHTS, model_name)) # Small version

        self.ret = None

    def inference(self, img):
        self.ret = self.model(img)

    @property
    def get_predicted_objects_names(self):
        self._predicted_objects_names = set()
        if self.ret is None: return set()
        # return {self.ret[0].names[int(cls) for cls in self.ret[0].boxes.cls}
        for cls in self.ret[0].boxes.cls:
            cls = int(cls)
            self._predicted_objects_names.add(self.ret[0].names[int(cls)])
        return self._predicted_objects_names

    def show(self):
        self.ret[0].show()




if __name__=='__main__':
    model = Model(model_name)
    img_path = os.path.join(DATA_DIR, "woman-riding-bike-with-cat-basket_1316799-24317.jpg")
    model.inference(img_path)
    # print(model.ret[0].boxes.cls)
    # print(model.ret[0].names)
    print(model.get_predicted_objects_names)
    print(model.ret[0].boxes.data)
    model.show(img_path)
