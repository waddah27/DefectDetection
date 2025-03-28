from consts import *
from ultralytics import YOLO

model_name = "yolov10s.pt"
class Model:
    def __init__(self, model_name):
        # Load model
        self.model = YOLO(os.path.join(MODEL_WEIGHTS, model_name)) # Small version

    def inference(self, img):
        return self.model(img)

    def show(self, img):
        self.inference(img)[0].show()



if __name__=='__main__':
    model = Model(model_name)
    img_path = os.path.join(DATA_DIR, "2025-03-13 14.13.41.jpg")
    ret = model.inference(img_path)
    model.show(img_path)
