import os
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")
MODEL_WEIGHTS = os.path.join(ROOT_DIR, 'weights')


print(DATA_DIR)