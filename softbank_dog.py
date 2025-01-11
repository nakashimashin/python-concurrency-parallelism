import numpy as np
from tensorflow import keras
# # データの拡張
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# # 画像の読込用
# import matplotlib.image as mpimg
# # 画像の表示用
# import matplotlib.pyplot as plt
# # 画像をPIL形式で読み込む。画像サイズの変更が可能
# from tensorflow.keras.preprocessing import image as image_utils
# # 画像の前処理
# from tensorflow.keras.applications.imagenet_utils import preprocess_input

pre_model = keras.applications.VGG16(
    weights='imagenet',
    input_shape=(224, 224, 3),
    include_top=False # 全結合層を含まない
)

pre_model.summary()