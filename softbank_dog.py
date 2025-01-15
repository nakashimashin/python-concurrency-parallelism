import numpy as np
from tensorflow import keras
# データの拡張
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# 画像の読込用
import matplotlib.image as mpimg
# 画像の表示用
import matplotlib.pyplot as plt
# 画像をPIL形式で読み込む。画像サイズの変更が可能
from tensorflow.keras.preprocessing import image as image_utils
# 画像の前処理
from tensorflow.keras.applications.imagenet_utils import preprocess_input

pre_model = keras.applications.VGG16(
    weights='imagenet',
    input_shape=(224, 224, 3),
    include_top=False # 全結合層を含まない
)

pre_model.summary()

# VGG16のモデルの重みを固定
# 学習中に重みが更新されないようにする
pre_model.trainable = False

# keras APIで入力層を定義
inputs = keras.Input(shape=(224, 224, 3))

# 入力テンソル(4次元テンソル)をVGG16のモデルに入力
# モデルが推論モードであることを明示するためにtraining=Falseを指定
x = pre_model(inputs, training=False)

# グローバル平均プーリング層を使用することで、特徴マップの空間次元を削減
x = keras.layers.GlobalAveragePooling2D()(x)

units = 2 # 出力空間の次元数
# 全結合層を追加, 活性化関数はsoftmax
outputs = keras.layers.Dense(units, activation='softmax')(x)

# モデルをインスタンス化
revised_model = keras.Model(inputs, outputs)

revised_model.summary()

# モデルのコンパイル
revised_model.compile(
    loss=keras.losses.BinaryCrossentropy(from_logits=True),  # 損失関数としてバイナリクロスエントロピーを使用。from_logits=True はモデルの出力がロジットであることを示す
    metrics=[keras.metrics.categorical_accuracy]  # 評価指標としてカテゴリカル精度を使用
)

#データ拡張
datagen = ImageDataGenerator(
        featurewise_center=True, # データセット全体で，入力の平均を0にする
        samplewise_center=True,  # 各サンプルの平均を0にする
        rotation_range=10,  # 整数．画像をランダムに回転する回転範囲．
        zoom_range = 0.1, # 浮動小数点数または[lower，upper]．ランダムにズームする範囲．
                          # 浮動小数点数が与えられた場合，[lower, upper] = [1-zoom_range, 1+zoom_range]
        width_shift_range=0.1,  # r浮動小数点数（横幅に対する割合）．ランダムに水平シフトする範囲
        height_shift_range=0.1,  # 浮動小数点数（縦幅に対する割合）．ランダムに垂直シフトする範囲
        horizontal_flip=True,  # 水平方向に入力をランダムに反転
        vertical_flip=True # 垂直方向に入力をランダムに反転
)