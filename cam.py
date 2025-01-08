import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import cv2
# import os

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

img_path = './content/sample_data/dog.png'

preprocessed_img = preprocess_image(img_path)

model = VGG16(weights='imagenet')
pred = model.predict(preprocessed_img)
decode_predictions(pred, top=5)

def get_gradcam_heatmap(model, img_array, layer_name):
    grad_model = tf.keras.models.Model(
        model.inputs, [model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, tf.argmax(predictions[0])]

    grads = tape.gradient(loss, conv_outputs) # 勾配を取得
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2)) # チャンネルごとの平均勾配を算出。バッチ、縦、横方向に平均する

    conv_outputs = conv_outputs[0] # layer_nameの出力する特徴量マップ
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis] # 特徴量マップにチャネルごとの平均勾配をかける(勾配の大きなチャネルがより重視される)
    heatmap = tf.squeeze(heatmap) # 不要な次元の削除。ex. (height, width, 1) -> (height, width)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# ヒートマップを画像に重ねる
def superimpose_heatmap(img_path, heatmap, alpha=0.4):
    img = cv2.imread(img_path)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * alpha + img # alphaを大きくするとheatmapの画像が強調される
    return superimposed_img

heatmap = get_gradcam_heatmap(model, preprocessed_img, 'block5_conv3')
superimposed_img = superimpose_heatmap(img_path, heatmap, alpha=0.6)
superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)

# 結果の表示
plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image.load_img(img_path))

plt.subplot(1, 2, 2)
plt.title('Grad-CAM')
plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
plt.show()