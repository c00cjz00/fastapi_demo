# python class01c_3_predict.py --input cat.png --output cat.png.txt
import json
import argparse
from io import BytesIO
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.imagenet_utils import decode_predictions

# 函式: 讀取模型
def load_model():
    model = tf.keras.applications.MobileNetV2(weights="imagenet")
    print("Model loaded")
    return model

# 函式: 預測圖片
def predict(image):
    # 取取模型
    model = load_model()
    
    # 影像處理
    image = np.asarray(image.resize((224, 224)))[..., :3]
    image = np.expand_dims(image, 0)
    image = image / 127.5 - 1.0
    
    # 預測結果輸出
    result = decode_predictions(model.predict(image), 3)[0]
    myclass = result[0][1]
    myconfidence = f"{result[0][2]*100:0.2f} %"
    output = myclass + ' : ' + myconfidence
    return output

# 函式: 帶入變數
def get_parser():
    parser = argparse.ArgumentParser(description="fastai demo for builtin configs")
    parser.add_argument(
        "--input",
        help="input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="output text; "
        "or a single glob pattern such as 'directory/*.text'",
    )
    return parser


# 執行狀況
args = get_parser().parse_args()
if args.input:
    inputImg = (args.input)
    outputTEXT = (args.output)
    image = Image.open(inputImg)
    data = predict(image)
    prediction = json.dumps(data)
    f = open(outputTEXT, "w")
    f.write(prediction)
    f.close()
    print(prediction)


