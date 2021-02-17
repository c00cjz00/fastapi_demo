# 13. FastAPI 教學  (class03_v1)

###### tags: `FastAPI`


## :memo: Where do I start?
http://203.145.218.138:8888/lab

### 完整安裝
```
conda create -n mytensorflow pip python=3.8

conda activate mytensorflow

pip install fastapi

pip install uvicorn[standard]

pip install Jinja2 aiofiles python-multipart Pillow starlette gunicorn tensorflow-cpu
```


## 案例一 class01a.py
```
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}
```

## 案例二 class01b.py
```
import uvicorn
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}


# Python 直接執行 python main.py
if __name__ == "__main__":
    uvicorn.run(app, port=8000, host='0.0.0.0')
```

## 案例三 class01c_3_predict.py
Tensorflow, MobileNetV2 預測案例
```
# 執行方法 # python class01c_3_predict.py --input cat.png --output cat.png.txt

# 帶入函示庫
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
```


## class01c_2_predict_process.py
```
import time
import subprocess

from io import BytesIO
import numpy as np
from PIL import Image

# 執行指令
def cmd(command):
    subp = subprocess.Popen(command,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,encoding="utf-8")
    subp.wait(300)
    if subp.poll() == 0:
        print(subp.communicate()[0])
    else:
        print("失败")

# 函式: 讀取讀片
def read_imagefile(file):
    # 內存讀取二進制數據, 轉為圖片
    image = Image.open(BytesIO(file))
    return image

# 函式: 預測圖片
def predict(mycmd):  
    cmd(mycmd)
    return mycmd 

'''
# 直接執行
prediction = predict(filename)
print(prediction)
'''




```
## class01c_1_predict_web.py

```
# 載入uvicorn, 網頁服務器
import uvicorn 
import os

# 載入fastapi, 功能api框架
from fastapi import FastAPI, File, UploadFile

# 載入starlette, 功能 Restful api
from starlette.responses import RedirectResponse

# 載入  class01c_2_predict_process.py 之 Def:  predict, read_imagefile
from class01c_2_predict_process import predict, read_imagefile

# 定義 app FastAPI()
app = FastAPI()

# 家目錄 /
@app.get("/", include_in_schema=False)
async def index():
    # 轉址到/dcos
    return RedirectResponse(url="/docs")

# 目錄 /predict/image, POST
# 檔案上傳函式, 採用 UploadFile
@app.post("/predict/image")
async def predict_api(file: UploadFile = File(...)):
    # 帶入參數
    filename = file.filename
    
    # 產生指令
    mycmd="conda run -n mytensorflow python class01c_3_predict.py \
    --input 'static/" + filename + "' \
    --output 'static/predict_" + filename + ".txt'"

    # 許可圖片格式
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    # 圖片格式判定, 中斷返回下列訊息
    if not extension:
        predictionMessage = "Image must be jpg or png format!"
    else:
        # 圖片讀取, 啟動等待作業
        image = read_imagefile(await file.read())
        image.save("static/" + filename)        
        # 圖片預測函式
        predict(mycmd)
        
        # 傳回預測結果, 檢查檔案是否存在
        if os.path.isfile("static/predict_" + filename +'.txt'):
            f = open("static/predict_" + filename + ".txt", "r")
            predictionMessage = (f.read())
        else:
            predictionMessage = "預測輸出檔案不存在, 失敗!"
                
    return predictionMessage

# Python 直接執行 python main.py
# 或  uvicorn main:app  --host 0.0.0.0 --port 7000
if __name__ == "__main__":
    #uvicorn.run(app, debug=True)
    uvicorn.run(app, port=7000, host='0.0.0.0')

```

## 網頁製作 form.html
https://www.w3schools.com/bootstrap/bootstrap_forms.asp

```
<!DOCTYPE html>
<html lang="en">
<head>
  <title>Tensorflow, MobileNetV2 預測案例</title>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.1/css/bootstrap.min.css">
  <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.5.1/jquery.min.js"></script>
  <script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.1/js/bootstrap.min.js"></script>
</head>
<body>
<div class="container">
	<h2>Tensorflow, MobileNetV2 預測案例</h2>
	<a href="static/bus2.jpg" download>Dowdload Example</a>
	<form action="/" enctype="multipart/form-data" method="post" data-toggle="validator">
		<div class="form-group">
		  <label for="email">Email:</label>
		  <input type="email" class="form-control" id="email"  name="email" placeholder="Enter email" value="{{ email }}" required>
		</div>
		<div class="form-group">
		  <label class="form-label" for="usr">Name:</label>
		  <input type="text" class="form-control" id="usr" name="usr" value="{{ usr }}" required>
		</div>
		<div class="form-group">
		  <label class="form-label" for="age">Age:</label>
		  <input type="number" class="form-control" id="age" name="age" value="{{ age }}" required>
		</div>
		<div class="form-group">
		  <label class="form-label" for="age">Image:</label>
		  <input type="file"  class="form-control" id="file" name="file" required><br>
		</div>
		<button type="submit" class="btn btn-default">Submit</button>
	</form>
	<br>
	{% if email %}
	<p>Email: {{ email }}</p>
	<p>Name: {{ usr }}</p>
	<p>Ages: {{ age }}</p>
	<p>Uploadfile: {{ filename }}</p>
	<p>Result: {{ prediction }}</p>	
	<img src="{{ url_for('static', path='/' + filename ) }}" width="500">
	{% endif %}
</div>
</body>
</html>

```

## class01d_3_predict.py 一樣
## class01d_2_predict_process.py 一樣
## class01d_1_predict_form.py
```
# 載入uvicorn, 網頁服務器
import uvicorn
import os

# NEW
import shutil

# 載入fastapi, 功能api框架
from fastapi import FastAPI, File, UploadFile, Request, Form

# 載入外掛資料夾
from fastapi.staticfiles import StaticFiles

# 載入Jinja2Templates, 網頁框架
from fastapi.templating import Jinja2Templates

# 載入starlette, 功能 Restful api
from starlette.responses import RedirectResponse

# 載入  class01d_2_predict_process.py 之 Def:  predict, read_imagefile
from class01d_2_predict_process import predict, read_imagefile



# 定義 app FastAPI()
app = FastAPI()

# 掛載資料夾
app.mount("/static", StaticFiles(directory="static"), name="static")


# 網頁範本
templates = Jinja2Templates(directory='webTemplates/')

# 家目錄 /
@app.get("/", include_in_schema=False)
async def index(request: Request):
    # 開啟網頁templates/form.html
    return templates.TemplateResponse('form.html', context={'request': request})

# 目錄 /predict/image, POST
# 檔案上傳函式, 採用 UploadFile
@app.post("/")
async def predict_api(request: Request, email: str = Form(...), usr: str = Form(...), age: int = Form(...), file: UploadFile = File(...)):
    # 帶入參數
    email = email
    usr = usr
    age = age
    filename = file.filename

    # 產生指令
    mycmd="conda run -n mytensorflow python class01d_3_predict.py \
    --input 'static/" + filename + "' \
    --output 'static/predict_" + filename + ".txt'"
    
    # 許可圖片格式
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    # 圖片格式判定, 中斷返回下列訊息
    if not extension:
        predictionMessage = "Image must be jpg or png format!"
    else:
        # 圖片讀取, 啟動等待作業
        image = read_imagefile(await file.read())
        image.save("static/" + filename)        
        # 圖片預測函式
        predict(mycmd)
        
        # 傳回預測結果, 檢查檔案是否存在
        if os.path.isfile("static/predict_" + filename +'.txt'):
            f = open("static/predict_" + filename + ".txt", "r")
            predictionMessage = (f.read())
        else:
            predictionMessage = "預測輸出檔案不存在, 失敗!"
                
    return templates.TemplateResponse('form.html', context={'request': request, 'email': email, 'usr': usr, 'age': age, 'filename': filename, 'prediction': predictionMessage})

    
# Python 直接執行 python main.py
# 或  uvicorn main:app  --host 0.0.0.0 --port 9999
if __name__ == "__main__":
    #uvicorn.run(app, debug=True)
    uvicorn.run(app, port=5000, host='0.0.0.0')

```
