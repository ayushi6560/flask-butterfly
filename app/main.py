from flask import Flask, request, jsonify, render_template,redirect, url_for
from PIL import Image
import torch 
import torchvision.transforms as transforms
import io
import pandas as pd
import os

from werkzeug.utils import secure_filename

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])


UPLOAD_FOLDER = 'static/uploads/'

model = torch.load("./model/cpu_model.pth")
model.eval()

butterfly = pd.read_csv("class_dict.csv")
butterfly = butterfly.dropna()
butterfly = dict(zip(butterfly["class_index"], butterfly["class"]))

def get_image():
    file = request.files['image']
    return file.read()


def transform_image(image):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])

    return my_transforms(image).unsqueeze(0)


app = Flask(__name__)
# ,static_folder='./app/static'


@app.route("/", methods= ["GET"])
def hello():
    return render_template("upload.html")


@app.route("/", methods=["POST"])
def get_prediction():
    imagefile = request.files["imagefile"]
    imagepath = "./app/static/images/" + imagefile.filename
    imagefile.save(imagepath)
    
    ########

    image = Image.open(imagepath)
    user_path = "../static/images/"+ imagefile.filename
    print(user_path)
    

    ########

    tensor = transform_image(image=image)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    return render_template("predict.html", user_image = user_path, predicted_class =  butterfly[int(y_hat)] )

    


    
   


