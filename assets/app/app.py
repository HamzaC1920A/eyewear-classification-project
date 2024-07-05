from flask import Flask, render_template, request,redirect,url_for
from keras.models import load_model
from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img

from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
from numpy import array
import numpy as np

app = Flask(__name__)
model = load_model('models/ResNet50_sexe.h5')
model1 = load_model('models/ResNet50_materiaux_verre.h5')
model2 = load_model('models/ResNet550_style.h5')
model3 = load_model('models/ResNet50type.h5')
model4 = load_model('models/ResNet50_couleur.h5')
model5 = load_model('models/ResNet50_prix.h5')

# routes
@app.route('/', methods=['GET'])
def main():
	return render_template("app.html")

@app.route('/', methods=['GET','POST'])
def predict():
	
    imagefile= request.files['imagefile']
    image_path = "./images/" + imagefile.filename
    imagefile.save(image_path)
    
    img = load_img(image_path, target_size=(224,224))
    im = np.array(img).reshape(-1,224,224,3)
    prediction = model.predict(im)
    prediction1 = model1.predict(im)
    prediction2 = model2.predict(im)
    prediction3 = model3.predict(im)
    prediction4 = model4.predict(im)
    prediction5 = model5.predict(im)
   
    type_mat = ["Cristal","Nylon","Personnalisable","Plastique","Polycarbonate"][prediction1.argmax()]
    sexe = ["Sexe_femme","Sexe_homme"][prediction.argmax()]
    style = ["Aviateur","Carrée","Enveloppante","Oeil de Chat","Ovale","Oversize","Papillon","Ronde","Verre Unique","Wayfarer"][prediction2.argmax()]
    type = ["Opticale","Soleil"][prediction3.argmax()]
    couleur = ["Blanc","Bleu","Gold","Marron","Noir","Rose","Rouge","Silver","Tortoise","Vert"][prediction4.argmax()]
    prix = ["between 118 and 180","less than 118","more than 180"][prediction5.argmax()]
    return render_template("app.html", a = sexe  ,b = type ,c = style ,d = type_mat,e = couleur ,f = prix)
    #return render_template('index.html')



     


if  __name__ =='__main__':
	#app.debug = True
	app.run(port=3010, debug = True)