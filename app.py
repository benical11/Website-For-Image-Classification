import os
import uuid
import flask
import urllib
from PIL import Image
from tensorflow.keras.models import load_model
import tensorflow as tf
import cv2
from flask import Flask , render_template  , request , send_file, Response
from tensorflow.keras.preprocessing.image import load_img , img_to_array

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#model = load_model(os.path.join(BASE_DIR , 'mobilenetv2.hdf5'))
model = tf.keras.models.load_model('Models/cnn')
video = cv2.VideoCapture(0)


ALLOWED_EXT = set(['jpg' , 'jpeg' , 'png' , 'jfif'])
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXT

classes = ['ayak_parmaklari_enfekte_ve_nerkotik', 'basinc_ulseri_1', 'cesitli_yaralar'
 'diyabetik_ayak_ulserleri', 'ektravazaston_yarasi', 
 'karin_yarasi', 'kotu_huylu_yaralar', 'menenjit_yaralari',
 'ortopedik_yaralar', 'venos_ulserler_arteriyel_ulserler_Set_1',
 'yanik_yarasi']



def predict(filename , model):
    img = load_img(filename , target_size = (224 , 224))
    img = img_to_array(img)
    img = img.reshape(1 , 224 ,224 ,3)

    img = img.astype('float32')
    img = img/255.0
    result = model.predict(img)

    dict_result = {}
    for i in range(10):
        dict_result[result[0][i]] = classes[i]

    res = result[0]
    res.sort()
    res = res[::-1]
    prob = res[:3]
    
    prob_result = []
    class_result = []
    for i in range(3):
        if prob[i]*100 <= 70 and i == 0:
            class_result.append("diger")
        else:
            class_result.append(dict_result[prob[i]])
            
        prob_result.append((prob[i]*100).round(2))
        
    return class_result , prob_result




@app.route('/')
def home():
        return render_template("index.html")

@app.route('/deneme')
def take_picture():
        return render_template("deneme.html")

@app.route('/success' , methods = ['GET' , 'POST'])
def success():
    error = ''
    target_img = os.path.join(os.getcwd() , 'static/images')
    if request.method == 'POST':
        if(request.form):
            link = request.form.get('link')
            try :
                resource = urllib.request.urlopen(link)
                unique_filename = str(uuid.uuid4())
                filename = unique_filename+".jpg"
                img_path = os.path.join(target_img , filename)
                output = open(img_path , "wb")
                output.write(resource.read())
                output.close()
                img = filename

                class_result , prob_result = predict(img_path , model)

                predictions = {
                      "class1":class_result[0],
                        "class2":class_result[1],
                        "class3":class_result[2],
                        "prob1": prob_result[0],
                        "prob2": prob_result[1],
                        "prob3": prob_result[2],
                }

            except Exception as e : 
                print(str(e))
                error = 'This image from this site is not accesible or inappropriate input'

            if(len(error) == 0):
                return  render_template('success.html' , img  = img , predictions = predictions)
            else:
                return render_template('index.html' , error = error) 

            
        elif (request.files):
            file = request.files['file']
            if file and allowed_file(file.filename):
                file.save(os.path.join(target_img , file.filename))
                img_path = os.path.join(target_img , file.filename)
                img = file.filename

                class_result , prob_result = predict(img_path , model)

                predictions = {
                      "class1":class_result[0],
                        "class2":class_result[1],
                        "class3":class_result[2],
                        "prob1": prob_result[0],
                        "prob2": prob_result[1],
                        "prob3": prob_result[2],
                }

            else:
                error = "Please upload images of jpg , jpeg and png extension only"

            if(len(error) == 0):
                return  render_template('success.html' , img  = img , predictions = predictions)
            else:
                return render_template('index.html' , error = error)

    else:
        return render_template('index.html')
    
def gen():
    """Video streaming generator function."""
    while True:
        rval, frame = video.read()
        cv2.imwrite('t.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + open('t.jpg', 'rb').read() + b'\r\n')
        
@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/takeimage', methods = ['POST'])
def takeimage():
    name = request.form['name']
    print(name)
    _, frame = video.read()
    cv2.imwrite(f'images/{name}.jpg', frame)
    return Response(status = 200)
        
if __name__ == "__main__":
    app.run(debug = True)


