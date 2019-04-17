#way to use
# python3 run_keras_api_server.py
# send request from application or requets package
# curl -X POST -F image=@dog.jpg 'http://localhost:5000/predict'


from keras.applications import ResNet50, imagenet_utils
from keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import flask
import io

#initalize our Flask application keras
app = flask.Flask(__name__)
model = None

def load_model():
    #use Trained Model ResNet50 To prodict
    global model
    model = ResNet50(weights='imagenet')

def prepare_image(image, target):
    #convert image to RGB
    if image.mode != "RGB":
        image = image.convert("RGB")

    #resize image and preprocessing
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)
    #return image
    return image

#setting Flask app
@app.route("/predict", methods=["POST"])
def predict():

    data = {}

    #ensure image was property uploaded to our enpoint
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            # read the image in PIL format
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))
            # preprocess the image and prepare it for classification
            image = prepare_image(image, target=(224, 224))
            # classify the input image and then initialize the list
            # of predictions to return to the client
            preds = model.predict(image)
            results = imagenet_utils.decode_predictions(preds)

            #loop over the results and add them to the list of
            #returned predictions
            for(imagenetID, label, prob) in results[0]:
                r = {"label":label, "probability": float(prob)}
                data["predictions"] = r
            #indicate that the request was a success
            data["success"] = True
    #return the data as JSON response
    return flask.jsonify(data)

#main fucntion
if __name__ == "__main__":
    print((" *Loading Keras model and Flask stating Server"))
    print("Wait untill server fully started")
    load_model()
    app.run(debug = False, threaded = False)
