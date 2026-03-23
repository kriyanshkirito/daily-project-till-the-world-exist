import numpy as np
from flask import Flask ,request,render_template
import pickle

flask_app=Flask(__name__)
model=pickle.load(open("clf.pkl","rb"))

@flask_app.route("/")
def Home():
    return render_template("index.html")
@flask_app.route("/predict",methods=["POST"])
def predict():
    float_features=[float(x) for x in request.form.values()]
    features=[np.array(float_features)]
    prediction=model.predict(features)
    # result="Purchase" if prediction==1 else "Not Purchase"  #we can change model output into String
    return render_template("index.html",prediction_text=" Student Will Score {}".format(prediction))


if __name__=="__main__":
    flask_app.run(debug=True)
