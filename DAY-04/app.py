import numpy as np
from flask import Flask ,request,render_template
from sklearn.preprocessing import PowerTransformer
import pickle

flask_app=Flask(__name__)
model=pickle.load(open("model.pkl","rb"))
transformer = pickle.load(open("transformer.pkl", "rb"))

@flask_app.route("/")
def Home():
    return render_template("index.html")
@flask_app.route("/predict",methods=["POST"])
def predict():
    float_features=[float(x) for x in request.form.values()]
    features=np.array(float_features).reshape(1, -1)
    transformed_features=transformer.transform(features)
    prediction=model.predict(transformed_features)
    return render_template("index.html", prediction_text="Predicted Concrete Strength is {:.2f} MPa".format(prediction[0]))


if __name__=="__main__":
    flask_app.run(debug=True)
