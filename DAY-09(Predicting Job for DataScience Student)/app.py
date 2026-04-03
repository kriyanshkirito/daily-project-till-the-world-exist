import numpy as np
from flask import Flask ,request,render_template
import pickle

flask_app=Flask(__name__)

def safe_log1p(x):
    return np.log1p(x.astype(float))

model=pickle.load(open("pipe.pkl","rb"))

@flask_app.route("/")
def Home():
    return render_template("index.html")
@flask_app.route("/predict",methods=["POST"])
def predict():
    raw_features=[x for x in request.form.values()]
    final_features=np.array(raw_features,dtype=object).reshape(1,-1)
    prediction=model.predict(final_features)
    result="Get Job" if prediction==1 else "Not Get"  #we can change model output into String
    return render_template("index.html",prediction_text="Student {}".format(result))


if __name__=="__main__":
    flask_app.run(debug=True)
