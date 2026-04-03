

# 💼  Data Science Job Prediction System (Flask + Machine Learning)

An **end-to-end Machine Learning web application** that predicts whether a candidate is likely to **get a job** based on features like education, experience, and training.

Built using **Flask + Scikit-Learn**, this project demonstrates full pipeline integration from **model training → deployment → user interaction**.

---

## 🚀 Features

* 🤖 ML-based job prediction
* 🌐 Flask web application
* 📊 Real-time prediction from user input
* 🧠 Pre-trained model (`pipe.pkl`)
* 🎯 Clean form-based UI
* ⚡ Instant result display

---

## 📸 Preview

 > 📷 ![alt text](<screenshot/Screen Recording 2026-04-03 233156.gif>)

### s1
> 📷 ![alt text](<screenshot/Screenshot 2026-04-03 233122.png>)

### s2
> 📷 ![alt text](<screenshot/Screenshot 2026-04-03 233222.png>)

### s3
> 📷 ![alt text](<screenshot/Screenshot 2026-04-03 233237.png>)

### s4
> 📷 ![alt text](<screenshot/Screenshot 2026-04-03 233324.png>)

### s5
> 📷 ![alt text](<screenshot/Screenshot 2026-04-03 233506.png>)




> 📷 

---

## 🛠️ Tech Stack

| Category     | Technology            |
| ------------ | --------------------- |
| **Backend**  | Flask (Python)        |
| **ML**       | Scikit-Learn, NumPy   |
| **Frontend** | HTML, CSS, JavaScript |
| **Model**    | Pickle (`pipe.pkl`)   |

---

## 📂 Project Structure

```id="jobstruct"
├── static/
│   ├── style.css
│   └── script.js
├── templates/
│   └── index.html
├── app.py              # Flask backend
├── pipe.pkl            # Trained ML model
├── requirements.txt
└── README.md
```

---

## ⚙️ How It Works

1. User enters details:

   * City Development Index
   * Course Enrollment
   * Education Level
   * Experience
   * Training Hours

2. Data is sent to Flask backend

3. Model processes input using:

   * Preprocessing pipeline
   * Trained ML model

4. Output:

   * ✅ **"Get Job"**
   * ❌ **"Not Get"**

---

## 💻 Core Backend Logic

```python id="jobcore"
@flask_app.route("/predict", methods=["POST"])
def predict():
    raw_features = [x for x in request.form.values()]
    final_features = np.array(raw_features, dtype=object).reshape(1, -1)
    
    prediction = model.predict(final_features)
    
    result = "Get Job" if prediction == 1 else "Not Get"
    
    return render_template("index.html", prediction_text="Student {}".format(result))
```

---

## ▶️ How to Run

### 🔧 Prerequisites

* Python 3.x
* pip

---

### ⚙️ Installation

```bash id="jobrun"
# Clone repository
git clone 

# Navigate to project
cd job-prediction-system

# Install dependencies
pip install -r requirements.txt

# Run Flask app
python app.py
```

---

### 🌐 Open in Browser

```id="joburl"
http://127.0.0.1:5000/
```

---

## 💡 Key Concepts Used

* Flask Routing (`@app.route`)
* Form Handling (`request.form`)
* Model Loading (`pickle`)
* NumPy Array Transformation
* ML Prediction Pipeline

---

## 🚧 Challenges Faced

### ⚠️ Data Type Handling

* Input comes as string from form
* ✔ Converted using NumPy before prediction

---

### ⚠️ Model Integration

* Difficulty loading `.pkl` model
* ✔ Solved using correct `pickle.load()`

---

### ⚠️ Prediction Output

* Model returns numeric value (0/1)
* ✔ Converted into readable text

---

## 📈 Future Improvements

* 📊 Show prediction probability
* 🌍 Deploy on cloud (Render / AWS)
* 🔐 Add user login system
* 🎨 Improve UI/UX design
* 🧠 Use advanced models (XGBoost)

---

## 🤝 Contributing

```bash id="jobcontri"
# Fork repo
# Create branch
# Make changes
# Submit PR
```

---

## 📜 License

This project is open-source under the **MIT License**

---

## ⭐ Support

If you like this project:

⭐ Star the repo
🍴 Fork it
📢 Share it

---

