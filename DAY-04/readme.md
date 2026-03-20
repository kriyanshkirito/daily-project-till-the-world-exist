
🏗️<h1> Concrete Compressive Strength Prediction</h1>
This is a Machine Learning-powered Web Application built with Flask. The app predicts the compressive strength of concrete based on its composition (cement, water, ash, age, etc.).

🚀 Features
 * Web Interface: A simple, user-friendly HTML form to input concrete ingredients.
 * ML Model: Uses a Linear Regression (or your specific model) trained on the UCI Concrete Strength dataset.
 * Real-time Prediction: Instant feedback on the concrete's strength in MegaPascals (MPa).
🛠️ Tech Stack
 * Backend: Python, Flask
 * Machine Learning: Scikit-Learn, NumPy, Pandas
 * Frontend: HTML5, CSS3
 * Deployment: Pickle (for model serialization)



📂 Project Structure
├── app.py              # Flask Application script
├── model.pkl           # Trained ML Model (Serialized)
├── transformer.pkl     # PowerTransformer/Scaler (Optional)
├── templates/
│   └── index.html      # Frontend Interface
└── static/
    └── css/            # (Optional) Custom Styling

📸 Project Screenshots

Screenshot 1
![alt text](Screenshot/s1.png)

Screenshot 2
![alt text](Screenshot/s2.png)

Screenshot 3
![alt text](Screenshot/s3.png)




⚙️ Installation & Setup
 * Clone the Repository
   git clone https://github.com/your-username/concrete-strength-prediction.git
cd concrete-strength-prediction

 * Create a Virtual Environment (Recommended)
   python -m venv venv
# Activate on Windows:
venv\Scripts\activate
# Activate on Mac/Linux:
source venv/bin/activate

 * Install Dependencies
   pip install flask numpy scikit-learn

 * Run the Application
   python app.py

   Open your browser and go to http://127.0.0.1:5000/.
#📊 How it Works
 * The user enters the quantities of ingredients (Cement, Blast Furnace Slag, Fly Ash, Water, Superplasticizer, Coarse Aggregate, Fine Aggregate, and Age).
 * The data is sent to the Flask backend via a POST request.
 * If applicable, the data is scaled using the transformer.pkl.
 * The model.pkl predicts the strength.
 * The result is rendered back to the index.html page.
