from flask import Flask, render_template, request
import pickle
import numpy as np


app = Flask(__name__)
#load the model
model = pickle.load(open('panic.pkl','rb'))

@app.route('/')
def home():
    result = ''
    return render_template('index.html',**locals())

@app.route('/predict',methods = ['POST'])        
def predict():
    # Participant_ID = request.form['Paticipant ID']
    Age= request.form['Age']
    Gender = request.form['Gender']
    # FamilyHistory = request.form['Family History']
    # PersonalHistory = request.form['Personal History']
    # CurrentStressors = request.form['Current Stressors']
    # SocialSupport = request.form['Social Support']
    # LifestyleFactors = request.form['Lifestyle Factors']
    Symptoms = request.form['Symptoms']
    Severity = request.form['Severity']
    
    result = model.predict([[float(Age),Gender,Symptoms,Severity]])[0]
    # return render_template('index.html',**locals())

    p = np.array(model.transform(result))
    p = p.astype(np.float32)
    
    prediction = model.predict(p)
    
    prediction = prediction > 0.5
    
    if (prediction == [[False]]):
        text = "Can suffer from panic disorder"
    else:
        text = "good mental health"
        
    return render_template("index.html",prediction_text = text )

    
    
if __name__ == "__main__":
    app.run(debug = True)
