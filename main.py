from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model


app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


model = load_model('wine_quality_model.h5')
dataset = pd.read_csv('winequality-white.csv')
feature_cols=['fixed acidity', 'citric acid', 'chlorides', 'free sulfur dioxide', 'density', 'sulphates', 'alcohol']
X_train = dataset[feature_cols]

scaler = StandardScaler()
scaler.fit(X_train)

@app.route('/api/predict', methods=['POST'])
@cross_origin()
def predict():

  data = request.json

  input_data = pd.DataFrame([data], columns=feature_cols)
  
  input_data_scaled = scaler.transform(input_data)

  predictions = model.predict(input_data_scaled)

  #Transformar as previsões em classes
  y_pred_class = predictions.argmax(axis=1)

  max_prob_index = predictions.argmax()
  max_prob_class = int(y_pred_class[0])
  max_prob_percentage = str(predictions[0][max_prob_index])


  response = {
    'prediction': max_prob_class,
    'probability': max_prob_percentage 
  }

  return jsonify(response), 200

if __name__ == '__main__':
  app.run()