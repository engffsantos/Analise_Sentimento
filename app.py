from flask import Flask, request, jsonify
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = Flask(__name__)

# Carregar o tokenizador e o modelo
tokenizer = AutoTokenizer.from_pretrained("turing-usp/FinBertPTBR")
model = AutoModelForSequenceClassification.from_pretrained("turing-usp/FinBertPTBR")

# Função para classificar o sentimento
def classify_sentiment(text):
    # Tokenizar o texto
    inputs = tokenizer(text, return_tensors="pt")
    # Classificar usando o modelo
    outputs = model(**inputs)
    # Obter a classe predita (0 = negativo, 1 = neutro, 2 = positivo)
    predicted_class = torch.argmax(outputs.logits, dim=1).item()
    # Mapear o índice da classe para o sentimento correspondente
    sentiment = {0: 'Negativo', 1: 'Neutro', 2: 'Positivo'}
    return sentiment[predicted_class]

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    text = data['text']
    sentiment = classify_sentiment(text)
    return jsonify({'sentiment': sentiment})

if __name__ == '__main__':
    app.run(debug=True)
