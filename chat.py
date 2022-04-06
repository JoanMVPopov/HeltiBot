import random
import json
import sys
import requests
import ast

import torch
import os

from model import NeuralNet
from nltk_utils import bagOfWords_BG
from bot_utils import StripOfChar

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intentsBGv2.json', 'r', encoding="utf8") as json_data:
    intents = json.load(json_data)

FILE = "dataBGv8.pth"  # using v8 of BoW model
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

###############################################
# ESTABLISH HUGGING FACE API FOR TEXT SIM MODEL

API_TOKEN = os.environ.get('API_TOKEN')
headers = {"Authorization": f"Bearer {API_TOKEN}"}
API_URL = "https://api-inference.huggingface.co/models/symanto/sn-xlm-roberta-base-snli-mnli-anli-xnli"

def query(payload):
    data = json.dumps(payload)
    response = requests.request("POST", API_URL, headers=headers, data=data)
    return json.loads(response.content.decode("utf-8"))

###############################################

bot_name = "HeltiBot"

def get_response(msg):
    URL = 'https://europe-west6-sharp-maxim-345614.cloudfunctions.net/lemmatizer'  # lemmatizer URL
    r = requests.post(URL, json={'message': msg})
    lemmatizedSentenceString = r.text
    lemmatizedSentence = ast.literal_eval(lemmatizedSentenceString)

    StripOfChar(lemmatizedSentence)

    print(lemmatizedSentence)

    X = bagOfWords_BG(lemmatizedSentence, all_words)

    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    print(f"TAG: {tag}")

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    print(f"PROB: {prob}")
    print(f"PROB.ITEM: {prob.item()}")

    if prob.item() > 0.30:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                samplePatterns = random.sample(intent['patterns'], 5)  # choose 5 random response sentences for compar.
                data = query(
                    {
                        "inputs": {
                            "source_sentence": msg,
                            "sentences": samplePatterns
                        }
                    })


                if type(data) != list:  # if hugging face has not loaded
                    return random.choice(intent['responses'])  # rely on BoW model
                else:
                    for probability in data:
                        if probability > 0.40:  # assure a proper answer
                            return random.choice(intent['responses'])
                    return "Не Ви разбрах. Моля, перифразирайте."  # filter out unrelated messages

    else:
        return "Не Ви разбрах. Моля, перифразирайте."
