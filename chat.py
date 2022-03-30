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

#CLASSLA_RESOURCES_DIR = os.getcwd()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intentsBG.json', 'r', encoding="utf8") as json_data:
    intents = json.load(json_data)

FILE = "dataBG.pth"
data = torch.load(FILE)

#print("\n\nCLASSLA DOWNLOAD CHAT.PY\n\n")
#print(os.getcwd())
#classla.download('bg', CLASSLA_RESOURCES_DIR)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()


###################################
# CREATE CHAT FOR DEBUGGING PURPOSES
###################################

bot_name = "HeltiBot"

def get_response(msg):
    #print(msg)
    #pipeline = classla.Pipeline('bg', dir=CLASSLA_RESOURCES_DIR, use_gpu=False, processors='tokenize, lemma')
    #sentenceToLemmatize = pipeline(msg)
    #print(f"SENTE TO LEM: {sentenceToLemmatize}")
    #lemmatizedSentence = [f'{word.lemma}' for sent in sentenceToLemmatize.sentences for word in sent.words]
    #print(f"LEMMA 1 : {lemmatizedSentence}")

    URL = 'https://europe-west6-sharp-maxim-345614.cloudfunctions.net/lemmaZIP'
    r = requests.post(URL, json={'message': msg})
    lemmatizedSentenceString = r.text
    lemmatizedSentence = ast.literal_eval(lemmatizedSentenceString)

    StripOfChar(lemmatizedSentence)

    print("\n\nGET_RESPONSE IN CHAT.PY\n\n")
    print(lemmatizedSentence)

    X = bagOfWords_BG(lemmatizedSentence, all_words)

    # X == -1 means that no corresponding words were found in all_words
    # This ends the current chat query

    #print(f"X: {X}")
    #if X == -1:
    #    return "За жалост не мога да отговоря на този въпрос. Моля, пробвайте отново."

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

    if prob.item() > 0.69:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])
    else:
        return "Не Ви разбрах. Моля, повторете"