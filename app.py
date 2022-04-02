from gevent import monkey
monkey.patch_all()

from flask import Flask, render_template, request, jsonify
#from langdetect import detect, detect_langs
from chat import get_response, requests, ast
from scout_apm.flask import ScoutApm
#import classla
#import langid
#from langid.langid import LanguageIdentifier, model

app = Flask(__name__)

ScoutApm(app)

## ERROR AND APP MANAGEMENT -> USING SCOUT ADD-ON
# Scout Settings
app.config["SCOUT_NAME"] = "HeltiScout"
# Utilize Error Monitoring:
app.config["SCOUT_ERRORS_ENABLED"] = True


@app.get("/")
def index_get():
    return render_template("base.html")

@app.post("/predict")
def predict():
    text = request.get_json().get("message")
    # TODO: check if text is valid
    if len(text) < 2 or text in ['?', '.', '!', '(', ')', '{', '}', '']:
        return jsonify({"answer": "Моля, въведете валидно съобщение."})

    URL = 'https://europe-west6-sharp-maxim-345614.cloudfunctions.net/lang-detect'
    r = requests.post(URL, json={'message': text})
    textDict = ast.literal_eval(r.text)


    if textDict["language"] != 'bg' and textDict["language"] != 'ru' and textDict["confidence"] > 0.85:
        print("\nText: {}".format(textDict["input"]))
        print("Confidence: {}".format(textDict["confidence"]))
        print("Language: {}".format(textDict["language"]))
        return jsonify({"answer": "Please, enter a valid message in Bulgarian or switch to the English version of the website."})
    print("Text: {}".format(textDict["input"]))
    print("Confidence: {}".format(textDict["confidence"]))
    print("Language: {}".format(textDict["language"]))


    response = get_response(text)
    message = {"answer": response}
    return jsonify(message)

if __name__ == "__main__":
    app.run(debug=False, use_reloader=False)