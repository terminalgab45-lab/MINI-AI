from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from flask import Flask, render_template, request
import random
import re

app = Flask(__name__)

# -------------------------
# UTILS
# -------------------------
def pulisci(testo):
    testo = testo.lower()
    testo = re.sub(r"[^a-zàèéìòù ]", "", testo)
    return testo

emoji = random.choice(["😀", "🐱", "🥳", "😁", "🥲"])

ultimo_intento = None

# -------------------------
# DATASET
# -------------------------
data = [
    ("ciao", "saluto"),
    ("buongiorno", "saluto"),
    ("buonasera", "saluto"),
    ("buonanotte", "saluto"),

    ("come stai", "stato"),
    ("come va", "stato"),
    ("come ti senti", "stato"),
    ("come butta", "stato"),
    ("come tira laria", "stato"),

    ("aiuto", "supporto"),
    ("assistenza", "supporto"),
    ("non ce la faccio piu", "supporto"),
    ("aiutami", "supporto"),
    ("assistimi", "supporto"),

    ("mi sento bene", "stato_bene"),
    ("sto bene", "stato_bene"),
    ("sono contenta", "stato_bene"),
    ("sono felice", "stato_bene"),
    ("sono contento", "stato_bene"),

    ("triste", "stato_male"),
    ("mi sento male", "stato_male"),
    ("sono arrabbiato", "stato_male"),
    ("sono arrabbiata", "stato_male"),
    ("sono triste", "stato_male"),

    ("arrivederci", "saluto_fine"),
    ("addio", "saluto_fine"),
    ("a presto", "saluto_fine"),
    ("a tra poco", "saluto_fine"),
    ("ci rivedremo", "saluto_fine"),

    ("chi ti ha creato", "creator"),
    ("cosa ti ha creato", "creator"),
    ("chi è il tuo creatore", "creator"),
    ("voglio sapere chi ti ha creato", "creator"),
    ("chi ti ha sviluppato", "creator")
    
]












vectorizer = CountVectorizer()
X = vectorizer.fit_transform([pulisci(t) for t, _ in data])
y = [classe for _, classe in data]

modello = MultinomialNB()
modello.fit(X, y)











risposte = {
    "saluto": [
        "Ciao! 👋 Come posso aiutarti?",
        "Hey! 😄 Dimmi pure",
        "Buongiorno 🌞",
        "ciao anche a te 😆",
        "こんにちは (buongiorno)　😇"
    ],
    "saluto_fine": [
        "arrivederci 👋",
        "a presto 🥳",
        "ciao 🙂",
        "a dopo 🔥"
    ],
    "stato": [
        "Sto bene grazie 😜",
        "Alla grande 💪",
        "Tutto ok 😎",
        "tutto bene 😁"
    ],
    "supporto": [
        "Dimmi pure 🔥",
        "Come posso aiutarti?",
        "Sono qui ✌️",
        "certo dimmi pure 😜"
    ],
    "stato_bene": [
        "mi fa piacere 😀",
        "che bello che sei felice 😁",
        "sono contento che tu sia allegro 😜",
        "meno male che sei felice 🙂"
    ],
    "stato_male": [
        "mi dispiace 😧",
        "che peccato ☹️",
        "non preoccuparti 😞",
        "spero starai meglio 😦"
    ],
    "creator": [
        "sono stato creato da @Terminalgab45 😎",
        "il mio creatore è @Terminalgab45 😜",
        "il mio creatore è @Terminalgab45 😁",
        "mi ha creato @Terminalgab45 😇"
    ],
    "non_capito": [
        "non ho capito bene, puoi ripetere 😴",
        "non capisco puoi ripetere di nuovo ? 😜",
        "non capisco bene potresti ripetere ? 😀",
        "mi dispiace, non ho capito bene ✨"
    ]    
}








def predici(frase):
    frase = pulisci(frase)
    frase_v = vectorizer.transform([frase])

    probabilita = modello.predict_proba(frase_v)[0]
    max_prob = max(probabilita)

    intento = modello.classes_[probabilita.argmax()]


    if max_prob < 0.2:
        return "non_capito"

    return intento
    









@app.route("/", methods=["GET", "POST"])
def index():
    global ultimo_intento
    risposta = ""



    
    if request.method == "POST":
        user_text = request.form["testo"]
        intento = predici(user_text)

        if intento == "stato" and ultimo_intento == "saluto":
            risposta = "bene 😄 Dimmi pure se posso aiutarti."
            
        elif intento == "supporto" and ultimo_intento == "stato":
            risposta = "Certo 🔥 che problema hai?"


        elif intento == "saluto" and ultimo_intento == "saluto":
            risposta = "ci rivediamo  😂"
            
        else:
            risposta = random.choice(risposte[intento])

        ultimo_intento = intento

    return render_template("index.html", risposta=risposta, emoji=emoji)








if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000, debug=True)