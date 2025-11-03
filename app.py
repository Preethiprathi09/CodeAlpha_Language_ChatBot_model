from flask import Flask, render_template, request, jsonify
import json
import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import datetime
import random

nltk.download('punkt')

app = Flask(__name__)

# Load FAQ data
with open("faq_data.json", "r") as file:
    faq_data = json.load(file)

# Extract questions
faq_questions = [item["question"] for item in faq_data]

# Vectorize the questions
vectorizer = TfidfVectorizer()
faq_vectors = vectorizer.fit_transform(faq_questions)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def chatbot_response():
    user_input = request.form["msg"].lower()

    # Handle dynamic questions
    if "time" in user_input:
        return jsonify({"response": f"The current time is {datetime.datetime.now().strftime('%I:%M %p')}."})
    elif "date" in user_input or "day" in user_input:
        return jsonify({"response": f"Today is {datetime.datetime.now().strftime('%A, %B %d, %Y')}."})

    # Calculate similarity
    user_vec = vectorizer.transform([user_input])
    similarities = cosine_similarity(user_vec, faq_vectors)
    index = similarities.argmax()

    # If not confident, handle unknowns
    if similarities[0][index] < 0.3:
        unknown_responses = [
            "I'm not sure I understand that, Noor 🌸",
            "Hmm… I don’t know that yet, but I can learn!",
            "Could you please rephrase your question?"
        ]
        return jsonify({"response": random.choice(unknown_responses)})

    return jsonify({"response": faq_data[index]["answer"]})

if __name__ == "__main__":
    app.run(debug=True)
