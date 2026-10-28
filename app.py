
from flask import Flask, request, jsonify, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

app = Flask(__name__)


def split_sentences(text):
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    return [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+", text)
        if s.strip()
    ]


def analyze_text(text, threshold=0.80):
    sentences = split_sentences(text)
    matches = []

    normalized = [
        re.sub(r"\W+", "", s.lower(), flags=re.UNICODE)
        for s in sentences
    ]

    # Find exact duplicate sentences
    exact_pairs = set()

    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            if normalized[i] and normalized[i] == normalized[j]:
                matches.append({
                    "sentence_1_number": i + 1,
                    "sentence_1": sentences[i],
                    "sentence_2_number": j + 1,
                    "sentence_2": sentences[j],
                    "similarity": 100.0,
                    "type": "Exact duplicate"
                })
                exact_pairs.add((i, j))

    # Find nearly identical sentences
    if len(sentences) >= 2:
        try:
            matrix = TfidfVectorizer(
                analyzer="char",
                ngram_range=(2, 4)
            ).fit_transform(sentences)

            scores = cosine_similarity(matrix)

            for i in range(len(sentences)):
                for j in range(i + 1, len(sentences)):
                    if (i, j) in exact_pairs:
                        continue

                    score = float(scores[i, j])

                    if score >= threshold:
                        matches.append({
                            "sentence_1_number": i + 1,
                            "sentence_1": sentences[i],
                            "sentence_2_number": j + 1,
                            "sentence_2": sentences[j],
                            "similarity": round(score * 100, 2),
                            "type": "Similar sentences"
                        })
        except ValueError:
            pass

    matches.sort(key=lambda x: x["similarity"], reverse=True)

    return {
        "total_sentences": len(sentences),
        "exact_duplicates": sum(
            x["type"] == "Exact duplicate" for x in matches
        ),
        "similar_pairs": sum(
            x["type"] == "Similar sentences" for x in matches
        ),
        "matches": matches
    }


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "")

    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "Please enter some text."}), 400

    try:
        threshold = float(data.get("threshold", 0.80))
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid similarity threshold."}), 400

    if not 0 <= threshold <= 1:
        return jsonify({"error": "Threshold must be between 0 and 1."}), 400

    if len(text) > 500_000:
        return jsonify({"error": "Text is too long."}), 413

    return jsonify(analyze_text(text, threshold))


if __name__ == "__main__":
    app.run(debug=True)
