from flask import Flask, render_template, request, jsonify
import boto3
import json
import csv
import os
from datetime import datetime

app = Flask(__name__)

# ── Config ──────────────────────────────────────────
REGION = "us-east-1"                        
REVIEWS_CSV = "new_reviews.csv"         

# Amazon Bedrock Nova Pro model ID
BEDROCK_MODEL_ID = "amazon.nova-pro-v1:0"

# ── Sample products (can be replaced with DB later) ─────────
PRODUCTS = {
    1: {
        "id": 1,
        "name": "Classic Fit Cotton Shirt",
        "brand": "QuantumWear",
        "price": "$49.99",
        "description": "A timeless classic fit shirt made from 100% premium cotton. Breathable, comfortable and perfect for any occasion.",
        "image": "/static/images/shirt.jpg",
        "clothing_id": 767,
        "sizes": ["XS", "S", "M", "L", "XL"],
        "colors": ["White", "Navy", "Black"],
    },
    2: {
        "id": 2,
        "name": "High-Rise Slim Jeans",
        "brand": "QuantumWear",
        "price": "$89.99",
        "description": "Flattering high-rise slim fit jeans with just the right amount of stretch for all-day comfort.",
        "image": "/static/images/jeans.jpg",
        "clothing_id": 1078,
        "sizes": ["XS", "S", "M", "L", "XL"],
        "colors": ["Light Wash", "Dark Wash", "Black"],
    },
}


# ── Helpers ──────────────────────────────────────────
def get_sentiment(review_text: str) -> dict:
    """
    Call Amazon Bedrock (Nova Pro) to analyze sentiment and intent
    of the given review text. Returns a dict with keys:
      - sentiment  : "positive" | "negative" | "neutral" | "mixed"
      - intent     : e.g. "return" | "compliment" | "complaint" | "other"
      - summary    : one-sentence summary of the review
      - score      : confidence score 0-100
    """
    prompt = f"""You are a sentiment analysis engine for an e-commerce clothing store.
Analyze the following customer review and respond ONLY with a valid JSON object — no explanation, no markdown, no extra text.

Review:
\"\"\"{review_text}\"\"\"

Respond with this exact JSON structure:
{{
  "sentiment": "<positive|negative|neutral|mixed>",
  "intent": "<compliment|complaint|return|size_issue|quality_issue|other>",
  "summary": "<one short sentence summarizing the review>",
  "score": <integer 0-100 representing confidence>,
  "highlights": ["<key phrase 1>", "<key phrase 2>"]
}}"""

    try:
        client = boto3.client("bedrock-runtime", region_name=REGION)

        body = json.dumps({
            "messages": [
                {
                    "role": "user",
                    "content": [{"text": prompt}]
                }
            ],
            "inferenceConfig": {
                "maxTokens": 300,     
                "temperature": 0.1,
                "topP": 0.9
            }
        })

        response = client.invoke_model(
            modelId=BEDROCK_MODEL_ID,
            contentType="application/json",
            accept="application/json",
            body=body
        )

        response_body = json.loads(response["body"].read())
        raw_text = response_body["output"]["message"]["content"][0]["text"].strip()

        # Strip markdown code fences if present
        if raw_text.startswith("```"):
            raw_text = raw_text.split("```")[1]
            if raw_text.startswith("json"):
                raw_text = raw_text[4:]
        raw_text = raw_text.strip()

        result = json.loads(raw_text)

        return {
            "sentiment": result.get("sentiment", "neutral"),
            "intent":    result.get("intent", "other"),
            "summary":   result.get("summary", ""),
            "score":     result.get("score", 0),
            "highlights": result.get("highlights", []),
        }

    except Exception as e:
        print(f"Bedrock error: {e}")
        return {
            "sentiment":  "neutral",
            "intent":     "other",
            "summary":    "",
            "score":      0,
            "highlights": [],
        }


def save_review(review: dict):
    file_exists = os.path.exists(REVIEWS_CSV)
    with open(REVIEWS_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "Clothing ID", "Age", "Rating", "Review Text",
            "sentiment", "intent", "summary", "score",
            "Recommended IND", "Department Name", "timestamp"
        ])
        if not file_exists:
            writer.writeheader()
        writer.writerow(review)


def load_reviews(clothing_id: int) -> list:
    reviews = []
    if not os.path.exists(REVIEWS_CSV):
        return reviews
    with open(REVIEWS_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                if int(float(row["Clothing ID"])) == clothing_id:
                    reviews.append(row)
            except Exception:
                continue
    return sorted(reviews, key=lambda x: x.get("timestamp", ""), reverse=True)


# ── Routes ───────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html", products=PRODUCTS.values())


@app.route("/product/<int:product_id>")
def product(product_id):
    p = PRODUCTS.get(product_id)
    if not p:
        return "Product not found", 404
    reviews = load_reviews(p["clothing_id"])
    avg_rating = round(
        sum(float(r["Rating"]) for r in reviews) / len(reviews), 1
    ) if reviews else 0
    return render_template(
        "product.html",
        product=p,
        reviews=reviews,
        avg_rating=avg_rating,
        review_count=len(reviews),
    )


@app.route("/submit-review", methods=["POST"])
def submit_review():
    data = request.get_json()

    product_id = int(data.get("product_id"))
    p = PRODUCTS.get(product_id)
    if not p:
        return jsonify({"error": "Product not found"}), 404

    review_text = data.get("review_text", "").strip()
    rating = int(data.get("rating", 3))
    age = int(data.get("age", 30))

    if not review_text:
        return jsonify({"error": "Review text is required"}), 400

    # ── Analyze sentiment with Amazon Bedrock Nova Pro ──
    prediction = get_sentiment(review_text)

    # ── Save to CSV ──────────────────────────────────────
    save_review({
        "Clothing ID":    p["clothing_id"],
        "Age":            age,
        "Rating":         rating,
        "Review Text":    review_text,
        "sentiment":      prediction["sentiment"],
        "intent":         prediction["intent"],
        "summary":        prediction["summary"],
        "score":          prediction["score"],
        "Recommended IND": 1 if rating >= 4 else 0,
        "Department Name": "Tops",
        "timestamp":      datetime.now().isoformat(),
    })

    return jsonify({
        "success":    True,
        "sentiment":  prediction["sentiment"],
        "intent":     prediction["intent"],
        "summary":    prediction["summary"],
        "score":      prediction["score"],
        "highlights": prediction["highlights"],
        "message":    "Review submitted successfully!",
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)
