from flask import Flask, render_template, request, jsonify
import boto3
import json
import csv
import os
from datetime import datetime

app = Flask(__name__)

# ── Config ──────────────────────────────────────────
SAGEMAKER_ENDPOINT = "sentiment-endpoint"   # your endpoint name
REGION = "us-east-1"                        # your region
REVIEWS_CSV = "cleaned_reviews.csv"         # same file Retail IQ reads

# ── Sample products (replace with DB later) ─────────
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
    try:
        client = boto3.client("sagemaker-runtime", region_name=REGION)
        response = client.invoke_endpoint(
            EndpointName=SAGEMAKER_ENDPOINT,
            ContentType="application/json",
            Body=json.dumps({"review": review_text}),
        )
        result = json.loads(response["Body"].read().decode())
        return {"sentiment": result[0], "intent": "other"}
    except Exception as e:
        print(f"SageMaker error: {e}")
        return {"sentiment": "neutral", "intent": "other"}


def save_review(review: dict):
    file_exists = os.path.exists(REVIEWS_CSV)
    with open(REVIEWS_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "Clothing ID", "Age", "Rating", "Review Text",
            "sentiment", "intent", "Recommended IND",
            "Department Name", "timestamp"
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

    # Get sentiment from SageMaker
    prediction = get_sentiment(review_text)

    # Save to CSV
    save_review({
        "Clothing ID": p["clothing_id"],
        "Age": age,
        "Rating": rating,
        "Review Text": review_text,
        "sentiment": prediction["sentiment"],
        "intent": prediction["intent"],
        "Recommended IND": 1 if rating >= 4 else 0,
        "Department Name": "Tops",
        "timestamp": datetime.now().isoformat(),
    })

    return jsonify({
        "success": True,
        "sentiment": prediction["sentiment"],
        "message": "Review submitted successfully!",
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)