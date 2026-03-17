**# 🛍️ RetailSense — AI-Powered Product Review Intelligence Platform

> **Top 15 Finalists — DAKSH 2026 Amazon AI Hackathon**
> Built in 24 hours by a team of 4. Deployed on AWS. Production-ready architecture.

---

## 📌 What is RetailSense?

RetailSense is a real-time sentiment analysis platform that automatically reads and classifies customer product reviews, giving retailers instant visibility into what their customers love, hate, and want changed — without reading a single review manually.

**The problem:** Small and mid-size retailers in India receive hundreds of reviews daily across platforms. They have no infrastructure to process this feedback at scale.

**Our solution:** A fully automated AWS pipeline that ingests reviews, predicts customer sentiment in under 100ms, stores everything in S3, and surfaces insights through an AI-powered Streamlit dashboard with a built-in Bedrock chatbot.

---

## 🏗️ Architecture

```
Customer (Frontend)
        │
        │  HTTP POST (review text)
        ▼
  ┌─────────────┐
  │   AWS EC2   │  ← Application Server (always-on)
  │  (Flask App)│
  └──────┬──────┘
         │
         │  calls inference.py
         ▼
  ┌─────────────────┐
  │  inference.py   │  ← Preprocesses + Vectorizes text (TF-IDF)
  └──────┬──────────┘
         │
         │  invokes endpoint
         ▼
  ┌──────────────────────┐
  │   AWS SageMaker      │  ← Hosted ML Endpoint
  │  (Logistic           │
  │   Regression Model)  │
  └──────┬───────────────┘
         │
         │  returns prediction
         ▼
  ┌─────────────┐
  │   AWS S3    │  ← Stores review + sentiment label
  └──────┬──────┘
         │
         ├─────────────────────────────┐
         ▼                             ▼
  ┌─────────────────┐     ┌────────────────────────┐
  │    Streamlit    │     │     AWS Bedrock         │
  │    Dashboard    │────▶│  (Amazon Nova Pro)      │
  │  (on EC2)       │     │  AI Chatbot for         │
  │  Retailer View  │     │  Retailer Queries       │
  └─────────────────┘     └────────────────────────┘
```

---

## ⚙️ Tech Stack

| Layer | Technology |
|---|---|
| Application Server | AWS EC2 (t2.micro) |
| ML Endpoint | AWS SageMaker |
| Storage | AWS S3 |
| AI Chatbot | AWS Bedrock — Amazon Nova Pro |
| Dashboard | Streamlit |
| ML Model | Logistic Regression + TF-IDF Vectorization |
| Language | Python 3.x |
| Serialization | Pickle |

---

## 📁 Project Structure

```
retailsense/
│
├── inference.py              # SageMaker entry point (model_fn + predict_fn)
├── dashboard.py              # Streamlit dashboard for retailers
├── model/
│   ├── sentiment_model.pkl   # Trained Logistic Regression model
│   └── vectorizer.pkl        # TF-IDF Vectorizer
├── model.tar.gz              # Packaged model artifact for SageMaker
├── app.py                    # Flask backend running on EC2
├── requirements.txt          # Python dependencies
└── README.md
```

---

## 🤖 How inference.py Works

SageMaker requires two specific functions as its entry point:

```python
def model_fn(model_dir):
    # Called ONCE when the endpoint starts up
    # Loads both pkl files into memory
    model = pickle.load(open(model_dir + "/sentiment_model.pkl", "rb"))
    vectorizer = pickle.load(open(model_dir + "/vectorizer.pkl", "rb"))
    return (model, vectorizer)

def predict_fn(input_data, model):
    # Called on EVERY review submission
    # Vectorizes text → runs model → returns prediction
    model, vectorizer = model
    text = [input_data["review"]]
    X = vectorizer.transform(text)
    prediction = model.predict(X)
    return prediction.tolist()
```

**Request format:**
```json
{
  "review": "The battery life on this product is terrible."
}
```

**Response format:**
```json
["negative"]
```

---

## 💰 Real Cost Breakdown (INR)

| Service | Usage | Monthly Cost |
|---|---|---|
| AWS EC2 t2.micro | 24/7 server | ₹620 – ₹750 |
| AWS SageMaker | ml.t2.medium endpoint | ₹1,400 – ₹1,800 |
| AWS S3 | 5GB review storage | ₹10 – ₹15 |
| AWS Bedrock Nova Pro | 1000 chatbot queries | ₹87 |
| **Total** | | **≈ ₹2,200 – ₹2,800/month** |

**Cost per review analysed: ₹0.003**
**Equivalent human analyst cost: ₹15,000 – ₹25,000/month**

---

## 🚀 Deployment Guide

### Prerequisites
- AWS Account with access to EC2, SageMaker, S3, and Bedrock
- Python 3.8+
- AWS CLI configured

### Step 1 — Package the model
```bash
tar -czvf model.tar.gz sentiment_model.pkl vectorizer.pkl inference.py
aws s3 cp model.tar.gz s3://your-bucket-name/model.tar.gz
```

### Step 2 — Deploy SageMaker Endpoint
```python
import boto3
client = boto3.client('sagemaker')

# Create model
client.create_model(
    ModelName='retailsense-sentiment',
    PrimaryContainer={
        'Image': '<sklearn-container-uri>',
        'ModelDataUrl': 's3://your-bucket-name/model.tar.gz'
    },
    ExecutionRoleArn='<your-sagemaker-role-arn>'
)

# Deploy endpoint
client.create_endpoint_config(
    EndpointConfigName='retailsense-config',
    ProductionVariants=[{
        'VariantName': 'default',
        'ModelName': 'retailsense-sentiment',
        'InstanceType': 'ml.t2.medium',
        'InitialInstanceCount': 1
    }]
)

client.create_endpoint(
    EndpointName='retailsense-endpoint',
    EndpointConfigName='retailsense-config'
)
```

### Step 3 — Launch EC2 and run the app
```bash
# SSH into your EC2 instance
ssh -i your-key.pem ec2-user@your-ec2-public-ip

# Install dependencies
pip install -r requirements.txt

# Run Flask backend
python app.py

# Run Streamlit dashboard (separate terminal)
streamlit run dashboard.py --server.port 8501
```

---

## 🔒 Security & Guardrails

- **Rate limiting** — max 10 review submissions per IP per minute to prevent spam and bot attacks
- **Input validation** — review text is validated before reaching the SageMaker endpoint
- **S3 privacy** — only review text and predicted label are stored; no PII collected
- **IAM policies** — EC2 and dashboard access S3 via scoped IAM roles only
- **Bedrock guardrails** — chatbot is restricted via system prompt to only respond to retailer queries about review data; jailbreak attempts are handled gracefully

---

## 📊 Model Performance

| Metric | Score |
|---|---|
| Accuracy | 86% |
---

## 📈 Scalability

- **S3** scales infinitely with no configuration changes
- **SageMaker** supports auto-scaling policies for traffic spikes
- **EC2** can be placed behind an AWS Application Load Balancer for horizontal scaling
- Architecture supports millions of reviews without structural changes

---

## 👥 Team

Built with ❤️ at DAKSH 2026 — Amazon AI Hackathon

| Name | Role |
|---|---|
| Pranav D | ML Pipeline & SageMaker |
| Sreshta kumar | Backend & EC2 |
| Krithik | Streamlit Dashboard |
| Prithvi iyer | Bedrock Integration |

---

## 📜 License

MIT License — feel free to use, modify, and build on this project.

---

> *"We did not just build a product. We built a business case."***
