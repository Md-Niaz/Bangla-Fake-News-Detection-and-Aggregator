from flask import Flask, jsonify, request
from flask_cors import CORS
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load pre-trained BERT model and tokenizer
model = BertForSequenceClassification.from_pretrained('./bert_fake_news_model')
tokenizer = BertTokenizer.from_pretrained('./bert_fake_news_model')

# Put the model in evaluation mode
model.eval()

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        content = data.get("content", "")
        if not content:
            return jsonify({"error": "No content provided"}), 400

        # Preprocess content (same as during training)
        content = content.lower()
        content = ''.join(e for e in content if e.isalnum() or e.isspace())

        # Tokenize input text
        inputs = tokenizer(content, return_tensors="pt", truncation=True, padding=True, max_length=512)

        # Predict using the BERT model
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)

        # Extract prediction and confidence
        predicted_class = torch.argmax(probabilities, dim=-1).item()
        confidence = probabilities.max().item() * 100

        result = "Authentic" if predicted_class == 1 else "Fake"

        return jsonify({"result": result, "confidence": round(confidence, 2)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/crawl", methods=["GET"])
def crawl():
    options = Options()
    options.add_argument("--headless")
    driver = webdriver.Chrome(options=options)

    url = "https://www.bd-pratidin.com/"
    list_of_articles = []

    try:
        driver.get(url)
        links = WebDriverWait(driver, 10).until(
            EC.presence_of_all_elements_located((By.TAG_NAME, "a"))
        )
        hrefs = [link.get_attribute("href") for link in links if link.get_attribute("href")]
        for href in hrefs:
            splitted_url = href.split('/')
            if len(splitted_url) == 8:
                list_of_articles.append(href)
        data = []

        count = 0
        for article in list_of_articles:
            count += 1
            driver.get(article)
            try:
                title = WebDriverWait(driver, 5).until(
                    EC.presence_of_element_located((By.CLASS_NAME, "n_head"))
                ).text
                data.append({"url": article, "title": title})
            except Exception as e:
                data.append({"url": article, "title": "Failed to fetch title"})
            if count == 10:  # Fetch 10 articles
                break
        return jsonify(data)
    finally:
        driver.quit()


if __name__ == "__main__":
    app.run(debug=True)














# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import torch
# from transformers import BertTokenizer, BertForSequenceClassification

# # Initialize Flask app
# app = Flask(__name__)
# CORS(app)

# # Load pre-trained BERT model and tokenizer
# model = BertForSequenceClassification.from_pretrained('./bert_fake_news_model')
# tokenizer = BertTokenizer.from_pretrained('./bert_fake_news_model')

# # Put the model in evaluation mode
# model.eval()

# @app.route("/predict", methods=["POST"])
# def predict():
#     try:
#         # Get the input data
#         data = request.get_json()
#         content = data.get("content", "")
#         if not content:
#             return jsonify({"error": "No content provided"}), 400

#         # Preprocess content (same as during training)
#         content = content.lower()
#         content = ''.join(e for e in content if e.isalnum() or e.isspace())

#         # Tokenize input text
#         inputs = tokenizer(content, return_tensors="pt", truncation=True, padding=True, max_length=512)

#         # Predict using the BERT model
#         with torch.no_grad():
#             outputs = model(**inputs)
#             logits = outputs.logits
#             probabilities = torch.nn.functional.softmax(logits, dim=-1)

#         # Extract prediction and confidence
#         predicted_class = torch.argmax(probabilities, dim=-1).item()
#         confidence = probabilities.max().item() * 100

#         # Map the prediction to label
#         result = "Authentic" if predicted_class == 1 else "Fake"

#         # Return the result as JSON
#         return jsonify({"result": result, "confidence": round(confidence, 2)})

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# if __name__ == "__main__":
#     app.run(debug=True)



















