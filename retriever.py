import os
import json
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

#State
faqs = None
faq_matrix = None
client = None

MODEL = "text-embedding-3-small"


#Init OpenAI client
def init_client():
    global client
    load_dotenv(override=True)

    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError("Missing OPENAI_API_KEY")

    client = OpenAI(api_key=key)


#Embedding function
def embed(text: str):
    if client is None:
        init_client()

    res = client.embeddings.create(
        model=MODEL,
        input=text
    )
    return res.data[0].embedding


#Load FAQs and embeddings
def load_faqs():
    global faqs, faq_matrix

    with open("faqs.json", "r") as f:
        data = json.load(f)

    for faq in data:
        #Embed question and answer together
        faq["embedding"] = embed(
            f"Question: {faq['question']} Answer: {faq['answer']}"
        )

    faqs = data
    faq_matrix = np.array([faq["embedding"] for faq in faqs])

    return faqs


#Init FAQs
def init_faqs():
    global faqs, faq_matrix
    if faqs is None or faq_matrix is None:
        load_faqs()


#Retrieval
def find_best_match(user_question: str, threshold: float = 0.30):

    if faqs is None or faq_matrix is None:
        init_faqs()

    user_question_clean = user_question.lower().strip()

    query_vec = np.array(embed(user_question_clean)).reshape(1, -1)

    scores = cosine_similarity(query_vec, faq_matrix)[0]

    best_idx = np.argmax(scores)
    best_score = scores[best_idx]

    if best_score >= threshold:
        return faqs[best_idx]["answer"], best_score

    return None, best_score