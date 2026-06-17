#RIVERSIDE BOOKS CHATBOT baseline

import json
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load FAQs
def load_faqs(file_path='faqs.json'):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print("Error: faqs.json not found!")
        sys.exit(1)

faqs = load_faqs()

# Prepare data
questions = [faq['question'] for faq in faqs]
answers = {faq['question']: faq['answer'] for faq in faqs}

# TF-IDF setup
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(questions)

def find_best_match(user_question, threshold=0.25):
    user_tfidf = vectorizer.transform([user_question])
    similarities = cosine_similarity(user_tfidf, tfidf_matrix).flatten()
    best_idx = similarities.argmax()
    best_score = similarities[best_idx]
    
    if best_score >= threshold:
        return questions[best_idx], answers[questions[best_idx]], best_score
    return None, None, best_score

# Main chatbot loop
print("=== Riverside Books Chatbot ===")
print("Ask me anything! (type 'quit' or 'exit' to stop)\n")

while True:
    try:
        user_input = input("You: ").strip()
        
        if user_input.lower() in ['quit', 'exit']:
            print("Bot: Goodbye! Come visit us at Riverside Books soon!")
            break
        
        if not user_input:
            continue
        
        match_q, answer, score = find_best_match(user_input)
        
        if answer:
            print(f"Bot: {answer}")
            # print(f"   (Confidence: {score:.2f})")  # Uncomment for debugging
        else:
            print("Bot: Sorry, I don’t know that one - please ask a member of staff.")
            
    except KeyboardInterrupt:
        print("\nBot: Goodbye!")
        break