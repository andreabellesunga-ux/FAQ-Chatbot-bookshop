#HOW TO RUN THE BOT
To run the chatbot, first install dependencies using pip install -r requirements.txt. The chatbot uses OpenAI embeddings for semantic retrieval, so an API key must be provided as an environment variable called OPENAI_API_KEY. This should not be hardcoded and is loaded securely at runtime by retriever.py.The key is excluded from version control via .gitignore. The system includes two interfaces: a command-line chatbot, which can be run using python riverside_books_chatbot.py, and a Streamlit web application, which can be launched with streamlit run chatbot_app.py. 

#DESIGN DECISIONS, TRADEOFFS AND ASSUMPTIONS 

**Matching approach chosen and why**

I implemented a semantic retrieval approach using OpenAI's text-embedding-3-small model combined with cosine similarity. Each FAQ question is converted into a vector embedding. At query time: the user input is embedded, cosine similarity is computed against all stored FAQ embeddings and the most similar FAQ is returned

Why this approach?
- It understands semantic meaning, not just keyword overlap 
- It handles paraphrased questions or indirect questions well
- It performs better than TF-IDF for natural language queries 

A TF-IDF model was used as a baseline before upgrading to embeddings. 

**Handling "No Good Answer" cases **

The system uses a similarity threshold mechanism. 

After computing embeddings for both the user's query and each FAQ entry (question and answer concatendated) the system computes cosine similarity scores between the query vector and all stored FAQ vectors. The final prediction is based on the highest similar score: 
- If the best score is greater than or equal to a threshold (0.30), the corresponding FAQ is returned. If the score is below this threshold, the system returns no match (This threshold was chosen to avoid random weak matches while still picking up paraphrased questions)

Instead, it returns a fallback response: 
"Sorry, I don't know that one - please ask a member of staff."

This ensures the chatbot does not return weak or incorrect matches and prioritises accuracy over guessing. 

**Trade offs**

Accuracy vs cost: 
- Embeddings significantly improve semantic understanding in comparison to TF-IDF. However, they require API calls to OpenAI, which introduces cost per request 

Latency: 
- TF-IDF is fast and runs locally. Embedding-based retrieval is slower due to API calls 
- Precomputing FAQ embeddings helps reduce repeated computation, but query embedding is still required
- OpenAI API calls adds time per query 

Scalability: 
- The system works well for small FAQ datasets but is not yet optimised for large-scale production use

Reliability: there is no hallucination risk (retrieval only) 

**What I'd do differently with more time**

With more time, I would improve the robustness and scalability of the retrieval system. One key improvement would be to introduce a proper vector database (such as FAISS) instead of performing linear similarity search, which would make retrieval more efficient as the FAQ dataset grows. I would also implement caching for embeddings to reduce repeated API calls and improve response time, especially for frequently asked questions. Also, I would build a small evaluation set to test different similarity thresholds more systematically rather than relying on manual tuning. This would help optimise the balance between accuracy and coverage. Finally, I would explore adding a lightweight reranking step or hybrid model (TF-IDF with embeddings) to improve retrieval accuracy in edge cases where semnatic similarity alone is not sufficient. 

**Assumptions**

- The FAQ dataset is clean 
- Each question has a single correct answer
- Semantic similarity is sufficient without conversational context 
- OpenAI embeddings are stable enough for precomputing FAQ embeddings
- Users will accept a fallback response when no match is found 
