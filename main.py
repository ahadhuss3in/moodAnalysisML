# backend/main.py
from fastapi import FastAPI
from pydantic import BaseModel
from collections import Counter
from sentence_transformers import SentenceTransformer, util
import numpy as np

app = FastAPI(title="Mental Health Semantic Classifier")

class XAIInput(BaseModel):
    keywords: list[str]

# -------------------------
# Representative sentences for each group
# -------------------------
group_descriptions = {
    0: "anxiety, stress, fatigue, worry, panic",
    1: "sadness, loneliness, isolation, crying, grief",
    2: "happiness, positive mindset, energy, motivation, recovery",
    3: "anger, frustration, irritability, aggression",
    4: "confusion, indecision, forgetfulness, uncertainty",
    5: "calm, relaxed, peaceful, mindfulness, meditation"
}

explanations = {
    0: "High anxiety and fatigue pattern",
    1: "Social withdrawal and sadness traits",
    2: "Positive mindset and recovery-oriented",
    3: "Anger and irritability issues",
    4: "Confusion and indecision pattern",
    5: "Calm, mindful, and relaxed state"
}

# -------------------------
# Load pre-trained embedding model
# -------------------------
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Precompute group embeddings
group_embeddings = {g: embedder.encode(desc, convert_to_tensor=True)
                    for g, desc in group_descriptions.items()}

# -------------------------
# Prediction endpoint
# -------------------------
@app.post("/predict")
def predict_subgroup(data: XAIInput):
    if not data.keywords:
        return {"error": "No keywords provided."}

    group_votes = []

    for word in data.keywords:
        word_emb = embedder.encode(word, convert_to_tensor=True)
        # Compute similarity with all groups
        similarities = {g: util.cos_sim(word_emb, emb).item() for g, emb in group_embeddings.items()}
        # Choose the group with highest similarity
        predicted_group = max(similarities, key=similarities.get)
        group_votes.append(predicted_group)

    # Majority vote
    vote_count = Counter(group_votes)
    final_group = vote_count.most_common(1)[0][0]

    return {
        "predicted_subgroup": final_group,
        "explanation": explanations[final_group],
        "votes": dict(vote_count)
    }

@app.get("/")
def root():
    return {"status": "Mental Health Semantic Classifier running"}
