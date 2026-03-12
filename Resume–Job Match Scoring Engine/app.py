"""
Resume-Job Match Scoring Engine
================================
Backend API using Flask + NLP (TF-IDF + Cosine Similarity)
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import re
import math
from collections import Counter

app = Flask(__name__)
CORS(app)

# ─── NLP Utilities (no external ML libs needed) ───────────────────────────────

STOP_WORDS = set([
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "used", "that", "this", "these", "those", "i", "you", "he", "she",
    "we", "they", "it", "my", "your", "his", "her", "our", "their",
    "as", "if", "then", "than", "so", "yet", "both", "either", "neither",
    "not", "no", "nor", "just", "very", "also", "too", "more", "most",
    "such", "each", "every", "any", "all", "some", "few", "many", "much"
])

TECH_SKILLS = [
    "python", "java", "javascript", "typescript", "react", "angular", "vue",
    "node", "django", "flask", "fastapi", "spring", "sql", "mysql", "postgres",
    "mongodb", "redis", "aws", "azure", "gcp", "docker", "kubernetes", "git",
    "ci/cd", "machine learning", "deep learning", "nlp", "tensorflow", "pytorch",
    "pandas", "numpy", "scikit-learn", "data analysis", "api", "rest", "graphql",
    "html", "css", "sass", "webpack", "linux", "bash", "agile", "scrum",
    "devops", "microservices", "kafka", "elasticsearch", "spark", "hadoop",
    "tableau", "powerbi", "excel", "r", "scala", "go", "rust", "c++", "c#"
]

def preprocess(text: str) -> list[str]:
    """Tokenize, lowercase, remove stop words."""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s/#+]', ' ', text)
    tokens = text.split()
    return [t for t in tokens if t not in STOP_WORDS and len(t) > 1]

def extract_skills(text: str) -> list[str]:
    """Extract known tech skills from text."""
    text_lower = text.lower()
    found = []
    for skill in TECH_SKILLS:
        if skill in text_lower:
            found.append(skill)
    return found

def tfidf_vector(tokens: list[str], corpus_tokens: list[list[str]]) -> dict:
    """Compute TF-IDF vector for a document."""
    tf = Counter(tokens)
    total = len(tokens) if tokens else 1
    n_docs = len(corpus_tokens)
    
    vector = {}
    for term, count in tf.items():
        tf_score = count / total
        docs_with_term = sum(1 for doc in corpus_tokens if term in doc)
        idf_score = math.log((n_docs + 1) / (docs_with_term + 1)) + 1
        vector[term] = tf_score * idf_score
    return vector

def cosine_similarity(vec_a: dict, vec_b: dict) -> float:
    """Compute cosine similarity between two TF-IDF vectors."""
    common = set(vec_a.keys()) & set(vec_b.keys())
    if not common:
        return 0.0
    
    dot = sum(vec_a[k] * vec_b[k] for k in common)
    mag_a = math.sqrt(sum(v**2 for v in vec_a.values()))
    mag_b = math.sqrt(sum(v**2 for v in vec_b.values()))
    
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)

def analyze_gaps(resume_skills: list, jd_skills: list) -> dict:
    """Find matching, missing, and extra skills."""
    resume_set = set(resume_skills)
    jd_set = set(jd_skills)
    return {
        "matched": sorted(resume_set & jd_set),
        "missing": sorted(jd_set - resume_set),
        "extra": sorted(resume_set - jd_set)
    }

def score_resume(resume_text: str, jd_text: str) -> dict:
    """Main scoring function."""
    resume_tokens = preprocess(resume_text)
    jd_tokens = preprocess(jd_text)
    corpus = [resume_tokens, jd_tokens]
    
    resume_vec = tfidf_vector(resume_tokens, corpus)
    jd_vec = tfidf_vector(jd_tokens, corpus)
    
    similarity = cosine_similarity(resume_vec, jd_vec)
    
    resume_skills = extract_skills(resume_text)
    jd_skills = extract_skills(jd_text)
    gaps = analyze_gaps(resume_skills, jd_skills)
    
    # Skill match bonus (0-20 points)
    skill_score = 0
    if jd_skills:
        skill_score = len(gaps["matched"]) / len(jd_skills)
    
    # Combined score (weighted)
    final_score = round((similarity * 0.6 + skill_score * 0.4) * 100, 1)
    final_score = min(final_score, 100.0)
    
    # Grade
    if final_score >= 80:
        grade, label = "A", "Excellent Match"
    elif final_score >= 65:
        grade, label = "B", "Good Match"
    elif final_score >= 50:
        grade, label = "C", "Average Match"
    elif final_score >= 35:
        grade, label = "D", "Weak Match"
    else:
        grade, label = "F", "Poor Match"
    
    # Recommendations
    recommendations = []
    if gaps["missing"]:
        recommendations.append(f"Add these missing skills: {', '.join(gaps['missing'][:5])}")
    if final_score < 50:
        recommendations.append("Tailor your resume language to match the job description more closely.")
    if len(resume_tokens) < 100:
        recommendations.append("Your resume seems short. Consider adding more detail about your experience.")
    if not gaps["matched"] and jd_skills:
        recommendations.append("No technical skills matched. Highlight relevant technologies prominently.")
    if not recommendations:
        recommendations.append("Great match! Make sure to highlight your key achievements quantitatively.")
    
    return {
        "score": final_score,
        "grade": grade,
        "label": label,
        "similarity": round(similarity * 100, 1),
        "skill_match_pct": round(skill_score * 100, 1),
        "skills": gaps,
        "resume_skill_count": len(resume_skills),
        "jd_skill_count": len(jd_skills),
        "recommendations": recommendations
    }

# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/api/score", methods=["POST"])
def score():
    data = request.json
    resume = data.get("resume", "").strip()
    jd = data.get("job_description", "").strip()
    
    if not resume or not jd:
        return jsonify({"error": "Both resume and job_description are required"}), 400
    
    result = score_resume(resume, jd)
    return jsonify(result)

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "engine": "Resume-Job Match Scoring Engine v1.0"})

if __name__ == "__main__":
    print("🚀 Resume Match Engine running at http://localhost:5000")
    app.run(debug=True, port=5000)
