# 🎯 Resume–Job Match Scoring Engine

An AI tool that scores resumes against job descriptions using NLP.

## Project Structure

```
resume-match-engine/
├── app.py              ← Flask backend API
├── requirements.txt    ← Python dependencies
├── src/
│   └── index.html      ← Frontend UI (open in browser)
└── README.md
```

## Tech Stack / Skills Used

| Category       | Tech                                |
|----------------|-------------------------------------|
| Backend        | Python, Flask, Flask-CORS           |
| NLP            | TF-IDF, Cosine Similarity           |
| Skill Matching | Text embeddings (bag-of-words)      |
| Frontend       | Vanilla HTML/CSS/JS                 |

## ⚙️ Setup & Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the backend API

```bash
python app.py
```

> Server starts at: http://localhost:5000

### 3. Open the frontend

Open `src/index.html` in your browser.  
(Or use VS Code Live Server extension for hot-reload)

## 📡 API Reference

### POST `/api/score`

**Request body:**
```json
{
  "resume": "Your full resume text here...",
  "job_description": "The full job description text here..."
}
```

**Response:**
```json
{
  "score": 72.4,
  "grade": "B",
  "label": "Good Match",
  "similarity": 61.2,
  "skill_match_pct": 80.0,
  "skills": {
    "matched": ["python", "docker", "aws"],
    "missing": ["kubernetes", "tensorflow"],
    "extra": ["react", "node"]
  },
  "recommendations": [
    "Add these missing skills: kubernetes, tensorflow",
    "Highlight your key achievements with metrics."
  ]
}
```

### GET `/api/health`

Returns API status.

## 🧠 How It Works

1. **Preprocessing** — Text is lowercased, punctuation removed, stop words filtered  
2. **TF-IDF Vectorization** — Terms are weighted by frequency & rarity  
3. **Cosine Similarity** — Measures directional angle between resume & JD vectors  
4. **Skill Extraction** — 40+ tech skills detected via keyword matching  
5. **Gap Analysis** — Matched, missing, and extra skills identified  
6. **Final Score** — Weighted blend: 60% text similarity + 40% skill match  

## 🚀 Upgrade Ideas

- [ ] Use sentence-transformers embeddings for deeper semantic matching
- [ ] Add PDF upload support (PyPDF2)
- [ ] Explain *why* score is low (GPT-4 integration)
- [ ] Save results to a database (SQLite / PostgreSQL)
- [ ] Deploy to Render / Railway with a public URL

## 📊 Scoring Grade Scale

| Score | Grade | Label          |
|-------|-------|----------------|
| 80–100 | A   | Excellent Match |
| 65–79  | B   | Good Match      |
| 50–64  | C   | Average Match   |
| 35–49  | D   | Weak Match      |
| 0–34   | F   | Poor Match      |
