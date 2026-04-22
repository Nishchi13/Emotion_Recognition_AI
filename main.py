from fastapi import FastAPI, Depends, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from transformers import pipeline
from sqlalchemy import create_engine, Column, Integer, String, Float, desc
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

# --- 1. DATABASE SETUP ---
DATABASE_URL = "sqlite:///./emotion_history.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class EmotionRecord(Base):
    __tablename__ = "emotion_results"
    id = Column(Integer, primary_key=True, index=True)
    text = Column(String)
    emotion = Column(String) # Stores: joy, anger, sadness, etc.
    confidence = Column(Float)

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try: yield db
    finally: db.close()

# --- 2. EMOTION MODEL SETUP ---
# Using a model specifically trained for 7+ emotions (Joy, Anger, Fear, Sadness, etc.)
# This model is open and does not require authentication
emotion_classifier = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion")


app = FastAPI()

class TextRequest(BaseModel):
    text: str

# --- 3. API ROUTES ---

@app.post("/analyze/single")
def analyze_and_save(request: TextRequest, db: Session = Depends(get_db)):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text is empty")
    
    # Run Emotion Analysis
    result = emotion_classifier(request.text)[0]
    
    db_record = EmotionRecord(
        text=request.text,
        emotion=result['label'],
        confidence=round(result['score'], 4)
    )
    db.add(db_record)
    db.commit()
    db.refresh(db_record)
    return db_record

@app.get("/history")
def get_history(db: Session = Depends(get_db)):
    return db.query(EmotionRecord).order_by(desc(EmotionRecord.id)).limit(10).all()

@app.delete("/history/clear")
def clear_history(db: Session = Depends(get_db)):
    db.query(EmotionRecord).delete()
    db.commit()
    return {"message": "Deleted"}

@app.get("/", response_class=HTMLResponse)
def get_ui():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Emotion AI Dashboard</title>
        <style>
            body { font-family: 'Segoe UI', sans-serif; max-width: 800px; margin: 40px auto; background: #f8f9fa; padding: 20px; }
            .card { background: white; padding: 25px; border-radius: 15px; box-shadow: 0 10px 25px rgba(0,0,0,0.05); margin-bottom: 25px; }
            textarea { width: 100%; height: 100px; padding: 15px; box-sizing: border-box; border-radius: 10px; border: 1px solid #ddd; font-size: 16px; transition: 0.3s; }
            textarea:focus { border-color: #4A90E2; outline: none; }
            button { background: #4A90E2; color: white; border: none; padding: 12px 20px; border-radius: 8px; cursor: pointer; width: 100%; font-size: 16px; margin-top: 15px; font-weight: bold; }
            .clear-btn { background: #ff4757; width: auto; font-size: 12px; margin-top: 20px; }
            table { width: 100%; border-collapse: collapse; margin-top: 20px; }
            th, td { text-align: left; padding: 15px; border-bottom: 1px solid #eee; }
            
            /* Emotion Badges */
            .badge { padding: 5px 12px; border-radius: 20px; font-size: 12px; text-transform: uppercase; font-weight: bold; }
            .joy { background: #ffeaa7; color: #d6a01e; }
            .anger { background: #ff7675; color: white; }
            .sadness { background: #74b9ff; color: white; }
            .fear { background: #a29bfe; color: white; }
            .surprise { background: #55efc4; color: #00b894; }
            .disgust { background: #81ecec; color: #008b8b; }
            .neutral { background: #dfe6e9; color: #636e72; }
        </style>
    </head>
    <body>
        <div class="card">
            <h2>🎭 Emotion Recognition AI</h2>
            <textarea id="userInput" placeholder="How are you feeling today? (e.g. 'I am so frustrated with this traffic!')"></textarea>
            <button onclick="analyze()">Identify Emotion</button>
        </div>

        <div class="card">
            <h3>📜 Emotion History</h3>
            <table>
                <thead><tr><th>What you said</th><th>Detected Emotion</th><th>Confidence</th></tr></thead>
                <tbody id="tableBody"></tbody>
            </table>
            <button class="clear-btn" onclick="clearAll()">Wipe Records</button>
        </div>

        <script>
            async function loadHistory() {
                const res = await fetch('/history');
                const data = await res.json();
                document.getElementById('tableBody').innerHTML = data.map(row => `
                    <tr>
                        <td>${row.text}</td>
                        <td><span class="badge ${row.emotion}">${row.emotion}</span></td>
                        <td>${(row.confidence * 100).toFixed(1)}%</td>
                    </tr>
                `).join('');
            }

            async function analyze() {
                const text = document.getElementById('userInput').value;
                if(!text.trim()) return;
                
                await fetch('/analyze/single', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({text})
                });
                
                document.getElementById('userInput').value = '';
                loadHistory();
            }

            async function clearAll() {
                if(confirm("Clear all emotional data?")) {
                    await fetch('/history/clear', { method: 'DELETE' });
                    loadHistory();
                }
            }

            window.onload = loadHistory;
        </script>
    </body>
    </html>
    """
