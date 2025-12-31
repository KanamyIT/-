from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import requests
from bs4 import BeautifulSoup
import uvicorn

app = FastAPI()

class URLRequest(BaseModel):
    url: str

# ВСТРОЕННЫЕ ТЕСТОВЫЕ СТАТЬИ (ВСЕГДА РАБОТАЮТ)
ARTICLES = {
    "programming": [
        {"title": "Learn Python in 30 Days", "url": "https://realpython.com/"},
        {"title": "FastAPI Tutorial", "url": "https://fastapi.tiangolo.com/"},
        {"title": "JavaScript Async Guide", "url": "https://developer.mozilla.org/en-US/docs/Learn/JavaScript/Asynchronous"},
        {"title": "React Hooks Explained", "url": "https://react.dev/reference/react"},
        {"title": "Docker for Developers", "url": "https://docs.docker.com/get-started/"},
    ],
    "gaming": [
        {"title": "Dota 2 Latest Patch", "url": "https://www.dota2.com/"},
        {"title": "Clash Royale Strategies", "url": "https://clashroyale.com/"},
        {"title": "Best Games 2025", "url": "https://store.steampowered.com/"},
    ],
    "history": [
        {"title": "Ancient Rome History", "url": "https://www.britannica.com/place/ancient-Rome"},
        {"title": "Egyptian Pyramids", "url": "https://www.nationalgeographic.com/history/article/giza-pyramids"},
    ],
    "movies": [
        {"title": "Top Movies 2025", "url": "https://www.imdb.com/"},
        {"title": "Netflix January 2025", "url": "https://www.netflix.com/"},
    ]
}

@app.get("/", response_class=HTMLResponse)
def home():
    return """<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <title>News Reader</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: Arial, sans-serif; background: #0a0a0a; color: #fff; }
        .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
        
        h1 { text-align: center; margin: 40px 0; font-size: 36px; }
        h1 span { color: #FF6B35; }
        
        .tabs { display: flex; gap: 10px; justify-content: center; margin: 30px 0; }
        .tab { padding: 12px 24px; background: #1a1a1a; border: none; color: #fff; cursor: pointer; border-radius: 6px; }
        .tab.active { background: #FF6B35; }
        
        .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 20px; margin: 30px 0; }
        .card { background: #1a1a1a; padding: 20px; border-radius: 8px; cursor: pointer; border: 2px solid transparent; }
        .card:hover { border-color: #FF6B35; }
        .card-title { font-size: 18px; margin-bottom: 10px; }
        
        #article { display: none; background: #1a1a1a; padding: 40px; border-radius: 8px; margin: 30px 0; }
        #article h1, #article h2 { color: #FF6B35; margin: 20px 0; }
        #article p { line-height: 1.8; margin: 15px 0; }
        #article img { max-width: 100%; border-radius: 8px; margin: 20px 0; }
        
        .back { display: inline-block; padding: 10px 20px; background: #FF6B35; color: #fff; cursor: pointer; border-radius: 6px; margin-bottom: 20px; }
        .loader { display: none; text-align: center; padding: 60px; font-size: 20px; }
    </style>
</head>
<body>

<div class="container">
    <h1>News <span>Reader</span></h1>
    
    <div id="news">
        <div class="tabs">
            <button class="tab active" onclick="show('programming')">Programming</button>
            <button class="tab" onclick="show('gaming')">Gaming</button>
            <button class="tab" onclick="show('history')">History</button>
            <button class="tab" onclick="show('movies')">Movies</button>
        </div>
        <div class="grid" id="grid"></div>
    </div>
    
    <div class="loader" id="loader">Loading article...</div>
    
    <div id="article">
        <div class="back" onclick="back()">← Back</div>
        <div id="content"></div>
    </div>
</div>

<script>
const data = """ + str(ARTICLES).replace("'", '"') + """;

function show(cat) {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    event.target.classList.add('active');
    
    const grid = document.getElementById('grid');
    grid.innerHTML = '';
    
    data[cat].forEach(article => {
        const card = document.createElement('div');
        card.className = 'card';
        card.innerHTML = '<div class="card-title">' + article.title + '</div>';
        card.onclick = () => open(article.url);
        grid.appendChild(card);
    });
}

async function open(url) {
    console.log('Opening:', url);
    
    document.getElementById('news').style.display = 'none';
    document.getElementById('article').style.display = 'none';
    document.getElementById('loader').style.display = 'block';
    
    try {
        const res = await fetch('/load', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({url: url})
        });
        
        const html = await res.text();
        
        document.getElementById('content').innerHTML = html;
        document.getElementById('article').style.display = 'block';
        window.scrollTo(0, 0);
        
    } catch(e) {
        alert('Error: ' + e);
        back();
    } finally {
        document.getElementById('loader').style.display = 'none';
    }
}

function back() {
    document.getElementById('article').style.display = 'none';
    document.getElementById('news').style.display = 'block';
}

show('programming');
</script>

</body>
</html>"""

@app.post("/load", response_class=HTMLResponse)
def load_article(req: URLRequest):
    print(f"\n🔄 Loading: {req.url}")
    
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        
        r = requests.get(req.url, headers=headers, timeout=15)
        print(f"✅ Status: {r.status_code}")
        
        if r.status_code != 200:
            return f"<p style='color:red'>Failed to load (HTTP {r.status_code})</p>"
        
        soup = BeautifulSoup(r.text, 'html.parser')
        
        # Get title
        title = "Article"
        if soup.find('h1'):
            title = soup.find('h1').get_text().strip()
        elif soup.find('title'):
            title = soup.find('title').get_text().strip()
        
        # Get content
        content = soup.find('article') or soup.find('main') or soup.find('body')
        
        if not content:
            return "<p>Content not found</p>"
        
        # Clean
        for tag in content(['script', 'style', 'nav', 'header', 'footer', 'aside']):
            tag.decompose()
        
        # Fix images
        for img in content.find_all('img'):
            img['style'] = 'max-width:100%;'
        
        html = f"<h1>{title}</h1>{str(content)}"
        
        print(f"✅ Loaded!\n")
        return html
        
    except Exception as e:
        print(f"❌ Error: {e}\n")
        return f"<p style='color:red'>Error: {e}</p>"

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 SIMPLE NEWS READER")
    print("="*60)
    print("📍 http://localhost:8000")
    print("✅ Articles load INSTANTLY (built-in)")
    print("✅ Click article → loads real content")
    print("✅ NO translation (just shows English)")
    print("="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
