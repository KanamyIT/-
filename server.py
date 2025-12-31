from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import requests
from bs4 import BeautifulSoup
import uvicorn

app = FastAPI()

class URLRequest(BaseModel):
    url: str

# Встроенные статьи
ARTICLES = {
    "programming": [
        {"title": "Learn Python Programming", "url": "https://realpython.com/"},
        {"title": "FastAPI Web Framework", "url": "https://fastapi.tiangolo.com/"},
        {"title": "JavaScript Async Guide", "url": "https://developer.mozilla.org/"},
        {"title": "React Development", "url": "https://react.dev/"},
        {"title": "Docker Tutorial", "url": "https://docs.docker.com/"},
    ],
    "gaming": [
        {"title": "Dota 2 Updates", "url": "https://www.dota2.com/"},
        {"title": "Gaming News", "url": "https://store.steampowered.com/"},
    ],
    "history": [
        {"title": "Ancient History", "url": "https://www.britannica.com/"},
        {"title": "World History", "url": "https://www.history.com/"},
    ],
    "movies": [
        {"title": "Top Movies", "url": "https://www.imdb.com/"},
        {"title": "Cinema News", "url": "https://variety.com/"},
    ]
}

@app.get("/", response_class=HTMLResponse)
def home():
    return """<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>News Reader</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: Arial, sans-serif;
            background: #111;
            color: #fff;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        
        h1 {
            text-align: center;
            font-size: 32px;
            margin: 30px 0;
            color: #fff;
        }
        
        h1 span {
            color: #FF6B35;
        }
        
        .tabs {
            display: flex;
            gap: 10px;
            justify-content: center;
            margin: 20px 0 40px 0;
            flex-wrap: wrap;
        }
        
        .tab {
            padding: 12px 24px;
            background: #222;
            border: 2px solid #333;
            color: #fff;
            cursor: pointer;
            border-radius: 6px;
            font-size: 15px;
        }
        
        .tab:hover {
            border-color: #FF6B35;
        }
        
        .tab.active {
            background: #FF6B35;
            border-color: #FF6B35;
        }
        
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }
        
        .card {
            background: #222;
            padding: 20px;
            border-radius: 8px;
            cursor: pointer;
            border: 2px solid #333;
            transition: all 0.2s;
        }
        
        .card:hover {
            border-color: #FF6B35;
            transform: translateY(-2px);
        }
        
        .card-title {
            font-size: 17px;
            line-height: 1.5;
            color: #fff;
        }
        
        #article {
            display: none;
            background: #222;
            padding: 40px;
            border-radius: 8px;
            margin: 30px 0;
        }
        
        #article h1,
        #article h2,
        #article h3 {
            color: #FF6B35;
            margin: 20px 0 10px 0;
        }
        
        #article p {
            line-height: 1.7;
            margin: 15px 0;
            color: #ddd;
        }
        
        #article img {
            max-width: 100%;
            border-radius: 8px;
            margin: 20px 0;
        }
        
        #article a {
            color: #FF6B35;
        }
        
        .back {
            display: inline-block;
            padding: 12px 24px;
            background: #FF6B35;
            color: #fff;
            cursor: pointer;
            border-radius: 6px;
            margin-bottom: 20px;
            border: none;
            font-size: 15px;
        }
        
        .back:hover {
            background: #ff8555;
        }
        
        .loader {
            display: none;
            text-align: center;
            padding: 60px;
            font-size: 18px;
            color: #888;
        }
        
        .spinner {
            width: 50px;
            height: 50px;
            border: 4px solid #333;
            border-top-color: #FF6B35;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }
        
        @keyframes spin {
            100% { transform: rotate(360deg); }
        }
        
        @media (max-width: 768px) {
            h1 { font-size: 24px; }
            .grid { grid-template-columns: 1fr; }
            #article { padding: 20px; }
        }
    </style>
</head>
<body>

<div class="container">
    <h1>News <span>Reader</span></h1>
    
    <div id="news">
        <div class="tabs">
            <button class="tab active" onclick="showCat('programming')">Programming</button>
            <button class="tab" onclick="showCat('gaming')">Gaming</button>
            <button class="tab" onclick="showCat('history')">History</button>
            <button class="tab" onclick="showCat('movies')">Movies</button>
        </div>
        <div class="grid" id="grid"></div>
    </div>
    
    <div class="loader" id="loader">
        <div class="spinner"></div>
        <p>Loading article...</p>
    </div>
    
    <div id="article">
        <button class="back" onclick="goBack()">← Back to news</button>
        <div id="content"></div>
    </div>
</div>

<script>
console.log('✅ Script loaded');

const articles = """ + str(ARTICLES).replace("'", '"') + """;

function showCat(cat) {
    console.log('📰 Category:', cat);
    
    // Update active tab
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    event.target.classList.add('active');
    
    // Render cards
    const grid = document.getElementById('grid');
    grid.innerHTML = '';
    
    articles[cat].forEach(article => {
        const card = document.createElement('div');
        card.className = 'card';
        card.innerHTML = '<div class="card-title">' + article.title + '</div>';
        card.onclick = () => openArticle(article.url);
        grid.appendChild(card);
    });
    
    console.log('✅ Rendered', articles[cat].length, 'cards');
}

async function openArticle(url) {
    console.log('🔄 Opening:', url);
    
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
        console.log('✅ Article loaded');
        
        document.getElementById('content').innerHTML = html;
        document.getElementById('article').style.display = 'block';
        window.scrollTo(0, 0);
        
    } catch(e) {
        console.error('❌ Error:', e);
        alert('Failed to load article: ' + e);
        goBack();
    } finally {
        document.getElementById('loader').style.display = 'none';
    }
}

function goBack() {
    document.getElementById('article').style.display = 'none';
    document.getElementById('news').style.display = 'block';
}

// Initial load
console.log('🚀 Initializing...');
showCat('programming');
console.log('✅ Ready!');
</script>

</body>
</html>"""

@app.post("/load", response_class=HTMLResponse)
def load_article(req: URLRequest):
    print(f"\n🔄 Loading: {req.url}")
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(req.url, headers=headers, timeout=15)
        print(f"✅ Status: {response.status_code}")
        
        if response.status_code != 200:
            return f"<p style='color:#ff6b6b'>Failed to load (HTTP {response.status_code})</p>"
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Title
        title = "Article"
        if soup.find('h1'):
            title = soup.find('h1').get_text().strip()
        elif soup.find('title'):
            title = soup.find('title').get_text().strip()
        
        # Content
        content = soup.find('article') or soup.find('main') or soup.find('body')
        
        if not content:
            return "<p>Content not found</p>"
        
        # Clean up
        for tag in content(['script', 'style', 'nav', 'header', 'footer', 'aside', 'iframe']):
            tag.decompose()
        
        # Fix images
        for img in content.find_all('img'):
            img['style'] = 'max-width:100%; height:auto;'
            if img.get('data-src') and not img.get('src'):
                img['src'] = img['data-src']
        
        result = f"<h1>{title}</h1>{str(content)}"
        
        print(f"✅ Success!\n")
        return result
        
    except Exception as e:
        print(f"❌ Error: {e}\n")
        return f"<p style='color:#ff6b6b'>Error: {str(e)}</p>"

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 NEWS READER - CLEAN VERSION")
    print("="*60)
    print("📍 http://localhost:8000")
    print("✅ NO gradients, NO effects")
    print("✅ JUST WORKS")
    print("="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
