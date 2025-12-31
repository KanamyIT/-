from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
from bs4 import BeautifulSoup
from deep_translator import GoogleTranslator
import random
from functools import lru_cache
import uvicorn

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class LinkRequest(BaseModel):
    url: str

class TextRequest(BaseModel):
    text: str

# RSS источники
RSS_SOURCES = {
    "programming": [
        "https://realpython.com/atom.xml",
        "https://devblogs.microsoft.com/dotnet/feed/",
    ],
    "history": [
        "https://www.history.com/.rss/full/this-day-in-history",
    ],
    "gaming": [
        "https://www.polygon.com/rss/index.xml",
    ],
    "movies": [
        "https://variety.com/feed/",
    ]
}

def translate_html_content(soup):
    """Переводит HTML"""
    translator = GoogleTranslator(source='auto', target='ru')
    
    content = None
    for selector in ['article', 'main', 'body']:
        content = soup.find(selector)
        if content:
            break
    
    if not content:
        return "<p>Контент не найден</p>", 0
    
    for junk in content(["script", "style", "iframe", "nav", "footer", "aside"]):
        junk.decompose()
    
    word_count = 0
    
    for node in content.find_all(text=True):
        original = str(node).strip()
        
        if len(original) < 5 or len(original) > 4000:
            continue
        
        if node.parent.name in ['pre', 'code', 'script', 'style']:
            continue
        
        try:
            translated = translator.translate(original)
            node.replace_with(translated)
            word_count += len(translated.split())
        except:
            pass
    
    for img in content.find_all('img'):
        img['style'] = "max-width: 100%; height: auto;"
    
    return str(content), word_count

@lru_cache(maxsize=20)
def fetch_feed_category(category):
    """Загружает RSS"""
    urls = RSS_SOURCES.get(category, [])
    all_articles = []
    translator = GoogleTranslator(source='auto', target='ru')
    
    for url in urls:
        try:
            print(f"📡 Loading RSS: {url}")
            resp = requests.get(url, timeout=8, headers={'User-Agent': 'Mozilla/5.0'})
            soup = BeautifulSoup(resp.content, "xml")
            
            items = soup.find_all("item")[:5] or soup.find_all("entry")[:5]
            print(f"✅ Found {len(items)} articles")
            
            for item in items:
                try:
                    title_tag = item.find("title")
                    link_tag = item.find("link")
                    
                    if not title_tag or not link_tag:
                        continue
                    
                    title = title_tag.text.strip()
                    link = link_tag.text.strip() if link_tag.text else link_tag.get('href')
                    
                    try:
                        ru_title = translator.translate(title[:150])
                    except:
                        ru_title = title
                    
                    all_articles.append({
                        "title": ru_title,
                        "link": link,
                        "tag": category.upper()
                    })
                except:
                    continue
        except Exception as e:
            print(f"❌ RSS Error: {e}")
            continue
    
    return all_articles

# ГЛАВНАЯ СТРАНИЦА - HTML ВСТРОЕН В КОД
@app.get("/", response_class=HTMLResponse)
def home():
    return """<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>News/Translate by Kanamy</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: Arial, sans-serif; background: #0a0a0a; color: #fff; padding: 20px; }
        .container { max-width: 1200px; margin: 0 auto; }
        
        h1 { text-align: center; font-size: 36px; margin: 40px 0; }
        h1 span { color: #FF6B35; }
        
        .input-box { max-width: 600px; margin: 0 auto 40px; background: #1a1a1a; padding: 24px; border-radius: 12px; }
        input { width: 100%; padding: 14px; background: #0a0a0a; border: 1px solid #333; border-radius: 8px; color: #fff; font-size: 15px; margin-bottom: 12px; }
        button { width: 100%; padding: 14px; background: #FF6B35; color: #fff; border: none; border-radius: 8px; font-size: 15px; cursor: pointer; }
        button:hover { background: #ff8555; }
        
        .categories { display: flex; gap: 12px; justify-content: center; margin: 30px 0; flex-wrap: wrap; }
        .cat-btn { padding: 10px 24px; background: #1a1a1a; border: 1px solid #333; color: #fff; cursor: pointer; border-radius: 8px; }
        .cat-btn.active { background: #FF6B35; border-color: #FF6B35; }
        
        .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 20px; margin: 40px 0; }
        .card { background: #1a1a1a; padding: 20px; border-radius: 12px; cursor: pointer; border: 2px solid #333; transition: 0.2s; }
        .card:hover { border-color: #FF6B35; transform: translateY(-4px); }
        .tag { font-size: 10px; color: #FF6B35; margin-bottom: 8px; font-weight: 600; }
        .card-title { font-size: 18px; line-height: 1.4; }
        
        #article { display: none; background: #1a1a1a; padding: 40px; border-radius: 12px; margin: 40px 0; }
        #article h1, #article h2 { color: #FF6B35; margin: 20px 0; }
        #article p { line-height: 1.7; margin: 15px 0; }
        #article img { max-width: 100%; border-radius: 8px; margin: 20px 0; }
        
        .back { display: inline-block; padding: 12px 24px; background: #FF6B35; color: #fff; cursor: pointer; border-radius: 8px; margin-bottom: 20px; }
        .loader { display: none; text-align: center; padding: 60px; font-size: 18px; }
        .spinner { width: 50px; height: 50px; border: 4px solid #333; border-top-color: #FF6B35; border-radius: 50%; animation: spin 1s linear infinite; margin: 0 auto 20px; }
        @keyframes spin { 100% { transform: rotate(360deg); } }
        .status { text-align: center; padding: 40px; color: #888; }
    </style>
</head>
<body>

<div class="container">
    <h1>Умный <span>переводчик</span></h1>
    
    <div class="input-box">
        <input type="text" id="urlInput" placeholder="https://example.com/article">
        <button onclick="translateUrl()">Перевести статью</button>
    </div>
    
    <div id="news">
        <div class="categories">
            <button class="cat-btn active" onclick="loadCat('programming')">Программирование</button>
            <button class="cat-btn" onclick="loadCat('history')">История</button>
            <button class="cat-btn" onclick="loadCat('gaming')">Игры</button>
            <button class="cat-btn" onclick="loadCat('movies')">Кино</button>
        </div>
        <div class="status" id="status">Загрузка...</div>
        <div class="grid" id="grid"></div>
    </div>
    
    <div class="loader" id="loader"><div class="spinner"></div><p>Загрузка статьи...</p></div>
    
    <div id="article">
        <div class="back" onclick="goBack()">← Назад</div>
        <div id="content"></div>
    </div>
</div>

<script>
console.log('🚀 App started');

async function loadCat(cat) {
    console.log('📰 Category:', cat);
    
    document.querySelectorAll('.cat-btn').forEach(t => t.classList.remove('active'));
    event.target.classList.add('active');
    
    const grid = document.getElementById('grid');
    const status = document.getElementById('status');
    
    grid.innerHTML = '';
    status.style.display = 'block';
    status.textContent = 'Загрузка статей...';
    
    try {
        const res = await fetch('/feed?category=' + cat);
        
        console.log('📡 Response status:', res.status);
        
        if (!res.ok) {
            throw new Error('HTTP ' + res.status);
        }
        
        const data = await res.json();
        console.log('✅ Articles:', data.articles.length);
        
        status.style.display = 'none';
        
        if (data.articles.length === 0) {
            status.textContent = 'Нет статей';
            status.style.display = 'block';
            return;
        }
        
        data.articles.forEach(art => {
            const card = document.createElement('div');
            card.className = 'card';
            card.innerHTML = '<div class="tag">' + art.tag + '</div><div class="card-title">' + art.title + '</div>';
            card.onclick = () => openArticle(art.link);
            grid.appendChild(card);
        });
        
        console.log('✅ Rendered', data.articles.length, 'cards');
        
    } catch(e) {
        console.error('❌ Error:', e);
        status.textContent = 'Ошибка загрузки: ' + e.message;
        status.style.color = '#ff6b6b';
    }
}

async function translateUrl() {
    const url = document.getElementById('urlInput').value.trim();
    if (!url) {
        alert('Введите URL!');
        return;
    }
    openArticle(url);
}

async function openArticle(url) {
    console.log('🔄 Loading:', url);
    
    document.getElementById('news').style.display = 'none';
    document.getElementById('article').style.display = 'none';
    document.getElementById('loader').style.display = 'block';
    
    try {
        const res = await fetch('/translate', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({url: url})
        });
        
        const data = await res.json();
        
        if (data.error) throw new Error(data.error);
        
        console.log('✅ Article loaded');
        
        document.getElementById('content').innerHTML = data.translated_html;
        document.getElementById('article').style.display = 'block';
        window.scrollTo(0, 0);
        
    } catch(e) {
        console.error('❌ Error:', e);
        alert('Ошибка: ' + e.message);
        goBack();
    } finally {
        document.getElementById('loader').style.display = 'none';
    }
}

function goBack() {
    document.getElementById('article').style.display = 'none';
    document.getElementById('news').style.display = 'block';
}

loadCat('programming');
</script>

</body>
</html>"""

# API ENDPOINTS
@app.get("/feed")
def get_feed(category: str = "programming"):
    print(f"\n📰 Feed request: {category}")
    
    if category not in RSS_SOURCES:
        category = "programming"
    
    articles = fetch_feed_category(category)
    random.shuffle(articles)
    
    print(f"📦 Returning {len(articles)} articles\n")
    
    return JSONResponse({"articles": articles[:12]})

@app.post("/translate")
def translate_article(request: LinkRequest):
    print(f"\n🔄 Translate: {request.url}")
    
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0'}
        response = requests.get(request.url, headers=headers, timeout=20)
        
        print(f"✅ Status: {response.status_code}")
        
        if response.status_code != 200:
            return JSONResponse({"error": f"HTTP {response.status_code}"}, status_code=400)
        
        soup = BeautifulSoup(response.text, 'html.parser')
        final_html, word_count = translate_html_content(soup)
        
        print(f"✅ Translated! Words: {word_count}\n")
        
        return JSONResponse({
            "translated_html": final_html,
            "word_count": word_count
        })
        
    except Exception as e:
        print(f"❌ Error: {e}\n")
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/translate_text")
def translate_text(request: TextRequest):
    try:
        translator = GoogleTranslator(source='auto', target='ru')
        translated = translator.translate(request.text[:4500])
        return JSONResponse({"result": translated})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🚀 NEWS TRANSLATOR")
    print("="*70)
    print("📍 http://localhost:8000")
    print("="*70 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
