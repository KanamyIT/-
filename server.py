from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
from bs4 import BeautifulSoup
from deep_translator import GoogleTranslator
import random
from functools import lru_cache
import uvicorn

app = FastAPI()

# ✅ ВАЖНО: CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class LinkRequest(BaseModel):
    url: str

# RSS источники
RSS_SOURCES = {
    "programming": [
        "https://realpython.com/atom.xml",
        "https://devblogs.microsoft.com/dotnet/feed/",
        "https://www.freecodecamp.org/news/rss/"
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
    """Переводит HTML контент"""
    translator = GoogleTranslator(source='auto', target='ru')
    
    content = None
    for selector in ['article', 'main', '.content', '.post-content', 'body']:
        if selector.startswith('.'):
            content = soup.find(class_=selector[1:])
        else:
            content = soup.find(selector)
        if content:
            break
    
    if not content:
        return "<p>Контент не найден</p>", 0
    
    # Удаляем мусор
    for junk in content(["script", "style", "iframe", "nav", "footer", "aside"]):
        junk.decompose()
    
    word_count = 0
    
    # Переводим текст
    for node in content.find_all(text=True):
        original = str(node).strip()
        
        if len(original) < 5:
            continue
        
        parent = node.parent
        if parent.name in ['pre', 'code', 'script', 'style']:
            continue
        
        if len(original) > 4000:
            continue
        
        try:
            translated = translator.translate(original)
            node.replace_with(translated)
            word_count += len(translated.split())
        except:
            pass
    
    # Стили для изображений
    for img in content.find_all('img'):
        img['style'] = "max-width: 100%; height: auto;"
    
    return str(content), word_count

@lru_cache(maxsize=20)
def fetch_feed_category(category):
    """Загружает и переводит RSS"""
    urls = RSS_SOURCES.get(category, [])
    all_articles = []
    translator = GoogleTranslator(source='auto', target='ru')
    
    for url in urls:
        try:
            print(f"📡 RSS: {url}")
            resp = requests.get(url, timeout=8, headers={'User-Agent': 'Mozilla/5.0'})
            soup = BeautifulSoup(resp.content, "xml")
            
            items = soup.find_all("item")[:5] or soup.find_all("entry")[:5]
            print(f"✅ Найдено: {len(items)} статей")
            
            for item in items:
                try:
                    title_tag = item.find("title")
                    if not title_tag:
                        continue
                    
                    title = title_tag.text.strip()
                    
                    link_tag = item.find("link")
                    if link_tag:
                        link = link_tag.text.strip() if link_tag.text else link_tag.get('href')
                    else:
                        continue
                    
                    # Переводим заголовок
                    try:
                        ru_title = translator.translate(title[:150])
                    except:
                        ru_title = title
                    
                    all_articles.append({
                        "title": ru_title,
                        "original_title": title,
                        "link": link,
                        "tag": category.upper()
                    })
                except:
                    continue
        except Exception as e:
            print(f"❌ Ошибка RSS: {e}")
            continue
    
    return all_articles

@app.get("/", response_class=HTMLResponse)
def home():
    # Твой оригинальный HTML (из скрина)
    return """<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>News/Translate by Kanamy</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg: #0a0a0a;
            --surface: #1a1a1a;
            --orange: #FF6B35;
            --text: #e8e8e8;
        }
        
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Inter', sans-serif; background: var(--bg); color: var(--text); }
        
        header { background: rgba(10,10,10,0.95); padding: 20px; text-align: center; }
        .logo { font-size: 20px; }
        .logo span { color: var(--orange); }
        
        .container { max-width: 1200px; margin: 0 auto; padding: 40px 20px; }
        h1 { text-align: center; font-size: 48px; margin: 40px 0; }
        
        .tabs { display: flex; gap: 10px; justify-content: center; margin: 30px 0; }
        .tab { padding: 12px 24px; background: var(--surface); border: 1px solid #333; color: #fff; cursor: pointer; border-radius: 8px; }
        .tab.active { background: var(--orange); }
        
        .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 20px; }
        .card { background: var(--surface); padding: 20px; border-radius: 12px; cursor: pointer; border: 1px solid #333; }
        .card:hover { border-color: var(--orange); transform: translateY(-4px); }
        .tag { font-size: 10px; color: var(--orange); margin-bottom: 8px; }
        
        #article { display: none; background: var(--surface); padding: 40px; border-radius: 12px; margin: 40px 0; }
        #article h1, #article h2 { color: var(--orange); }
        #article img { max-width: 100%; }
        
        .loader { display: none; text-align: center; padding: 60px; }
        .spinner { width: 50px; height: 50px; border: 3px solid #333; border-top-color: var(--orange); border-radius: 50%; animation: spin 1s linear infinite; margin: 0 auto; }
        @keyframes spin { 100% { transform: rotate(360deg); } }
        
        .back { display: inline-block; padding: 10px 20px; background: var(--orange); color: #fff; border-radius: 8px; cursor: pointer; margin-bottom: 20px; }
        .status { text-align: center; padding: 20px; color: #888; }
    </style>
</head>
<body>

<header>
    <div class="logo">News/Translate <span>by Kanamy</span></div>
</header>

<div class="container">
    <h1>Умный переводчик</h1>
    
    <div id="news">
        <div class="tabs">
            <button class="tab active" onclick="load('programming')">Программирование</button>
            <button class="tab" onclick="load('history')">История</button>
            <button class="tab" onclick="load('gaming')">Игры</button>
            <button class="tab" onclick="load('movies')">Кино</button>
        </div>
        <div class="status" id="status">Загрузка статей...</div>
        <div class="grid" id="grid"></div>
    </div>
    
    <div class="loader" id="loader"><div class="spinner"></div></div>
    
    <div id="article">
        <div class="back" onclick="back()">← Назад</div>
        <div id="content"></div>
    </div>
</div>

<script>
console.log('🚀 Start');

async function load(cat) {
    console.log('📰 Category:', cat);
    
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    event.target.classList.add('active');
    
    const grid = document.getElementById('grid');
    const status = document.getElementById('status');
    
    grid.innerHTML = '';
    status.style.display = 'block';
    
    try {
        const res = await fetch('/feed?category=' + cat);
        const data = await res.json();
        
        console.log('✅ Articles:', data.articles.length);
        
        status.style.display = 'none';
        
        data.articles.forEach(art => {
            const card = document.createElement('div');
            card.className = 'card';
            card.innerHTML = '<div class="tag">' + art.tag + '</div><div>' + art.title + '</div>';
            card.onclick = () => open(art.link);
            grid.appendChild(card);
        });
    } catch(e) {
        console.error('❌', e);
        status.textContent = 'Ошибка: ' + e.message;
    }
}

async function open(url) {
    console.log('🔄 Opening:', url);
    
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
        
        if(data.error) throw new Error(data.error);
        
        console.log('✅ Loaded');
        
        document.getElementById('content').innerHTML = data.translated_html;
        document.getElementById('article').style.display = 'block';
        window.scrollTo(0,0);
    } catch(e) {
        alert('Ошибка: ' + e.message);
        back();
    } finally {
        document.getElementById('loader').style.display = 'none';
    }
}

function back() {
    document.getElementById('article').style.display = 'none';
    document.getElementById('news').style.display = 'block';
}

load('programming');
</script>

</body>
</html>"""

@app.get("/feed")
def get_feed(category: str = "programming"):
    print(f"\n📰 Feed: {category}")
    
    if category not in RSS_SOURCES:
        category = "programming"
    
    articles = fetch_feed_category(category)
    random.shuffle(articles)
    
    print(f"📦 Returning: {len(articles)}\n")
    
    return {"articles": articles[:12]}

@app.post("/translate")
def translate_article(request: LinkRequest):
    print(f"\n🔄 Translate: {request.url}")
    
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0'}
        response = requests.get(request.url, headers=headers, timeout=20)
        
        print(f"✅ Status: {response.status_code}")
        
        try:
            soup = BeautifulSoup(response.text, 'lxml')
        except:
            soup = BeautifulSoup(response.text, 'html.parser')
        
        final_html, word_count = translate_html_content(soup)
        
        print(f"✅ Translated! Words: {word_count}\n")
        
        return {
            "translated_html": final_html,
            "word_count": word_count
        }
    except Exception as e:
        print(f"❌ Error: {e}\n")
        return {"error": str(e)}

if __name__ == "__main__":
    print("\n🚀 NEWS TRANSLATOR")
    print("📍 http://localhost:8000\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
