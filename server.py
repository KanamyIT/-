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
import os

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
    """Переводит HTML контент"""
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

# ЧИТАЕМ index.html
@app.get("/", response_class=HTMLResponse)
def home():
    try:
        # Пытаемся найти файл в разных местах
        possible_paths = [
            'index.html',
            './index.html',
            os.path.join(os.path.dirname(__file__), 'index.html')
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                print(f"✅ Found index.html at: {path}")
                with open(path, 'r', encoding='utf-8') as f:
                    return f.read()
        
        # Если не нашли - ошибка
        print("❌ index.html not found!")
        return HTMLResponse(
            content="<h1>Error: index.html not found</h1>",
            status_code=500
        )
        
    except Exception as e:
        print(f"❌ Error reading index.html: {e}")
        return HTMLResponse(
            content=f"<h1>Error: {str(e)}</h1>",
            status_code=500
        )

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
    
    # Проверяем есть ли index.html
    if os.path.exists('index.html'):
        print("✅ index.html found")
    else:
        print("❌ WARNING: index.html NOT found!")
    
    print("="*70 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
