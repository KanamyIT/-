from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
from bs4 import BeautifulSoup
from deep_translator import GoogleTranslator
import random
import uvicorn
import os

app = FastAPI()

# CORS для работы с GitHub Pages
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

RSS_SOURCES = {
    "programming": [
        "https://realpython.com/atom.xml",
        "https://dev.to/feed",
    ],
    "history": [
        "https://www.smithsonianmag.com/rss/history/",
    ],
    "gaming": [
        "https://www.ign.com/articles?tags=news&format=rss",
    ],
    "movies": [
        "https://variety.com/feed/",
    ]
}

def translate_html_content(soup):
    try:
        translator = GoogleTranslator(source='auto', target='ru')
    except Exception as e:
        print(f"❌ Translator error: {e}")
        return "<p>Ошибка переводчика</p>", 0
    
    content = None
    for selector in ['article', 'main', '.post-content', '.entry-content', 'body']:
        content = soup.find(selector)
        if content:
            break
    
    if not content:
        return "<p>Контент не найден</p>", 0
    
    for junk in content(["script", "style", "iframe", "nav", "footer", "aside", "header"]):
        junk.decompose()
    
    word_count = 0
    text_nodes = content.find_all(text=True)
    
    # Переводим первые 30 блоков для скорости
    for node in text_nodes[:30]:
        original = str(node).strip()
        if len(original) < 10 or len(original) > 1000:
            continue
        if node.parent.name in ['pre', 'code', 'script', 'style']:
            continue
        
        try:
            translated = translator.translate(original)
            if translated:
                node.replace_with(translated)
                word_count += len(translated.split())
        except:
            continue
    
    for img in content.find_all('img'):
        img['style'] = "max-width: 100%; height: auto;"
    
    return str(content), word_count

def fetch_feed_category(category):
    urls = RSS_SOURCES.get(category, [])
    all_articles = []
    
    try:
        translator = GoogleTranslator(source='auto', target='ru')
    except:
        return []
    
    for url in urls:
        try:
            print(f"📡 Loading: {url}")
            resp = requests.get(url, timeout=8, headers={'User-Agent': 'Mozilla/5.0'})
            
            if resp.status_code != 200:
                continue
            
            soup = BeautifulSoup(resp.content, "xml")
            items = soup.find_all("item")
            
            if not items:
                soup = BeautifulSoup(resp.content, "html.parser")
                items = soup.find_all("entry")
            
            for item in items[:8]:
                try:
                    title_tag = item.find("title")
                    if not title_tag:
                        continue
                    title = title_tag.text.strip()
                    
                    link = None
                    link_tag = item.find("link")
                    if link_tag:
                        if link_tag.text and link_tag.text.strip():
                            link = link_tag.text.strip()
                        elif link_tag.get('href'):
                            link = link_tag.get('href')
                    
                    if not link or not link.startswith('http'):
                        continue
                    
                    try:
                        ru_title = translator.translate(title[:200])
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
            print(f"❌ RSS error: {e}")
            continue
    
    return all_articles

@app.get("/")
def home():
    return JSONResponse({
        "status": "ok",
        "message": "News Translator API is running",
        "endpoints": {
            "/feed": "GET - Load RSS feed",
            "/translate": "POST - Translate article",
            "/translate_text": "POST - Translate text",
            "/health": "GET - Health check"
        }
    })

@app.get("/feed")
def get_feed(category: str = "programming"):
    print(f"\n📰 Feed: {category}")
    
    if category not in RSS_SOURCES:
        category = "programming"
    
    articles = fetch_feed_category(category)
    
    if not articles:
        return JSONResponse({"articles": []})
    
    random.shuffle(articles)
    return JSONResponse({"articles": articles[:12]})

@app.post("/translate")
def translate_article(request: LinkRequest):
    print(f"\n🔄 Translate: {request.url}")
    
    if not request.url.startswith('http'):
        return JSONResponse({"error": "Invalid URL"}, status_code=400)
    
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(request.url, headers=headers, timeout=15)
        
        if response.status_code != 200:
            return JSONResponse({"error": f"HTTP {response.status_code}"}, status_code=400)
        
        soup = BeautifulSoup(response.text, 'html.parser')
        html, words = translate_html_content(soup)
        
        if words == 0:
            return JSONResponse({"error": "No text found"}, status_code=400)
        
        return JSONResponse({"translated_html": html, "word_count": words})
        
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/translate_text")
def translate_text(request: TextRequest):
    if not request.text:
        return JSONResponse({"error": "Empty text"}, status_code=400)
    
    try:
        translator = GoogleTranslator(source='auto', target='ru')
        translated = translator.translate(request.text[:4500])
        return JSONResponse({"result": translated})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/health")
def health():
    return {"status": "ok", "message": "Server is healthy"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    print("\n" + "="*70)
    print("🚀 NEWS TRANSLATOR SERVER")
    print("="*70)
    print(f"📍 Running on port: {port}")
    print("="*70 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=port)
