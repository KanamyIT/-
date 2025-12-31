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
import time

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
    try:
        translator = GoogleTranslator(source='auto', target='ru')
    except Exception as e:
        print(f"❌ Translator init error: {e}")
        return "<p>Ошибка инициализации переводчика</p>", 0
    
    content = None
    for selector in ['article', 'main', '.post-content', '.entry-content', 'body']:
        content = soup.find(selector)
        if content:
            print(f"✅ Found content in: {selector}")
            break
    
    if not content:
        print("❌ No content found")
        return "<p>Контент не найден</p>", 0
    
    # Удаляем мусор
    for junk in content(["script", "style", "iframe", "nav", "footer", "aside", "header"]):
        junk.decompose()
    
    word_count = 0
    translated_count = 0
    
    # Находим все текстовые узлы
    text_nodes = content.find_all(text=True)
    print(f"📝 Found {len(text_nodes)} text nodes")
    
    for node in text_nodes:
        original = str(node).strip()
        
        # Пропускаем короткие и очень длинные
        if len(original) < 3 or len(original) > 4500:
            continue
        
        # Пропускаем код
        if node.parent.name in ['pre', 'code', 'script', 'style']:
            continue
        
        try:
            # Переводим с задержкой чтобы не словить rate limit
            translated = translator.translate(original)
            if translated and translated != original:
                node.replace_with(translated)
                word_count += len(translated.split())
                translated_count += 1
            time.sleep(0.1)  # Небольшая задержка
        except Exception as e:
            print(f"⚠️ Translation error for text: {original[:50]}... Error: {e}")
            continue
    
    print(f"✅ Translated {translated_count} text blocks, {word_count} words")
    
    # Оптимизируем изображения
    for img in content.find_all('img'):
        img['style'] = "max-width: 100%; height: auto;"
    
    return str(content), word_count

def fetch_feed_category(category):
    """Загружает RSS - БЕЗ кеширования для отладки"""
    urls = RSS_SOURCES.get(category, [])
    all_articles = []
    
    try:
        translator = GoogleTranslator(source='auto', target='ru')
    except Exception as e:
        print(f"❌ Translator init failed: {e}")
        return []
    
    for url in urls:
        try:
            print(f"📡 Loading RSS: {url}")
            resp = requests.get(url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
            
            if resp.status_code != 200:
                print(f"❌ HTTP {resp.status_code}")
                continue
                
            soup = BeautifulSoup(resp.content, "xml")
            
            # Пробуем оба формата RSS
            items = soup.find_all("item")[:8] if soup.find_all("item") else soup.find_all("entry")[:8]
            print(f"✅ Found {len(items)} items")
            
            for item in items:
                try:
                    # Заголовок
                    title_tag = item.find("title")
                    if not title_tag:
                        continue
                    title = title_tag.text.strip()
                    
                    # Ссылка - ИСПРАВЛЕНО
                    link = None
                    link_tag = item.find("link")
                    
                    if link_tag:
                        # Если есть текст внутри тега
                        if link_tag.text and link_tag.text.strip():
                            link = link_tag.text.strip()
                        # Если ссылка в атрибуте href
                        elif link_tag.get('href'):
                            link = link_tag.get('href')
                    
                    # Проверяем что ссылка валидная
                    if not link or not link.startswith('http'):
                        print(f"⚠️ Invalid link for: {title[:40]}")
                        continue
                    
                    # Переводим заголовок
                    try:
                        ru_title = translator.translate(title[:200])
                        time.sleep(0.1)
                    except Exception as e:
                        print(f"⚠️ Title translation error: {e}")
                        ru_title = title
                    
                    all_articles.append({
                        "title": ru_title,
                        "link": link,
                        "tag": category.upper()
                    })
                    
                    print(f"✅ Added: {ru_title[:50]}...")
                    
                except Exception as e:
                    print(f"⚠️ Item parse error: {e}")
                    continue
                    
        except Exception as e:
            print(f"❌ RSS Error for {url}: {e}")
            continue
    
    print(f"📦 Total articles collected: {len(all_articles)}")
    return all_articles

# ГЛАВНАЯ СТРАНИЦА
@app.get("/", response_class=HTMLResponse)
def home():
    try:
        possible_paths = [
            'index.html',
            './index.html',
            'index-4.html',
            './index-4.html',
            os.path.join(os.path.dirname(__file__), 'index.html'),
            os.path.join(os.path.dirname(__file__), 'index-4.html')
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                print(f"✅ Found HTML at: {path}")
                with open(path, 'r', encoding='utf-8') as f:
                    return f.read()
        
        print("❌ HTML file not found!")
        return HTMLResponse(
            content="<h1>Error: index.html not found</h1><p>Please place index.html in the same directory as server-3.py</p>",
            status_code=500
        )
        
    except Exception as e:
        print(f"❌ Error reading HTML: {e}")
        return HTMLResponse(
            content=f"<h1>Error: {str(e)}</h1>",
            status_code=500
        )

@app.get("/feed")
def get_feed(category: str = "programming"):
    print(f"\n{'='*60}")
    print(f"📰 Feed request: {category}")
    print(f"{'='*60}")
    
    # Валидация категории
    if category not in RSS_SOURCES:
        print(f"⚠️ Unknown category, using 'programming'")
        category = "programming"
    
    # Загружаем статьи
    articles = fetch_feed_category(category)
    
    if not articles:
        print("❌ No articles loaded!")
        return JSONResponse({"articles": []})
    
    # Перемешиваем
    random.shuffle(articles)
    
    # Ограничиваем количество
    result = articles[:12]
    
    print(f"📦 Returning {len(result)} articles")
    print(f"{'='*60}\n")
    
    return JSONResponse({"articles": result})

@app.post("/translate")
def translate_article(request: LinkRequest):
    print(f"\n{'='*60}")
    print(f"🔄 Translate request")
    print(f"URL: {request.url}")
    print(f"{'='*60}")
    
    # Валидация URL
    if not request.url.startswith('http'):
        print("❌ Invalid URL")
        return JSONResponse({"error": "Неверный URL"}, status_code=400)
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        
        print("📡 Fetching page...")
        response = requests.get(request.url, headers=headers, timeout=20)
        
        print(f"📊 Status: {response.status_code}")
        print(f"📊 Content length: {len(response.text)} chars")
        
        if response.status_code != 200:
            return JSONResponse(
                {"error": f"Не удалось загрузить страницу (HTTP {response.status_code})"},
                status_code=400
            )
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        print("🔄 Starting translation...")
        final_html, word_count = translate_html_content(soup)
        
        if word_count == 0:
            print("⚠️ No words translated!")
            return JSONResponse(
                {"error": "Не удалось найти текст для перевода"},
                status_code=400
            )
        
        print(f"✅ Translation complete!")
        print(f"📊 Words translated: {word_count}")
        print(f"{'='*60}\n")
        
        return JSONResponse({
            "translated_html": final_html,
            "word_count": word_count
        })
        
    except requests.Timeout:
        print("❌ Timeout")
        return JSONResponse({"error": "Превышено время ожидания"}, status_code=500)
    except Exception as e:
        print(f"❌ Error: {type(e).__name__}: {e}")
        return JSONResponse({"error": f"Ошибка: {str(e)}"}, status_code=500)

@app.post("/translate_text")
def translate_text(request: TextRequest):
    print(f"\n🔄 Text translation request ({len(request.text)} chars)")
    
    if not request.text or len(request.text) < 1:
        return JSONResponse({"error": "Пустой текст"}, status_code=400)
    
    try:
        translator = GoogleTranslator(source='auto', target='ru')
        
        # Обрезаем если слишком длинный
        text_to_translate = request.text[:4500]
        
        translated = translator.translate(text_to_translate)
        
        print(f"✅ Text translated\n")
        
        return JSONResponse({"result": translated})
    except Exception as e:
        print(f"❌ Translation error: {e}\n")
        return JSONResponse({"error": f"Ошибка перевода: {str(e)}"}, status_code=500)

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🚀 NEWS TRANSLATOR SERVER")
    print("="*70)
    print("📍 URL: http://localhost:8000")
    print("📍 API: http://localhost:8000/docs")
    
    # Проверяем файлы
    html_files = ['index.html', 'index-4.html']
    found = False
    for f in html_files:
        if os.path.exists(f):
            print(f"✅ Found: {f}")
            found = True
    
    if not found:
        print("❌ WARNING: No HTML files found!")
    
    print("="*70 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
