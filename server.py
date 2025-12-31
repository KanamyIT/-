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

# RSS источники - ОБНОВЛЕНЫ
RSS_SOURCES = {
    "programming": [
        "https://realpython.com/atom.xml",
        "https://dev.to/feed",
        "https://www.freecodecamp.org/news/rss/",
    ],
    "history": [
        "https://www.smithsonianmag.com/rss/history/",
        "https://www.historytoday.com/feed",
    ],
    "gaming": [
        "https://www.ign.com/articles?tags=news&format=rss",
        "https://www.gamespot.com/feeds/news/",
    ],
    "movies": [
        "https://variety.com/feed/",
        "https://www.hollywoodreporter.com/feed/",
    ]
}

def translate_html_content(soup):
    """Переводит HTML контент - ОПТИМИЗИРОВАНО"""
    try:
        translator = GoogleTranslator(source='auto', target='ru')
    except Exception as e:
        print(f"❌ Translator init error: {e}")
        return "<p>Ошибка инициализации переводчика</p>", 0
    
    content = None
    for selector in ['article', 'main', '.post-content', '.entry-content', '.article-body', 'body']:
        content = soup.find(selector)
        if content:
            print(f"✅ Found content in: {selector}")
            break
    
    if not content:
        print("❌ No content found")
        return "<p>Контент не найден</p>", 0
    
    # Удаляем мусор
    for junk in content(["script", "style", "iframe", "nav", "footer", "aside", "header", "form", "button"]):
        junk.decompose()
    
    word_count = 0
    translated_count = 0
    
    # Собираем весь текст единым блоком (БЫСТРЕЕ!)
    all_text = []
    text_nodes = content.find_all(text=True)
    
    for node in text_nodes:
        original = str(node).strip()
        if len(original) >= 10 and node.parent.name not in ['pre', 'code', 'script', 'style']:
            all_text.append(original)
    
    print(f"📝 Found {len(all_text)} text blocks")
    
    # Переводим большими блоками (НАМНОГО БЫСТРЕЕ)
    if all_text:
        try:
            # Объединяем по 5 предложений
            batch_size = 5
            for i in range(0, len(all_text), batch_size):
                batch = all_text[i:i+batch_size]
                combined = " ".join(batch)
                
                # Ограничиваем длину
                if len(combined) > 4500:
                    combined = combined[:4500]
                
                try:
                    translated = translator.translate(combined)
                    if translated:
                        # Разбиваем обратно
                        sentences = translated.split(". ")
                        for j, sent in enumerate(sentences):
                            if i+j < len(text_nodes):
                                word_count += len(sent.split())
                        translated_count += len(batch)
                except Exception as e:
                    print(f"⚠️ Batch translation error: {e}")
                    continue
            
            # Простая замена текста
            content_str = str(content)
            for orig in all_text[:10]:  # Переводим первые 10 блоков для превью
                try:
                    trans = translator.translate(orig[:1000])
                    content_str = content_str.replace(orig, trans, 1)
                except:
                    continue
            
            content = BeautifulSoup(content_str, 'html.parser')
            
        except Exception as e:
            print(f"❌ Translation error: {e}")
    
    print(f"✅ Translated {translated_count} blocks, ~{word_count} words")
    
    # Оптимизируем изображения
    for img in content.find_all('img'):
        img['style'] = "max-width: 100%; height: auto;"
    
    return str(content), max(word_count, len(all_text) * 5)  # Примерная оценка

def fetch_feed_category(category):
    """Загружает RSS - ОПТИМИЗИРОВАНО"""
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
            resp = requests.get(url, timeout=8, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            
            if resp.status_code != 200:
                print(f"❌ HTTP {resp.status_code}")
                continue
            
            # Пробуем и XML и HTML парсинг
            soup = BeautifulSoup(resp.content, "xml")
            items = soup.find_all("item")
            
            if not items:
                soup = BeautifulSoup(resp.content, "html.parser")
                items = soup.find_all("entry")
            
            print(f"✅ Found {len(items)} items")
            
            for item in items[:8]:
                try:
                    # Заголовок
                    title_tag = item.find("title")
                    if not title_tag:
                        continue
                    title = title_tag.text.strip()
                    
                    # Ссылка
                    link = None
                    link_tag = item.find("link")
                    
                    if link_tag:
                        if link_tag.text and link_tag.text.strip():
                            link = link_tag.text.strip()
                        elif link_tag.get('href'):
                            link = link_tag.get('href')
                        elif link_tag.get('url'):
                            link = link_tag.get('url')
                    
                    # Проверяем ссылку
                    if not link or not link.startswith('http'):
                        print(f"⚠️ Invalid link for: {title[:40]}")
                        continue
                    
                    # Переводим заголовок БЕЗ задержки
                    try:
                        ru_title = translator.translate(title[:200])
                    except:
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
    
    print(f"📦 Total articles: {len(all_articles)}")
    return all_articles

# ГЛАВНАЯ СТРАНИЦА
@app.get("/", response_class=HTMLResponse)
def home():
    try:
        possible_paths = [
            'index.html',
            'index-4.html',
            './index.html',
            './index-4.html',
            os.path.join(os.path.dirname(__file__), 'index.html'),
            os.path.join(os.path.dirname(__file__), 'index-4.html')
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                print(f"✅ Found HTML at: {path}")
                with open(path, 'r', encoding='utf-8') as f:
                    return f.read()
        
        return HTMLResponse(
            content="<h1>Server is running!</h1><p>Access your site via GitHub Pages</p>",
            status_code=200
        )
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return HTMLResponse(content=f"<h1>Error: {str(e)}</h1>", status_code=500)

@app.get("/feed")
def get_feed(category: str = "programming"):
    print(f"\n📰 Feed: {category}")
    
    if category not in RSS_SOURCES:
        category = "programming"
    
    articles = fetch_feed_category(category)
    
    if not articles:
        return JSONResponse({"articles": []})
    
    random.shuffle(articles)
    result = articles[:12]
    
    print(f"📦 Returning {len(result)} articles\n")
    return JSONResponse({"articles": result})

@app.post("/translate")
def translate_article(request: LinkRequest):
    print(f"\n🔄 Translating: {request.url}")
    
    if not request.url.startswith('http'):
        return JSONResponse({"error": "Неверный URL"}, status_code=400)
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(request.url, headers=headers, timeout=15)
        
        if response.status_code != 200:
            return JSONResponse(
                {"error": f"HTTP {response.status_code}"},
                status_code=400
            )
        
        soup = BeautifulSoup(response.text, 'html.parser')
        final_html, word_count = translate_html_content(soup)
        
        if word_count == 0:
            return JSONResponse(
                {"error": "Не удалось найти текст"},
                status_code=400
            )
        
        print(f"✅ Done! Words: {word_count}\n")
        
        return JSONResponse({
            "translated_html": final_html,
            "word_count": word_count
        })
        
    except Exception as e:
        print(f"❌ Error: {e}\n")
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/translate_text")
def translate_text(request: TextRequest):
    print(f"\n🔄 Text translation ({len(request.text)} chars)")
    
    if not request.text:
        return JSONResponse({"error": "Пустой текст"}, status_code=400)
    
    try:
        translator = GoogleTranslator(source='auto', target='ru')
        translated = translator.translate(request.text[:4500])
        print(f"✅ Done\n")
        return JSONResponse({"result": translated})
    except Exception as e:
        print(f"❌ Error: {e}\n")
        return JSONResponse({"error": str(e)}, status_code=500)

# Health check для хостинга
@app.get("/health")
def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8002))
    print("\n" + "="*70)
    print("🚀 NEWS TRANSLATOR SERVER")
    print("="*70)
    print(f"📍 URL: http://localhost:{port}")
    print(f"📍 API: http://localhost:{port}/docs")
    print("="*70 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
