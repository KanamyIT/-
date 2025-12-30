from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import requests
from bs4 import BeautifulSoup
from deep_translator import GoogleTranslator
import random
from functools import lru_cache
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
import io
from gtts import gTTS

app = FastAPI(title="Article Translator API", version="3.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# GZIP
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Thread pool
executor = ThreadPoolExecutor(max_workers=4)

# Переводчик
_translator = GoogleTranslator(source='auto', target='ru')

# ========== МОДЕЛИ ==========

class LinkRequest(BaseModel):
    url: str

class TextRequest(BaseModel):
    text: str

class TTSRequest(BaseModel):
    text: str
    voice: Optional[str] = "ru"
    speed: Optional[float] = 1.0

# ========== RSS ==========

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

# ========== ENDPOINTS ==========

@app.get("/")
async def root():
    return {
        "status": "Alive", 
        "version": "3.0",
        "message": "API работает корректно"
    }

@app.get("/health")
async def health():
    return {"status": "OK", "translator": "active"}

# ========== TTS ==========

@app.post("/text_to_speech")
async def text_to_speech(request: TTSRequest):
    try:
        if not request.text or len(request.text.strip()) == 0:
            raise HTTPException(status_code=400, detail="Empty text")
        
        text = request.text[:5000]
        
        def generate_speech():
            tts = gTTS(text=text, lang=request.voice, slow=False)
            fp = io.BytesIO()
            tts.write_to_fp(fp)
            fp.seek(0)
            return fp.read()
        
        audio_data = await asyncio.get_event_loop().run_in_executor(
            executor, generate_speech
        )
        
        return StreamingResponse(
            io.BytesIO(audio_data),
            media_type="audio/mpeg",
            headers={"Content-Disposition": "attachment; filename=audio.mp3"}
        )
        
    except Exception as e:
        print(f"TTS Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ========== ПЕРЕВОД ТЕКСТА ==========

@app.post("/translate_text")
async def translate_text(request: TextRequest):
    try:
        if not request.text:
            raise HTTPException(status_code=400, detail="Empty text")
        
        text = request.text[:4500]
        
        def do_translate():
            return _translator.translate(text)
        
        result = await asyncio.get_event_loop().run_in_executor(
            executor, do_translate
        )
        
        return {"result": result, "success": True}
        
    except Exception as e:
        print(f"Text translation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ========== ПЕРЕВОД HTML ==========

def translate_html_content(soup):
    try:
        content = None
        for selector in ['article', 'main', '.content', '.post-content', '.entry-content', 'body']:
            if selector.startswith('.'):
                content = soup.find(class_=selector[1:])
            else:
                content = soup.find(selector)
            if content:
                break
        
        if not content:
            return "<p>Контент не найден</p>", 0

        for tag in content(["script", "style", "iframe", "noscript", "svg", "form", "nav", "footer", "aside"]):
            tag.decompose()

        for code in content.find_all(['pre', 'code']):
            code['data-skip'] = 'true'

        word_count = 0
        
        for text_node in content.find_all(text=True):
            if not text_node.strip():
                continue
            
            parent = text_node.parent
            if parent.name in ['script', 'style', 'code', 'pre']:
                continue
            if parent.get('data-skip'):
                continue
            
            original = str(text_node).strip()
            if len(original) < 5 or len(original) > 3000:
                continue
            
            if any(original.startswith(kw) for kw in ['def ', 'class ', 'import ', 'const ', 'var ', 'function ']):
                continue
            
            try:
                translated = _translator.translate(original)
                text_node.replace_with(translated)
                word_count += len(translated.split())
            except Exception as e:
                print(f"Translation error: {e}")
                continue

        for img in content.find_all('img'):
            img['style'] = "max-width:100%;height:auto;border-radius:12px;margin:20px 0"
            img['loading'] = 'lazy'
            if not img.get('src') and img.get('data-src'):
                img['src'] = img['data-src']
        
        return str(content), word_count
        
    except Exception as e:
        print(f"HTML translation error: {e}")
        return f"<p>Ошибка обработки: {str(e)}</p>", 0

# ========== RSS ==========

@lru_cache(maxsize=50)
def fetch_feed(category: str):
    urls = RSS_SOURCES.get(category, [])
    articles = []
    
    for url in urls:
        try:
            resp = requests.get(url, timeout=5, headers={'User-Agent': 'Mozilla/5.0'})
            soup = BeautifulSoup(resp.content, "xml")
            
            items = soup.find_all("item")[:5] or soup.find_all("entry")[:5]
            
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
                    
                    try:
                        ru_title = _translator.translate(title[:200])
                    except:
                        ru_title = title
                    
                    articles.append({
                        "title": ru_title,
                        "original_title": title,
                        "link": link,
                        "tag": category.upper()
                    })
                except Exception as e:
                    print(f"Item parsing error: {e}")
                    continue
                    
        except Exception as e:
            print(f"RSS fetch error for {url}: {e}")
            continue
    
    return articles

@app.get("/feed")
async def get_feed(category: str = "programming"):
    try:
        if category not in RSS_SOURCES:
            category = "programming"
        
        articles = await asyncio.get_event_loop().run_in_executor(
            executor, fetch_feed, category
        )
        
        random.shuffle(articles)
        
        return {
            "articles": articles[:12],
            "category": category,
            "total": len(articles)
        }
    except Exception as e:
        print(f"Feed error: {e}")
        return {"articles": [], "error": str(e)}

@app.get("/categories")
async def get_categories():
    return {"categories": list(RSS_SOURCES.keys())}

# ========== ПЕРЕВОД СТАТЬИ ==========

@app.post("/translate")
async def translate_article(request: LinkRequest):
    print(f"\n=== Запрос на перевод: {request.url} ===")
    
    try:
        url = str(request.url)
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'ru,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }
        
        print(f"Загружаю страницу...")
        
        def fetch_page():
            return requests.get(url, headers=headers, timeout=20, allow_redirects=True)
        
        response = await asyncio.get_event_loop().run_in_executor(
            executor, fetch_page
        )
        
        print(f"Статус: {response.status_code}")
        
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail="Не удалось загрузить страницу")
        
        print("Парсинг HTML...")
        soup = BeautifulSoup(response.text, 'html.parser')
        
        title_tag = soup.find('h1') or soup.find('title')
        title = title_tag.text.strip() if title_tag else "Статья"
        
        print(f"Заголовок: {title}")
        print("Переводим контент...")
        
        translated_html, word_count = translate_html_content(soup)
        
        read_time = max(1, round(word_count / 200))
        
        print(f"Перевод завершен. Слов: {word_count}, Время чтения: {read_time} мин")
        
        result = {
            "title": title,
            "translated_html": translated_html,
            "read_time": read_time,
            "word_count": word_count,
            "url": url,
            "success": True
        }
        
        return result
        
    except requests.Timeout:
        print("ОШИБКА: Timeout")
        raise HTTPException(status_code=504, detail="Время ожидания истекло")
    except requests.RequestException as e:
        print(f"ОШИБКА REQUEST: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка загрузки: {str(e)}")
    except Exception as e:
        print(f"ОШИБКА: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Ошибка: {str(e)}")

# ========== ЗАПУСК ==========

if __name__ == "__main__":
    import uvicorn
    print("\n🚀 Запуск сервера...")
    print("📍 URL: http://localhost:8000")
    print("📖 Docs: http://localhost:8000/docs\n")
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info"
    )
