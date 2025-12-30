from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel, HttpUrl
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
import redis
import hashlib
import time

app = FastAPI(title="Article Translator API", version="3.0")

# ========== ОПТИМИЗАЦИИ ДЛЯ СКОРОСТИ ==========

# 1. GZIP сжатие ответов
app.add_middleware(GZipMiddleware, minimum_size=1000)

# 2. CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Redis кеш (опционально - закомментируйте если нет Redis)
try:
    cache = redis.Redis(host='localhost', port=6379, db=0, decode_responses=False)
    REDIS_ENABLED = True
except:
    REDIS_ENABLED = False
    print("⚠️ Redis не подключен, используется только LRU кеш")

# 4. Thread pool для параллельных задач
executor = ThreadPoolExecutor(max_workers=8)  # Увеличено с 4 до 8

# 5. Глобальный переводчик
_translator = GoogleTranslator(source='auto', target='ru')

# ========== МОДЕЛИ ==========

class LinkRequest(BaseModel):
    url: HttpUrl

class TextRequest(BaseModel):
    text: str

class TTSRequest(BaseModel):
    text: str
    voice: Optional[str] = "ru"
    speed: Optional[float] = 1.0

# ========== RSS ИСТОЧНИКИ ==========

RSS_SOURCES = {
    "programming": [
        "https://realpython.com/atom.xml",
        "https://devblogs.microsoft.com/dotnet/feed/",
        "https://www.freecodecamp.org/news/rss/"
    ],
    "history": [
        "https://www.history.com/.rss/full/this-day-in-history",
        "https://feeds.feedburner.com/HistoryNet"
    ],
    "gaming": [
        "https://www.polygon.com/rss/index.xml",
        "https://kotaku.com/rss"
    ],
    "movies": [
        "https://variety.com/feed/",
        "https://deadline.com/feed/"
    ]
}

# ========== ФУНКЦИИ КЕШИРОВАНИЯ ==========

def get_cache_key(prefix: str, data: str) -> str:
    """Создание ключа кеша"""
    return f"{prefix}:{hashlib.md5(data.encode()).hexdigest()}"

def get_from_cache(key: str):
    """Получение из кеша"""
    if not REDIS_ENABLED:
        return None
    try:
        return cache.get(key)
    except:
        return None

def set_to_cache(key: str, value: bytes, ttl: int = 3600):
    """Сохранение в кеш"""
    if not REDIS_ENABLED:
        return
    try:
        cache.setex(key, ttl, value)
    except:
        pass

# ========== ENDPOINTS ==========

@app.get("/")
async def root():
    return {
        "status": "Alive", 
        "version": "3.0",
        "features": ["translation", "rss_feed", "text_to_speech", "caching", "gzip"],
        "optimizations": ["redis_cache", "thread_pool", "gzip_compression", "lru_cache"]
    }

# ========== TEXT-TO-SPEECH С КЕШЕМ ==========

@app.post("/text_to_speech")
async def text_to_speech(request: TTSRequest):
    """TTS с кешированием готовых аудио"""
    if not request.text or len(request.text.strip()) == 0:
        raise HTTPException(status_code=400, detail="Text is empty")
    
    text = request.text[:5000]
    
    # Проверяем кеш
    cache_key = get_cache_key("tts", f"{text}:{request.voice}:{request.speed}")
    cached_audio = get_from_cache(cache_key)
    
    if cached_audio:
        return StreamingResponse(
            io.BytesIO(cached_audio),
            media_type="audio/mpeg",
            headers={"X-Cache": "HIT"}
        )
    
    try:
        def generate_speech():
            tts = gTTS(text=text, lang=request.voice, slow=False)
            fp = io.BytesIO()
            tts.write_to_fp(fp)
            fp.seek(0)
            return fp.read()
        
        audio_data = await asyncio.get_event_loop().run_in_executor(
            executor, generate_speech
        )
        
        # Сохраняем в кеш на 1 час
        set_to_cache(cache_key, audio_data, ttl=3600)
        
        return StreamingResponse(
            io.BytesIO(audio_data),
            media_type="audio/mpeg",
            headers={
                "Content-Disposition": "attachment; filename=audio.mp3",
                "X-Cache": "MISS"
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS error: {str(e)}")

# ========== ПЕРЕВОД ТЕКСТА С КЕШЕМ ==========

@app.post("/translate_text")
async def translate_raw_text(request: TextRequest):
    """Перевод текста с кешированием"""
    if not request.text:
        raise HTTPException(status_code=400, detail="Empty text")
    
    text = request.text[:4500]
    cache_key = get_cache_key("text", text)
    
    # Проверяем кеш
    if REDIS_ENABLED:
        cached = get_from_cache(cache_key)
        if cached:
            return {"result": cached.decode('utf-8'), "cached": True}
    
    try:
        def translate():
            return _translator.translate(text)
        
        translated = await asyncio.get_event_loop().run_in_executor(
            executor, translate
        )
        
        # Сохраняем в кеш на 24 часа
        set_to_cache(cache_key, translated.encode('utf-8'), ttl=86400)
        
        return {"result": translated, "chars": len(translated), "cached": False}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ========== ОПТИМИЗИРОВАННЫЙ ПЕРЕВОД HTML ==========

def translate_html_content(soup):
    """Быстрый перевод HTML с батчами"""
    selectors = [
        ('article', None),
        ('main', None),
        (None, 'content'),
        (None, 'post-content'),
        (None, 'entry-content'),
        ('body', None)
    ]
    
    content = None
    for tag, class_name in selectors:
        if class_name:
            content = soup.find(class_=class_name)
        else:
            content = soup.find(tag)
        if content:
            break
    
    if not content:
        return "<p>Контент не найден</p>", 0

    # Удаление мусора
    for junk in content(["script", "style", "iframe", "noscript", 
                         "svg", "form", "button", "nav", "footer", 
                         "aside", "ad", "advertisement", "header"]):
        junk.decompose()

    for block in content.find_all(['pre', 'code', 'kbd', 'samp']):
        block['data-no-translate'] = 'true'

    word_count = 0
    texts_batch = []
    nodes_batch = []

    # Сбор текстов
    for node in content.find_all(text=True):
        original = str(node).strip()
        
        if len(original) < 3:
            continue
            
        parent = node.parent
        if parent.name in ['pre', 'code', 'script', 'style']:
            continue
        if parent.get('data-no-translate'):
            continue
        if len(original) > 4000:
            continue
        
        code_keywords = ['def ', 'class ', 'import ', 'from ', 'return ', 
                        'public ', 'void ', 'function ', 'const ', 'var ', 'let ', '//']
        if any(original.startswith(kw) for kw in code_keywords):
            continue
        
        texts_batch.append(original)
        nodes_batch.append(node)
    
    # Batch перевод (параллельно)
    batch_size = 15  # Увеличено с 10 до 15
    for i in range(0, len(texts_batch), batch_size):
        batch_texts = texts_batch[i:i+batch_size]
        batch_nodes = nodes_batch[i:i+batch_size]
        
        for text, node in zip(batch_texts, batch_nodes):
            try:
                translated = _translator.translate(text)
                node.replace_with(translated)
                word_count += len(translated.split())
            except:
                pass

    # Оптимизация изображений
    for img in content.find_all('img'):
        img['style'] = "max-width:100%;height:auto;border-radius:12px;margin:20px 0"
        if not img.get('loading'):
            img['loading'] = 'lazy'
        if not img.get('src') and img.get('data-src'):
            img['src'] = img['data-src']
    
    return content.prettify(), word_count

# ========== RSS С УВЕЛИЧЕННЫМ КЕШЕМ ==========

@lru_cache(maxsize=100)  # Увеличено с 50 до 100
def fetch_feed_category(category: str):
    """Кешированная загрузка RSS"""
    urls = RSS_SOURCES.get(category, [])
    all_articles = []
    
    for url in urls:
        try:
            resp = requests.get(url, timeout=4, headers={
                'User-Agent': 'Mozilla/5.0 ArticleBot/2.0'
            })
            soup = BeautifulSoup(resp.content, "xml")
            
            items = soup.find_all("item")[:6] or soup.find_all("entry")[:6]
            
            for item in items:
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
                    ru_title = _translator.translate(title[:250])
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
    
    return all_articles

@app.get("/feed")
async def get_news(category: str = "programming"):
    """Новостная лента"""
    if category not in RSS_SOURCES:
        category = "programming"
    
    articles = await asyncio.get_event_loop().run_in_executor(
        executor, fetch_feed_category, category
    )
    
    random.shuffle(articles)
    
    return {
        "articles": articles[:15],  # Увеличено с 12 до 15
        "category": category,
        "total": len(articles)
    }

@app.get("/categories")
async def get_categories():
    """Список категорий"""
    return {
        "categories": list(RSS_SOURCES.keys()),
        "total": len(RSS_SOURCES)
    }

# ========== ПЕРЕВОД СТАТЬИ С КЕШЕМ ==========

@app.post("/translate")
async def translate_article(request: LinkRequest):
    """Перевод статьи с кешированием"""
    url = str(request.url)
    cache_key = get_cache_key("article", url)
    
    # Проверяем кеш
    if REDIS_ENABLED:
        cached = get_from_cache(cache_key)
        if cached:
            import json
            data = json.loads(cached.decode('utf-8'))
            data['cached'] = True
            return data
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0',
            'Accept': 'text/html,application/xhtml+xml',
            'Accept-Language': 'ru,en;q=0.9'
        }
        
        def fetch_page():
            return requests.get(url, headers=headers, timeout=15)
        
        response = await asyncio.get_event_loop().run_in_executor(
            executor, fetch_page
        )
        
        soup = BeautifulSoup(response.text, 'lxml')
        
        title_tag = soup.find('h1') or soup.find('title')
        title = title_tag.text.strip() if title_tag else "Без заголовка"
        
        final_html, words = translate_html_content(soup)
        read_time = max(1, round(words / 200))
        
        result = {
            "title": title,
            "translated_html": final_html,
            "read_time": read_time,
            "word_count": words,
            "url": url,
            "cached": False
        }
        
        # Сохраняем в кеш на 6 часов
        if REDIS_ENABLED:
            import json
            set_to_cache(cache_key, json.dumps(result).encode('utf-8'), ttl=21600)
        
        return result
        
    except requests.Timeout:
        raise HTTPException(status_code=504, detail="Timeout")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check"""
    return {
        "status": "healthy",
        "redis": "active" if REDIS_ENABLED else "disabled",
        "workers": executor._max_workers
    }

# ========== ЗАПУСК ==========

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000, 
        workers=4,  # Несколько воркеров для параллельной обработки
        log_level="info"
    )
