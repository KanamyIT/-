from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import requests
from bs4 import BeautifulSoup
from deep_translator import GoogleTranslator
import random
from functools import lru_cache
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
import io

# Для TTS - можно выбрать один из вариантов:
# 1. gTTS (бесплатный, но простой голос)
from gtts import gTTS

# 2. Для Google Cloud TTS (качественнее):
# from google.cloud import texttospeech

# 3. Для ElevenLabs (лучшее качество):
# from elevenlabs import generate, set_api_key

app = FastAPI(title="Article Translator & TTS API", version="2.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Модели данных
class LinkRequest(BaseModel):
    url: HttpUrl

class TextRequest(BaseModel):
    text: str

class TTSRequest(BaseModel):
    text: str
    voice: Optional[str] = "ru"  # ru, en, etc.
    speed: Optional[float] = 1.0

# RSS источники
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

# Thread pool для блокирующих операций
executor = ThreadPoolExecutor(max_workers=4)

# Кеш для переводчика
_translator = GoogleTranslator(source='auto', target='ru')

@app.get("/")
async def root():
    return {
        "status": "Alive", 
        "version": "2.0",
        "features": ["translation", "rss_feed", "text_to_speech"]
    }

# ==================== НОВАЯ ФУНКЦИЯ: TEXT-TO-SPEECH ====================

@app.post("/text_to_speech")
async def text_to_speech(request: TTSRequest):
    """
    Преобразование текста в речь с улучшенным голосом
    Варианты голосов: ru, en, fr, de и др.
    """
    if not request.text or len(request.text.strip()) == 0:
        raise HTTPException(status_code=400, detail="Text is empty")
    
    # Ограничение длины текста
    text = request.text[:5000]
    
    try:
        # Вариант 1: gTTS (бесплатный, простой)
        def generate_speech():
            tts = gTTS(text=text, lang=request.voice, slow=False)
            fp = io.BytesIO()
            tts.write_to_fp(fp)
            fp.seek(0)
            return fp
        
        audio_fp = await asyncio.get_event_loop().run_in_executor(
            executor, generate_speech
        )
        
        return StreamingResponse(
            audio_fp, 
            media_type="audio/mpeg",
            headers={"Content-Disposition": "attachment; filename=article_audio.mp3"}
        )
        
        # Вариант 2: Google Cloud TTS (раскомментируйте, если используете)
        """
        client = texttospeech.TextToSpeechClient()
        
        synthesis_input = texttospeech.SynthesisInput(text=text)
        
        # Настройка голоса - WaveNet для лучшего качества
        voice = texttospeech.VoiceSelectionParams(
            language_code="ru-RU",
            name="ru-RU-Wavenet-D",  # Женский голос
            # name="ru-RU-Wavenet-B",  # Мужской голос
            ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
        )
        
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=request.speed,
            pitch=0.0
        )
        
        response = client.synthesize_speech(
            input=synthesis_input, 
            voice=voice, 
            audio_config=audio_config
        )
        
        audio_fp = io.BytesIO(response.audio_content)
        return StreamingResponse(audio_fp, media_type="audio/mpeg")
        """
        
        # Вариант 3: ElevenLabs (раскомментируйте, если используете)
        """
        # set_api_key("YOUR_API_KEY")
        audio = generate(
            text=text,
            voice="Bella",  # или другой голос
            model="eleven_multilingual_v2"
        )
        audio_fp = io.BytesIO(audio)
        return StreamingResponse(audio_fp, media_type="audio/mpeg")
        """
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS error: {str(e)}")

# ==================== ОПТИМИЗИРОВАННЫЕ ФУНКЦИИ ====================

@app.post("/translate_text")
async def translate_raw_text(request: TextRequest):
    """Перевод обычного текста"""
    if not request.text:
        raise HTTPException(status_code=400, detail="Empty text")
    
    try:
        # Асинхронный перевод
        def translate():
            return _translator.translate(request.text[:4500])
        
        translated = await asyncio.get_event_loop().run_in_executor(
            executor, translate
        )
        return {"result": translated, "chars": len(translated)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def translate_html_content(soup):
    """Оптимизированный перевод HTML контента"""
    content = None
    
    # Приоритетные селекторы для контента
    selectors = [
        ('article', None),
        ('main', None),
        (None, 'content'),
        (None, 'post-content'),
        (None, 'entry-content'),
        ('body', None)
    ]
    
    for tag, class_name in selectors:
        if class_name:
            content = soup.find(class_=class_name)
        else:
            content = soup.find(tag)
        if content:
            break
    
    if not content:
        return "<p>Не удалось найти контент</p>", 0

    # Удаление мусора
    for junk in content(["script", "style", "iframe", "noscript", 
                         "svg", "form", "button", "nav", "footer", 
                         "aside", "ad", "advertisement"]):
        junk.decompose()

    # Защита кода от перевода
    for block in content.find_all(['pre', 'code', 'kbd', 'samp']):
        block['data-no-translate'] = 'true'

    word_count = 0
    texts_to_translate = []
    nodes_to_update = []

    # Сбор текстов для batch перевода
    for node in content.find_all(text=True):
        original = str(node).strip()
        
        if len(original) < 3:
            continue
            
        parent = node.parent
        
        # Пропуск кода и специальных элементов
        if parent.name in ['pre', 'code', 'script', 'style']:
            continue
        if parent.get('data-no-translate'):
            continue
        if len(original) > 4000:
            continue
        
        # Пропуск кода (улучшенная проверка)
        code_keywords = ['def ', 'class ', 'import ', 'from ', 'return ', 
                        'public ', 'void ', 'function ', 'const ', 'var ', 'let ']
        if any(original.startswith(kw) for kw in code_keywords):
            continue
        
        texts_to_translate.append(original)
        nodes_to_update.append(node)
    
    # Batch перевод (по частям)
    batch_size = 10
    for i in range(0, len(texts_to_translate), batch_size):
        batch = texts_to_translate[i:i+batch_size]
        batch_nodes = nodes_to_update[i:i+batch_size]
        
        for text, node in zip(batch, batch_nodes):
            try:
                translated = _translator.translate(text)
                node.replace_with(translated)
                word_count += len(translated.split())
            except:
                pass  # Оставляем оригинальный текст при ошибке

    # Оптимизация изображений
    for img in content.find_all('img'):
        img['style'] = ("max-width: 100%; height: auto; border-radius: 12px; "
                       "margin: 20px 0; display: block; box-shadow: 0 4px 8px rgba(0,0,0,0.1);")
        # Lazy loading
        if not img.get('loading'):
            img['loading'] = 'lazy'
        # Data-src fix
        if not img.get('src') and img.get('data-src'):
            img['src'] = img['data-src']
    
    return content.prettify(), word_count

@lru_cache(maxsize=50)  # Увеличен размер кеша
def fetch_feed_category(category: str):
    """Кешированная загрузка RSS"""
    urls = RSS_SOURCES.get(category, [])
    all_articles = []
    
    for url in urls:
        try:
            resp = requests.get(url, timeout=5, headers={
                'User-Agent': 'Mozilla/5.0 (compatible; ArticleBot/1.0)'
            })
            soup = BeautifulSoup(resp.content, "xml")
            
            items = soup.find_all("item")[:5] or soup.find_all("entry")[:5]
            
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
                    ru_title = _translator.translate(title[:200])
                except:
                    ru_title = title
                
                all_articles.append({
                    "title": ru_title,
                    "original_title": title,
                    "link": link,
                    "tag": category.upper()
                })
        except Exception as e:
            print(f"Error fetching {url}: {e}")
            continue
    
    return all_articles

@app.get("/feed")
async def get_news(category: str = "programming"):
    """Получение новостной ленты по категории"""
    if category not in RSS_SOURCES:
        category = "programming"
    
    # Асинхронная загрузка
    articles = await asyncio.get_event_loop().run_in_executor(
        executor, fetch_feed_category, category
    )
    
    random.shuffle(articles)
    
    return {
        "articles": articles[:12],
        "category": category,
        "total": len(articles)
    }

@app.get("/categories")
async def get_categories():
    """Список доступных категорий"""
    return {
        "categories": list(RSS_SOURCES.keys()),
        "total": len(RSS_SOURCES)
    }

@app.post("/translate")
async def translate_article(request: LinkRequest):
    """Перевод статьи по URL"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml',
            'Accept-Language': 'en-US,en;q=0.9'
        }
        
        # Асинхронный запрос
        def fetch_page():
            return requests.get(str(request.url), headers=headers, timeout=20)
        
        response = await asyncio.get_event_loop().run_in_executor(
            executor, fetch_page
        )
        
        # Парсинг
        try:
            soup = BeautifulSoup(response.text, 'lxml')
        except:
            soup = BeautifulSoup(response.text, 'html.parser')
        
        # Извлечение заголовка
        title_tag = soup.find('h1') or soup.find('title')
        title = title_tag.text.strip() if title_tag else "Без заголовка"
        
        # Перевод контента
        final_html, words = translate_html_content(soup)
        
        # Время чтения: 200 слов/мин
        read_time = max(1, round(words / 200))
        
        return {
            "title": title,
            "translated_html": final_html,
            "read_time": read_time,
            "word_count": words,
            "url": str(request.url)
        }
        
    except requests.Timeout:
        raise HTTPException(status_code=504, detail="Request timeout")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Health check
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "translator": "active",
        "tts": "active"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=2)
