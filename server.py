from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
from bs4 import BeautifulSoup
import json

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class LinkRequest(BaseModel):
    url: str

class TextRequest(BaseModel):
    text: str

# ========== РАБОЧИЕ RSS ИСТОЧНИКИ ==========

FEEDS = {
    "programming": [
        {
            "url": "https://dev.to/feed",
            "type": "rss"
        },
        {
            "url": "https://www.reddit.com/r/programming/.rss",
            "type": "rss"
        }
    ],
    "history": [
        {
            "url": "https://www.reddit.com/r/history/.rss",
            "type": "rss"
        }
    ],
    "gaming": [
        {
            "url": "https://www.reddit.com/r/gaming/.rss",
            "type": "rss"
        }
    ],
    "movies": [
        {
            "url": "https://www.reddit.com/r/movies/.rss",
            "type": "rss"
        }
    ]
}

# ========== ПРОСТАЯ ФУНКЦИЯ ПЕРЕВОДА (без Google) ==========

def simple_translate(text):
    """Упрощенный 'перевод' - возвращает текст как есть"""
    # В будущем можете подключить любой переводчик
    return text

# ========== ENDPOINTS ==========

@app.get("/")
def root():
    print("✅ Root endpoint вызван")
    return {"status": "OK", "message": "Сервер работает!"}

@app.get("/health")
def health():
    print("✅ Health check")
    return {"status": "OK", "message": "Сервер активен"}

@app.get("/feed")
def get_feed(category: str = "programming"):
    print(f"\n{'='*60}")
    print(f"📰 ЗАПРОС ЛЕНТЫ: {category}")
    print(f"{'='*60}")
    
    if category not in FEEDS:
        category = "programming"
    
    articles = []
    
    for feed_info in FEEDS[category]:
        try:
            print(f"📡 Загружаю: {feed_info['url']}")
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(feed_info['url'], headers=headers, timeout=10)
            print(f"✅ Статус: {response.status_code}")
            
            if response.status_code != 200:
                print(f"❌ Ошибка загрузки")
                continue
            
            soup = BeautifulSoup(response.content, 'xml')
            
            # Ищем записи
            entries = soup.find_all('entry')[:10]
            if not entries:
                entries = soup.find_all('item')[:10]
            
            print(f"📊 Найдено записей: {len(entries)}")
            
            for entry in entries:
                try:
                    # Заголовок
                    title_tag = entry.find('title')
                    if not title_tag:
                        continue
                    title = title_tag.get_text().strip()
                    
                    # Ссылка
                    link_tag = entry.find('link')
                    if link_tag:
                        link = link_tag.get('href') or link_tag.get_text().strip()
                    else:
                        continue
                    
                    # Очистка заголовка от HTML
                    title = BeautifulSoup(title, 'html.parser').get_text()
                    
                    articles.append({
                        "title": title[:150],  # Ограничиваем длину
                        "original_title": title[:150],
                        "link": link,
                        "tag": category.upper()
                    })
                    
                except Exception as e:
                    print(f"⚠️ Ошибка парсинга записи: {e}")
                    continue
            
            print(f"✅ Добавлено статей: {len(articles)}")
            
        except Exception as e:
            print(f"❌ ОШИБКА загрузки фида: {e}")
            continue
    
    print(f"\n📦 ИТОГО статей: {len(articles)}")
    print(f"{'='*60}\n")
    
    # Если ничего не нашли, добавляем тестовые
    if len(articles) == 0:
        print("⚠️ Добавляю тестовые статьи")
        articles = [
            {
                "title": "Тестовая статья 1 - Программирование",
                "original_title": "Test Article 1",
                "link": "https://example.com/test1",
                "tag": category.upper()
            },
            {
                "title": "Тестовая статья 2 - Туториал",
                "original_title": "Test Article 2",
                "link": "https://example.com/test2",
                "tag": category.upper()
            }
        ]
    
    return {
        "articles": articles[:15],
        "category": category,
        "total": len(articles)
    }

@app.post("/translate")
def translate_article(request: LinkRequest):
    print(f"\n{'='*60}")
    print(f"🔄 ПЕРЕВОД СТАТЬИ")
    print(f"URL: {request.url}")
    print(f"{'='*60}")
    
    try:
        url = str(request.url)
        
        # Проверка на тестовые URL
        if "example.com" in url:
            print("ℹ️ Тестовый URL - возвращаю заглушку")
            return {
                "title": "Тестовая статья",
                "translated_html": """
                    <div style="padding: 20px;">
                        <h2>Это тестовая статья</h2>
                        <p>Если вы видите это сообщение, значит сервер работает корректно!</p>
                        <p>Попробуйте вставить реальную ссылку на статью, например:</p>
                        <ul>
                            <li>https://dev.to/...</li>
                            <li>https://medium.com/...</li>
                            <li>Любую статью с интернета</li>
                        </ul>
                    </div>
                """,
                "read_time": 1,
                "word_count": 50,
                "url": url,
                "success": True
            }
        
        print(f"📡 Загружаю страницу...")
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
        }
        
        response = requests.get(url, headers=headers, timeout=15, allow_redirects=True)
        
        print(f"✅ Статус: {response.status_code}")
        
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail=f"Не удалось загрузить страницу (статус {response.status_code})")
        
        print(f"📄 Парсинг HTML...")
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Получаем заголовок
        title = "Статья"
        title_tag = soup.find('h1')
        if title_tag:
            title = title_tag.get_text().strip()
        elif soup.find('title'):
            title = soup.find('title').get_text().strip()
        
        print(f"📌 Заголовок: {title}")
        
        # Ищем контент
        content = None
        selectors = [
            'article',
            'main',
            {'class': 'post-content'},
            {'class': 'article-content'},
            {'class': 'entry-content'},
            {'class': 'content'},
            {'id': 'content'},
            'body'
        ]
        
        for selector in selectors:
            if isinstance(selector, str):
                content = soup.find(selector)
            else:
                content = soup.find(**selector)
            
            if content:
                print(f"✅ Контент найден: {selector}")
                break
        
        if not content:
            print("❌ Контент не найден")
            raise HTTPException(status_code=400, detail="Не удалось найти контент статьи")
        
        # Очищаем от мусора
        for tag in content(['script', 'style', 'nav', 'header', 'footer', 'aside', 'iframe', 'noscript']):
            tag.decompose()
        
        print(f"🧹 Очистка завершена")
        
        # Подсчитываем слова
        text = content.get_text()
        words = len(text.split())
        read_time = max(1, round(words / 200))
        
        print(f"📊 Слов: {words}, Время чтения: {read_time} мин")
        
        # Получаем HTML
        html_content = str(content)
        
        # Обрабатываем изображения
        for img in content.find_all('img'):
            if img.get('src'):
                img['style'] = 'max-width:100%;height:auto;border-radius:8px;margin:20px 0;'
                img['loading'] = 'lazy'
        
        print(f"✅ ПЕРЕВОД ЗАВЕРШЕН")
        print(f"{'='*60}\n")
        
        return {
            "title": title,
            "translated_html": html_content,
            "read_time": read_time,
            "word_count": words,
            "url": url,
            "success": True
        }
        
    except requests.Timeout:
        print(f"❌ TIMEOUT")
        raise HTTPException(status_code=504, detail="Время ожидания истекло")
    
    except requests.RequestException as e:
        print(f"❌ REQUEST ERROR: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка загрузки: {str(e)}")
    
    except Exception as e:
        print(f"❌ ОБЩАЯ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Ошибка: {str(e)}")

@app.post("/translate_text")
def translate_text(request: TextRequest):
    print(f"📝 Перевод текста (длина: {len(request.text)})")
    
    # Просто возвращаем текст (можете добавить переводчик позже)
    return {
        "result": request.text,
        "success": True
    }

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("🚀 ЗАПУСК СЕРВЕРА")
    print("="*60)
    print("📍 URL: http://localhost:8000")
    print("📖 Документация: http://localhost:8000/docs")
    print("="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
