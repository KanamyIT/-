from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import requests
from bs4 import BeautifulSoup, NavigableString
from deep_translator import GoogleTranslator
import time
import random

app = FastAPI()

# Настройки доступа (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class LinkRequest(BaseModel):
    url: str

# === ИСТОЧНИКИ НОВОСТЕЙ ===
RSS_FEEDS = {
    "python": "https://realpython.com/atom.xml",
    "java": "https://www.baeldung.com/feed",
    "csharp": "https://devblogs.microsoft.com/dotnet/feed/"
}

def translate_html_content(soup, translator):
    """
    Умный переводчик:
    1. Сохраняет верстку (картинки, заголовки).
    2. Защищает код от перевода.
    3. Переводит только текст.
    """
    # 1. Находим основной контент
    content = soup.find('article') or soup.find('main') or soup.find('div', class_='content') or soup.body
    
    if not content:
        return "<p>Не удалось найти контент :(</p>"

    # 2. Чистка мусора
    for junk in content(["script", "style", "iframe", "noscript", "svg"]):
        junk.decompose()

    # 3. CODE GUARD (Защита кода)
    # Находим все блоки с кодом и помечаем их
    code_blocks = content.find_all(['pre', 'code', 'kbd', 'samp', 'var'])
    for block in code_blocks:
        block['data-no-translate'] = 'true'
    
    # 4. Перевод текста (Проход по дереву)
    text_nodes = content.find_all(text=True)

    for node in text_nodes:
        # Условия пропуска (SKIP):
        if len(node.strip()) < 3: continue # Пустой текст
        
        parent = node.parent
        # Если внутри кода
        if parent.name in ['pre', 'code', 'kbd', 'samp', 'script', 'style']:
            continue
        
        # Если класс родителя похож на код
        parent_classes = parent.get('class', [])
        if any('code' in cls or 'highlight' in cls for cls in parent_classes):
            continue

        # Если помечено нами
        if parent.get('data-no-translate'):
            continue

        # Если текст похож на код (def, class, import)
        original_text = str(node)
        if any(original_text.strip().startswith(kw) for kw in ['def ', 'class ', 'import ', 'from ', 'return ', 'public ', 'void ']):
             continue

        # ПЕРЕВОД
        try:
            translated_text = translator.translate(original_text)
            node.replace_with(translated_text)
        except:
            pass

    # 5. Обработка картинок
    for img in content.find_all('img'):
        img['style'] = "max-width: 100%; height: auto; border-radius: 12px; margin: 20px 0; display: block;"
        if not img.get('src') and img.get('data-src'):
            img['src'] = img['data-src']

    return content.prettify()

def get_feed_data(tag):
    """Скачивает новости по тегу из RSS"""
    url = RSS_FEEDS.get(tag)
    if not url: return []
    
    try:
        resp = requests.get(url, timeout=10)
        # Используем xml парсер для RSS
        soup = BeautifulSoup(resp.content, "xml") 
        articles = []
        
        # Берем первые 4 статьи
        items = soup.find_all("entry")[:4] or soup.find_all("item")[:4]
        
        for item in items:
            title = item.find("title").text
            # Ищем ссылку (в Atom и RSS она разная)
            link_tag = item.find("link")
            if link_tag:
                link = link_tag.text if link_tag.text else link_tag.get('href')
            else:
                continue

            # Переводим заголовок для удобства
            try:
                ru_title = GoogleTranslator(source='auto', target='ru').translate(title)
            except:
                ru_title = title
                
            articles.append({
                "title": ru_title,
                "original_title": title,
                "link": link,
                "tag": tag
            })
        return articles
    except Exception as e:
        print(f"Ошибка RSS {tag}: {e}")
        return []

@app.get("/feed")
async def get_news():
    """API для получения ленты новостей"""
    all_news = []
    for tag in ["python", "java", "csharp"]:
        news = get_feed_data(tag)
        all_news.extend(news)
    
    random.shuffle(all_news)
    return {"articles": all_news}

@app.post("/translate")
async def translate_article(request: LinkRequest):
    """API для перевода статьи"""
    start_time = time.time()
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/100.0.0.0 Safari/537.36'}
        response = requests.get(request.url, headers=headers, timeout=25)
        soup = BeautifulSoup(response.text, 'html.parser')
        translator = GoogleTranslator(source='auto', target='ru')

        final_html = translate_html_content(soup, translator)
        
        return {
            "translated_html": final_html,
            "stats": f"Готово за {round(time.time() - start_time, 2)} сек."
        }
    except Exception as e:
        return {"error": f"Ошибка: {str(e)}"}

# Запуск для локального компа (Render это игнорирует)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
