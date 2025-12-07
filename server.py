from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import requests
from bs4 import BeautifulSoup, NavigableString
from deep_translator import GoogleTranslator
import time
import random
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class LinkRequest(BaseModel):
    url: str

RSS_FEEDS = {
    "python": "https://realpython.com/atom.xml",
    "java": "https://www.baeldung.com/feed",
    "csharp": "https://devblogs.microsoft.com/dotnet/feed/"
}

# Кэш для новостей (глобальная переменная)
NEWS_CACHE = {}

def translate_text_parallel(text_nodes, translator):
    """
    Безопасный параллельный перевод (3 потока)
    """
    def process_node(node):
        original = str(node)
        # Если текст огромный - лучше не трогать (риск таймаута)
        if len(original) > 3000: 
            return original
            
        # Защита кода (простая эвристика)
        if any(original.strip().startswith(kw) for kw in ['def ', 'class ', 'import ', 'from ', 'return ', 'public ', 'void ']):
             return None
             
        try:
            # Переводим
            return translator.translate(original)
        except:
            return None

    # Используем 3 потока (оптимально для Render Free Tier)
    with ThreadPoolExecutor(max_workers=3) as executor:
        results = list(executor.map(process_node, text_nodes))
    
    return results

def translate_html_content(soup):
    translator = GoogleTranslator(source='auto', target='ru')
    
    # 1. Поиск контента
    content = soup.find('article') or soup.find('main') or soup.find('div', class_='content') or soup.body
    if not content: return "<p>Контент не найден :(</p>"

    # 2. Чистка
    for junk in content(["script", "style", "iframe", "noscript", "svg"]):
        junk.decompose()

    # 3. Code Guard (помечаем теги кода)
    code_blocks = content.find_all(['pre', 'code', 'kbd', 'samp'])
    for block in code_blocks: 
        block['data-no-translate'] = 'true'

    # 4. Сбор текста для перевода
    text_nodes_to_translate = []
    
    # Ищем только текст, который не внутри кода
    for node in content.find_all(text=True):
        if len(node.strip()) < 3: continue # Пропускаем короткий мусор
        
        parent = node.parent
        if parent.name in ['pre', 'code', 'script', 'style']: continue
        if any('code' in c for c in parent.get('class', [])): continue
        if parent.get('data-no-translate'): continue
        
        text_nodes_to_translate.append(node)

    # 5. Параллельный перевод
    if text_nodes_to_translate:
        translated_texts = translate_text_parallel(text_nodes_to_translate, translator)

        # 6. Замена текста
        for node, translated in zip(text_nodes_to_translate, translated_texts):
            if translated:
                node.replace_with(translated)

    # 7. Картинки (стилизация)
    for img in content.find_all('img'):
        img['style'] = "max-width: 100%; height: auto; border-radius: 12px; margin: 20px 0; display: block;"
        if not img.get('src') and img.get('data-src'): 
            img['src'] = img['data-src']

    return content.prettify()

@lru_cache(maxsize=10)
def fetch_feed(tag):
    """Скачивание новостей (с кэшированием)"""
    url = RSS_FEEDS.get(tag)
    if not url: return []
    try:
        resp = requests.get(url, timeout=5)
        soup = BeautifulSoup(resp.content, "xml")
        articles = []
        items = soup.find_all("entry")[:4] or soup.find_all("item")[:4]
        
        translator = GoogleTranslator(source='auto', target='ru')
        
        for item in items:
            title = item.find("title").text
            # Поиск ссылки (для Atom и RSS)
            link_tag = item.find("link")
            if link_tag:
                link = link_tag.text if link_tag.text else link_tag.get('href')
            else:
                continue

            # Перевод заголовка (один раз)
            try:
                ru_title = translator.translate(title)
            except:
                ru_title = title
                
            articles.append({"title": ru_title, "original_title": title, "link": link, "tag": tag})
        return articles
    except Exception as e:
        print(f"Ошибка ленты {tag}: {e}")
        return []

@app.get("/feed")
async def get_news():
    """Отдает новости из кэша (быстро)"""
    all_news = []
    current_time = time.time()
    
    for tag in ["python", "java", "csharp"]:
        # Проверяем кэш (время жизни 1 час = 3600 сек)
        if tag in NEWS_CACHE and (current_time - NEWS_CACHE[tag]['time'] < 3600):
             all_news.extend(NEWS_CACHE[tag]['data'])
        else:
             data = fetch_feed(tag)
             # Если скачалось успешно - обновляем кэш
             if data:
                NEWS_CACHE[tag] = {'data': data, 'time': current_time}
                all_news.extend(data)
             # Если ошибка сети - отдаем старый кэш если есть
             elif tag in NEWS_CACHE:
                all_news.extend(NEWS_CACHE[tag]['data'])
    
    random.shuffle(all_news)
    return {"articles": all_news}

@app.post("/translate")
async def translate_article(request: LinkRequest):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/100.0.0.0 Safari/537.36'}
        # Таймаут на скачивание 15 сек
        response = requests.get(request.url, headers=headers, timeout=15)
        
        # Используем lxml для скорости (если он есть в requirements.txt)
        # Если нет - упадет на html.parser, что тоже норм
        try:
            soup = BeautifulSoup(response.text, 'lxml')
        except:
            soup = BeautifulSoup(response.text, 'html.parser')
        
        final_html = translate_html_content(soup)
        
        return {"translated_html": final_html}
    except Exception as e:
        return {"error": str(e)}

# Запуск локально
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
