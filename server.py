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

# Кэш для новостей (живет пока работает сервер)
NEWS_CACHE = {}

def translate_text_parallel(text_nodes, translator):
    """
    Переводит список текстовых узлов ПАРАЛЛЕЛЬНО.
    Это ускоряет перевод в 5-10 раз.
    """
    def process_node(node):
        original = str(node)
        # Проверка на код (быстрая)
        if any(original.strip().startswith(kw) for kw in ['def ', 'class ', 'import ', 'from ', 'return ', 'public ']):
             return None
        try:
            return translator.translate(original)
        except:
            return None

    # Используем 10 потоков для перевода
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(process_node, text_nodes))
    
    return results

def translate_html_content(soup):
    translator = GoogleTranslator(source='auto', target='ru')
    
    content = soup.find('article') or soup.find('main') or soup.find('div', class_='content') or soup.body
    if not content: return "<p>Контент не найден</p>"

    for junk in content(["script", "style", "iframe", "noscript", "svg"]):
        junk.decompose()

    code_blocks = content.find_all(['pre', 'code', 'kbd', 'samp'])
    for block in code_blocks: block['data-no-translate'] = 'true'

    # Собираем ВСЕ подходящие текстовые узлы в список
    text_nodes_to_translate = []
    
    for node in content.find_all(text=True):
        if len(node.strip()) < 3: continue
        parent = node.parent
        if parent.name in ['pre', 'code', 'script', 'style']: continue
        if any('code' in c for c in parent.get('class', [])): continue
        if parent.get('data-no-translate'): continue
        
        text_nodes_to_translate.append(node)

    # Переводим их пачкой (быстро)
    translated_texts = translate_text_parallel(text_nodes_to_translate, translator)

    # Применяем переводы обратно
    for node, translated in zip(text_nodes_to_translate, translated_texts):
        if translated:
            node.replace_with(translated)

    for img in content.find_all('img'):
        img['style'] = "max-width: 100%; height: auto; border-radius: 12px; margin: 20px 0; display: block;"
        if not img.get('src') and img.get('data-src'): img['src'] = img['data-src']

    return content.prettify()

@lru_cache(maxsize=10) # Кэшируем результат этой функции
def fetch_feed(tag):
    """Эта функция кэшируется, чтобы не долбить сайты новостей"""
    url = RSS_FEEDS.get(tag)
    if not url: return []
    try:
        resp = requests.get(url, timeout=5)
        soup = BeautifulSoup(resp.content, "xml")
        articles = []
        items = soup.find_all("entry")[:4] or soup.find_all("item")[:4]
        
        # Переводим заголовки тоже параллельно? Нет, их мало, можно так.
        # Но для скорости переведем только первый раз
        translator = GoogleTranslator(source='auto', target='ru')
        
        for item in items:
            title = item.find("title").text
            link = item.find("link").text if item.find("link").text else item.find("link")['href']
            try:
                ru_title = translator.translate(title)
            except:
                ru_title = title
            articles.append({"title": ru_title, "original_title": title, "link": link, "tag": tag})
        return articles
    except:
        return []

@app.get("/feed")
async def get_news():
    all_news = []
    # Теперь это работает мгновенно из кэша
    for tag in ["python", "java", "csharp"]:
        # Обертка для кэша (простая реализация через глобальную переменную для надежности)
        current_time = time.time()
        if tag in NEWS_CACHE and (current_time - NEWS_CACHE[tag]['time'] < 3600): # 1 час
             all_news.extend(NEWS_CACHE[tag]['data'])
        else:
             data = fetch_feed(tag)
             NEWS_CACHE[tag] = {'data': data, 'time': current_time}
             all_news.extend(data)
    
    random.shuffle(all_news)
    return {"articles": all_news}

@app.post("/translate")
async def translate_article(request: LinkRequest):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/100.0.0.0 Safari/537.36'}
        response = requests.get(request.url, headers=headers, timeout=15)
        soup = BeautifulSoup(response.text, 'html.parser') # lxml быстрее, если установлен
        
        final_html = translate_html_content(soup)
        
        return {"translated_html": final_html}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
