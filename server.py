from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import requests
from bs4 import BeautifulSoup, NavigableString
from deep_translator import GoogleTranslator
import time
import random
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

# === ГЛАВНАЯ РУЧКА ДЛЯ ПИНГА ===
@app.get("/")
async def root():
    return {"status": "Alive and Kicking!"}

# --- КЭШ ---
NEWS_CACHE = {}

def translate_html_content(soup):
    translator = GoogleTranslator(source='auto', target='ru')
    
    # 1. Поиск контента (расширенный список классов)
    content = None
    for selector in ['article', 'main', '.content', '.post', '.entry-content', '#content', 'body']:
        if selector.startswith('.'):
            content = soup.find(class_=selector[1:])
        elif selector.startswith('#'):
            content = soup.find(id=selector[1:])
        else:
            content = soup.find(selector)
        if content: break
        
    if not content: return "<p>Не удалось найти контент :(</p>"

    # 2. Чистка
    for junk in content(["script", "style", "iframe", "noscript", "svg", "form", "button"]):
        junk.decompose()

    # 3. Code Guard
    for block in content.find_all(['pre', 'code', 'kbd', 'samp']): 
        block['data-no-translate'] = 'true'

    # 4. ПОСЛЕДОВАТЕЛЬНЫЙ ПЕРЕВОД (Самый надежный)
    # Ищем только текст
    for node in content.find_all(text=True):
        original = str(node)
        if len(original.strip()) < 3: continue 
        
        parent = node.parent
        if parent.name in ['pre', 'code', 'script', 'style']: continue
        if any('code' in c for c in parent.get('class', [])): continue
        if parent.get('data-no-translate'): continue
        
        # Защита от длинных текстов (таймаут)
        if len(original) > 4000: continue

        # Защита кода в тексте
        if any(original.strip().startswith(kw) for kw in ['def ', 'class ', 'import ', 'from ', 'return ', 'public ', 'void ']):
             continue

        try:
            # Переводим по одному. Это не так быстро, зато не жрет память.
            res = translator.translate(original)
            node.replace_with(res)
        except:
            pass

    # 5. Картинки
    for img in content.find_all('img'):
        img['style'] = "max-width: 100%; height: auto; border-radius: 12px; margin: 20px 0; display: block;"
        if not img.get('src') and img.get('data-src'): 
            img['src'] = img['data-src']

    return content.prettify()

@lru_cache(maxsize=10)
def fetch_feed(tag):
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
            link = item.find("link")
            if link:
                link = link.text if link.text else link.get('href')
            else:
                continue

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
    current_time = time.time()
    for tag in ["python", "java", "csharp"]:
        if tag in NEWS_CACHE and (current_time - NEWS_CACHE[tag]['time'] < 3600):
             all_news.extend(NEWS_CACHE[tag]['data'])
        else:
             data = fetch_feed(tag)
             if data:
                NEWS_CACHE[tag] = {'data': data, 'time': current_time}
                all_news.extend(data)
    random.shuffle(all_news)
    return {"articles": all_news}

@app.post("/translate")
async def translate_article(request: LinkRequest):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/100.0.0.0 Safari/537.36'}
        response = requests.get(request.url, headers=headers, timeout=20)
        
        # Пробуем lxml, если нет - html.parser
        try:
            soup = BeautifulSoup(response.text, 'lxml')
        except:
            soup = BeautifulSoup(response.text, 'html.parser')
        
        final_html = translate_html_content(soup)
        return {"translated_html": final_html}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
