from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import requests
from bs4 import BeautifulSoup
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

class TextRequest(BaseModel):
    text: str

RSS_SOURCES = {
    "programming": ["https://realpython.com/atom.xml", "https://devblogs.microsoft.com/dotnet/feed/", "https://www.freecodecamp.org/news/rss/"],
    "history": ["https://www.history.com/.rss/full/this-day-in-history", "https://feeds.feedburner.com/HistoryNet"],
    "gaming": ["https://www.polygon.com/rss/index.xml", "https://kotaku.com/rss"],
    "movies": ["https://www.cinemablend.com/rss/news"]
}

@app.get("/")
async def root():
    return {"status": "Alive"}

# --- ПЕРЕВОД ТЕКСТА (Новая функция) ---
@app.post("/translate_text")
async def translate_raw_text(request: TextRequest):
    """Переводит обычный текст, не ссылку"""
    if not request.text: return {"error": "Пустой текст"}
    try:
        # Разбиваем на куски по 4000 символов, если нужно
        translator = GoogleTranslator(source='auto', target='ru')
        translated = translator.translate(request.text[:4500]) # Ограничение 4500
        return {"result": translated}
    except Exception as e:
        return {"error": str(e)}

# --- СТАРЫЕ ФУНКЦИИ (Сократил для удобства, они те же) ---
def translate_html_content(soup):
    translator = GoogleTranslator(source='auto', target='ru')
    content = None
    for selector in ['article', 'main', '.content', '.post-content', '.entry-content', '#content', 'body']:
        if selector.startswith('.'): content = soup.find(class_=selector[1:])
        elif selector.startswith('#'): content = soup.find(id=selector[1:])
        else: content = soup.find(selector)
        if content: break
    if not content: return "<p>Контент не найден :(</p>"

    for junk in content(["script", "style", "iframe", "noscript", "svg", "form", "button", "nav", "footer"]):
        junk.decompose()

    for block in content.find_all(['pre', 'code', 'kbd', 'samp']): block['data-no-translate'] = 'true'

    for node in content.find_all(text=True):
        original = str(node)
        if len(original.strip()) < 3: continue 
        parent = node.parent
        if parent.name in ['pre', 'code', 'script', 'style']: continue
        if any('code' in c for c in parent.get('class', [])): continue
        if parent.get('data-no-translate'): continue
        if len(original) > 4000: continue
        if any(original.strip().startswith(kw) for kw in ['def ', 'class ', 'import ', 'from ', 'return ', 'public ', 'void ']): continue
        try:
            res = translator.translate(original)
            node.replace_with(res)
        except: pass

    for img in content.find_all('img'):
        img['style'] = "max-width: 100%; height: auto; border-radius: 12px; margin: 20px 0; display: block;"
        if not img.get('src') and img.get('data-src'): img['src'] = img['data-src']
    return content.prettify()

@lru_cache(maxsize=20)
def fetch_feed_category(category):
    urls = RSS_SOURCES.get(category, [])
    all_articles = []
    translator = GoogleTranslator(source='auto', target='ru')
    for url in urls:
        try:
            resp = requests.get(url, timeout=4)
            soup = BeautifulSoup(resp.content, "xml")
            items = soup.find_all("item")[:3] or soup.find_all("entry")[:3]
            for item in items:
                title = item.find("title").text
                link_tag = item.find("link")
                if link_tag: link = link_tag.text.strip() if link_tag.text else link_tag.get('href')
                else: continue
                try: ru_title = translator.translate(title)
                except: ru_title = title
                all_articles.append({"title": ru_title, "original_title": title, "link": link, "tag": category.upper()})
        except: continue
    return all_articles

@app.get("/feed")
async def get_news(category: str = "programming"):
    if category not in RSS_SOURCES: category = "programming"
    articles = fetch_feed_category(category)
    random.shuffle(articles)
    return {"articles": articles[:12]}

@app.post("/translate")
async def translate_article(request: LinkRequest):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(request.url, headers=headers, timeout=20)
        try: soup = BeautifulSoup(response.text, 'lxml')
        except: soup = BeautifulSoup(response.text, 'html.parser')
        final_html = translate_html_content(soup)
        return {"translated_html": final_html}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
