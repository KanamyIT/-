from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
from bs4 import BeautifulSoup
from deep_translator import GoogleTranslator
import random
from functools import lru_cache
import uvicorn

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

# ========== ФУНКЦИЯ ПЕРЕВОДА HTML КОНТЕНТА ==========

def translate_html_content(soup):
    """Переводит текст в HTML с английского на русский"""
    
    translator = GoogleTranslator(source='auto', target='ru')
    
    # Ищем контент
    content = None
    for selector in ['article', 'main', '.content', '.post-content', '.entry-content', '#content', 'body']:
        if selector.startswith('.'):
            content = soup.find(class_=selector[1:])
        elif selector.startswith('#'):
            content = soup.find(id=selector[1:])
        else:
            content = soup.find(selector)
        if content:
            break
    
    if not content:
        return "<p>Не удалось найти контент</p>", 0
    
    # Удаляем мусор
    for junk in content(["script", "style", "iframe", "noscript", "svg", "form", "button", "nav", "footer", "aside"]):
        junk.decompose()
    
    # Помечаем код
    for block in content.find_all(['pre', 'code', 'kbd', 'samp']):
        block['data-no-translate'] = 'true'
    
    word_count = 0
    
    # ПЕРЕВОДИМ КАЖДЫЙ ТЕКСТОВЫЙ УЗЕЛ
    for node in content.find_all(text=True):
        original = str(node).strip()
        
        if len(original) < 3:
            continue
        
        parent = node.parent
        
        # Пропускаем код
        if parent.name in ['pre', 'code', 'script', 'style']:
            continue
        
        if parent.get('data-no-translate'):
            continue
        
        if len(original) > 4000:
            continue
        
        # Пропускаем строки кода
        if any(original.startswith(kw) for kw in ['def ', 'class ', 'import ', 'from ', 'return ', 'public ', 'void ']):
            continue
        
        try:
            # РЕАЛЬНЫЙ ПЕРЕВОД
            translated = translator.translate(original)
            node.replace_with(translated)
            word_count += len(translated.split())
        except:
            pass
    
    # Обрабатываем изображения
    for img in content.find_all('img'):
        img['style'] = "max-width: 100%; height: auto; border-radius: 12px; margin: 20px 0;"
        if not img.get('src') and img.get('data-src'):
            img['src'] = img['data-src']
    
    return str(content), word_count

# ========== КЭШИРОВАННАЯ ЗАГРУЗКА RSS ==========

@lru_cache(maxsize=20)
def fetch_feed_category(category):
    """Парсит RSS и переводит заголовки"""
    
    urls = RSS_SOURCES.get(category, [])
    all_articles = []
    translator = GoogleTranslator(source='auto', target='ru')
    
    for url in urls:
        try:
            print(f"📡 Загружаю RSS: {url}")
            
            resp = requests.get(url, timeout=8, headers={'User-Agent': 'Mozilla/5.0'})
            soup = BeautifulSoup(resp.content, "xml")
            
            items = soup.find_all("item")[:5] or soup.find_all("entry")[:5]
            
            print(f"✅ Найдено: {len(items)} статей")
            
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
                    
                    # ПЕРЕВОДИМ ЗАГОЛОВОК
                    try:
                        ru_title = translator.translate(title[:150])
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
                    
        except Exception as e:
            print(f"❌ Ошибка RSS: {e}")
            continue
    
    return all_articles

# ========== ENDPOINTS ==========

@app.get("/", response_class=HTMLResponse)
def home():
    return """<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>News Translator</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, sans-serif; background: #0a0a0a; color: #e8e8e8; padding: 20px; }
        .container { max-width: 1200px; margin: 0 auto; }
        h1 { text-align: center; font-size: 40px; margin: 40px 0; font-weight: 300; }
        h1 span { color: #FF6B35; font-weight: 600; }
        
        .tabs { display: flex; gap: 12px; justify-content: center; margin-bottom: 40px; flex-wrap: wrap; }
        .tab { padding: 10px 24px; background: #1a1a1a; border: 1px solid #333; border-radius: 8px; cursor: pointer; transition: 0.2s; }
        .tab.active { background: #FF6B35; border-color: #FF6B35; }
        
        #grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 20px; }
        .card { background: #1a1a1a; padding: 20px; border-radius: 12px; border: 1px solid #333; cursor: pointer; transition: 0.2s; }
        .card:hover { transform: translateY(-4px); border-color: #FF6B35; }
        .tag { font-size: 10px; color: #FF6B35; text-transform: uppercase; margin-bottom: 8px; font-weight: 600; }
        .title { font-size: 16px; line-height: 1.4; margin-bottom: 6px; }
        .original { font-size: 13px; color: #666; }
        
        #article { display: none; background: #1a1a1a; padding: 40px; border-radius: 12px; margin-top: 40px; }
        #article h1, #article h2, #article h3 { color: #FF6B35; margin: 20px 0 10px; }
        #article p { margin: 12px 0; line-height: 1.7; }
        #article img { max-width: 100%; border-radius: 8px; margin: 20px 0; }
        #article a { color: #FF6B35; }
        
        .loader { display: none; text-align: center; padding: 60px; }
        .spinner { width: 40px; height: 40px; border: 3px solid #333; border-top-color: #FF6B35; border-radius: 50%; animation: spin 1s linear infinite; margin: 0 auto 20px; }
        @keyframes spin { 100% { transform: rotate(360deg); } }
        
        .back { display: inline-block; padding: 10px 20px; background: #333; border-radius: 8px; cursor: pointer; margin-bottom: 20px; }
        .status { text-align: center; padding: 40px; color: #888; }
    </style>
</head>
<body>

<div class="container">
    <h1>Переводчик <span>новостей</span></h1>

    <div id="news">
        <div class="tabs">
            <div class="tab active" onclick="loadCat('programming')">Программирование</div>
            <div class="tab" onclick="loadCat('history')">История</div>
            <div class="tab" onclick="loadCat('gaming')">Игры</div>
            <div class="tab" onclick="loadCat('movies')">Кино</div>
        </div>
        <div class="status" id="status">Загрузка статей...</div>
        <div id="grid"></div>
    </div>

    <div class="loader" id="loader"><div class="spinner"></div><p>Перевод статьи...</p></div>
    
    <div id="article">
        <div class="back" onclick="goBack()">← Назад к новостям</div>
        <div id="content"></div>
    </div>
</div>

<script>
console.log('🚀 Запуск');

async function loadCat(cat) {
    console.log('📰 Категория:', cat);
    
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    event.target.classList.add('active');
    
    const grid = document.getElementById('grid');
    const status = document.getElementById('status');
    
    grid.innerHTML = '';
    status.style.display = 'block';
    status.textContent = 'Загрузка и перевод заголовков...';
    
    try {
        const res = await fetch('/feed?category=' + cat);
        const data = await res.json();
        
        console.log('✅ Получено:', data.articles.length, 'статей');
        
        status.style.display = 'none';
        
        if (data.articles.length === 0) {
            status.textContent = 'Нет статей';
            status.style.display = 'block';
            return;
        }
        
        data.articles.forEach(art => {
            const card = document.createElement('div');
            card.className = 'card';
            card.innerHTML = `
                <div class="tag">${art.tag}</div>
                <div class="title">${art.title}</div>
                <div class="original">${art.original_title}</div>
            `;
            card.onclick = () => openArticle(art.link);
            grid.appendChild(card);
        });
        
    } catch (e) {
        console.error('❌', e);
        status.textContent = 'Ошибка загрузки';
        status.style.color = '#ff6b6b';
    }
}

async function openArticle(url) {
    console.log('🔄 Загружаю и перевожу:', url);
    
    document.getElementById('news').style.display = 'none';
    document.getElementById('article').style.display = 'none';
    document.getElementById('loader').style.display = 'block';
    
    try {
        const res = await fetch('/translate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ url: url })
        });
        
        const data = await res.json();
        
        if (data.error) {
            throw new Error(data.error);
        }
        
        console.log('✅ Переведено! Слов:', data.word_count);
        
        document.getElementById('content').innerHTML = data.translated_html;
        document.getElementById('article').style.display = 'block';
        window.scrollTo(0, 0);
        
    } catch (e) {
        console.error('❌', e);
        alert('Ошибка: ' + e.message);
        goBack();
    } finally {
        document.getElementById('loader').style.display = 'none';
    }
}

function goBack() {
    document.getElementById('article').style.display = 'none';
    document.getElementById('news').style.display = 'block';
}

// Загружаем сразу
loadCat('programming');
</script>

</body>
</html>"""

@app.get("/feed")
def get_feed(category: str = "programming"):
    print(f"\n📰 Запрос категории: {category}")
    
    if category not in RSS_SOURCES:
        category = "programming"
    
    articles = fetch_feed_category(category)
    random.shuffle(articles)
    
    print(f"📦 Возвращаю: {len(articles)} статей\n")
    
    return {"articles": articles[:12]}

@app.post("/translate")
def translate_article(request: LinkRequest):
    print(f"\n🔄 Перевод статьи: {request.url}")
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0 Safari/537.36'
        }
        
        response = requests.get(request.url, headers=headers, timeout=20)
        
        print(f"✅ Загружено, статус: {response.status_code}")
        
        # Парсим HTML
        try:
            soup = BeautifulSoup(response.text, 'lxml')
        except:
            soup = BeautifulSoup(response.text, 'html.parser')
        
        # ПЕРЕВОДИМ КОНТЕНТ
        print(f"🔄 Перевожу контент...")
        final_html, word_count = translate_html_content(soup)
        
        # Время чтения
        read_time = max(1, round(word_count / 200))
        
        print(f"✅ Переведено! Слов: {word_count}, Время: {read_time} мин\n")
        
        return {
            "translated_html": final_html,
            "read_time": read_time,
            "word_count": word_count
        }
        
    except Exception as e:
        print(f"❌ Ошибка: {e}\n")
        return {"error": str(e)}

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🚀 NEWS TRANSLATOR - С РЕАЛЬНЫМ ПЕРЕВОДОМ!")
    print("="*70)
    print("📍 http://localhost:8000")
    print("✅ Парсит английские RSS")
    print("✅ Переводит заголовки на русский")
    print("✅ Переводит статьи целиком через Google Translate")
    print("="*70 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
