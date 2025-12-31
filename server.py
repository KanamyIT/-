from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import requests
from bs4 import BeautifulSoup
import uvicorn

app = FastAPI()

class TranslateRequest(BaseModel):
    url: str

# ========== ВСТРОЕННЫЕ СТАТЬИ ==========

ARTICLES = {
    "programming": [
        {"title": "Python для начинающих - полный курс", "url": "https://realpython.com/"},
        {"title": "FastAPI - современный веб-фреймворк", "url": "https://fastapi.tiangolo.com/"},
        {"title": "JavaScript async/await объяснение", "url": "https://developer.mozilla.org/"},
        {"title": "React Hooks - подробный гайд", "url": "https://react.dev/"},
        {"title": "Docker контейнеры для разработчиков", "url": "https://docs.docker.com/"},
    ],
    "history": [
        {"title": "Древний Рим - история империи", "url": "https://www.britannica.com/"},
        {"title": "Египетские пирамиды - тайны", "url": "https://www.nationalgeographic.com/"},
    ],
    "gaming": [
        {"title": "Dota 2 - новый патч обзор", "url": "https://www.dota2.com/"},
        {"title": "Clash Royale - лучшие стратегии", "url": "https://clashroyale.com/"},
    ],
    "movies": [
        {"title": "Топ фильмов 2025 года", "url": "https://www.imdb.com/"},
        {"title": "Netflix новинки января", "url": "https://www.netflix.com/"},
    ]
}

# ========== ГЛАВНАЯ СТРАНИЦА ==========

@app.get("/", response_class=HTMLResponse)
def home(category: str = "programming"):
    
    articles = ARTICLES.get(category, ARTICLES["programming"])
    
    html = f"""
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>News/Translate by Kanamy</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: #0a0a0a;
            color: #e8e8e8;
            line-height: 1.6;
            min-height: 100vh;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}

        header {{
            text-align: center;
            padding: 40px 20px;
            border-bottom: 1px solid #2a2a2a;
            margin-bottom: 40px;
        }}

        h1 {{
            font-size: 42px;
            font-weight: 300;
            margin-bottom: 30px;
        }}

        h1 span {{
            color: #FF6B35;
            font-weight: 600;
        }}

        .input-box {{
            max-width: 600px;
            margin: 0 auto 40px;
            background: #1a1a1a;
            padding: 24px;
            border-radius: 12px;
            border: 1px solid #2a2a2a;
        }}

        input {{
            width: 100%;
            padding: 14px;
            background: #0a0a0a;
            border: 1px solid #2a2a2a;
            border-radius: 8px;
            color: #e8e8e8;
            font-size: 15px;
            margin-bottom: 12px;
        }}

        input:focus {{
            outline: none;
            border-color: #FF6B35;
        }}

        button {{
            width: 100%;
            padding: 14px;
            background: #FF6B35;
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 15px;
            font-weight: 500;
            cursor: pointer;
            transition: 0.2s;
        }}

        button:hover {{
            background: #ff8555;
            transform: translateY(-1px);
        }}

        .categories {{
            display: flex;
            gap: 12px;
            justify-content: center;
            flex-wrap: wrap;
            margin-bottom: 40px;
        }}

        .cat-btn {{
            padding: 10px 24px;
            background: #1a1a1a;
            color: #888;
            text-decoration: none;
            border-radius: 8px;
            border: 1px solid #2a2a2a;
            transition: 0.2s;
            display: inline-block;
        }}

        .cat-btn:hover {{
            border-color: #FF6B35;
            color: #FF6B35;
        }}

        .cat-btn.active {{
            background: #FF6B35;
            color: white;
            border-color: #FF6B35;
        }}

        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 40px;
        }}

        .card {{
            background: #1a1a1a;
            padding: 24px;
            border-radius: 12px;
            border: 1px solid #2a2a2a;
            cursor: pointer;
            transition: 0.2s;
            text-decoration: none;
            color: inherit;
            display: block;
        }}

        .card:hover {{
            transform: translateY(-4px);
            border-color: #FF6B35;
            box-shadow: 0 8px 20px rgba(0,0,0,0.4);
        }}

        .tag {{
            font-size: 10px;
            color: #FF6B35;
            text-transform: uppercase;
            font-weight: 600;
            margin-bottom: 10px;
            letter-spacing: 1px;
        }}

        .card-title {{
            font-size: 17px;
            font-weight: 500;
            line-height: 1.4;
        }}

        .loading {{
            text-align: center;
            padding: 60px 20px;
            color: #888;
            display: none;
        }}

        .spinner {{
            width: 40px;
            height: 40px;
            border: 3px solid #2a2a2a;
            border-top-color: #FF6B35;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 20px auto;
        }}

        @keyframes spin {{
            100% {{ transform: rotate(360deg); }}
        }}

        @media (max-width: 768px) {{
            h1 {{ font-size: 32px; }}
            .grid {{ grid-template-columns: 1fr; }}
        }}
    </style>
</head>
<body>

<div class="container">
    <header>
        <h1>Умный <span>переводчик</span></h1>
        
        <div class="input-box">
            <form action="/translate" method="post" id="translateForm">
                <input type="url" name="url" placeholder="https://example.com/article" required>
                <button type="submit">Перевести статью</button>
            </form>
        </div>
    </header>

    <div class="categories">
        <a href="/?category=programming" class="cat-btn {'active' if category == 'programming' else ''}">Программирование</a>
        <a href="/?category=history" class="cat-btn {'active' if category == 'history' else ''}">История</a>
        <a href="/?category=gaming" class="cat-btn {'active' if category == 'gaming' else ''}">Игры</a>
        <a href="/?category=movies" class="cat-btn {'active' if category == 'movies' else ''}">Кино</a>
    </div>

    <div class="grid">
"""
    
    for article in articles:
        html += f"""
        <a href="/article?url={article['url']}" class="card">
            <div class="tag">{category.upper()}</div>
            <div class="card-title">{article['title']}</div>
        </a>
"""
    
    html += """
    </div>

    <div class="loading" id="loading">
        <div class="spinner"></div>
        <p>Загрузка статьи...</p>
    </div>
</div>

<script>
document.getElementById('translateForm').addEventListener('submit', function(e) {
    e.preventDefault();
    const url = this.url.value;
    if (url) {
        window.location.href = '/article?url=' + encodeURIComponent(url);
    }
});
</script>

</body>
</html>
"""
    
    return HTMLResponse(content=html)

# ========== СТРАНИЦА СТАТЬИ ==========

@app.get("/article", response_class=HTMLResponse)
def show_article(url: str):
    
    print(f"\n{'='*60}")
    print(f"🔄 Загрузка статьи: {url}")
    
    try:
        # Загружаем страницу
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        print(f"✅ Статус: {response.status_code}")
        
        if response.status_code != 200:
            raise Exception(f"HTTP {response.status_code}")
        
        # Парсим
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Заголовок
        title = "Статья"
        if soup.find('h1'):
            title = soup.find('h1').get_text().strip()
        elif soup.find('title'):
            title = soup.find('title').get_text().strip()
        
        # Контент
        content = soup.find('article') or soup.find('main') or soup.find('body')
        
        if content:
            # Удаляем мусор
            for tag in content(['script', 'style', 'nav', 'header', 'footer', 'aside']):
                tag.decompose()
            
            # Стили для изображений
            for img in content.find_all('img'):
                img['style'] = 'max-width:100%; border-radius:8px;'
            
            content_html = str(content)
        else:
            content_html = "<p>Контент не найден</p>"
        
        print(f"✅ Успешно загружено")
        print(f"{'='*60}\n")
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        print(f"{'='*60}\n")
        title = "Ошибка"
        content_html = f"<p style='color:#ff6b6b;'>Не удалось загрузить статью: {str(e)}</p>"
    
    html = f"""
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: #0a0a0a;
            color: #e8e8e8;
            line-height: 1.7;
        }}

        .container {{
            max-width: 900px;
            margin: 0 auto;
            padding: 40px 20px;
        }}

        .back-btn {{
            display: inline-block;
            padding: 10px 20px;
            background: #1a1a1a;
            color: #e8e8e8;
            text-decoration: none;
            border-radius: 8px;
            margin-bottom: 30px;
            border: 1px solid #2a2a2a;
            transition: 0.2s;
        }}

        .back-btn:hover {{
            border-color: #FF6B35;
            background: #2a2a2a;
        }}

        article {{
            background: #1a1a1a;
            padding: 40px;
            border-radius: 12px;
            border: 1px solid #2a2a2a;
        }}

        h1, h2, h3 {{
            color: #FF6B35;
            margin: 20px 0 12px;
        }}

        p {{
            margin: 12px 0;
            line-height: 1.8;
        }}

        img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            margin: 20px 0;
        }}

        a {{
            color: #FF6B35;
        }}

        @media (max-width: 768px) {{
            article {{ padding: 24px; }}
        }}
    </style>
</head>
<body>

<div class="container">
    <a href="/" class="back-btn">← Назад к новостям</a>
    
    <article>
        <h1>{title}</h1>
        {content_html}
    </article>
</div>

</body>
</html>
"""
    
    return HTMLResponse(content=html)

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🚀 СЕРВЕР ЗАПУЩЕН")
    print("="*70)
    print("📍 Откройте: http://localhost:8000")
    print("✅ Статьи рендерятся на сервере - никаких fetch запросов!")
    print("✅ Минимум JavaScript - максимум надежности!")
    print("="*70 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
