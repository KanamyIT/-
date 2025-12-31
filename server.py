from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import requests
from bs4 import BeautifulSoup
import uvicorn

app = FastAPI()

class LinkRequest(BaseModel):
    url: str

# ========== HTML СТРАНИЦА ==========

HTML = '''<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>News/Translate by Kanamy</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: #0a0a0a;
            color: #e8e8e8;
            line-height: 1.6;
        }

        .container { max-width: 1200px; margin: 0 auto; padding: 40px 20px; }

        header { 
            background: rgba(10, 10, 10, 0.95);
            border-bottom: 1px solid #2a2a2a;
            padding: 20px 0;
            position: sticky;
            top: 0;
            z-index: 100;
        }
        
        .logo { 
            font-size: 24px;
            font-weight: 300;
            text-align: center;
        }
        .logo span { color: #FF6B35; font-weight: 600; }

        .hero { text-align: center; margin: 60px 0; }
        .hero h1 { 
            font-size: 48px;
            font-weight: 300;
            margin-bottom: 30px;
        }
        .hero h1 span { color: #FF6B35; }

        .input-box {
            max-width: 600px;
            margin: 0 auto 40px auto;
            background: #1a1a1a;
            padding: 30px;
            border-radius: 16px;
            border: 1px solid #2a2a2a;
        }

        input {
            width: 100%;
            padding: 16px;
            background: #0a0a0a;
            border: 1px solid #2a2a2a;
            border-radius: 8px;
            color: #e8e8e8;
            font-size: 15px;
            margin-bottom: 16px;
        }
        input:focus { outline: none; border-color: #FF6B35; }

        .btn {
            width: 100%;
            padding: 14px;
            background: #FF6B35;
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 500;
            cursor: pointer;
            transition: 0.3s;
        }
        .btn:hover { transform: translateY(-2px); box-shadow: 0 8px 20px rgba(255,107,53,0.4); }

        .categories {
            display: flex;
            gap: 12px;
            justify-content: center;
            margin-bottom: 40px;
            flex-wrap: wrap;
        }

        .cat-btn {
            padding: 10px 24px;
            background: transparent;
            border: 1px solid #2a2a2a;
            color: #888;
            border-radius: 8px;
            cursor: pointer;
            transition: 0.3s;
        }
        .cat-btn.active { background: #FF6B35; color: white; border-color: #FF6B35; }
        .cat-btn:hover { border-color: #FF6B35; }

        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 24px;
            margin-top: 40px;
        }

        .card {
            background: #1a1a1a;
            padding: 24px;
            border-radius: 12px;
            border: 1px solid #2a2a2a;
            cursor: pointer;
            transition: 0.3s;
        }
        .card:hover { 
            transform: translateY(-4px);
            border-color: #FF6B35;
            box-shadow: 0 12px 24px rgba(0,0,0,0.5);
        }

        .tag { 
            font-size: 10px;
            color: #FF6B35;
            text-transform: uppercase;
            font-weight: 600;
            margin-bottom: 12px;
        }

        .card-title {
            font-size: 17px;
            font-weight: 500;
            line-height: 1.5;
        }

        .article {
            background: #1a1a1a;
            padding: 50px;
            border-radius: 16px;
            border: 1px solid #2a2a2a;
            display: none;
            margin-top: 40px;
        }
        .article h1, .article h2, .article h3 { color: #FF6B35; margin: 20px 0 10px; }
        .article p { margin: 12px 0; line-height: 1.8; }
        .article img { max-width: 100%; border-radius: 8px; margin: 20px 0; }
        .article a { color: #FF6B35; }

        .loader {
            display: none;
            width: 50px;
            height: 50px;
            border: 4px solid #2a2a2a;
            border-top-color: #FF6B35;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 60px auto;
        }
        @keyframes spin { 100% { transform: rotate(360deg); } }

        .status {
            text-align: center;
            padding: 40px;
            color: #888;
            font-size: 15px;
        }

        .back-btn {
            display: inline-block;
            padding: 10px 20px;
            background: #2a2a2a;
            color: #e8e8e8;
            border-radius: 8px;
            margin-bottom: 20px;
            cursor: pointer;
            transition: 0.3s;
        }
        .back-btn:hover { background: #FF6B35; }

        @media (max-width: 768px) {
            .hero h1 { font-size: 32px; }
            .grid { grid-template-columns: 1fr; }
            .article { padding: 24px; }
        }
    </style>
</head>
<body>

<header>
    <div class="logo">News/Translate <span>by Kanamy</span></div>
</header>

<div class="container">
    <div class="hero">
        <h1>Умный <span>переводчик</span></h1>
        
        <div class="input-box">
            <input type="text" id="urlInput" placeholder="https://example.com/article">
            <button class="btn" onclick="translateArticle()">Перевести статью</button>
        </div>
    </div>

    <div id="newsSection">
        <div class="categories">
            <button class="cat-btn active" onclick="loadNews('programming')">Программирование</button>
            <button class="cat-btn" onclick="loadNews('history')">История</button>
            <button class="cat-btn" onclick="loadNews('gaming')">Игры</button>
            <button class="cat-btn" onclick="loadNews('movies')">Кино</button>
        </div>
        
        <div class="status" id="status">Загрузка статей...</div>
        <div class="grid" id="grid"></div>
    </div>

    <div class="loader" id="loader"></div>
    
    <div id="articleView" class="article"></div>
</div>

<script>
console.log('🚀 Приложение запущено');

let currentCategory = 'programming';

window.onload = () => {
    loadNews('programming');
};

async function loadNews(category) {
    console.log('📰 Загрузка категории:', category);
    
    currentCategory = category;
    
    // Обновляем активную кнопку
    document.querySelectorAll('.cat-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    event.target.classList.add('active');
    
    const grid = document.getElementById('grid');
    const status = document.getElementById('status');
    
    grid.innerHTML = '';
    status.textContent = 'Загрузка...';
    status.style.display = 'block';
    
    try {
        const response = await fetch(`/api/feed?category=${category}`);
        
        if (!response.ok) {
            throw new Error('Ошибка загрузки');
        }
        
        const data = await response.json();
        console.log('✅ Получено статей:', data.articles.length);
        
        status.style.display = 'none';
        
        if (data.articles.length === 0) {
            status.textContent = 'Нет статей';
            status.style.display = 'block';
            return;
        }
        
        data.articles.forEach(article => {
            const card = document.createElement('div');
            card.className = 'card';
            card.innerHTML = `
                <div class="tag">${article.tag}</div>
                <div class="card-title">${article.title}</div>
            `;
            card.onclick = () => openArticle(article.link);
            grid.appendChild(card);
        });
        
    } catch (error) {
        console.error('❌ Ошибка:', error);
        status.textContent = 'Ошибка загрузки статей';
        status.style.color = '#ff6b6b';
    }
}

async function translateArticle() {
    const url = document.getElementById('urlInput').value.trim();
    
    if (!url) {
        alert('Введите URL статьи!');
        return;
    }
    
    openArticle(url);
}

async function openArticle(url) {
    console.log('🔄 Открываю статью:', url);
    
    document.getElementById('newsSection').style.display = 'none';
    document.getElementById('articleView').style.display = 'none';
    document.getElementById('loader').style.display = 'block';
    
    try {
        const response = await fetch('/api/translate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ url: url })
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Ошибка перевода');
        }
        
        const data = await response.json();
        console.log('✅ Статья загружена');
        
        const articleView = document.getElementById('articleView');
        articleView.innerHTML = `
            <div class="back-btn" onclick="goBack()">← Назад к новостям</div>
            <h1>${data.title}</h1>
            ${data.content}
        `;
        articleView.style.display = 'block';
        
        window.scrollTo({ top: 0, behavior: 'smooth' });
        
    } catch (error) {
        console.error('❌ Ошибка:', error);
        alert('Ошибка загрузки статьи: ' + error.message);
        goBack();
    } finally {
        document.getElementById('loader').style.display = 'none';
    }
}

function goBack() {
    document.getElementById('articleView').style.display = 'none';
    document.getElementById('newsSection').style.display = 'block';
    loadNews(currentCategory);
}
</script>

</body>
</html>'''

# ========== ДАННЫЕ ==========

ARTICLES_DB = {
    "programming": [
        {
            "title": "Как изучить Python за 30 дней - полное руководство",
            "link": "https://realpython.com/",
            "tag": "PROGRAMMING"
        },
        {
            "title": "FastAPI: современный фреймворк для API",
            "link": "https://fastapi.tiangolo.com/",
            "tag": "PROGRAMMING"
        },
        {
            "title": "Асинхронное программирование в JavaScript",
            "link": "https://developer.mozilla.org/en-US/docs/Learn/JavaScript/Asynchronous",
            "tag": "PROGRAMMING"
        },
        {
            "title": "React Hooks - детальное руководство",
            "link": "https://react.dev/reference/react",
            "tag": "PROGRAMMING"
        },
        {
            "title": "Docker для начинающих разработчиков",
            "link": "https://docs.docker.com/get-started/",
            "tag": "PROGRAMMING"
        },
        {
            "title": "Git и GitHub - лучшие практики",
            "link": "https://docs.github.com/en/get-started",
            "tag": "PROGRAMMING"
        }
    ],
    "history": [
        {
            "title": "Древний Рим: взлет и падение империи",
            "link": "https://www.britannica.com/place/ancient-Rome",
            "tag": "HISTORY"
        },
        {
            "title": "Египетские пирамиды: тайны строительства",
            "link": "https://www.nationalgeographic.com/history/article/giza-pyramids",
            "tag": "HISTORY"
        },
        {
            "title": "Средневековая Европа и крестовые походы",
            "link": "https://www.britannica.com/event/Crusades",
            "tag": "HISTORY"
        }
    ],
    "gaming": [
        {
            "title": "Лучшие игры 2025 года - топ-10",
            "link": "https://store.steampowered.com/",
            "tag": "GAMING"
        },
        {
            "title": "Dota 2: обновление патча 7.35",
            "link": "https://www.dota2.com/",
            "tag": "GAMING"
        },
        {
            "title": "Clash Royale: гайд по новым картам",
            "link": "https://clashroyale.com/",
            "tag": "GAMING"
        }
    ],
    "movies": [
        {
            "title": "Топ фильмов 2025 по версии IMDb",
            "link": "https://www.imdb.com/",
            "tag": "MOVIES"
        },
        {
            "title": "Новинки Netflix в январе 2025",
            "link": "https://www.netflix.com/",
            "tag": "MOVIES"
        }
    ]
}

# ========== API ENDPOINTS ==========

@app.get("/")
def home():
    return HTMLResponse(content=HTML)

@app.get("/api/feed")
def get_feed(category: str = "programming"):
    """Получить список статей по категории"""
    
    print(f"\n{'='*60}")
    print(f"📰 Запрос категории: {category}")
    
    if category not in ARTICLES_DB:
        category = "programming"
    
    articles = ARTICLES_DB[category].copy()
    
    # Попытка загрузить реальные RSS (опционально)
    rss_feeds = {
        "programming": "https://www.reddit.com/r/programming/.rss",
        "history": "https://www.reddit.com/r/history/.rss",
        "gaming": "https://www.reddit.com/r/gaming/.rss",
        "movies": "https://www.reddit.com/r/movies/.rss"
    }
    
    if category in rss_feeds:
        try:
            response = requests.get(
                rss_feeds[category],
                headers={'User-Agent': 'Mozilla/5.0'},
                timeout=5
            )
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'xml')
                entries = soup.find_all('entry')[:3] or soup.find_all('item')[:3]
                
                for entry in entries:
                    title_tag = entry.find('title')
                    link_tag = entry.find('link')
                    
                    if title_tag and link_tag:
                        title = BeautifulSoup(title_tag.get_text(), 'html.parser').get_text()[:100]
                        link = link_tag.get('href') or link_tag.get_text()
                        
                        articles.append({
                            "title": title,
                            "link": link,
                            "tag": category.upper()
                        })
                
                print(f"✅ Загружено из RSS: {len(entries)} статей")
        except Exception as e:
            print(f"⚠️ RSS не загрузился: {e}")
    
    print(f"📦 Возвращаю статей: {len(articles)}")
    print(f"{'='*60}\n")
    
    return {
        "articles": articles[:10],
        "category": category,
        "total": len(articles)
    }

@app.post("/api/translate")
def translate_article(request: LinkRequest):
    """Перевести/загрузить статью"""
    
    print(f"\n{'='*60}")
    print(f"🔄 Перевод статьи: {request.url}")
    
    try:
        # Загружаем страницу
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9'
        }
        
        response = requests.get(request.url, headers=headers, timeout=15)
        
        print(f"📡 Статус: {response.status_code}")
        
        if response.status_code != 200:
            raise HTTPException(
                status_code=400,
                detail=f"Не удалось загрузить страницу (HTTP {response.status_code})"
            )
        
        # Парсим HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Извлекаем заголовок
        title = "Статья"
        if soup.find('h1'):
            title = soup.find('h1').get_text().strip()
        elif soup.find('title'):
            title = soup.find('title').get_text().strip()
        
        print(f"📌 Заголовок: {title}")
        
        # Извлекаем контент
        content = None
        for selector in ['article', 'main', '[role="main"]', '.content', '.post', 'body']:
            if selector.startswith('.'):
                content = soup.find(class_=selector[1:])
            elif selector.startswith('['):
                attr, value = selector.strip('[]').split('=')
                content = soup.find(attrs={attr: value.strip('"')})
            else:
                content = soup.find(selector)
            
            if content:
                print(f"✅ Контент найден через селектор: {selector}")
                break
        
        if not content:
            raise HTTPException(status_code=400, detail="Не удалось найти контент статьи")
        
        # Очищаем от мусора
        for tag in content(['script', 'style', 'nav', 'header', 'footer', 'aside', 'iframe', 'form']):
            tag.decompose()
        
        # Обрабатываем изображения
        for img in content.find_all('img'):
            if img.get('src'):
                img['style'] = 'max-width: 100%; height: auto; border-radius: 8px; margin: 20px 0;'
            if img.get('data-src'):
                img['src'] = img['data-src']
        
        # Обрабатываем ссылки
        for link in content.find_all('a'):
            link['style'] = 'color: #FF6B35;'
            link['target'] = '_blank'
        
        content_html = str(content)
        
        print(f"✅ Статья успешно обработана")
        print(f"{'='*60}\n")
        
        return {
            "title": title,
            "content": content_html,
            "success": True
        }
        
    except requests.Timeout:
        print(f"❌ Timeout")
        raise HTTPException(status_code=504, detail="Время ожидания истекло")
    
    except requests.RequestException as e:
        print(f"❌ Ошибка загрузки: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка загрузки: {str(e)}")
    
    except Exception as e:
        print(f"❌ Общая ошибка: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Ошибка: {str(e)}")

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🚀 СЕРВЕР ЗАПУЩЕН")
    print("="*70)
    print("📍 Откройте в браузере: http://localhost:8000")
    print("💡 Статьи всегда загружаются (встроенные тестовые данные)")
    print("📰 RSS как бонус - если загрузится, добавится к тестовым")
    print("="*70 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
