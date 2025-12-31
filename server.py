from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import requests
from bs4 import BeautifulSoup

app = FastAPI()

# ========== ПРАВИЛЬНЫЙ CORS ==========
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # Изменено на False для * origins
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Добавляем обработчик OPTIONS для всех маршрутов
@app.options("/{full_path:path}")
async def options_handler(full_path: str):
    return JSONResponse(
        content={},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "*",
        }
    )

class LinkRequest(BaseModel):
    url: str

class TextRequest(BaseModel):
    text: str

# ========== HTML ==========

HTML_PAGE = """
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>News/Translate by Kanamy</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg: #0a0a0a;
            --surface: #1a1a1a;
            --orange: #FF6B35;
            --orange-dim: rgba(255, 107, 53, 0.1);
            --text: #e8e8e8;
            --text-dim: #888;
            --border: #2a2a2a;
        }
        
        * { margin: 0; padding: 0; box-sizing: border-box; }
        html { scroll-behavior: smooth; }
        
        body { 
            font-family: 'Inter', -apple-system, sans-serif; 
            background: var(--bg); 
            color: var(--text); 
            line-height: 1.6;
            overflow-x: hidden;
        }

        body::before {
            content: '';
            position: fixed;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle, var(--orange-dim) 0%, transparent 70%);
            animation: pulse 15s ease-in-out infinite;
            pointer-events: none;
            z-index: 0;
        }
        
        @keyframes pulse {
            0%, 100% { transform: translate(0, 0) scale(1); opacity: 0.3; }
            50% { transform: translate(10%, 10%) scale(1.1); opacity: 0.5; }
        }

        #particles {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: 0;
            pointer-events: none;
        }

        header { 
            position: sticky;
            top: 0;
            z-index: 100;
            background: rgba(10, 10, 10, 0.8);
            backdrop-filter: blur(20px);
            border-bottom: 1px solid var(--border);
            padding: 20px 5%;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .logo { 
            font-size: 20px; 
            font-weight: 300;
            cursor: pointer;
        }
        .logo span { color: var(--orange); font-weight: 600; }

        .container { 
            max-width: 1100px; 
            margin: 0 auto; 
            padding: 60px 20px;
            position: relative;
            z-index: 1;
        }

        .hero { 
            text-align: center; 
            margin-bottom: 80px;
        }
        
        .hero h1 { 
            font-size: 48px; 
            font-weight: 300;
            margin-bottom: 40px;
            color: var(--text);
            min-height: 60px;
        }
        
        .hero h1 span { color: var(--orange); }

        .tabs { 
            display: inline-flex;
            gap: 4px;
            background: var(--surface);
            padding: 4px;
            border-radius: 50px;
            margin-bottom: 40px;
            border: 1px solid var(--border);
        }
        
        .tab-btn { 
            padding: 10px 28px;
            border: none;
            background: transparent;
            color: var(--text-dim);
            cursor: pointer;
            font-weight: 500;
            font-size: 14px;
            border-radius: 50px;
            transition: all 0.3s;
        }
        .tab-btn.active { 
            background: var(--orange);
            color: #fff;
        }

        .input-area { display: none; max-width: 600px; margin: 0 auto; }
        .input-area.active { display: block; }

        input, textarea { 
            width: 100%;
            padding: 16px 20px;
            border: 1px solid var(--border);
            border-radius: 12px;
            font-size: 15px;
            background: var(--surface);
            color: var(--text);
            font-family: inherit;
        }
        input:focus, textarea:focus { 
            border-color: var(--orange);
            outline: none;
        }
        textarea { resize: vertical; min-height: 120px; }

        .btn-main { 
            margin-top: 16px;
            padding: 14px 32px;
            background: var(--orange);
            color: #fff;
            border: none;
            border-radius: 8px;
            font-weight: 500;
            cursor: pointer;
            font-size: 15px;
            transition: all 0.3s;
            width: 100%;
        }
        .btn-main:hover { 
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(255, 107, 53, 0.4);
        }

        .categories { 
            display: flex;
            gap: 12px;
            justify-content: center;
            margin-bottom: 40px;
            flex-wrap: wrap;
        }
        
        .cat-btn { 
            padding: 10px 24px;
            border: 1px solid var(--border);
            background: transparent;
            color: var(--text-dim);
            font-weight: 500;
            cursor: pointer;
            transition: all 0.3s;
            border-radius: 8px;
        }
        .cat-btn.active { 
            background: var(--orange);
            color: #fff;
        }

        .news-grid { 
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 24px;
            min-height: 200px;
        }
        
        .news-card { 
            background: var(--surface);
            padding: 24px;
            border-radius: 12px;
            cursor: pointer;
            transition: all 0.3s;
            border: 1px solid var(--border);
        }
        .news-card:hover { 
            transform: translateY(-4px);
            border-color: var(--orange);
        }
        
        .tag-badge { 
            font-size: 10px;
            font-weight: 600;
            color: var(--orange);
            text-transform: uppercase;
            margin-bottom: 12px;
        }
        
        .news-title { 
            font-size: 17px;
            font-weight: 500;
            line-height: 1.5;
        }

        .article-view { 
            background: var(--surface);
            padding: 50px;
            border-radius: 16px;
            margin-top: 40px;
            display: none;
            line-height: 1.8;
            font-size: 17px;
            border: 1px solid var(--border);
        }

        .loader { 
            display: none;
            margin: 40px auto;
            width: 40px;
            height: 40px;
            border: 3px solid var(--border);
            border-top-color: var(--orange);
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        @keyframes spin { 100% { transform: rotate(360deg); } }

        .error-msg {
            background: rgba(255, 50, 50, 0.1);
            border: 1px solid rgba(255, 50, 50, 0.3);
            color: #ff6b6b;
            padding: 16px 20px;
            border-radius: 12px;
            margin: 20px auto;
            max-width: 600px;
            display: none;
        }

        @media (max-width: 768px) {
            .hero h1 { font-size: 32px; }
            .article-view { padding: 24px; }
            .news-grid { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>

<canvas id="particles"></canvas>

<header>
    <div class="logo" onclick="goHome()">News/Translate <span>by Kanamy</span></div>
</header>

<div class="container">
    <div class="hero">
        <h1>Умный <span>переводчик</span></h1>
        
        <div class="tabs">
            <button class="tab-btn active" onclick="switchMode('url')">🔗 Ссылка</button>
            <button class="tab-btn" onclick="switchMode('text')">📝 Текст</button>
        </div>

        <div id="urlMode" class="input-area active">
            <input type="text" id="urlInput" placeholder="Вставьте ссылку на статью..." />
            <button class="btn-main" onclick="translateUrl()">Перевести</button>
        </div>

        <div id="textMode" class="input-area">
            <textarea id="textInput" placeholder="Вставьте текст..."></textarea>
            <button class="btn-main" onclick="translateText()">Перевести</button>
        </div>
    </div>

    <div class="error-msg" id="errorMsg"></div>

    <div id="newsSection">
        <div class="categories">
            <button class="cat-btn active" onclick="loadCategory('programming', this)">Программирование</button>
            <button class="cat-btn" onclick="loadCategory('history', this)">История</button>
            <button class="cat-btn" onclick="loadCategory('gaming', this)">Игры</button>
            <button class="cat-btn" onclick="loadCategory('movies', this)">Кино</button>
        </div>
        <div class="news-grid" id="newsGrid">
            <p style="grid-column: 1/-1; text-align: center; color: var(--text-dim);">Загрузка...</p>
        </div>
    </div>

    <div class="loader" id="loader"></div>
    <div id="result" class="article-view"></div>
</div>

<script>
    const canvas = document.getElementById('particles');
    const ctx = canvas.getContext('2d');
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;

    let particles = [];

    class Particle {
        constructor() {
            this.x = Math.random() * canvas.width;
            this.y = Math.random() * canvas.height;
            this.size = Math.random() * 2 + 0.5;
            this.speedX = Math.random() * 0.5 - 0.25;
            this.speedY = Math.random() * 0.5 - 0.25;
            this.color = Math.random() > 0.7 ? '#FF6B35' : '#333';
        }
        
        update() {
            this.x += this.speedX;
            this.y += this.speedY;
            if(this.x > canvas.width) this.x = 0;
            if(this.x < 0) this.x = canvas.width;
            if(this.y > canvas.height) this.y = 0;
            if(this.y < 0) this.y = canvas.height;
        }
        
        draw() {
            ctx.fillStyle = this.color;
            ctx.beginPath();
            ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
            ctx.fill();
        }
    }

    function initParticles() {
        for(let i = 0; i < 60; i++) particles.push(new Particle());
    }

    function animateParticles() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        particles.forEach(p => { p.update(); p.draw(); });
        requestAnimationFrame(animateParticles);
    }

    initParticles();
    animateParticles();

    window.addEventListener('resize', () => {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
    });

    window.onload = function() {
        console.log('🚀 Инициализация');
        loadCategory('programming', document.querySelector('.cat-btn.active'));
    };

    function showError(msg) {
        const errorDiv = document.getElementById('errorMsg');
        errorDiv.textContent = msg;
        errorDiv.style.display = 'block';
        setTimeout(() => errorDiv.style.display = 'none', 5000);
    }

    function resetView() {
        document.getElementById('result').style.display = 'none';
        document.getElementById('loader').style.display = 'none';
        document.getElementById('errorMsg').style.display = 'none';
    }

    function goHome() {
        resetView();
        document.getElementById('newsSection').style.display = 'block';
        switchMode('url');
        window.scrollTo({top: 0, behavior: 'smooth'});
    }

    function switchMode(mode) {
        resetView();
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.input-area').forEach(a => a.classList.remove('active'));
        
        if(mode === 'url') {
            document.querySelector('.tab-btn:nth-child(1)').classList.add('active');
            document.getElementById('urlMode').classList.add('active');
            document.getElementById('newsSection').style.display = 'block';
        } else {
            document.querySelector('.tab-btn:nth-child(2)').classList.add('active');
            document.getElementById('textMode').classList.add('active');
            document.getElementById('newsSection').style.display = 'none';
        }
    }

    async function loadCategory(cat, btn) {
        console.log('📰 Загрузка категории:', cat);
        resetView();
        
        document.querySelectorAll('.cat-btn').forEach(b => b.classList.remove('active'));
        if(btn) btn.classList.add('active');
        
        const grid = document.getElementById('newsGrid');
        grid.innerHTML = '<p style="grid-column: 1/-1; text-align: center; color: var(--text-dim);">Загрузка...</p>';

        try {
            console.log('📡 Запрос к /feed?category=' + cat);
            
            const res = await fetch('/feed?category=' + cat, {
                method: 'GET',
                headers: {
                    'Accept': 'application/json'
                }
            });
            
            console.log('📨 Статус:', res.status);
            
            if(!res.ok) {
                throw new Error('Ошибка сервера: ' + res.status);
            }
            
            const data = await res.json();
            console.log('✅ Получено статей:', data.articles?.length || 0);
            
            grid.innerHTML = '';
            
            if(!data.articles || data.articles.length === 0) {
                grid.innerHTML = '<p style="grid-column: 1/-1; text-align: center; color: var(--text-dim);">Нет статей</p>';
                return;
            }

            data.articles.forEach(art => {
                const card = document.createElement('div');
                card.className = 'news-card';
                card.innerHTML = `
                    <div class="tag-badge">${art.tag}</div>
                    <div class="news-title">${art.title}</div>
                `;
                card.onclick = () => translateUrl(art.link);
                grid.appendChild(card);
            });
            
        } catch(e) { 
            console.error('❌ Ошибка загрузки:', e);
            grid.innerHTML = '<p style="grid-column: 1/-1; text-align: center; color: #ff6b6b;">Ошибка загрузки: ' + e.message + '</p>';
            showError('Ошибка загрузки новостей: ' + e.message);
        }
    }

    async function translateUrl(url = null) {
        const inputUrl = url || document.getElementById('urlInput').value;
        if(!inputUrl) {
            showError('Введите ссылку!');
            return;
        }
        
        console.log('🔄 Перевод:', inputUrl);
        
        document.getElementById('newsSection').style.display = 'none';
        document.getElementById('result').style.display = 'none';
        document.getElementById('loader').style.display = 'block';
        document.getElementById('errorMsg').style.display = 'none';

        try {
            const res = await fetch('/translate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                },
                body: JSON.stringify({url: inputUrl})
            });
            
            console.log('📨 Статус:', res.status);
            
            if(!res.ok) {
                const errorData = await res.json();
                throw new Error(errorData.detail || 'Ошибка сервера');
            }
            
            const data = await res.json();
            console.log('✅ Перевод получен');
            
            document.getElementById('result').innerHTML = data.translated_html;
            document.getElementById('result').style.display = 'block';
            
        } catch(e) {
            console.error('❌ Ошибка:', e);
            showError('Ошибка перевода: ' + e.message);
            goHome();
        } finally {
            document.getElementById('loader').style.display = 'none';
        }
    }

    async function translateText() {
        const text = document.getElementById('textInput').value;
        if(!text) {
            showError('Введите текст!');
            return;
        }
        
        console.log('📝 Перевод текста...');
        document.getElementById('loader').style.display = 'block';

        try {
            const res = await fetch('/translate_text', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                },
                body: JSON.stringify({text: text})
            });
            
            const data = await res.json();
            console.log('✅ Готово');
            
            document.getElementById('result').innerHTML = '<p>' + data.result + '</p>';
            document.getElementById('result').style.display = 'block';
            
        } catch(e) { 
            console.error('❌', e);
            showError('Ошибка перевода текста');
        } finally { 
            document.getElementById('loader').style.display = 'none'; 
        }
    }
</script>

</body>
</html>
"""

# ========== FEEDS ==========

FEEDS = {
    "programming": [
        "https://www.reddit.com/r/programming/.rss",
        "https://dev.to/feed"
    ],
    "history": [
        "https://www.reddit.com/r/history/.rss"
    ],
    "gaming": [
        "https://www.reddit.com/r/gaming/.rss"
    ],
    "movies": [
        "https://www.reddit.com/r/movies/.rss"
    ]
}

# ========== ENDPOINTS ==========

@app.get("/", response_class=HTMLResponse)
async def home():
    return HTML_PAGE

@app.get("/health")
async def health():
    return JSONResponse(
        content={"status": "OK"},
        headers={"Access-Control-Allow-Origin": "*"}
    )

@app.get("/feed")
async def get_feed(category: str = "programming"):
    print(f"\n{'='*60}")
    print(f"📰 Категория: {category}")
    
    if category not in FEEDS:
        category = "programming"
    
    articles = []
    
    for feed_url in FEEDS[category]:
        try:
            print(f"📡 {feed_url}")
            
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(feed_url, headers=headers, timeout=8)
            
            if response.status_code != 200:
                print(f"❌ Статус: {response.status_code}")
                continue
            
            soup = BeautifulSoup(response.content, 'xml')
            entries = soup.find_all('entry')[:8] or soup.find_all('item')[:8]
            
            print(f"✅ Найдено: {len(entries)}")
            
            for entry in entries:
                try:
                    title_tag = entry.find('title')
                    if not title_tag:
                        continue
                    
                    title = BeautifulSoup(title_tag.get_text(), 'html.parser').get_text().strip()
                    
                    link_tag = entry.find('link')
                    if link_tag:
                        link = link_tag.get('href') or link_tag.get_text().strip()
                    else:
                        continue
                    
                    if title and link:
                        articles.append({
                            "title": title[:120],
                            "original_title": title[:120],
                            "link": link,
                            "tag": category.upper()
                        })
                except:
                    continue
            
        except Exception as e:
            print(f"❌ {e}")
            continue
    
    # Фоллбэк если ничего не загрузилось
    if len(articles) == 0:
        print("⚠️ Тестовые данные")
        articles = [
            {"title": "Python Tutorial", "original_title": "Python Tutorial", "link": "https://dev.to/", "tag": category.upper()},
            {"title": "Web Development", "original_title": "Web Development", "link": "https://dev.to/", "tag": category.upper()}
        ]
    
    print(f"📦 Итого: {len(articles)}\n")
    
    return JSONResponse(
        content={"articles": articles[:12], "category": category},
        headers={"Access-Control-Allow-Origin": "*"}
    )

@app.post("/translate")
async def translate_article(request: LinkRequest):
    print(f"\n{'='*60}")
    print(f"🔄 URL: {request.url}")
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html',
        }
        
        response = requests.get(request.url, headers=headers, timeout=12)
        
        print(f"✅ Статус: {response.status_code}")
        
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail=f"Не удалось загрузить (статус {response.status_code})")
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Заголовок
        title = "Статья"
        if soup.find('h1'):
            title = soup.find('h1').get_text().strip()
        elif soup.find('title'):
            title = soup.find('title').get_text().strip()
        
        # Контент
        content = None
        for selector in ['article', 'main', {'class': 'content'}, 'body']:
            if isinstance(selector, str):
                content = soup.find(selector)
            else:
                content = soup.find(**selector)
            if content:
                break
        
        if not content:
            raise HTTPException(status_code=400, detail="Контент не найден")
        
        # Очистка
        for tag in content(['script', 'style', 'nav', 'header', 'footer', 'aside']):
            tag.decompose()
        
        # Обработка изображений
        for img in content.find_all('img'):
            if img.get('src'):
                img['style'] = 'max-width:100%;height:auto;border-radius:8px;'
        
        print(f"✅ Готово\n")
        
        return JSONResponse(
            content={
                "title": title,
                "translated_html": str(content),
                "success": True
            },
            headers={"Access-Control-Allow-Origin": "*"}
        )
        
    except Exception as e:
        print(f"❌ {e}\n")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/translate_text")
async def translate_text(request: TextRequest):
    return JSONResponse(
        content={"result": request.text, "success": True},
        headers={"Access-Control-Allow-Origin": "*"}
    )

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("🚀 СЕРВЕР")
    print("="*60)
    print("📍 http://localhost:8000")
    print("="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
