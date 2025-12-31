from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import requests
from bs4 import BeautifulSoup

app = FastAPI()

# CORS - максимально открытый
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

# ========== HTML СТРАНИЦА ==========

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

        .cursor-trail {
            position: absolute;
            width: 4px;
            height: 4px;
            background: var(--orange);
            border-radius: 50%;
            pointer-events: none;
            animation: trailFade 1s ease-out forwards;
            z-index: 9999;
        }
        
        @keyframes trailFade {
            to { transform: scale(3); opacity: 0; }
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
            background: linear-gradient(135deg, var(--text) 0%, var(--orange) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            min-height: 60px;
        }

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
            box-shadow: 0 4px 12px rgba(255, 107, 53, 0.3);
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
            transition: all 0.3s;
        }
        input:focus, textarea:focus { 
            border-color: var(--orange);
            outline: none;
            box-shadow: 0 0 0 3px var(--orange-dim);
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
        }
        .btn-main:hover { 
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(255, 107, 53, 0.4);
        }

        .history-box { margin-top: 30px; display: none; }
        .history-title { 
            font-size: 11px;
            font-weight: 600;
            color: var(--text-dim);
            text-transform: uppercase;
            margin-bottom: 12px;
        }
        .history-item { 
            background: var(--surface);
            padding: 12px 16px;
            border-radius: 8px;
            margin-bottom: 8px;
            font-size: 13px;
            cursor: pointer;
            display: flex;
            gap: 12px;
            border: 1px solid transparent;
            transition: all 0.2s;
        }
        .history-item:hover { 
            border-color: var(--orange);
            transform: translateX(4px);
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
            font-size: 14px;
        }
        .cat-btn.active { 
            background: var(--orange);
            color: #fff;
            border-color: var(--orange);
        }

        .news-grid { 
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 24px;
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
            box-shadow: 0 12px 24px rgba(0,0,0,0.3);
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

        .text-result { 
            background: var(--surface);
            padding: 30px;
            border-radius: 12px;
            margin-top: 30px;
            display: none;
            border-left: 3px solid var(--orange);
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
        <h1 id="mainTitle"></h1>
        
        <div class="tabs">
            <button class="tab-btn active" onclick="switchMode('url')">🔗 Ссылка</button>
            <button class="tab-btn" onclick="switchMode('text')">📝 Текст</button>
        </div>

        <div id="urlMode" class="input-area active">
            <input type="text" id="urlInput" placeholder="Вставьте ссылку на статью..." />
            <button class="btn-main" onclick="translateUrl()">Перевести</button>
            
            <div id="historyBox" class="history-box">
                <div class="history-title">Недавно просмотрено</div>
                <div id="historyList"></div>
            </div>
        </div>

        <div id="textMode" class="input-area">
            <textarea id="textInput" placeholder="Вставьте текст..."></textarea>
            <button class="btn-main" onclick="translateText()">Перевести</button>
        </div>
    </div>

    <div id="newsSection">
        <div class="categories">
            <button class="cat-btn active" onclick="loadCategory('programming', this)">Программирование</button>
            <button class="cat-btn" onclick="loadCategory('history', this)">История</button>
            <button class="cat-btn" onclick="loadCategory('gaming', this)">Игры</button>
            <button class="cat-btn" onclick="loadCategory('movies', this)">Кино</button>
        </div>
        <div class="news-grid" id="newsGrid"></div>
    </div>

    <div class="loader" id="loader"></div>
    <div id="result" class="article-view">
        <div id="articleContent"></div>
    </div>
    <div id="textResult" class="text-result"></div>
</div>

<script>
    const API = '';  // Пустая строка = тот же сервер
    let isSpeaking = false;

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
        for(let i = 0; i < 80; i++) particles.push(new Particle());
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

    let lastTrailTime = 0;
    document.addEventListener('mousemove', (e) => {
        const now = Date.now();
        if(now - lastTrailTime < 50) return;
        lastTrailTime = now;
        const trail = document.createElement('div');
        trail.className = 'cursor-trail';
        trail.style.left = e.pageX + 'px';
        trail.style.top = e.pageY + 'px';
        document.body.appendChild(trail);
        setTimeout(() => trail.remove(), 1000);
    });

    const titleElement = document.getElementById('mainTitle');
    const titleText = 'Умный переводчик';
    let titleIndex = 0;

    function typeWriter() {
        if(titleIndex < titleText.length) {
            titleElement.textContent += titleText.charAt(titleIndex);
            titleIndex++;
            setTimeout(typeWriter, 100);
        }
    }

    window.onload = function() {
        console.log('🚀 Инициализация');
        setTimeout(typeWriter, 300);
        loadCategory('programming', document.querySelector('.cat-btn.active'));
        renderHistory();
    };

    function resetView() {
        document.getElementById('result').style.display = 'none';
        document.getElementById('textResult').style.display = 'none';
        document.getElementById('loader').style.display = 'none';
        document.querySelector('.hero').style.display = 'block';
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

    function saveToHistory(url) {
        let history = JSON.parse(localStorage.getItem('kanamyHistory') || '[]');
        history = history.filter(item => item !== url);
        history.unshift(url);
        if(history.length > 3) history.pop();
        localStorage.setItem('kanamyHistory', JSON.stringify(history));
        renderHistory();
    }

    function renderHistory() {
        const list = document.getElementById('historyList');
        const box = document.getElementById('historyBox');
        const history = JSON.parse(localStorage.getItem('kanamyHistory') || '[]');
        
        if(history.length === 0) {
            box.style.display = 'none';
            return;
        }
        
        box.style.display = 'block';
        list.innerHTML = "";
        history.forEach(url => {
            const item = document.createElement('div');
            item.className = 'history-item';
            const shortUrl = url.length > 50 ? url.substring(0, 50) + '...' : url;
            item.innerHTML = `<span style="opacity:0.5">🕒</span>${shortUrl}`;
            item.onclick = () => translateUrl(url);
            list.appendChild(item);
        });
    }

    async function loadCategory(cat, btn) {
        console.log('📰 Загрузка:', cat);
        resetView();
        document.querySelectorAll('.cat-btn').forEach(b => b.classList.remove('active'));
        if(btn) btn.classList.add('active');
        
        const grid = document.getElementById('newsGrid');
        grid.innerHTML = "";
        document.getElementById('loader').style.display = 'block';

        try {
            const res = await fetch(`/feed?category=${cat}`);
            const data = await res.json();
            console.log('✅ Статей:', data.articles?.length);
            
            document.getElementById('loader').style.display = 'none';
            
            if(!data.articles || data.articles.length === 0) {
                grid.innerHTML = "<p style='grid-column: 1/-1; text-align: center;'>Нет статей</p>";
                return;
            }

            data.articles.forEach(art => {
                const card = document.createElement('div');
                card.className = 'news-card';
                card.innerHTML = `
                    <div class="tag-badge">${art.tag}</div>
                    <div class="news-title">${art.title}</div>
                    <div style="font-size:12px; color: #888; margin-top: 8px;">${art.original_title}</div>
                `;
                card.onclick = () => translateUrl(art.link);
                grid.appendChild(card);
            });
        } catch(e) { 
            console.error('❌ Ошибка:', e);
            document.getElementById('loader').style.display = 'none';
        }
    }

    async function translateUrl(url = null) {
        const inputUrl = url || document.getElementById('urlInput').value;
        if(!inputUrl) return alert("Введите ссылку!");
        
        console.log('🔄 Перевод:', inputUrl);
        saveToHistory(inputUrl);
        
        document.getElementById('newsSection').style.display = 'none';
        document.getElementById('result').style.display = 'none';
        document.getElementById('loader').style.display = 'block';

        try {
            const res = await fetch('/translate', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({url: inputUrl})
            });
            
            if(!res.ok) throw new Error('Ошибка сервера');
            
            const data = await res.json();
            console.log('✅ Готово');
            
            document.getElementById('articleContent').innerHTML = data.translated_html;
            document.getElementById('result').style.display = 'block';
            
        } catch(e) {
            console.error('❌', e);
            alert("Ошибка: " + e.message);
            goHome();
        } finally {
            document.getElementById('loader').style.display = 'none';
        }
    }

    async function translateText() {
        const text = document.getElementById('textInput').value;
        if(!text) return alert("Введите текст!");
        
        document.getElementById('loader').style.display = 'block';

        try {
            const res = await fetch('/translate_text', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({text: text})
            });
            const data = await res.json();
            document.getElementById('textResult').innerText = data.result;
            document.getElementById('textResult').style.display = 'block';
        } catch(e) { 
            alert("Ошибка"); 
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
        {"url": "https://dev.to/feed", "type": "rss"},
        {"url": "https://www.reddit.com/r/programming/.rss", "type": "rss"}
    ],
    "history": [
        {"url": "https://www.reddit.com/r/history/.rss", "type": "rss"}
    ],
    "gaming": [
        {"url": "https://www.reddit.com/r/gaming/.rss", "type": "rss"}
    ],
    "movies": [
        {"url": "https://www.reddit.com/r/movies/.rss", "type": "rss"}
    ]
}

# ========== ГЛАВНАЯ СТРАНИЦА ==========

@app.get("/", response_class=HTMLResponse)
def home():
    return HTML_PAGE

@app.get("/health")
def health():
    return {"status": "OK"}

# ========== RSS ЛЕНТА ==========

@app.get("/feed")
def get_feed(category: str = "programming"):
    print(f"\n{'='*60}")
    print(f"📰 Категория: {category}")
    
    if category not in FEEDS:
        category = "programming"
    
    articles = []
    
    for feed_info in FEEDS[category]:
        try:
            print(f"📡 {feed_info['url']}")
            
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(feed_info['url'], headers=headers, timeout=10)
            
            if response.status_code != 200:
                continue
            
            soup = BeautifulSoup(response.content, 'xml')
            entries = soup.find_all('entry')[:10] or soup.find_all('item')[:10]
            
            for entry in entries:
                try:
                    title_tag = entry.find('title')
                    if not title_tag:
                        continue
                    title = title_tag.get_text().strip()
                    
                    link_tag = entry.find('link')
                    if link_tag:
                        link = link_tag.get('href') or link_tag.get_text().strip()
                    else:
                        continue
                    
                    title = BeautifulSoup(title, 'html.parser').get_text()
                    
                    articles.append({
                        "title": title[:150],
                        "original_title": title[:150],
                        "link": link,
                        "tag": category.upper()
                    })
                except:
                    continue
            
        except Exception as e:
            print(f"❌ {e}")
            continue
    
    if len(articles) == 0:
        articles = [
            {"title": "Тестовая статья 1", "original_title": "Test", "link": "https://dev.to/", "tag": category.upper()},
            {"title": "Тестовая статья 2", "original_title": "Test", "link": "https://dev.to/", "tag": category.upper()}
        ]
    
    print(f"✅ Статей: {len(articles)}\n")
    
    return {"articles": articles[:15], "category": category}

# ========== ПЕРЕВОД ==========

@app.post("/translate")
def translate_article(request: LinkRequest):
    print(f"\n{'='*60}")
    print(f"🔄 URL: {request.url}")
    
    try:
        url = str(request.url)
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html',
        }
        
        response = requests.get(url, headers=headers, timeout=15)
        
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail=f"Статус {response.status_code}")
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        title = "Статья"
        title_tag = soup.find('h1') or soup.find('title')
        if title_tag:
            title = title_tag.get_text().strip()
        
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
        
        for tag in content(['script', 'style', 'nav', 'header', 'footer', 'aside']):
            tag.decompose()
        
        words = len(content.get_text().split())
        read_time = max(1, round(words / 200))
        
        print(f"✅ Готово\n")
        
        return {
            "title": title,
            "translated_html": str(content),
            "read_time": read_time,
            "word_count": words,
            "url": url,
            "success": True
        }
        
    except Exception as e:
        print(f"❌ {e}\n")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/translate_text")
def translate_text(request: TextRequest):
    return {"result": request.text, "success": True}

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("🚀 СЕРВЕР ЗАПУЩЕН")
    print("="*60)
    print("📍 Откройте: http://localhost:8000")
    print("="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
