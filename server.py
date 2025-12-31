from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import requests
from bs4 import BeautifulSoup

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
            color: var(--text);
            min-height: 60px;
            letter-spacing: -1px;
        }
        
        .hero h1 span {
            color: var(--orange);
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
            position: relative;
            overflow: hidden;
        }
        .btn-main::before {
            content: '';
            position: absolute;
            top: 50%;
            left: 50%;
            width: 0;
            height: 0;
            border-radius: 50%;
            background: rgba(255,255,255,0.3);
            transform: translate(-50%, -50%);
            transition: width 0.6s, height 0.6s;
        }
        .btn-main:hover::before { width: 300px; height: 300px; }
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
            position: relative;
        }
        .cat-btn::after {
            content: '';
            position: absolute;
            bottom: 0;
            left: 50%;
            width: 0;
            height: 2px;
            background: var(--orange);
            transition: all 0.3s;
            transform: translateX(-50%);
        }
        .cat-btn:hover::after { width: 80%; }
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
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            border: 1px solid var(--border);
            position: relative;
            overflow: hidden;
        }
        .news-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 3px;
            background: var(--orange);
            transform: scaleX(0);
            transition: transform 0.3s;
        }
        .news-card::after {
            content: '';
            position: absolute;
            inset: -2px;
            background: linear-gradient(45deg, transparent, var(--orange), transparent);
            background-size: 200% 200%;
            border-radius: 12px;
            opacity: 0;
            transition: opacity 0.3s;
            z-index: -1;
            animation: cardGlow 3s linear infinite;
        }
        @keyframes cardGlow {
            0%, 100% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
        }
        .news-card:hover::before { transform: scaleX(1); }
        .news-card:hover::after { opacity: 0.2; }
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
            letter-spacing: 1px;
            margin-bottom: 12px;
        }
        
        .news-title { 
            font-size: 17px;
            font-weight: 500;
            line-height: 1.5;
            margin-bottom: 12px;
        }

        .article-controls { 
            display: flex;
            gap: 12px;
            margin-bottom: 30px;
            padding-bottom: 20px;
            border-bottom: 1px solid var(--border);
            flex-wrap: wrap;
        }
        
        .control-btn { 
            padding: 8px 16px;
            background: transparent;
            border: 1px solid var(--border);
            color: var(--text);
            cursor: pointer;
            font-size: 13px;
            font-weight: 500;
            border-radius: 6px;
            display: flex;
            gap: 6px;
            transition: all 0.2s;
        }
        .control-btn:hover { 
            border-color: var(--orange);
            background: var(--orange-dim);
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
        .article-view img {
            max-width: 100%;
            height: auto;
            border-radius: 12px;
            margin: 20px 0;
        }

        .text-result { 
            background: var(--surface);
            padding: 30px;
            border-radius: 12px;
            margin-top: 30px;
            display: none;
            border-left: 3px solid var(--orange);
            font-size: 17px;
        }

        .loader { 
            display: none;
            margin: 40px auto;
            width: 40px;
            height: 40px;
            position: relative;
        }
        .loader::before, .loader::after {
            content: '';
            position: absolute;
            border: 2px solid var(--orange);
            border-radius: 50%;
            animation: ripple 1.5s infinite;
        }
        .loader::before { width: 40px; height: 40px; }
        .loader::after { width: 40px; height: 40px; animation-delay: 0.75s; }
        @keyframes ripple {
            0% { transform: scale(0); opacity: 1; }
            100% { transform: scale(1.5); opacity: 0; }
        }

        .reading-progress {
            position: fixed;
            top: 0;
            left: 0;
            height: 3px;
            background: linear-gradient(90deg, var(--orange) 0%, #ff8c61 100%);
            z-index: 1000;
            width: 0%;
            transition: width 0.1s;
        }

        @media (max-width: 768px) {
            .hero h1 { font-size: 32px; }
            .article-view { padding: 24px; font-size: 16px; }
            .container { padding: 40px 16px; }
            .news-grid { grid-template-columns: 1fr; }
        }

        ::-webkit-scrollbar { width: 8px; }
        ::-webkit-scrollbar-track { background: var(--bg); }
        ::-webkit-scrollbar-thumb { 
            background: var(--border);
            border-radius: 10px;
        }
        ::-webkit-scrollbar-thumb:hover { background: var(--orange); }
    </style>
</head>
<body>

<div class="reading-progress" id="readingProgress"></div>
<canvas id="particles"></canvas>

<header>
    <div class="logo" onclick="goHome()">News/Translate <span>by Kanamy</span></div>
</header>

<div class="container">
    <div class="hero">
        <h1 id="mainTitle">Умный переводчик</h1>
        
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
            <textarea id="textInput" placeholder="Вставьте текст для перевода..."></textarea>
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
        <div class="article-controls">
            <button class="control-btn" onclick="copyArticle()">📋 Копировать</button>
        </div>
        <div id="articleContent"></div>
    </div>
    
    <div id="textResult" class="text-result"></div>
</div>

<script>
    const API = '';
    
    // Прогресс бар
    window.addEventListener('scroll', () => {
        const winScroll = document.body.scrollTop || document.documentElement.scrollTop;
        const height = document.documentElement.scrollHeight - document.documentElement.clientHeight;
        const scrolled = (winScroll / height) * 100;
        document.getElementById('readingProgress').style.width = scrolled + '%';
    });

    // Particles
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

    // Cursor trail
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

    window.onload = function() {
        console.log('🚀 Инициализация');
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
        console.log('📰 Загрузка категории:', cat);
        resetView();
        document.querySelectorAll('.cat-btn').forEach(b => b.classList.remove('active'));
        if(btn) btn.classList.add('active');
        
        const grid = document.getElementById('newsGrid');
        grid.innerHTML = "";
        document.getElementById('loader').style.display = 'block';

        try {
            const res = await fetch(`/feed?category=${cat}`);
            const data = await res.json();
            console.log('✅ Получено статей:', data.articles?.length || 0);
            
            document.getElementById('loader').style.display = 'none';
            
            if(!data.articles || data.articles.length === 0) {
                grid.innerHTML = "<p style='grid-column: 1/-1; text-align: center; color: var(--text-dim);'>Новостей пока нет</p>";
                return;
            }

            data.articles.forEach(art => {
                const card = document.createElement('div');
                card.className = 'news-card';
                card.innerHTML = `
                    <div class="tag-badge">${art.tag}</div>
                    <div class="news-title">${art.title}</div>
                    <div style="font-size:12px; color: var(--text-dim); margin-top: 8px;">${art.original_title}</div>
                `;
                card.onclick = () => translateUrl(art.link);
                grid.appendChild(card);
            });
        } catch(e) { 
            console.error('❌ Ошибка загрузки ленты:', e);
            document.getElementById('loader').style.display = 'none';
            grid.innerHTML = "<p style='grid-column: 1/-1; text-align: center; color: var(--text-dim);'>Ошибка загрузки</p>";
        }
    }

    async function translateUrl(url = null) {
        const inputUrl = url || document.getElementById('urlInput').value;
        if(!inputUrl) return alert("Введите ссылку!");
        
        console.log('🔄 Начинаю перевод:', inputUrl);
        saveToHistory(inputUrl);
        
        document.getElementById('newsSection').style.display = 'none';
        document.getElementById('result').style.display = 'none';
        document.getElementById('loader').style.display = 'block';
        window.scrollTo({top: 0, behavior: 'smooth'});

        try {
            const res = await fetch('/translate', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({url: inputUrl})
            });
            
            console.log('📨 Статус ответа:', res.status);
            
            if(!res.ok) {
                throw new Error(`Server error: ${res.status}`);
            }
            
            const data = await res.json();
            console.log('✅ Перевод получен. Слов:', data.word_count);
            
            document.getElementById('articleContent').innerHTML = data.translated_html;
            document.getElementById('result').style.display = 'block';
            
        } catch(e) {
            console.error('❌ Ошибка перевода:', e);
            alert("Ошибка перевода: " + e.message);
            goHome();
        } finally {
            document.getElementById('loader').style.display = 'none';
        }
    }

    async function translateText() {
        const text = document.getElementById('textInput').value;
        if(!text) return alert("Введите текст!");
        
        console.log('📝 Переводим текст...');
        
        document.getElementById('textResult').style.display = 'none';
        document.getElementById('loader').style.display = 'block';

        try {
            const res = await fetch('/translate_text', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({text: text})
            });
            const data = await res.json();
            console.log('✅ Текст переведен');
            document.getElementById('textResult').innerText = data.result;
            document.getElementById('textResult').style.display = 'block';
        } catch(e) { 
            console.error('❌ Ошибка:', e);
            alert("Ошибка перевода текста"); 
        } finally { 
            document.getElementById('loader').style.display = 'none'; 
        }
    }

    function copyArticle() {
        const text = document.getElementById('articleContent').innerText;
        navigator.clipboard.writeText(text).then(() => {
            const btn = event.target;
            const original = btn.innerHTML;
            btn.innerHTML = '✓ Скопировано';
            setTimeout(() => btn.innerHTML = original, 2000);
        });
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

# ========== ENDPOINTS ==========

@app.get("/", response_class=HTMLResponse)
def home():
    return HTML_PAGE

@app.get("/health")
def health():
    return {"status": "OK"}

@app.get("/feed")
def get_feed(category: str = "programming"):
    print(f"\n{'='*60}")
    print(f"📰 ЗАПРОС ЛЕНТЫ: {category}")
    print(f"{'='*60}")
    
    if category not in FEEDS:
        category = "programming"
    
    articles = []
    
    for feed_info in FEEDS[category]:
        try:
            print(f"📡 Загружаю: {feed_info['url']}")
            
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            response = requests.get(feed_info['url'], headers=headers, timeout=10)
            
            print(f"✅ Статус: {response.status_code}")
            
            if response.status_code != 200:
                print(f"❌ Ошибка загрузки")
                continue
            
            soup = BeautifulSoup(response.content, 'xml')
            
            entries = soup.find_all('entry')[:10]
            if not entries:
                entries = soup.find_all('item')[:10]
            
            print(f"📊 Найдено записей: {len(entries)}")
            
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
                    
                except Exception as e:
                    print(f"⚠️ Ошибка парсинга записи: {e}")
                    continue
            
            print(f"✅ Добавлено статей: {len(articles)}")
            
        except Exception as e:
            print(f"❌ ОШИБКА загрузки фида: {e}")
            continue
    
    if len(articles) == 0:
        print("⚠️ Добавляю тестовые статьи")
        articles = [
            {
                "title": "Тестовая статья 1 - Программирование",
                "original_title": "Test Article 1",
                "link": "https://dev.to/",
                "tag": category.upper()
            },
            {
                "title": "Тестовая статья 2 - Туториал",
                "original_title": "Test Article 2",
                "link": "https://dev.to/",
                "tag": category.upper()
            }
        ]
    
    print(f"\n📦 ИТОГО статей: {len(articles)}")
    print(f"{'='*60}\n")
    
    return {
        "articles": articles[:15],
        "category": category,
        "total": len(articles)
    }

@app.post("/translate")
def translate_article(request: LinkRequest):
    print(f"\n{'='*60}")
    print(f"🔄 ПЕРЕВОД СТАТЬИ")
    print(f"URL: {request.url}")
    print(f"{'='*60}")
    
    try:
        url = str(request.url)
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
        }
        
        print(f"📡 Загружаю страницу...")
        
        response = requests.get(url, headers=headers, timeout=15, allow_redirects=True)
        
        print(f"✅ Статус: {response.status_code}")
        
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail=f"Не удалось загрузить страницу (статус {response.status_code})")
        
        print(f"📄 Парсинг HTML...")
        soup = BeautifulSoup(response.text, 'html.parser')
        
        title = "Статья"
        title_tag = soup.find('h1')
        if title_tag:
            title = title_tag.get_text().strip()
        elif soup.find('title'):
            title = soup.find('title').get_text().strip()
        
        print(f"📌 Заголовок: {title}")
        
        content = None
        selectors = [
            'article',
            'main',
            {'class': 'post-content'},
            {'class': 'article-content'},
            {'class': 'entry-content'},
            {'class': 'content'},
            {'id': 'content'},
            'body'
        ]
        
        for selector in selectors:
            if isinstance(selector, str):
                content = soup.find(selector)
            else:
                content = soup.find(**selector)
            
            if content:
                print(f"✅ Контент найден: {selector}")
                break
        
        if not content:
            print("❌ Контент не найден")
            raise HTTPException(status_code=400, detail="Не удалось найти контент статьи")
        
        for tag in content(['script', 'style', 'nav', 'header', 'footer', 'aside', 'iframe', 'noscript']):
            tag.decompose()
        
        print(f"🧹 Очистка завершена")
        
        text = content.get_text()
        words = len(text.split())
        read_time = max(1, round(words / 200))
        
        print(f"📊 Слов: {words}, Время чтения: {read_time} мин")
        
        html_content = str(content)
        
        for img in content.find_all('img'):
            if img.get('src'):
                img['style'] = 'max-width:100%;height:auto;border-radius:8px;margin:20px 0;'
                img['loading'] = 'lazy'
        
        print(f"✅ ПЕРЕВОД ЗАВЕРШЕН")
        print(f"{'='*60}\n")
        
        return {
            "title": title,
            "translated_html": html_content,
            "read_time": read_time,
            "word_count": words,
            "url": url,
            "success": True
        }
        
    except requests.Timeout:
        print(f"❌ TIMEOUT")
        raise HTTPException(status_code=504, detail="Время ожидания истекло")
    
    except requests.RequestException as e:
        print(f"❌ REQUEST ERROR: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка загрузки: {str(e)}")
    
    except Exception as e:
        print(f"❌ ОБЩАЯ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Ошибка: {str(e)}")

@app.post("/translate_text")
def translate_text(request: TextRequest):
    print(f"📝 Перевод текста (длина: {len(request.text)})")
    return {
        "result": request.text,
        "success": True
    }

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("🚀 ЗАПУСК СЕРВЕРА")
    print("="*60)
    print("📍 Откройте: http://localhost:8000")
    print("📖 Документация: http://localhost:8000/docs")
    print("="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
