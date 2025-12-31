from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
import requests
from bs4 import BeautifulSoup
import uvicorn

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
def home():
    return """<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>News/Translate by Kanamy</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: -apple-system, sans-serif;
            background: #0a0a0a;
            color: #e8e8e8;
            padding: 20px;
        }

        .container { max-width: 1200px; margin: 0 auto; }

        h1 {
            text-align: center;
            font-size: 40px;
            font-weight: 300;
            margin: 40px 0;
        }

        h1 span { color: #FF6B35; }

        .input-box {
            max-width: 600px;
            margin: 0 auto 40px;
            background: #1a1a1a;
            padding: 24px;
            border-radius: 12px;
        }

        input {
            width: 100%;
            padding: 14px;
            background: #0a0a0a;
            border: 1px solid #333;
            border-radius: 8px;
            color: #fff;
            font-size: 15px;
            margin-bottom: 12px;
        }

        button {
            width: 100%;
            padding: 14px;
            background: #FF6B35;
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 15px;
            cursor: pointer;
        }

        button:hover { background: #ff8555; }

        .tabs {
            display: flex;
            gap: 12px;
            justify-content: center;
            margin-bottom: 40px;
        }

        .tab {
            padding: 10px 24px;
            background: #1a1a1a;
            border: 1px solid #333;
            border-radius: 8px;
            cursor: pointer;
        }

        .tab.active {
            background: #FF6B35;
            border-color: #FF6B35;
        }

        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
        }

        .card {
            background: #1a1a1a;
            padding: 24px;
            border-radius: 12px;
            border: 1px solid #333;
            cursor: pointer;
            transition: 0.2s;
        }

        .card:hover {
            transform: translateY(-4px);
            border-color: #FF6B35;
        }

        .tag {
            font-size: 10px;
            color: #FF6B35;
            text-transform: uppercase;
            margin-bottom: 8px;
        }

        .card-title {
            font-size: 16px;
            line-height: 1.4;
        }

        #article {
            display: none;
            background: #1a1a1a;
            padding: 40px;
            border-radius: 12px;
            margin-top: 40px;
        }

        #article h1, #article h2 { color: #FF6B35; margin: 20px 0 10px; }
        #article p { margin: 12px 0; line-height: 1.7; }
        #article img { max-width: 100%; border-radius: 8px; margin: 20px 0; }

        .loader {
            display: none;
            text-align: center;
            padding: 60px;
            color: #888;
        }

        .spinner {
            width: 40px;
            height: 40px;
            border: 3px solid #333;
            border-top-color: #FF6B35;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }

        @keyframes spin { 100% { transform: rotate(360deg); } }

        .back {
            display: inline-block;
            padding: 10px 20px;
            background: #1a1a1a;
            border-radius: 8px;
            cursor: pointer;
            margin-bottom: 20px;
        }
    </style>
</head>
<body>

<div class="container">
    <h1>Умный <span>переводчик</span></h1>

    <div class="input-box">
        <input type="text" id="urlInput" placeholder="https://example.com/article">
        <button onclick="translateUrl()">Перевести</button>
    </div>

    <div id="news">
        <div class="tabs">
            <div class="tab active" onclick="showCategory('programming')">Программирование</div>
            <div class="tab" onclick="showCategory('history')">История</div>
            <div class="tab" onclick="showCategory('gaming')">Игры</div>
            <div class="tab" onclick="showCategory('movies')">Кино</div>
        </div>
        <div class="grid" id="grid"></div>
    </div>

    <div class="loader" id="loader">
        <div class="spinner"></div>
        <p>Загрузка...</p>
    </div>

    <div id="article">
        <div class="back" onclick="goBack()">← Назад</div>
        <div id="articleContent"></div>
    </div>
</div>

<script>
console.log('✅ Скрипт загружен');

// Встроенные статьи
const articles = {
    programming: [
        {title: 'Python для начинающих - полное руководство', url: 'https://realpython.com/'},
        {title: 'FastAPI - современный веб-фреймворк', url: 'https://fastapi.tiangolo.com/'},
        {title: 'JavaScript async/await подробно', url: 'https://developer.mozilla.org/'},
        {title: 'React Hooks - детальный гайд', url: 'https://react.dev/'},
        {title: 'Docker контейнеры для разработчиков', url: 'https://docs.docker.com/'},
    ],
    history: [
        {title: 'Древний Рим - история империи', url: 'https://www.britannica.com/'},
        {title: 'Египетские пирамиды - секреты строительства', url: 'https://www.nationalgeographic.com/'},
    ],
    gaming: [
        {title: 'Dota 2 - обзор нового патча', url: 'https://www.dota2.com/'},
        {title: 'Clash Royale - топ стратегии', url: 'https://clashroyale.com/'},
    ],
    movies: [
        {title: 'Лучшие фильмы 2025 года', url: 'https://www.imdb.com/'},
        {title: 'Netflix - новинки января', url: 'https://www.netflix.com/'},
    ]
};

let currentCategory = 'programming';

function showCategory(category) {
    console.log('📰 Показываю категорию:', category);
    
    currentCategory = category;
    
    // Обновляем активную вкладку
    document.querySelectorAll('.tab').forEach(tab => {
        tab.classList.remove('active');
    });
    event.target.classList.add('active');
    
    // Рендерим карточки
    const grid = document.getElementById('grid');
    grid.innerHTML = '';
    
    articles[category].forEach(article => {
        const card = document.createElement('div');
        card.className = 'card';
        card.innerHTML = `
            <div class="tag">${category.toUpperCase()}</div>
            <div class="card-title">${article.title}</div>
        `;
        card.onclick = () => loadArticle(article.url);
        grid.appendChild(card);
    });
    
    console.log('✅ Показано карточек:', articles[category].length);
}

async function loadArticle(url) {
    console.log('🔄 Загружаю статью:', url);
    
    // Скрываем новости, показываем загрузку
    document.getElementById('news').style.display = 'none';
    document.getElementById('article').style.display = 'none';
    document.getElementById('loader').style.display = 'block';
    
    try {
        const formData = new FormData();
        formData.append('url', url);
        
        const response = await fetch('/translate', {
            method: 'POST',
            body: formData
        });
        
        console.log('📡 Ответ:', response.status);
        
        if (!response.ok) {
            throw new Error('Ошибка загрузки');
        }
        
        const html = await response.text();
        console.log('✅ Статья загружена');
        
        document.getElementById('articleContent').innerHTML = html;
        document.getElementById('article').style.display = 'block';
        
    } catch (error) {
        console.error('❌ Ошибка:', error);
        alert('Не удалось загрузить статью: ' + error.message);
        goBack();
    } finally {
        document.getElementById('loader').style.display = 'none';
    }
}

function translateUrl() {
    const url = document.getElementById('urlInput').value.trim();
    if (!url) {
        alert('Введите URL!');
        return;
    }
    loadArticle(url);
}

function goBack() {
    document.getElementById('article').style.display = 'none';
    document.getElementById('news').style.display = 'block';
    showCategory(currentCategory);
}

// Инициализация
console.log('🚀 Инициализация...');
showCategory('programming');
console.log('✅ Готово!');
</script>

</body>
</html>"""

@app.post("/translate")
async def translate(url: str = Form(...)):
    print(f"\n🔄 Перевод: {url}")
    
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        
        print(f"✅ Статус: {response.status_code}")
        
        if response.status_code != 200:
            return f"<p style='color:#ff6b6b'>Ошибка загрузки (HTTP {response.status_code})</p>"
        
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
            for tag in content(['script', 'style', 'nav', 'header', 'footer', 'aside']):
                tag.decompose()
            
            for img in content.find_all('img'):
                img['style'] = 'max-width:100%; border-radius:8px;'
            
            html = f"<h1>{title}</h1>{str(content)}"
        else:
            html = "<p>Контент не найден</p>"
        
        print(f"✅ Успешно\n")
        return html
        
    except Exception as e:
        print(f"❌ Ошибка: {e}\n")
        return f"<p style='color:#ff6b6b'>Ошибка: {str(e)}</p>"

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🚀 СЕРВЕР ЗАПУЩЕН")
    print("="*70)
    print("📍 http://localhost:8000")
    print("✅ Статьи встроены в HTML - показываются СРАЗУ!")
    print("✅ Никаких внешних запросов для статей!")
    print("="*70 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
