from fastapi import FastAPI
from fastapi.responses import HTMLResponse, PlainTextResponse
from pydantic import BaseModel
import requests
from bs4 import BeautifulSoup
import uvicorn

app = FastAPI()

class URLRequest(BaseModel):
    url: str

@app.get("/", response_class=HTMLResponse)
def home():
    return """<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DEBUG MODE</title>
    <style>
        body {
            background: white;
            color: black;
            font-family: monospace;
            padding: 20px;
            font-size: 16px;
        }
        
        .status {
            background: yellow;
            padding: 10px;
            margin: 10px 0;
            border: 2px solid black;
        }
        
        .ok {
            background: lightgreen;
        }
        
        .error {
            background: red;
            color: white;
        }
        
        button {
            padding: 15px 30px;
            font-size: 18px;
            margin: 10px;
            cursor: pointer;
            background: blue;
            color: white;
            border: none;
        }
        
        .card {
            background: #ddd;
            padding: 15px;
            margin: 10px 0;
            border: 2px solid black;
            cursor: pointer;
        }
        
        .card:hover {
            background: yellow;
        }
    </style>
</head>
<body>

<h1>🔍 DEBUG MODE</h1>

<div class="status ok">
    ✅ STEP 1: HTML загружен
</div>

<div class="status" id="status2">
    ⏳ STEP 2: Ждём загрузку JavaScript...
</div>

<div class="status" id="status3">
    ⏳ STEP 3: Ждём отрисовку карточек...
</div>

<div class="status" id="status4">
    ⏳ STEP 4: Ждём клик на карточку...
</div>

<hr>

<h2>📰 СТАТЬИ:</h2>
<div id="cards"></div>

<hr>

<h2>📋 ЛОГИ:</h2>
<div id="log" style="background: black; color: lime; padding: 10px; min-height: 200px;"></div>

<script>
function log(msg) {
    const logDiv = document.getElementById('log');
    logDiv.innerHTML += msg + '<br>';
    console.log(msg);
}

log('🚀 JavaScript запущен!');

// STEP 2
document.getElementById('status2').className = 'status ok';
document.getElementById('status2').innerHTML = '✅ STEP 2: JavaScript работает!';
log('✅ JavaScript работает');

// Данные
const articles = [
    {title: "Test Article 1", url: "https://example.com/1"},
    {title: "Test Article 2", url: "https://example.com/2"},
    {title: "Test Article 3", url: "https://example.com/3"},
];

log('📦 Данные загружены: ' + articles.length + ' статей');

// Отрисовка
const container = document.getElementById('cards');

articles.forEach((article, index) => {
    log('🎨 Рисую карточку ' + (index + 1));
    
    const card = document.createElement('div');
    card.className = 'card';
    card.innerHTML = '<strong>' + article.title + '</strong><br>Нажми для теста загрузки';
    
    card.onclick = function() {
        log('🖱️ Клик на: ' + article.title);
        testLoad(article.url);
    };
    
    container.appendChild(card);
});

log('✅ Карточки отрисованы: ' + articles.length + ' шт.');

// STEP 3
document.getElementById('status3').className = 'status ok';
document.getElementById('status3').innerHTML = '✅ STEP 3: Карточки отрисованы (' + articles.length + ' шт)';

async function testLoad(url) {
    document.getElementById('status4').className = 'status';
    document.getElementById('status4').innerHTML = '⏳ STEP 4: Загружаю статью...';
    
    log('📡 Отправляю запрос на сервер...');
    log('URL: ' + url);
    
    try {
        const res = await fetch('/test');
        const text = await res.text();
        
        log('✅ Ответ получен: ' + text);
        
        document.getElementById('status4').className = 'status ok';
        document.getElementById('status4').innerHTML = '✅ STEP 4: Сервер отвечает!';
        
        alert('SUCCESS! Сервер работает!\\n\\nОтвет: ' + text);
        
    } catch(e) {
        log('❌ ОШИБКА: ' + e);
        
        document.getElementById('status4').className = 'status error';
        document.getElementById('status4').innerHTML = '❌ STEP 4: ОШИБКА - ' + e;
        
        alert('ERROR: ' + e);
    }
}

log('✅ Всё готово к работе!');
log('');
log('📌 ИНСТРУКЦИЯ:');
log('1. Видишь ли ты 3 карточки выше?');
log('2. Кликни на любую карточку');
log('3. Должен появиться alert "SUCCESS!"');
</script>

</body>
</html>"""

@app.get("/test", response_class=PlainTextResponse)
def test():
    return "Server is working! Time: 2025-12-31 13:45"

@app.post("/load", response_class=HTMLResponse)
def load_article(req: URLRequest):
    print(f"\n🔄 Loading: {req.url}")
    
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        r = requests.get(req.url, headers=headers, timeout=10)
        
        print(f"✅ Status: {r.status_code}")
        
        soup = BeautifulSoup(r.text, 'html.parser')
        title = soup.find('title')
        
        return f"<h1>Loaded!</h1><p>Title: {title.get_text() if title else 'No title'}</p>"
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return f"<p>Error: {e}</p>"

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🔍 DEBUG MODE")
    print("="*60)
    print("📍 http://localhost:8000")
    print("="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
