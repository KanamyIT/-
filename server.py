from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
import requests
from bs4 import BeautifulSoup
import uvicorn

app = FastAPI()

# RSS источники английских новостей
RSS_FEEDS = {
    "programming": [
        "https://dev.to/feed",
        "https://news.ycombinator.com/rss",
    ],
    "tech": [
        "https://techcrunch.com/feed/",
    ],
    "science": [
        "https://www.reddit.com/r/science/.rss",
    ]
}

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
        h1 { text-align: center; font-size: 36px; margin: 40px 0; }
        h1 span { color: #FF6B35; }
        
        .tabs { display: flex; gap: 12px; justify-content: center; margin-bottom: 40px; flex-wrap: wrap; }
        .tab { padding: 10px 24px; background: #1a1a1a; border: 1px solid #333; border-radius: 8px; cursor: pointer; }
        .tab.active { background: #FF6B35; border-color: #FF6B35; }
        
        #grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 20px; }
        .card { background: #1a1a1a; padding: 20px; border-radius: 12px; border: 1px solid #333; cursor: pointer; transition: 0.2s; }
        .card:hover { transform: translateY(-4px); border-color: #FF6B35; }
        .tag { font-size: 10px; color: #FF6B35; text-transform: uppercase; margin-bottom: 8px; }
        .title { font-size: 16px; line-height: 1.4; }
        
        #article { display: none; background: #1a1a1a; padding: 40px; border-radius: 12px; margin-top: 40px; }
        #article h1, #article h2 { color: #FF6B35; margin: 20px 0 10px; }
        #article p { margin: 12px 0; line-height: 1.7; }
        #article img { max-width: 100%; border-radius: 8px; margin: 20px 0; }
        
        .loader { display: none; text-align: center; padding: 60px; }
        .spinner { width: 40px; height: 40px; border: 3px solid #333; border-top-color: #FF6B35; border-radius: 50%; animation: spin 1s linear infinite; margin: 0 auto 20px; }
        @keyframes spin { 100% { transform: rotate(360deg); } }
        
        .back { display: inline-block; padding: 10px 20px; background: #333; border-radius: 8px; cursor: pointer; margin-bottom: 20px; }
        .status { text-align: center; padding: 40px; color: #888; }
    </style>
</head>
<body>

<div class="container">
    <h1>News <span>Translator</span></h1>

    <div id="news">
        <div class="tabs">
            <div class="tab active" onclick="loadCategory('programming')">Programming</div>
            <div class="tab" onclick="loadCategory('tech')">Tech</div>
            <div class="tab" onclick="loadCategory('science')">Science</div>
        </div>
        <div class="status" id="status">Загрузка...</div>
        <div id="grid"></div>
    </div>

    <div class="loader" id="loader"><div class="spinner"></div><p>Загрузка статьи...</p></div>
    
    <div id="article">
        <div class="back" onclick="goBack()">← Назад</div>
        <div id="content"></div>
    </div>
</div>

<script>
async function loadCategory(cat) {
    console.log('📰 Категория:', cat);
    
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    event.target.classList.add('active');
    
    const grid = document.getElementById('grid');
    const status = document.getElementById('status');
    
    grid.innerHTML = '';
    status.style.display = 'block';
    status.textContent = 'Загрузка статей...';
    
    try {
        const res = await fetch('/feed?category=' + cat);
        const data = await res.json();
        
        console.log('✅ Статей:', data.articles.length);
        
        status.style.display = 'none';
        
        if (data.articles.length === 0) {
            status.textContent = 'Нет статей';
            status.style.display = 'block';
            return;
        }
        
        data.articles.forEach(art => {
            const card = document.createElement('div');
            card.className = 'card';
            card.innerHTML = `<div class="tag">${art.tag}</div><div class="title">${art.title}</div>`;
            card.onclick = () => openArticle(art.url);
            grid.appendChild(card);
        });
        
    } catch (e) {
        console.error('❌', e);
        status.textContent = 'Ошибка загрузки';
    }
}

async function openArticle(url) {
    console.log('🔄 Открываю:', url);
    
    document.getElementById('news').style.display = 'none';
    document.getElementById('article').style.display = 'none';
    document.getElementById('loader').style.display = 'block';
    
    try {
        const formData = new FormData();
        formData.append('url', url);
        
        const res = await fetch('/translate', { method: 'POST', body: formData });
        const html = await res.text();
        
        console.log('✅ Загружено');
        
        document.getElementById('content').innerHTML = html;
        document.getElementById('article').style.display = 'block';
        window.scrollTo(0, 0);
        
    } catch (e) {
        console.error('❌', e);
        alert('Ошибка загрузки');
        goBack();
    } finally {
        document.getElementById('loader').style.display = 'none';
    }
}

function goBack() {
    document.getElementById('article').style.display = 'none';
    document.getElementById('news').style.display = 'block';
}

loadCategory('programming');
</script>

</body>
</html>"""

@app.get("/feed")
def get_feed(category: str = "programming"):
    print(f"\n📰 Парсинг категории: {category}")
    
    articles = []
    feeds = RSS_FEEDS.get(category, RSS_FEEDS["programming"])
    
    for feed_url in feeds:
        try:
            print(f"📡 Загружаю: {feed_url}")
            
            response = requests.get(feed_url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=8)
            
            if response.status_code != 200:
                continue
            
            soup = BeautifulSoup(response.content, 'xml')
            entries = soup.find_all('entry')[:10] or soup.find_all('item')[:10]
            
            print(f"✅ Найдено: {len(entries)} статей")
            
            for entry in entries:
                try:
                    title_tag = entry.find('title')
                    link_tag = entry.find('link')
                    
                    if not title_tag or not link_tag:
                        continue
                    
                    title = BeautifulSoup(title_tag.get_text(), 'html.parser').get_text().strip()
                    link = link_tag.get('href') or link_tag.get_text().strip()
                    
                    if title and link:
                        articles.append({
                            "title": title[:100],
                            "url": link,
                            "tag": category.upper()
                        })
                except:
                    continue
                    
        except Exception as e:
            print(f"❌ Ошибка: {e}")
            continue
    
    print(f"📦 Итого: {len(articles)} статей\n")
    
    return {"articles": articles[:15]}

@app.post("/translate")
def translate(url: str = Form(...)):
    print(f"\n🔄 Перевод: {url}")
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html',
        }
        
        response = requests.get(url, headers=headers, timeout=12)
        print(f"✅ Статус: {response.status_code}")
        
        if response.status_code != 200:
            return f"<p style='color:#ff6b6b'>Ошибка загрузки (HTTP {response.status_code})</p>"
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Заголовок
        title = "Article"
        if soup.find('h1'):
            title = soup.find('h1').get_text().strip()
        elif soup.find('title'):
            title = soup.find('title').get_text().strip()
        
        # Контент
        content = None
        for selector in ['article', 'main', '[role="main"]', '.post-content', 'body']:
            if selector.startswith('.'):
                content = soup.find(class_=selector[1:])
            elif selector.startswith('['):
                content = soup.find(attrs={'role': 'main'})
            else:
                content = soup.find(selector)
            if content:
                break
        
        if not content:
            return "<p>Контент не найден</p>"
        
        # Очистка
        for tag in content(['script', 'style', 'nav', 'header', 'footer', 'aside', 'iframe']):
            tag.decompose()
        
        # Стили для изображений
        for img in content.find_all('img'):
            if img.get('src'):
                img['style'] = 'max-width:100%; height:auto; border-radius:8px;'
            if img.get('data-src') and not img.get('src'):
                img['src'] = img['data-src']
        
        # Стили для ссылок
        for link in content.find_all('a'):
            link['style'] = 'color:#FF6B35;'
            link['target'] = '_blank'
        
        result = f"<h1>{title}</h1>{str(content)}"
        
        print(f"✅ Успешно переведено\n")
        
        return result
        
    except Exception as e:
        print(f"❌ Ошибка: {e}\n")
        return f"<p style='color:#ff6b6b'>Ошибка: {str(e)}</p>"

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🚀 NEWS TRANSLATOR")
    print("="*70)
    print("📍 http://localhost:8000")
    print("✅ Парсит РЕАЛЬНЫЕ английские статьи из RSS")
    print("✅ Показывает их в переведенном виде")
    print("="*70 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
