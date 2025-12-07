from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import requests
from bs4 import BeautifulSoup, NavigableString
from deep_translator import GoogleTranslator
import time

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

def translate_html_content(soup, translator):
    """
    Умный переводчик с защитой кода (Code Guard).
    """
    # 1. Находим основной контент
    content = soup.find('article') or soup.find('main') or soup.find('div', class_='content') or soup.body
    
    if not content:
        return "<p>Не удалось найти контент :(</p>"

    # 2. Чистка мусора
    for junk in content(["script", "style", "iframe", "noscript", "svg"]):
        junk.decompose()

    # 3. CODE GUARD (Защита кода)
    # Находим все блоки с кодом и помечаем их специальным атрибутом, чтобы не трогать
    code_blocks = content.find_all(['pre', 'code', 'kbd', 'samp', 'var'])
    for block in code_blocks:
        block['data-no-translate'] = 'true'
        # Дополнительно: часто код лежит в div с классом highlight или gist
    
    # 4. Перевод текста
    # Ищем все текстовые узлы
    text_nodes = content.find_all(text=True)

    for node in text_nodes:
        # Условия пропуска (SKIP):
        # 1. Пустой текст
        if len(node.strip()) < 3: continue
        
        # 2. Родитель - это тег кода?
        parent = node.parent
        if parent.name in ['pre', 'code', 'kbd', 'samp', 'script', 'style']:
            continue
        
        # 3. У родителя есть класс, похожий на код? (highlight, gist, syntax)
        parent_classes = parent.get('class', [])
        if any('code' in cls or 'highlight' in cls for cls in parent_classes):
            continue

        # 4. Если помечено нами как 'data-no-translate'
        if parent.get('data-no-translate'):
            continue

        # Если проверки пройдены - ПЕРЕВОДИМ
        original_text = str(node)
        try:
            # Маленький хак: если текст начинается с def, class, import - скорее всего это кусок кода без тегов
            if any(original_text.strip().startswith(kw) for kw in ['def ', 'class ', 'import ', 'from ', 'return ']):
                 continue

            translated_text = translator.translate(original_text)
            node.replace_with(translated_text)
        except:
            pass

    # 5. Обработка картинок
    for img in content.find_all('img'):
        img['style'] = "max-width: 100%; height: auto; border-radius: 12px; margin: 20px 0; display: block;"
        if not img.get('src') and img.get('data-src'):
            img['src'] = img['data-src']

    return content.prettify()

@app.post("/translate")
async def translate_article(request: LinkRequest):
    start_time = time.time()
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/100.0.0.0 Safari/537.36'}
        response = requests.get(request.url, headers=headers, timeout=25)
        soup = BeautifulSoup(response.text, 'html.parser')
        translator = GoogleTranslator(source='auto', target='ru')

        final_html = translate_html_content(soup, translator)
        
        return {
            "translated_html": final_html,
            "stats": f"Готово за {round(time.time() - start_time, 2)} сек."
        }
    except Exception as e:
        return {"error": f"Ошибка: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
