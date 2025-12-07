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
    Проходит по HTML, переводит текст, но оставляет теги (картинки, жирный шрифт)
    """
    # Теги, которые мы хотим оставить
    allowed_tags = ['p', 'h1', 'h2', 'h3', 'h4', 'li', 'ul', 'ol', 'img', 'figure', 'figcaption', 'strong', 'b', 'blockquote']
    
    translated_html = ""

    # Ищем основной контент
    content = soup.find('article') or soup.find('main') or soup.find('div', class_='content') or soup.body
    
    if not content:
        return "<p>Не удалось найти контент :(</p>"

    # Пробегаем по всем элементам внутри контента
    for element in content.find_all(allowed_tags, recursive=True):
        # Если это картинка - просто добавляем её HTML
        if element.name == 'img':
            # Исправляем относительные ссылки (src="/img.jpg" -> src="site.com/img.jpg")
            if element.get('src') and not element['src'].startswith('http'):
                # (Тут можно доработать логику, пока оставим как есть или пропустим)
                continue 
            # Добавляем класс для красоты
            element['style'] = "max-width: 100%; height: auto; border-radius: 10px; margin: 20px 0;"
            translated_html += str(element)
        
        # Если это текст (заголовок, параграф)
        elif element.name in ['h1', 'h2', 'h3', 'p', 'li', 'figcaption']:
            text = element.get_text(strip=True)
            if len(text) > 3: # Переводим, если текст не мусор
                try:
                    # Переводим текст
                    ru_text = translator.translate(text)
                    # Заменяем текст внутри тега
                    element.string = ru_text
                    translated_html += str(element)
                except:
                    translated_html += str(element) # Если ошибка, оставляем оригинал

    return translated_html

@app.post("/translate")
async def translate_article(request: LinkRequest):
    start_time = time.time()
    
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(request.url, headers=headers, timeout=20)
        
        soup = BeautifulSoup(response.text, 'html.parser')
        translator = GoogleTranslator(source='auto', target='ru')

        # Основная магия: переводим HTML
        final_html = translate_html_content(soup, translator)
        
        duration = round(time.time() - start_time, 2)
        
        return {
            "translated_html": final_html,
            "stats": f"Готово за {duration} сек."
        }
        
    except Exception as e:
        return {"error": f"Ошибка: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
