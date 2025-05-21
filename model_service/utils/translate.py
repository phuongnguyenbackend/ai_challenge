import requests
import os
import langid
import textwrap
from pydantic import BaseModel
from google import genai
from dotenv import load_dotenv


load_dotenv()

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

GOOGLE_API_KEY=os.getenv('GOOGLE_API_KEY')

class TranslationRequest(BaseModel):
    text: str
    src_lang: str
    tgt_lang: str
    
def detect_language(text):
    lang, _ = langid.classify(text)
    return lang
 
class Translate:
    def __init__(self, text):
        self.text = text.strip()
        self.client = genai.Client(api_key=GEMINI_API_KEY)


    def gemini_correct(self):
        prompt = textwrap.dedent(f"""
        [ROLE]
        You are a professional multilingual proofreader.

        [INSTRUCTIONS]
        - Correct spelling, grammar, and word usage in the given text.
        - Maintain the original meaning, tone, and structure.
        - Do NOT rewrite or rephrase unnecessarily.
        - Support all languages (e.g., English, Vietnamese, Spanish, Japanese, etc.)
        - Return ONLY the corrected version of the text, no explanation.

        [TEXT]
        {self.text}
        """)
        
        response = self.client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )

        return response.candidates[0].content.parts[0].text.strip()


    def translate(self, src_lang, tgt_lang):
        url = "https://translation.googleapis.com/language/translate/v2"
        data = {
            "q": self.text,
            "source": src_lang,
            "target": tgt_lang,
            "key": GOOGLE_API_KEY
        }

        response = requests.post(url, params=data)

        if response.status_code == 200:
            return response.json()["data"]["translations"][0]["translatedText"]

        else:
            raise Exception(response.text)

