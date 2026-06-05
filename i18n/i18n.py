# Made And Checked By DELTA SYNTH & Gemini AI
# Original Developers: RVC Project

import json
import locale
import os
import sys

def load_language_list(language: str) -> dict:
    """ฟังก์ชันสำหรับโหลดไฟล์ภาษาจากแฟ้ม i18n/locale"""
    try:
        with open(f"./i18n/locale/{language}.json", "r", encoding="utf-8") as f:
            language_list = json.load(f)
        return language_list
    except FileNotFoundError:
        return {}

class I18nAuto:
    def __init__(self, language=None):
        # ตรวจสอบภาษาและตั้งค่าเริ่มต้นเป็นภาษาไทยทันที
        if language in ["Auto", None]:
            try:
                sys_lang = locale.getlocale()[0]
                # หากตรวจพบภาษาไทยในระบบ หรือเป็นค่าว่าง จะบังคับใช้ภาษาไทย
                if sys_lang == "th_TH" or not sys_lang:
                    language = "th_TH"
                else:
                    language = "th_TH" # บังคับเปลี่ยนเป็น UI ภาษาไทยให้โดยอัตโนมัติ ทันที
            except Exception:
                language = "th_TH" 

        # ตรวจสอบว่าไฟล์ภาษาไทยมีอยู่จริงหรือไม่ หากไม่มีให้ใช้ภาษาอังกฤษเพื่อป้องกันโปรแกรมล่ม
        if not os.path.exists(f"./i18n/locale/{language}.json"):
            language = "en_US"

        self.language = language
        self.language_map = load_language_list(language)

    def __call__(self, key: str) -> str:
        # ดึงคำแปลจากคีย์ หากไม่พบให้คืนค่าเป็นคีย์เดิม
        return self.language_map.get(key, key)

    def __repr__(self) -> str:
        return f"Use Language: {self.language}"