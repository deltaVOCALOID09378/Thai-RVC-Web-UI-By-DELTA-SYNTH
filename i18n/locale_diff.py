# Made And Checked By DELTA SYNTH & Gemini AI
# Original Developers: RVC Project

import json
import os
from collections import OrderedDict

def sync_locale_files():
    # กำหนดไฟล์ต้นแบบ
    dir_path = "locale/"
    standard_file_name = "zh_CN.json"
    standard_file = os.path.join(dir_path, standard_file_name)

    # ตรวจสอบว่ามีโฟลเดอร์และไฟล์ต้นแบบอยู่จริงหรือไม่
    if not os.path.exists(standard_file):
        print(f"[ข้อผิดพลาด] ไม่พบไฟล์ต้นแบบที่: {standard_file}")
        return

    # ค้นหาไฟล์ JSON ทั้งหมดในโฟลเดอร์ ยกเว้นไฟล์ต้นแบบ
    languages = [
        os.path.join(dir_path, f)
        for f in os.listdir(dir_path)
        if f.endswith(".json") and f != standard_file_name
    ]

    # โหลดไฟล์ต้นแบบและรักษาลำดับคีย์ไว้
    with open(standard_file, "r", encoding="utf-8") as f:
        standard_data = json.load(f, object_pairs_hook=OrderedDict)

    print(f"กำลังอ้างอิงโครงสร้างคีย์จาก: {standard_file_name}")
    print("-" * 50)

    # วนลูปเพื่อปรับสมดุลแต่ละไฟล์ภาษา
    for lang_file in languages:
        with open(lang_file, "r", encoding="utf-8") as f:
            lang_data = json.load(f, object_pairs_hook=OrderedDict)

        # หาจุดต่าง: คีย์ที่ขาด และ คีย์ที่เกิน
        missing_keys = set(standard_data.keys()) - set(lang_data.keys())
        extra_keys = set(lang_data.keys()) - set(standard_data.keys())

        # เพิ่มคีย์ที่ขาดหายไป
        for key in missing_keys:
            lang_data[key] = key

        # ลบคีย์ที่เกินมา
        for key in extra_keys:
            del lang_data[key]

        # แก้ไขบั๊กการเรียงลำดับ: จัดเรียงคีย์ใหม่ให้ตรงตามลำดับของไฟล์ต้นแบบ 100%
        # (วิธีนี้ประหยัดทรัพยากรเครื่องกว่าการใช้ lambda index)
        sorted_lang_data = OrderedDict()
        for key in standard_data.keys():
            if key in lang_data:
                sorted_lang_data[key] = lang_data[key]

        # บันทึกไฟล์ที่ถูกปรับปรุงแล้ว
        # หมายเหตุสำคัญ: ต้องใช้ sort_keys=False เพื่อไม่ให้ JSON ทำลายการจัดเรียงของ OrderedDict
        with open(lang_file, "w", encoding="utf-8") as f:
            json.dump(sorted_lang_data, f, ensure_ascii=False, indent=4, sort_keys=False)
            f.write("\n")
        
        file_name_only = os.path.basename(lang_file)
        print(f"ปรับสมดุลไฟล์ [{file_name_only}] สำเร็จ -> เพิ่มคีย์: {len(missing_keys)}, ลบคีย์เก่า: {len(extra_keys)}")

if __name__ == "__main__":
    print("=====================================================")
    print("ระบบจัดการและปรับสมดุลภาษา (Locale Sync) - DELTA SYNTH")
    print("=====================================================")
    try:
        sync_locale_files()
    except Exception as e:
        print(f"\n[เกิดข้อผิดพลาด]: {e}")
    print("=====================================================")
    print("กระบวนการปรับสมดุลไฟล์ภาษาเสร็จสิ้นสมบูรณ์")