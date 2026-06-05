# Made And Checked By DELTA SYNTH & Gemini AI
# Original Developers: RVC Project

import ast
import glob
import json
import os
from collections import OrderedDict

def extract_i18n_strings(node):
    i18n_strings = []

    # ตรวจสอบว่าโหนดนี้คือการเรียกใช้ฟังก์ชัน i18n() หรือไม่
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "i18n"
    ):
        for arg in node.args:
            # รองรับทั้ง Python เวอร์ชั่นเก่าและใหม่ (ast.Str ถูกยกเลิกในรุ่นใหม่ๆ และใช้ ast.Constant แทน)
            if hasattr(ast, 'Constant') and isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                i18n_strings.append(arg.value)
            elif hasattr(ast, 'Str') and isinstance(arg, ast.Str):
                i18n_strings.append(arg.s)

    # วนลูปตรวจสอบโหนดลูกต่อไปเรื่อยๆ
    for child_node in ast.iter_child_nodes(node):
        i18n_strings.extend(extract_i18n_strings(child_node))

    return i18n_strings

print("=====================================================")
print("ระบบสแกนและดึงข้อมูลคีย์ภาษา (AST Scanner) - DELTA SYNTH")
print("=====================================================")

# โฟลเดอร์ที่ต้องการเพิกเฉย เพื่อให้สแกนได้เร็วขึ้น
ignore_dirs = {"runtime", "venv", "env", ".git", "__pycache__"}

strings = []
print("กำลังสแกนไฟล์ .py ในโปรเจกต์...")

for filename in glob.iglob("**/*.py", recursive=True):
    # ข้ามโฟลเดอร์ระบบที่ไม่จำเป็น
    if any(ignored in filename for ignored in ignore_dirs):
        continue

    try:
        with open(filename, "r", encoding="utf-8") as f:
            code = f.read()
            # สแกนเฉพาะไฟล์ที่มีการเรียกใช้คลาสหรือฟังก์ชันภาษาเพื่อประหยัดเวลา
            if "i18n" in code or "I18nAuto" in code:
                tree = ast.parse(code)
                i18n_strings = extract_i18n_strings(tree)
                if i18n_strings:
                    print(f"พบในไฟล์: {filename} -> {len(i18n_strings)} คีย์")
                strings.extend(i18n_strings)
    except Exception as e:
        print(f"[ข้ามไฟล์] ไม่สามารถอ่าน {filename} ได้: {e}")

code_keys = set(strings)
print(f"\nรวมคีย์ที่ไม่ซ้ำกันทั้งหมด: {len(code_keys)} คีย์")

# กำหนดไฟล์ต้นแบบ
standard_file = "i18n/locale/zh_CN.json"

if not os.path.exists(standard_file):
    print(f"\n[แจ้งเตือน] ไม่พบไฟล์ต้นแบบที่: {standard_file}")
    print("ระบบจะสร้างไฟล์ใหม่ให้โดยอัตโนมัติ")
    standard_data = OrderedDict()
else:
    with open(standard_file, "r", encoding="utf-8") as f:
        standard_data = json.load(f, object_pairs_hook=OrderedDict)

standard_keys = set(standard_data.keys())

# หาคีย์ที่ไม่ได้ใช้งานแล้ว (ลบทิ้งในอนาคตได้)
unused_keys = standard_keys - code_keys
print(f"\nคีย์ที่ไม่ได้ใช้งานแล้ว (Unused keys): {len(unused_keys)} คีย์")
for unused_key in unused_keys:
    print(f"   - {unused_key}")

# หาคีย์ที่ต้องเพิ่มเข้าไปใหม่ (ตกหล่น)
missing_keys = code_keys - standard_keys
print(f"\nคีย์ที่ตกหล่นและต้องเพิ่มใหม่ (Missing keys): {len(missing_keys)} คีย์")
for missing_key in missing_keys:
    print(f"   + {missing_key}")

# อัปเดตข้อมูล: รักษาคำแปลเดิม เพิ่มคำใหม่ และลบคำที่ไม่ได้ใช้
updated_data = OrderedDict()

# รักษากระบวนการจัดเรียงตามลำดับที่พบในโค้ด
for s in strings:
    if s not in updated_data:
        # ถ้ามีคำแปลเดิมอยู่แล้ว ให้ใช้คำแปลเดิม ถ้าไม่มีให้ใช้ตัวคีย์เป็นคำแปลชั่วคราว
        updated_data[s] = standard_data.get(s, s)

# บันทึกกลับลงไฟล์ (ปิด sort_keys เพื่อรักษาโครงสร้างของ OrderedDict)
os.makedirs(os.path.dirname(standard_file), exist_ok=True)
with open(standard_file, "w", encoding="utf-8") as f:
    json.dump(updated_data, f, ensure_ascii=False, indent=4, sort_keys=False)
    f.write("\n")

print("\n=====================================================")
print("อัปเดตไฟล์ต้นแบบสำเร็จเรียบร้อยแล้ว!")
print("=====================================================")