import os
import torch

# ==========================================
# 1. การตั้งค่าไดเรกทอรีหลัก (Directory Configuration)
# ==========================================
# กำหนดเส้นทางพื้นฐานสำหรับไฟล์เสียงต้นฉบับและไฟล์เสียงที่ส่งออก
inp_root = os.getenv("inp_root", r"H:\Export to UTAU")
opt_root = os.getenv("opt_root", r"H:\Render file to UTAU")

# ==========================================
# 2. การตั้งค่าคุณลักษณะเสียง (Voice Characteristics)
# ==========================================
f0_up_key = int(os.getenv("f0_up_key", 0))  # ค่าการปรับระดับเสียง (Pitch)
person = os.getenv("person", "A")           # ชื่อหรือหมายเลขเป้าหมายของผู้พูด

# ==========================================
# 3. ระบบบริหารจัดการฮาร์ดแวร์ (Hardware Management)
# ==========================================
# ระบบจะทำการตรวจสอบและเลือกใช้อุปกรณ์ประมวลผลที่เหมาะสมที่สุดโดยอัตโนมัติ
if torch.cuda.is_available():
    device = "cuda:0"
    is_half = True  # เปิดใช้งานความแม่นยำแบบครึ่ง (FP16) เพื่อประหยัด VRAM ของการ์ดจอ
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
    is_half = False
else:
    device = "cpu"
    is_half = False # หน่วยประมวลผลกลางมักจะทำงานได้เสถียรกว่าในโหมดความแม่นยำเต็ม (FP32)