@echo off
title ระบบ AI Voice Conversion (RVC) โดย DELTA SYNTH
color 0A

:: บังคับให้เริ่มทำงานในโฟลเดอร์ปัจจุบันเสมอ (แก้ไขจุดที่พิมพ์ผิด)
cd /d "%~dp0"

echo =======================================================
echo    กำลังตรวจสอบและอัปเดตเอนจิน (pip) ให้เป็นเวอร์ชันล่าสุด...
echo =======================================================
:: ใช้ python จากโฟลเดอร์ runtime เพื่อให้ตรงกับสภาพแวดล้อมของ RVC
runtime\python.exe -m pip install --upgrade pip

echo =======================================================
echo    กำลังเริ่มต้นระบบ RVC WebUI กรุณารอสักครู่...
echo =======================================================

:: ตั้งค่า Environment Variable ป้องกัน Error จากบางไลบรารี
set USE_LIBUV=0

:: หากต้องการให้สคริปต์ Train รันพร้อมกับ WebUI ในหน้าต่างใหม่ ให้ใช้คำสั่ง start
:: ลบเครื่องหมาย :: ด้านล่างออก หากต้องการเปิดหน้าต่าง Train แยก
:: start "RVC Training" runtime\python.exe infer\modules\train\train.py

:: เปิด WebUI (สคริปต์จะทำงานค้างอยู่ที่บรรทัดนี้เพื่อเปิดเซิร์ฟเวอร์)
runtime\python.exe infer-web.py --pycmd runtime\python.exe --port 9378

echo.
echo ระบบทำงานเสร็จสิ้น หรือ WebUI ถูกปิดลง
pause