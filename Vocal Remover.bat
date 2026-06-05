@echo off
chcp 65001 >nul
title DELTA SYNTH AI - UVR5 Core Processor V1.3
color 0D

echo ===================================================
echo   Made And Checked By DELTA SYNTH ^& Gemini AI
echo   ระบบศูนย์กลางการแยกเสียง Vocal / Instrumental
echo ===================================================
echo.

if "%~1"=="" (
    echo [คำแนะนำ] การใช้งานแบบรวดเร็ว
    echo ให้ลากโฟลเดอร์ที่มีไฟล์เพลง หรือลากไฟล์เพลง มาวางทับที่ไฟล์ .bat นี้นะครับ
    echo.
    echo [ระบบ] กำลังเปิดใช้งานโหมดปกติ (WebUI)...
    python infer_web.py
    pause
    exit /b
)

echo [ข้อมูล] ได้รับเป้าหมาย: "%~1"
echo [ระบบ] กำลังเข้าสู่โหมดประมวลผลอัตโนมัติ กรุณารอสักครู่...
echo.

:: รันคำสั่งเชื่อมต่อไปยัง CLI (หากโปรเจกต์มีการรองรับ CLI)
python tools/uvr5_cli.py --input_path "%~1"

echo.
echo [เสร็จสิ้น] ระบบทำการแยกเสียงเรียบร้อยแล้วครับ!
pause