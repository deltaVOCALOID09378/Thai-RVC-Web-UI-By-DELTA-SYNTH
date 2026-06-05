# Made And Checked By DELTA SYNTH & Gemini AI | Original by UVR5/RVC Project
# Version: 1.3

import os
import traceback
import logging
import subprocess

import ffmpeg
import torch

from configs.config import Config
from infer.modules.uvr5.mdxnet import MDXNetDereverb
from infer.modules.uvr5.vr import AudioPre, AudioPreDeEcho

logger = logging.getLogger(__name__)
config = Config()

def uvr(model_name, inp_root, save_root_vocal, paths, save_root_ins, agg, format0):
    """
    ฟังก์ชันหลักสำหรับสั่งการโมเดล UVR5 เพื่อแยกเสียงร้องและเสียงดนตรี
    รองรับการส่งค่ากลับ (Yield) เพื่อแสดงสถานะการทำงานบนหน้าต่าง UI แบบเรียลไทม์
    """
    infos = []
    pre_fun = None  # กำหนดค่าเริ่มต้นเป็น None ป้องกัน Error ตอนล้างหน่วยความจำ (finally)
    
    try:
        # 1. ทำความสะอาดข้อความ Path ลบช่องว่างและอักขระขยะ
        inp_root = inp_root.strip(' "\n')
        save_root_vocal = save_root_vocal.strip(' "\n')
        save_root_ins = save_root_ins.strip(' "\n')
        
        # 2. ตรวจสอบและโหลดโมเดลตามชื่อที่ระบุ
        if model_name == "onnx_dereverb_By_FoxJoy":
            pre_fun = MDXNetDereverb(15, config.device)
        else:
            func = AudioPre if "DeEcho" not in model_name else AudioPreDeEcho
            model_path = os.path.join(os.getenv("weight_uvr5_root", ""), f"{model_name}.pth")
            pre_fun = func(
                agg=int(agg),
                model_path=model_path,
                device=config.device,
                is_half=config.is_half,
            )
            
        # ตรวจสอบว่าเป็นโมเดล HP3 (ที่ร้องและดนตรีสลับกัน) หรือไม่
        is_hp3 = "HP3" in model_name
        
        # 3. รวบรวมรายชื่อไฟล์ที่ต้องการประมวลผล
        if inp_root != "":
            # อ่านจากโฟลเดอร์โดยตรง
            file_paths = [os.path.join(inp_root, name) for name in os.listdir(inp_root) if os.path.isfile(os.path.join(inp_root, name))]
        else:
            # อ่านจาก List ไฟล์ที่ส่งมาจาก UI
            file_paths = [path.name for path in paths] if paths else []

        # 4. ลูปประมวลผลทีละไฟล์
        for inp_path in file_paths:
            need_reformat = True
            file_basename = os.path.basename(inp_path)
            
            # ตรวจสอบความสมบูรณ์และรูปแบบของไฟล์เสียงด้วย ffprobe
            try:
                info = ffmpeg.probe(inp_path, cmd="ffprobe")
                if (
                    info["streams"][0]["channels"] == 2
                    and str(info["streams"][0]["sample_rate"]) == "44100"
                ):
                    need_reformat = False
            except Exception as e:
                logger.warning(f"ไม่สามารถตรวจสอบไฟล์ {file_basename} ได้ ระบบจะบังคับแปลงไฟล์เพื่อความปลอดภัย: {e}")
                need_reformat = True

            # 5. แปลงไฟล์เป็น 44100Hz Stereo หากรูปแบบไม่ถูกต้อง
            if need_reformat:
                tmp_path = os.path.join(os.environ.get("TEMP", "/tmp"), f"{file_basename}.reformatted.wav")
                try:
                    # ใช้ subprocess แทน os.system เพื่อความเสถียร
                    subprocess.run(
                        ['ffmpeg', '-i', inp_path, '-vn', '-acodec', 'pcm_s16le', '-ac', '2', '-ar', '44100', tmp_path, '-y'],
                        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                    )
                    inp_path = tmp_path
                except subprocess.CalledProcessError:
                    infos.append(f"{file_basename} -> ล้มเหลว (เกิดข้อผิดพลาดในการแปลงไฟล์เสียง)")
                    yield "\n".join(infos)
                    continue  # ข้ามไปทำไฟล์ต่อไป

            # 6. ส่งไฟล์เข้าสู่กระบวนการแยกเสียงด้วย AI
            try:
                # แก้ไขบั๊กของเก่า โดยส่ง is_hp3 เข้าไปเสมอไม่ว่าจะผ่านการ Reformat หรือไม่
                pre_fun._path_audio_(inp_path, save_root_ins, save_root_vocal, format0, is_hp3=is_hp3)
                infos.append(f"{file_basename} -> สำเร็จเรียบร้อย")
                yield "\n".join(infos)
            except Exception:
                infos.append(f"{file_basename} -> ล้มเหลวระหว่างประมวลผล: {traceback.format_exc()}")
                yield "\n".join(infos)

    except Exception:
        infos.append(f"เกิดข้อผิดพลาดร้ายแรงในระบบหลัก: {traceback.format_exc()}")
        yield "\n".join(infos)
        
    finally:
        # 7. คืนค่าหน่วยความจำ (VRAM) หลังจากประมวลผลเสร็จสิ้น
        try:
            if pre_fun is not None:
                if model_name == "onnx_dereverb_By_FoxJoy":
                    del pre_fun.pred.model
                    del pre_fun.pred.model_
                else:
                    del pre_fun.model
                del pre_fun
        except Exception:
            logger.error(f"เกิดข้อผิดพลาดระหว่างการล้างโมเดลออกจากหน่วยความจำ: {traceback.format_exc()}")
            
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("ระบบได้เคลียร์หน่วยความจำ GPU (torch.cuda.empty_cache) เรียบร้อยแล้ว")
            
    yield "\n".join(infos)