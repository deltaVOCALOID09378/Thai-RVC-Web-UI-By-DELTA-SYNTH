# Made And Checked By DELTA SYNTH & Gemini AI
# Original Owner: RVC Project
# Version: 1.4
# ชื่อไฟล์: extract_f0_print.py

import os
import sys
import traceback
import logging
from multiprocessing import Process
from typing import List, Tuple

import numpy as np
import parselmouth
import pyworld

# กำหนดเส้นทางโปรเจกต์
now_dir = os.getcwd()
sys.path.append(now_dir)

from infer.lib.audio import load_audio

# ปิดการแจ้งเตือนรบกวนจาก numba
logging.getLogger("numba").setLevel(logging.WARNING)

# ==========================================
# [ส่วนที่ 1: การรับค่าพารามิเตอร์จากระบบหลัก (System Arguments)]
# ==========================================
exp_dir: str = sys.argv[1]       # เส้นทางโฟลเดอร์ของการทดลอง (Experiment Directory)
n_p: int = int(sys.argv[2])      # จำนวนคอร์ CPU ที่ใช้ประมวลผล (Number of Processes)
f0method: str = sys.argv[3]      # วิธีการสกัดระดับเสียง (F0 Extraction Method)

# ==========================================
# [ส่วนที่ 2: ระบบบันทึกการทำงานแบบสองภาษา (Dual-Language Logging System)]
# ==========================================
def printt(strr: str) -> None:
    """
    ฟังก์ชันสำหรับแสดงผลลัพธ์บนหน้าจอและบันทึกลงไฟล์ Log อย่างปลอดภัย 
    รองรับการทำงานแบบ Multi-processing โดยไม่ทำให้ไฟล์ค้าง (File Lock)
    """
    print(strr)
    log_path = os.path.join(exp_dir, "extract_f0_feature.log")
    with open(log_path, "a+", encoding="utf-8") as f:
        f.write("%s\n" % strr)

# ==========================================
# [ส่วนที่ 3: คลาสหลักสำหรับการวิเคราะห์ระดับเสียง (F0 Feature Extraction)]
# ==========================================
class FeatureInput(object):
    def __init__(self, samplerate: int = 16000, hop_size: int = 160):
        self.fs: int = samplerate
        self.hop: int = hop_size
        self.f0_bin: int = 256
        self.f0_max: float = 1100.0
        self.f0_min: float = 50.0
        self.f0_mel_min: float = 1127 * np.log(1 + self.f0_min / 700)
        self.f0_mel_max: float = 1127 * np.log(1 + self.f0_max / 700)

    def compute_f0(self, path: str, f0_method: str) -> np.ndarray:
        """
        สกัดค่าความถี่ F0 ด้วยอัลกอริทึมที่ผู้ใช้เลือก (PM, Harvest, DIO, หรือ RMVPE)
        """
        x = load_audio(path, self.fs)
        p_len = x.shape[0] // self.hop
        
        if f0_method == "pm":
            time_step = 160 / 16000 * 1000
            f0 = (
                parselmouth.Sound(x, self.fs)
                .to_pitch_ac(
                    time_step=time_step / 1000,
                    voicing_threshold=0.6,
                    pitch_floor=self.f0_min,
                    pitch_ceiling=self.f0_max,
                )
                .selected_array["frequency"]
            )
            pad_size = (p_len - len(f0) + 1) // 2
            if pad_size > 0 or p_len - len(f0) - pad_size > 0:
                f0 = np.pad(f0, [[pad_size, p_len - len(f0) - pad_size]], mode="constant")
                
        elif f0_method == "harvest":
            f0, t = pyworld.harvest(
                x.astype(np.double),
                fs=self.fs,
                f0_ceil=self.f0_max,
                f0_floor=self.f0_min,
                frame_period=1000 * self.hop / self.fs,
            )
            f0 = pyworld.stonemask(x.astype(np.double), f0, t, self.fs)
            
        elif f0_method == "dio":
            f0, t = pyworld.dio(
                x.astype(np.double),
                fs=self.fs,
                f0_ceil=self.f0_max,
                f0_floor=self.f0_min,
                frame_period=1000 * self.hop / self.fs,
            )
            f0 = pyworld.stonemask(x.astype(np.double), f0, t, self.fs)
            
        elif f0_method == "rmvpe":
            if not hasattr(self, "model_rmvpe"):
                from infer.lib.rmvpe import RMVPE
                printt("⏳ กำลังโหลดโมเดล RMVPE บน CPU... (Loading RMVPE model on CPU...)")
                # บังคับใช้ CPU สำหรับ Multi-processing เพื่อป้องกัน GPU VRAM ทะลุ
                self.model_rmvpe = RMVPE("assets/rmvpe/rmvpe.pt", is_half=False, device="cpu")
            f0 = self.model_rmvpe.infer_from_audio(x, thred=0.03)
            
        return f0

    def coarse_f0(self, f0: np.ndarray) -> np.ndarray:
        """
        แปลงค่า F0 ให้เป็นสเกล Mel (1-255) สำหรับให้โมเดล AI นำไปฝึกสอน
        """
        f0_mel = 1127 * np.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - self.f0_mel_min) * (
            self.f0_bin - 2
        ) / (self.f0_mel_max - self.f0_mel_min) + 1

        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > self.f0_bin - 1] = self.f0_bin - 1
        f0_coarse = np.rint(f0_mel).astype(int)
        
        assert f0_coarse.max() <= 255 and f0_coarse.min() >= 1, (f0_coarse.max(), f0_coarse.min())
        return f0_coarse

    def go(self, paths: List[Tuple[str, str, str]], f0_method: str) -> None:
        """
        ระบบรันคิวแยกตาม Process เพื่อประมวลผลไฟล์เสียง
        """
        if len(paths) == 0:
            printt("✔️ ไม่มีไฟล์เสียงใหม่ที่ต้องประมวลผล (No F0 files to process)")
            return
            
        printt(f"🎯 ตรวจพบไฟล์ที่ต้องประมวลผล (Tasks): {len(paths)} ไฟล์")
        n = max(len(paths) // 5, 1)  # ลดความหนาแน่นของการแสดงผล (Print limit)
        
        for idx, (inp_path, opt_path1, opt_path2) in enumerate(paths):
            try:
                if idx % n == 0:
                    printt("▶ กำลังประมวลผล (Processing) [%s/%s] ไฟล์: %s" % (idx, len(paths), inp_path))
                
                # ข้ามไฟล์ที่เคยสกัดเสร็จแล้ว
                if os.path.exists(opt_path1 + ".npy") and os.path.exists(opt_path2 + ".npy"):
                    continue
                    
                featur_pit = self.compute_f0(inp_path, f0_method)
                np.save(opt_path2, featur_pit, allow_pickle=False)  # NSF
                
                coarse_pit = self.coarse_f0(featur_pit)
                np.save(opt_path1, coarse_pit, allow_pickle=False)  # Ori
                
            except Exception as e:
                printt("❌ ล้มเหลว (Failed) [%s]: %s\nรายละเอียด (Error): %s" % (idx, inp_path, traceback.format_exc()))

# ==========================================
# [ส่วนที่ 4: การกระจายงาน Multi-Processing]
# ==========================================
if __name__ == "__main__":
    printt(f"⚙️ ข้อมูลคำสั่งระบบ (System Arguments): {sys.argv}")
    featureInput = FeatureInput()
    paths = []
    
    inp_root = os.path.join(exp_dir, "1_16k_wavs")
    opt_root1 = os.path.join(exp_dir, "2a_f0")
    opt_root2 = os.path.join(exp_dir, "2b-f0nsf")

    os.makedirs(opt_root1, exist_ok=True)
    os.makedirs(opt_root2, exist_ok=True)
    
    for name in sorted(os.listdir(inp_root)):
        inp_path = os.path.join(inp_root, name)
        if "spec" in inp_path:
            continue
        opt_path1 = os.path.join(opt_root1, name)
        opt_path2 = os.path.join(opt_root2, name)
        paths.append((inp_path, opt_path1, opt_path2))

    # กระจายไฟล์ลงในแต่ละคอร์ CPU แบบอัตโนมัติ (Multiprocessing Pool Distribution)
    ps = []
    for i in range(n_p):
        p = Process(
            target=featureInput.go,
            args=(paths[i::n_p], f0method),
        )
        ps.append(p)
        p.start()
        
    # รอให้ทุกคอร์ทำงานเสร็จสิ้น
    for i in range(n_p):
        ps[i].join()
        
    printt("✨ กระบวนการสกัดค่าระดับเสียงด้วย CPU เสร็จสมบูรณ์! (CPU F0 Extraction Completed!)")