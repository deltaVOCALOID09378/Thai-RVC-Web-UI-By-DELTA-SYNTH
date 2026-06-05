# Made And Checked By DELTA SYNTH & Gemini AI | Original by UVR5/RVC Project
# Version: 1.5

import os
import logging
import subprocess
import librosa
import numpy as np
import soundfile as sf
import torch

from infer.lib.uvr5_pack.lib_v5 import nets_61968KB as Nets
from infer.lib.uvr5_pack.lib_v5 import spec_utils
from infer.lib.uvr5_pack.lib_v5.model_param_init import ModelParameters
from infer.lib.uvr5_pack.lib_v5.nets_new import CascadedNet
from infer.lib.uvr5_pack.utils import inference

logger = logging.getLogger(__name__)

class AudioPre:
    """
    คลาสสำหรับประมวลผลและแยกเสียงร้อง (Vocal) ออกจากเสียงดนตรี (Instrumental)
    """
    def __init__(self, agg, model_path, device, is_half, tta=False):
        self.model_path = model_path
        self.device = device
        self.data = {
            "postprocess": False,
            "tta": tta,
            "window_size": 512,
            "agg": agg,
            "high_end_process": "mirroring",
        }
        self.mp = ModelParameters("infer/lib/uvr5_pack/lib_v5/modelparams/4band_v2.json")
        model = Nets.CascadedASPPNet(self.mp.param["bins"] * 2)
        cpk = torch.load(model_path, map_location="cpu")
        model.load_state_dict(cpk)
        model.eval()
        
        if is_half:
            model = model.half().to(device)
        else:
            model = model.to(device)

        self.model = model

    def _auto_clean_vocal(self, wav_data):
        """
        ฟังก์ชันสำหรับทำความสะอาดเสียงร้องให้ดีที่สุดโดยอัตโนมัติ
        (Normalize ให้เสียงพุ่งชัดเจน และทำ Noise Gate ตัดเสียงซ่าที่เบามากๆ ทิ้ง)
        """
        # 1. Normalization
        max_amp = np.max(np.abs(wav_data))
        if max_amp > 0:
            wav_data = (wav_data / max_amp) * 0.98  # ปรับความดังสูงสุดและเผื่อ Headroom 2% ป้องกันเสียงแตก
            
        # 2. Soft Noise Gate (ตัดสัญญาณที่เบากว่า 0.1% ทิ้งเพื่อความเงียบสงัด)
        noise_threshold = 0.001
        wav_data[np.abs(wav_data) < noise_threshold] = 0.0
        
        return wav_data

    def _save_audio_file(self, path, audio_data, sample_rate, target_format):
        sf.write(path, (np.array(audio_data) * 32768).astype("int16"), sample_rate)
        if target_format not in ["wav", "flac"] and os.path.exists(path):
            opt_format_path = path[:-4] + f".{target_format}"
            try:
                subprocess.run(
                    ['ffmpeg', '-i', path, '-vn', opt_format_path, '-q:a', '2', '-y'],
                    check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
                if os.path.exists(opt_format_path):
                    os.remove(path)
            except Exception as e:
                logger.error(f"เกิดข้อผิดพลาดระหว่างการแปลงไฟล์ด้วย FFmpeg: {e}")

    def _path_audio_(self, music_file, ins_root=None, vocal_root=None, format="flac", is_hp3=False):
        if ins_root is None and vocal_root is None:
            return "ไม่พบโฟลเดอร์ปลายทางสำหรับการบันทึกไฟล์"
            
        name = os.path.basename(music_file)
        if ins_root is not None:
            os.makedirs(ins_root, exist_ok=True)
        if vocal_root is not None:
            os.makedirs(vocal_root, exist_ok=True)
            
        logger.info(f"[ความคืบหน้า 1/5] เริ่มต้นอ่านไฟล์ {name} และเตรียมย่านความถี่เสียง...")
        X_wave, X_spec_s = {}, {}
        bands_n = len(self.mp.param["band"])

        for d in range(bands_n, 0, -1):
            bp = self.mp.param["band"][d]
            if d == bands_n:
                X_wave[d], _ = librosa.load(music_file, sr=bp["sr"], mono=False, dtype=np.float32, res_type=bp["res_type"])
                if X_wave[d].ndim == 1:
                    X_wave[d] = np.asfortranarray([X_wave[d], X_wave[d]])
            else:
                X_wave[d] = librosa.resample(X_wave[d + 1], orig_sr=self.mp.param["band"][d + 1]["sr"], target_sr=bp["sr"], res_type=bp["res_type"])
                
            X_spec_s[d] = spec_utils.wave_to_spectrogram_mt(
                X_wave[d], bp["hl"], bp["n_fft"], self.mp.param["mid_side"],
                self.mp.param["mid_side_b2"], self.mp.param["reverse"],
            )

            if d == bands_n and self.data["high_end_process"] != "none":
                input_high_end_h = (bp["n_fft"] // 2 - bp["crop_stop"]) + (self.mp.param["pre_filter_stop"] - self.mp.param["pre_filter_start"])
                input_high_end = X_spec_s[d][:, bp["n_fft"] // 2 - input_high_end_h : bp["n_fft"] // 2, :]

        logger.info("[ความคืบหน้า 2/5] แปลงสัญญาณเป็น Spectrogram และรวมย่านความถี่เสร็จสิ้น...")
        X_spec_m = spec_utils.combine_spectrograms(X_spec_s, self.mp)
        aggressiveness = {"value": float(self.data["agg"] / 100), "split_bin": self.mp.param["band"][1]["crop_stop"]}
        
        logger.info("[ความคืบหน้า 3/5] กำลังให้ AI ประมวลผลแยกเลเยอร์เสียงร้องและดนตรี (Inference)...")
        with torch.no_grad():
            pred, X_mag, X_phase = inference(X_spec_m, self.device, self.model, aggressiveness, self.data)
            
        logger.info("[ความคืบหน้า 4/5] ปรับสมดุลคุณภาพเสียงหลังการประมวลผล (Post-processing)...")
        if self.data["postprocess"]:
            pred_inv = np.clip(X_mag - pred, 0, np.inf)
            pred = spec_utils.mask_silence(pred, pred_inv)
            
        y_spec_m = pred * X_phase
        v_spec_m = X_spec_m - y_spec_m

        logger.info("[ความคืบหน้า 5/5] กำลังทำความสะอาดเสียง (Auto-Clean) และบันทึกไฟล์...")
        
        if ins_root is not None:
            if self.data["high_end_process"].startswith("mirroring"):
                input_high_end_ = spec_utils.mirroring(self.data["high_end_process"], y_spec_m, input_high_end, self.mp)
                wav_instrument = spec_utils.cmb_spectrogram_to_wave(y_spec_m, self.mp, input_high_end_h, input_high_end_)
            else:
                wav_instrument = spec_utils.cmb_spectrogram_to_wave(y_spec_m, self.mp)
                
            head = "vocal_" if is_hp3 else "instrument_"
            ext = format if format in ["wav", "flac"] else "wav"
            save_path = os.path.join(ins_root, f"{head}{name}_{self.data['agg']}.{ext}")
            self._save_audio_file(save_path, wav_instrument, self.mp.param["sr"], format)

        if vocal_root is not None:
            if self.data["high_end_process"].startswith("mirroring"):
                input_high_end_ = spec_utils.mirroring(self.data["high_end_process"], v_spec_m, input_high_end, self.mp)
                wav_vocals = spec_utils.cmb_spectrogram_to_wave(v_spec_m, self.mp, input_high_end_h, input_high_end_)
            else:
                wav_vocals = spec_utils.cmb_spectrogram_to_wave(v_spec_m, self.mp)
                
            # เรียกใช้ฟังก์ชันทำความสะอาดเสียงร้องอัตโนมัติ
            wav_vocals = self._auto_clean_vocal(wav_vocals)
                
            head = "instrument_" if is_hp3 else "vocal_"
            ext = format if format in ["wav", "flac"] else "wav"
            # ถึงแม้ไม่สามารถแยกรายบุคคลได้ แต่สามารถตั้งชื่อกำกับให้ดูเป็นระเบียบได้
            save_path = os.path.join(vocal_root, f"{head}All_Singers_{name}_{self.data['agg']}.{ext}")
            self._save_audio_file(save_path, wav_vocals, self.mp.param["sr"], format)
            
        logger.info(f"==> ประมวลผลไฟล์ {name} เสร็จสมบูรณ์! <==")


class AudioPreDeEcho:
    """
    คลาสสำหรับประมวลผลการลบเสียงสะท้อน (De-Echo / De-Reverb)
    """
    def __init__(self, agg, model_path, device, is_half, tta=False):
        self.model_path = model_path
        self.device = device
        self.data = {
            "postprocess": False,
            "tta": tta,
            "window_size": 512,
            "agg": agg,
            "high_end_process": "mirroring",
        }
        self.mp = ModelParameters("infer/lib/uvr5_pack/lib_v5/modelparams/4band_v3.json")
        nout = 64 if "DeReverb" in model_path else 48
        model = CascadedNet(self.mp.param["bins"] * 2, nout)
        cpk = torch.load(model_path, map_location="cpu")
        model.load_state_dict(cpk)
        model.eval()
        
        if is_half:
            model = model.half().to(device)
        else:
            model = model.to(device)

        self.model = model

    def _auto_clean_vocal(self, wav_data):
        """ฟังก์ชันทำความสะอาดเสียง (ใช้งานเหมือนคลาสบน)"""
        max_amp = np.max(np.abs(wav_data))
        if max_amp > 0:
            wav_data = (wav_data / max_amp) * 0.98
        noise_threshold = 0.001
        wav_data[np.abs(wav_data) < noise_threshold] = 0.0
        return wav_data

    def _save_audio_file(self, path, audio_data, sample_rate, target_format):
        sf.write(path, (np.array(audio_data) * 32768).astype("int16"), sample_rate)
        if target_format not in ["wav", "flac"] and os.path.exists(path):
            opt_format_path = path[:-4] + f".{target_format}"
            try:
                subprocess.run(
                    ['ffmpeg', '-i', path, '-vn', opt_format_path, '-q:a', '2', '-y'],
                    check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
                if os.path.exists(opt_format_path):
                    os.remove(path)
            except Exception as e:
                logger.error(f"เกิดข้อผิดพลาดระหว่างการแปลงไฟล์ด้วย FFmpeg: {e}")

    def _path_audio_(self, music_file, vocal_root=None, ins_root=None, format="flac", is_hp3=False):
        if ins_root is None and vocal_root is None:
            return "ไม่พบโฟลเดอร์ปลายทางสำหรับการบันทึกไฟล์"
            
        name = os.path.basename(music_file)
        if ins_root is not None:
            os.makedirs(ins_root, exist_ok=True)
        if vocal_root is not None:
            os.makedirs(vocal_root, exist_ok=True)
            
        logger.info(f"[ความคืบหน้า 1/5] เริ่มโหลดไฟล์ {name} สำหรับการลบเสียงสะท้อน...")
        X_wave, X_spec_s = {}, {}
        bands_n = len(self.mp.param["band"])

        for d in range(bands_n, 0, -1):
            bp = self.mp.param["band"][d]
            if d == bands_n:
                X_wave[d], _ = librosa.load(music_file, sr=bp["sr"], mono=False, dtype=np.float32, res_type=bp["res_type"])
                if X_wave[d].ndim == 1:
                    X_wave[d] = np.asfortranarray([X_wave[d], X_wave[d]])
            else:
                X_wave[d] = librosa.resample(X_wave[d + 1], orig_sr=self.mp.param["band"][d + 1]["sr"], target_sr=bp["sr"], res_type=bp["res_type"])
                
            X_spec_s[d] = spec_utils.wave_to_spectrogram_mt(
                X_wave[d], bp["hl"], bp["n_fft"], self.mp.param["mid_side"],
                self.mp.param["mid_side_b2"], self.mp.param["reverse"],
            )

            if d == bands_n and self.data["high_end_process"] != "none":
                input_high_end_h = (bp["n_fft"] // 2 - bp["crop_stop"]) + (self.mp.param["pre_filter_stop"] - self.mp.param["pre_filter_start"])
                input_high_end = X_spec_s[d][:, bp["n_fft"] // 2 - input_high_end_h : bp["n_fft"] // 2, :]

        logger.info("[ความคืบหน้า 2/5] วิเคราะห์ Spectrogram สำหรับเสียงสะท้อนเสร็จสิ้น...")
        X_spec_m = spec_utils.combine_spectrograms(X_spec_s, self.mp)
        aggressiveness = {"value": float(self.data["agg"] / 100), "split_bin": self.mp.param["band"][1]["crop_stop"]}
        
        logger.info("[ความคืบหน้า 3/5] โมเดลกำลังคำนวณและหักล้าง Reverb (Inference)...")
        with torch.no_grad():
            pred, X_mag, X_phase = inference(X_spec_m, self.device, self.model, aggressiveness, self.data)
            
        logger.info("[ความคืบหน้า 4/5] ประมวลผลคลื่นเสียงกลับ (Inverse Post-processing)...")
        if self.data["postprocess"]:
            pred_inv = np.clip(X_mag - pred, 0, np.inf)
            pred = spec_utils.mask_silence(pred, pred_inv)
            
        y_spec_m = pred * X_phase
        v_spec_m = X_spec_m - y_spec_m

        logger.info("[ความคืบหน้า 5/5] ทำความสะอาดเสียงร้อง (Auto-Clean) และเขียนลงไฟล์...")
        
        if ins_root is not None:
            if self.data["high_end_process"].startswith("mirroring"):
                input_high_end_ = spec_utils.mirroring(self.data["high_end_process"], y_spec_m, input_high_end, self.mp)
                wav_instrument = spec_utils.cmb_spectrogram_to_wave(y_spec_m, self.mp, input_high_end_h, input_high_end_)
            else:
                wav_instrument = spec_utils.cmb_spectrogram_to_wave(y_spec_m, self.mp)
                
            wav_instrument = self._auto_clean_vocal(wav_instrument)
            ext = format if format in ["wav", "flac"] else "wav"
            save_path = os.path.join(ins_root, f"vocal_Cleaned_{name}_{self.data['agg']}.{ext}")
            self._save_audio_file(save_path, wav_instrument, self.mp.param["sr"], format)

        if vocal_root is not None:
            if self.data["high_end_process"].startswith("mirroring"):
                input_high_end_ = spec_utils.mirroring(self.data["high_end_process"], v_spec_m, input_high_end, self.mp)
                wav_vocals = spec_utils.cmb_spectrogram_to_wave(v_spec_m, self.mp, input_high_end_h, input_high_end_)
            else:
                wav_vocals = spec_utils.cmb_spectrogram_to_wave(v_spec_m, self.mp)
                
            ext = format if format in ["wav", "flac"] else "wav"
            save_path = os.path.join(vocal_root, f"instrument_Reverb_{name}_{self.data['agg']}.{ext}")
            self._save_audio_file(save_path, wav_vocals, self.mp.param["sr"], format)
            
        logger.info(f"==> ประมวลผล De-Echo ของไฟล์ {name} เสร็จสมบูรณ์! <==")