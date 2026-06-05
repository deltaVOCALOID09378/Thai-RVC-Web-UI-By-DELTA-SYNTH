# Made And Checked By DELTA SYNTH & Gemini AI | Original by UVR5/RVC Project
# Version: 1.4

import os
import logging
import subprocess
import librosa
import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)
cpu = torch.device("cpu")


class ConvTDFNetTrim:
    """
    คลาสสถาปัตยกรรมเครือข่าย Conv-TDF สำหรับการแยกและประมวลผลสัญญาณเสียง
    """
    def __init__(self, device, model_name, target_name, L, dim_f, dim_t, n_fft, hop=1024):
        super(ConvTDFNetTrim, self).__init__()

        self.dim_f = dim_f
        self.dim_t = 2**dim_t
        self.n_fft = n_fft
        self.hop = hop
        self.n_bins = self.n_fft // 2 + 1
        self.chunk_size = hop * (self.dim_t - 1)
        self.window = torch.hann_window(window_length=self.n_fft, periodic=True).to(device)
        self.target_name = target_name
        self.blender = "blender" in model_name

        self.dim_c = 4
        out_c = self.dim_c * 4 if target_name == "*" else self.dim_c
        self.freq_pad = torch.zeros([1, out_c, self.n_bins - self.dim_f, self.dim_t]).to(device)
        self.n = L // 2

    def stft(self, x):
        """แปลงคลื่นเสียงเป็นสเปกโตรแกรม (STFT)"""
        x = x.reshape([-1, self.chunk_size])
        x = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop,
            window=self.window,
            center=True,
            return_complex=True,
        )
        x = torch.view_as_real(x)
        x = x.permute([0, 3, 1, 2])
        x = x.reshape([-1, 2, 2, self.n_bins, self.dim_t]).reshape(
            [-1, self.dim_c, self.n_bins, self.dim_t]
        )
        return x[:, :, : self.dim_f]

    def istft(self, x, freq_pad=None):
        """แปลงสเปกโตรแกรมกลับเป็นคลื่นเสียง (ISTFT)"""
        freq_pad = (
            self.freq_pad.repeat([x.shape[0], 1, 1, 1])
            if freq_pad is None
            else freq_pad
        )
        x = torch.cat([x, freq_pad], -2)
        c = 4 * 2 if self.target_name == "*" else 2
        x = x.reshape([-1, c, 2, self.n_bins, self.dim_t]).reshape(
            [-1, 2, self.n_bins, self.dim_t]
        )
        x = x.permute([0, 2, 3, 1])
        x = x.contiguous()
        x = torch.view_as_complex(x)
        x = torch.istft(
            x, n_fft=self.n_fft, hop_length=self.hop, window=self.window, center=True
        )
        return x.reshape([-1, c, self.chunk_size])


def get_models(device, dim_f, dim_t, n_fft):
    """ฟังก์ชันโหลดโมเดล Conv-TDF"""
    return ConvTDFNetTrim(
        device=device,
        model_name="Conv-TDF",
        target_name="vocals",
        L=11,
        dim_f=dim_f,
        dim_t=dim_t,
        n_fft=n_fft,
    )


class Predictor:
    """
    คลาสสำหรับรันการพยากรณ์และแยกเสียงสะท้อนผ่าน ONNX Runtime
    """
    def __init__(self, args):
        import onnxruntime as ort

        logger.info(f"ผู้ให้บริการ ONNX ที่รองรับ: {ort.get_available_providers()}")
        self.args = args
        self.model_ = get_models(
            device=cpu, dim_f=args.dim_f, dim_t=args.dim_t, n_fft=args.n_fft
        )
        
        # โหลดโมเดล ONNX พร้อมกำหนด Execution Providers
        onnx_path = os.path.join(args.onnx, f"{self.model_.target_name}.onnx")
        self.model = ort.InferenceSession(
            onnx_path,
            providers=[
                "CUDAExecutionProvider",
                "DmlExecutionProvider",
                "CPUExecutionProvider",
            ],
        )
        logger.info("โหลดโมเดล ONNX เสร็จสมบูรณ์")

    def demix(self, mix):
        """ฟังก์ชันจัดกลุ่มเสียงและตัดแบ่ง Chunk เพื่อประมวลผล"""
        samples = mix.shape[-1]
        margin = self.args.margin
        chunk_size = self.args.chunks * 44100
        
        assert margin != 0, "ค่า Margin ต้องไม่เป็นศูนย์!"
        if margin > chunk_size:
            margin = chunk_size

        segmented_mix = {}

        if self.args.chunks == 0 or samples < chunk_size:
            chunk_size = samples

        counter = -1
        for skip in range(0, samples, chunk_size):
            counter += 1
            s_margin = 0 if counter == 0 else margin
            end = min(skip + chunk_size + margin, samples)
            start = skip - s_margin
            segmented_mix[skip] = mix[:, start:end].copy()
            if end == samples:
                break

        sources = self.demix_base(segmented_mix, margin_size=margin)
        return sources

    def demix_base(self, mixes, margin_size):
        """ประมวลผลการแยกเสียงด้วยโมเดล AI"""
        chunked_sources = []
        progress_bar = tqdm(total=len(mixes), desc="กำลังประมวลผลเสียง (MDXNet)")
        
        for mix_index in mixes:
            cmix = mixes[mix_index]
            sources = []
            n_sample = cmix.shape[1]
            model = self.model_
            trim = model.n_fft // 2
            gen_size = model.chunk_size - 2 * trim
            pad = gen_size - n_sample % gen_size
            
            mix_p = np.concatenate(
                (np.zeros((2, trim)), cmix, np.zeros((2, pad)), np.zeros((2, trim))), 1
            )
            mix_waves = []
            i = 0
            while i < n_sample + pad:
                waves = np.array(mix_p[:, i : i + model.chunk_size])
                mix_waves.append(waves)
                i += gen_size
                
            mix_waves = torch.tensor(mix_waves, dtype=torch.float32).to(cpu)
            
            with torch.no_grad():
                _ort = self.model
                spek = model.stft(mix_waves)
                if self.args.denoise:
                    spec_pred = (
                        -_ort.run(None, {"input": -spek.cpu().numpy()})[0] * 0.5
                        + _ort.run(None, {"input": spek.cpu().numpy()})[0] * 0.5
                    )
                    tar_waves = model.istft(torch.tensor(spec_pred))
                else:
                    tar_waves = model.istft(
                        torch.tensor(_ort.run(None, {"input": spek.cpu().numpy()})[0])
                    )
                    
                tar_signal = (
                    tar_waves[:, :, trim:-trim]
                    .transpose(0, 1)
                    .reshape(2, -1)
                    .numpy()[:, :-pad]
                )

                start = 0 if mix_index == 0 else margin_size
                end = None if mix_index == list(mixes.keys())[-1] else -margin_size
                
                if margin_size == 0:
                    end = None
                    
                sources.append(tar_signal[:, start:end])
                progress_bar.update(1)

            chunked_sources.append(sources)
            
        _sources = np.concatenate(chunked_sources, axis=-1)
        progress_bar.close()
        return _sources

    def _save_audio_file(self, path, audio_data, sample_rate, target_format):
        """ฟังก์ชันช่วยเหลือสำหรับการบันทึกและแปลงไฟล์อย่างปลอดภัย"""
        sf.write(path, audio_data, sample_rate)
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

    def prediction(self, m, vocal_root, others_root, target_format):
        """จัดการการทำนายและแยกไฟล์เสียงต้นฉบับไปยังโฟลเดอร์เป้าหมาย"""
        os.makedirs(vocal_root, exist_ok=True)
        os.makedirs(others_root, exist_ok=True)
        basename = os.path.basename(m)
        
        # โหลดไฟล์เสียงด้วย Librosa (44.1kHz)
        mix, rate = librosa.load(m, mono=False, sr=44100)
        if mix.ndim == 1:
            mix = np.asfortranarray([mix, mix])
            
        mix = mix.T
        sources = self.demix(mix.T)
        opt = sources[0].T
        
        # คำนวณเลเยอร์เสียง
        vocal_data = mix - opt
        others_data = opt

        # ตั้งชื่อและบันทึกไฟล์
        logger.info(f"กำลังบันทึกไฟล์ผลลัพธ์สำหรับ: {basename}")
        vocal_path = os.path.join(vocal_root, f"{basename}_main_vocal.wav")
        others_path = os.path.join(others_root, f"{basename}_others.wav")

        if target_format in ["wav", "flac"]:
            vocal_path = os.path.join(vocal_root, f"{basename}_main_vocal.{target_format}")
            others_path = os.path.join(others_root, f"{basename}_others.{target_format}")
            sf.write(vocal_path, vocal_data, rate)
            sf.write(others_path, others_data, rate)
        else:
            # ใช้ฟังก์ชันช่วยเหลือเพื่อความปลอดภัยในการแปลงไฟล์
            self._save_audio_file(vocal_path, vocal_data, rate, target_format)
            self._save_audio_file(others_path, others_data, rate, target_format)


class MDXNetDereverb:
    """
    คลาสหลักสำหรับเรียกใช้งานโมเดล MDXNet Dereverb (FoxJoy)
    """
    def __init__(self, chunks, device):
        self.onnx = "assets/uvr5_weights/onnx_dereverb_By_FoxJoy"
        self.shifts = 10  # Predict with randomised equivariant stabilisation
        self.mixing = "min_mag"  # ['default','min_mag','max_mag']
        self.chunks = chunks
        self.margin = 44100
        self.dim_t = 9
        self.dim_f = 3072
        self.n_fft = 6144
        self.denoise = True
        self.pred = Predictor(self)
        self.device = device

    def _path_audio_(self, input_file, vocal_root, others_root, format_type, is_hp3=False):
        """สั่งให้โมเดลเริ่มกระบวนการลบเสียงสะท้อน"""
        self.pred.prediction(input_file, vocal_root, others_root, format_type)