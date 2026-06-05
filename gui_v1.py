# Made And Checked By DELTA SYNTH & Gemini AI
# Original Author: Xiaokai (RVC Realtime GUI)
# Version: 1.3

import os
import sys
import io
import json
import base64
import multiprocessing
import re
import threading
import time
import traceback
from multiprocessing import Queue, cpu_count
from queue import Empty
from dotenv import load_dotenv
import shutil

load_dotenv()

os.environ["OMP_NUM_THREADS"] = "4"
if sys.platform == "darwin":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

now_dir = os.getcwd()
sys.path.append(now_dir)

import librosa
from tools.torchgate import TorchGate
import numpy as np
import FreeSimpleGUI as sg
import sounddevice as sd
import torch
import torch.nn.functional as F
import torchaudio.transforms as tat
from PIL import Image

from infer.lib import rtrvc as rvc_for_realtime
from configs.config import Config

flag_vc = False

def printt(strr, *args):
    if len(args) == 0:
        print(strr)
    else:
        print(strr % args)

def phase_vocoder(a, b, fade_out, fade_in):
    window = torch.sqrt(fade_out * fade_in)
    fa = torch.fft.rfft(a * window)
    fb = torch.fft.rfft(b * window)
    absab = torch.abs(fa) + torch.abs(fb)
    n = a.shape[0]
    if n % 2 == 0:
        absab[1:-1] *= 2
    else:
        absab[1:] *= 2
    phia = torch.angle(fa)
    phib = torch.angle(fb)
    deltaphase = phib - phia
    deltaphase = deltaphase - 2 * np.pi * torch.floor(deltaphase / 2 / np.pi + 0.5)
    w = 2 * np.pi * torch.arange(n // 2 + 1).to(a) + deltaphase
    t = torch.arange(n).unsqueeze(-1).to(a) / n
    result = (
        a * (fade_out**2)
        + b * (fade_in**2)
        + torch.sum(absab * torch.cos(w * t + phia), -1) * window / n
    )
    return result

class Harvest(multiprocessing.Process):
    def __init__(self, inp_q, opt_q):
        multiprocessing.Process.__init__(self)
        self.inp_q = inp_q
        self.opt_q = opt_q

    def run(self):
        import numpy as np
        import pyworld

        while 1:
            idx, x, res_f0, n_cpu, ts = self.inp_q.get()
            f0, t = pyworld.harvest(
                x.astype(np.double),
                fs=16000,
                f0_ceil=1100,
                f0_floor=50,
                frame_period=10,
            )
            res_f0[idx] = f0
            if len(res_f0.keys()) >= n_cpu:
                self.opt_q.put(ts)

# -------------------------------------------------------------------
# ระบบฝังและอ่านข้อมูลโปรไฟล์ลงในไฟล์ .pth โดยตรง
# -------------------------------------------------------------------
def update_pth_metadata(pth_path, name, bio, img_path):
    try:
        data = torch.load(pth_path, map_location="cpu")
        if 'delta_info' not in data:
            data['delta_info'] = {}
        
        data['delta_info']['name'] = name
        data['delta_info']['bio'] = bio
        
        if img_path and os.path.exists(img_path):
            with Image.open(img_path) as img:
                img.thumbnail((250, 250)) # ย่อขนาดเพื่อไม่ให้ไฟล์ .pth ใหญ่เกินไป
                buffer = io.BytesIO()
                img.save(buffer, format="PNG")
                img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
                data['delta_info']['image'] = img_b64
                
        torch.save(data, pth_path)
        return True
    except Exception as e:
        print(f"เกิดข้อผิดพลาดในการบันทึกข้อมูล: {e}")
        return False

def read_pth_metadata(pth_path):
    try:
        data = torch.load(pth_path, map_location="cpu")
        return data.get('delta_info', {})
    except:
        return {}

if __name__ == "__main__":
    current_dir = os.getcwd()
    inp_q = Queue()
    opt_q = Queue()
    n_cpu = min(cpu_count(), 8)
    for _ in range(n_cpu):
        p = Harvest(inp_q, opt_q)
        p.daemon = True
        p.start()

    class GUIConfig:
        def __init__(self) -> None:
            self.pth_path: str = ""
            self.index_path: str = ""
            self.pitch: int = 0
            self.formant=0.0
            self.sr_type: str = "sr_model"
            self.block_time: float = 0.25 
            self.threhold: int = -60
            self.crossfade_time: float = 0.05
            self.extra_time: float = 2.5
            self.I_noise_reduce: bool = False
            self.O_noise_reduce: bool = False
            self.use_pv: bool = False
            self.rms_mix_rate: float = 0.0
            self.index_rate: float = 0.0
            self.n_cpu: int = min(n_cpu, 4)
            self.f0method: str = "fcpe"
            self.sg_hostapi: str = ""
            self.wasapi_exclusive: bool = False
            self.sg_input_device: str = ""
            self.sg_output_device: str = ""

    class GUI:
        def __init__(self) -> None:
            self.gui_config = GUIConfig()
            self.config = Config()
            self.function = "vc"
            self.delay_time = 0
            self.hostapis = None
            self.input_devices = None
            self.output_devices = None
            self.input_devices_indices = None
            self.output_devices_indices = None
            self.stream = None
            self.char_info = {}
            self.update_devices()
            self.launcher()

        def load(self):
            try:
                if not os.path.exists("configs/inuse/config.json"):
                    shutil.copy("configs/config.json", "configs/inuse/config.json")
                with open("configs/inuse/config.json", "r", encoding="utf-8") as j:
                    data = json.load(j)
                    data["sr_model"] = data.get("sr_type") == "sr_model"
                    data["sr_device"] = data.get("sr_type") == "sr_device"
                    data["pm"] = data.get("f0method") == "pm"
                    data["harvest"] = data.get("f0method") == "harvest"
                    data["crepe"] = data.get("f0method") == "crepe"
                    data["rmvpe"] = data.get("f0method") == "rmvpe"
                    data["fcpe"] = data.get("f0method") == "fcpe"
                    if data.get("sg_hostapi") in self.hostapis:
                        self.update_devices(hostapi_name=data["sg_hostapi"])
                        if (
                            data.get("sg_input_device") not in self.input_devices
                            or data.get("sg_output_device") not in self.output_devices
                        ):
                            self.update_devices()
                            data["sg_hostapi"] = self.hostapis[0]
                            data["sg_input_device"] = self.input_devices[
                                self.input_devices_indices.index(sd.default.device[0])
                            ]
                            data["sg_output_device"] = self.output_devices[
                                self.output_devices_indices.index(sd.default.device[1])
                            ]
                    else:
                        data["sg_hostapi"] = self.hostapis[0]
                        data["sg_input_device"] = self.input_devices[
                            self.input_devices_indices.index(sd.default.device[0])
                        ]
                        data["sg_output_device"] = self.output_devices[
                            self.output_devices_indices.index(sd.default.device[1])
                        ]
            except:
                with open("configs/inuse/config.json", "w", encoding="utf-8") as j:
                    data = {
                        "pth_path": "",
                        "index_path": "",
                        "sg_hostapi": self.hostapis[0],
                        "sg_wasapi_exclusive": False,
                        "sg_input_device": self.input_devices[
                            self.input_devices_indices.index(sd.default.device[0])
                        ],
                        "sg_output_device": self.output_devices[
                            self.output_devices_indices.index(sd.default.device[1])
                        ],
                        "sr_type": "sr_model",
                        "threhold": -60,
                        "pitch": 0,
                        "formant": 0.0,
                        "index_rate": 0,
                        "rms_mix_rate": 0,
                        "block_time": 0.25,
                        "crossfade_length": 0.05,
                        "extra_time": 2.5,
                        "n_cpu": 4,
                        "f0method": "rmvpe",
                        "use_jit": False,
                        "use_pv": False,
                    }
                    data["sr_model"] = data["sr_type"] == "sr_model"
                    data["sr_device"] = data["sr_type"] == "sr_device"
                    data["pm"] = data["f0method"] == "pm"
                    data["harvest"] = data["f0method"] == "harvest"
                    data["crepe"] = data["f0method"] == "crepe"
                    data["rmvpe"] = data["f0method"] == "rmvpe"
                    data["fcpe"] = data["f0method"] == "fcpe"
            return data

        def launcher(self):
            data = self.load()
            self.config.use_jit = False 
            sg.theme("DarkTeal12") # ปรับธีมให้ดูทันสมัยและเป็นทางการ

            left_col = [
                [
                    sg.Frame(
                        title="ส่วนการโหลดโมเดลเสียง",
                        layout=[
                            [
                                sg.Input(default_text=data.get("pth_path", ""), key="pth_path", enable_events=True),
                                sg.FileBrowse("เลือกไฟล์ .pth", initial_folder=os.path.join(os.getcwd(), "assets/weights"), file_types=((".pth"),)),
                            ],
                            [
                                sg.Input(default_text=data.get("index_path", ""), key="index_path"),
                                sg.FileBrowse("เลือกไฟล์ .index", initial_folder=os.path.join(os.getcwd(), "logs"), file_types=((".index"),)),
                            ],
                        ],
                    )
                ],
                [
                    sg.Frame(
                        layout=[
                            [
                                sg.Text("ระบบปฏิบัติการเสียง"),
                                sg.Combo(self.hostapis, key="sg_hostapi", default_value=data.get("sg_hostapi", ""), enable_events=True, size=(20, 1)),
                                sg.Checkbox("ใช้งาน WASAPI แบบผูกขาด", key="sg_wasapi_exclusive", default=data.get("sg_wasapi_exclusive", False), enable_events=True),
                            ],
                            [
                                sg.Text("อุปกรณ์รับเสียง (ไมค์)"),
                                sg.Combo(self.input_devices, key="sg_input_device", default_value=data.get("sg_input_device", ""), enable_events=True, size=(45, 1)),
                            ],
                            [
                                sg.Text("อุปกรณ์ส่งเสียง (ลำโพง)"),
                                sg.Combo(self.output_devices, key="sg_output_device", default_value=data.get("sg_output_device", ""), enable_events=True, size=(45, 1)),
                            ],
                            [
                                sg.Button("โหลดอุปกรณ์ใหม่", key="reload_devices"),
                                sg.Radio("อิงตามโมเดล", "sr_type", key="sr_model", default=data.get("sr_model", True), enable_events=True),
                                sg.Radio("อิงตามอุปกรณ์", "sr_type", key="sr_device", default=data.get("sr_device", False), enable_events=True),
                                sg.Text("อัตราสุ่มเสียง:"),
                                sg.Text("", key="sr_stream"),
                            ],
                        ],
                        title="ตั้งค่าอุปกรณ์เสียง",
                    )
                ],
                [
                    sg.Frame(
                        layout=[
                            [
                                sg.Text("ระดับเสียงเริ่มทำงาน"),
                                sg.Slider(range=(-60, 0), key="threhold", resolution=1, orientation="h", default_value=data.get("threhold", -60), enable_events=True),
                            ],
                            [
                                sg.Text("ปรับระดับคีย์เสียง"),
                                sg.Slider(range=(-16, 16), key="pitch", resolution=1, orientation="h", default_value=data.get("pitch", 0), enable_events=True),
                            ],
                            [
                                sg.Text("ฟอร์แมนท์ (ความทุ้มแหลม)"),
                                sg.Slider(range=(-2, 2), key="formant", resolution=0.05, orientation="h", default_value=data.get("formant", 0.0), enable_events=True),
                            ],
                            [
                                sg.Text("อัตราการใช้งาน Index"),
                                sg.Slider(range=(0.0, 1.0), key="index_rate", resolution=0.01, orientation="h", default_value=data.get("index_rate", 0), enable_events=True),
                            ],
                            [
                                sg.Text("อัตราส่วนผสมเสียงเดิม"),
                                sg.Slider(range=(0.0, 1.0), key="rms_mix_rate", resolution=0.01, orientation="h", default_value=data.get("rms_mix_rate", 0), enable_events=True),
                            ],
                            [
                                sg.Text("อัลกอริทึมวิเคราะห์เสียง"),
                                sg.Radio("pm", "f0method", key="pm", default=data.get("pm", False), enable_events=True),
                                sg.Radio("harvest", "f0method", key="harvest", default=data.get("harvest", False), enable_events=True),
                                sg.Radio("crepe", "f0method", key="crepe", default=data.get("crepe", False), enable_events=True),
                                sg.Radio("rmvpe", "f0method", key="rmvpe", default=data.get("rmvpe", False), enable_events=True),
                                sg.Radio("fcpe", "f0method", key="fcpe", default=data.get("fcpe", True), enable_events=True),
                            ],
                        ],
                        title="การปรับแต่งเสียงโดยรวม",
                    ),
                    sg.Frame(
                        layout=[
                            [
                                sg.Text("ช่วงประมวลผล (วินาที)"),
                                sg.Slider(range=(0.02, 1.5), key="block_time", resolution=0.01, orientation="h", default_value=data.get("block_time", 0.25), enable_events=True),
                            ],
                            [
                                sg.Text("การทำงาน CPU (Harvest)"),
                                sg.Slider(range=(1, n_cpu), key="n_cpu", resolution=1, orientation="h", default_value=data.get("n_cpu", min(self.gui_config.n_cpu, n_cpu)), enable_events=True),
                            ],
                            [
                                sg.Text("ความยาวรอยต่อเสียง"),
                                sg.Slider(range=(0.01, 0.15), key="crossfade_length", resolution=0.01, orientation="h", default_value=data.get("crossfade_length", 0.05), enable_events=True),
                            ],
                            [
                                sg.Text("ประมวลผลล่วงหน้า"),
                                sg.Slider(range=(0.05, 5.00), key="extra_time", resolution=0.01, orientation="h", default_value=data.get("extra_time", 2.5), enable_events=True),
                            ],
                            [
                                sg.Checkbox("ลดนอยส์ขาเข้า", key="I_noise_reduce", enable_events=True),
                                sg.Checkbox("ลดนอยส์ขาออก", key="O_noise_reduce", enable_events=True),
                                sg.Checkbox("เปิด Phase Vocoder", key="use_pv", default=data.get("use_pv", False), enable_events=True),
                            ],
                        ],
                        title="ตั้งค่าประสิทธิภาพระบบ",
                    ),
                ],
                [
                    sg.Button("เริ่มประมวลผลเสียง", key="start_vc", button_color=("white", "green")),
                    sg.Button("หยุดประมวลผลเสียง", key="stop_vc", button_color=("white", "red")),
                    sg.Radio("ฟังเสียงต้นฉบับ", "function", key="im", default=False, enable_events=True),
                    sg.Radio("แปลงเสียงร้อง", "function", key="vc", default=True, enable_events=True),
                    sg.Text("ความหน่วงอัลกอริทึม (ms):"),
                    sg.Text("0", key="delay_time"),
                    sg.Text("ระยะเวลาประมวลผล (ms):"),
                    sg.Text("0", key="infer_time"),
                ],
            ]

            right_col = [
                [
                    sg.Frame(
                        "โปรไฟล์นักร้อง (Character Profile)",
                        layout=[
                            [sg.Image(data=b'', key="char_img", size=(250, 250), background_color="black")],
                            [sg.Text("ชื่อตัวละคร:", font=("Helvetica", 10, "bold")), sg.Text("-", key="char_name", size=(25, 1))],
                            [sg.Text("ประวัติ / รายละเอียด:", font=("Helvetica", 10, "bold"))],
                            [sg.Multiline("-", key="char_bio", size=(35, 10), disabled=True, no_scrollbar=True)],
                            [sg.Button("แก้ไขและบันทึกลงโมเดล .pth", key="edit_profile")]
                        ],
                        element_justification='c'
                    )
                ]
            ]

            layout = [
                [sg.Column(left_col), sg.VSeparator(), sg.Column(right_col, element_justification='c')]
            ]
            
            self.window = sg.Window("DELTA SYNTH - Realtime RVC", layout=layout, finalize=True)
            self.refresh_profile(data.get("pth_path", ""))
            self.event_handler()

        def refresh_profile(self, pth_path):
            if os.path.exists(pth_path):
                info = read_pth_metadata(pth_path)
                self.window["char_name"].update(info.get("name", "ไม่ระบุชื่อ"))
                self.window["char_bio"].update(info.get("bio", "ไม่มีข้อมูลประวัติ"))
                img_b64 = info.get("image", "")
                if img_b64:
                    try:
                        self.window["char_img"].update(data=img_b64)
                    except:
                        self.window["char_img"].update(data=b'')
                else:
                    self.window["char_img"].update(data=b'')

        def open_edit_profile_window(self):
            pth_path = self.gui_config.pth_path or self.window["pth_path"].get()
            if not os.path.exists(pth_path):
                sg.popup_error("กรุณาเลือกไฟล์ .pth ก่อนทำการแก้ไขโปรไฟล์")
                return

            info = read_pth_metadata(pth_path)
            layout = [
                [sg.Text("ชื่อตัวละคร:"), sg.Input(default_text=info.get("name", ""), key="ep_name")],
                [sg.Text("ประวัติโดยย่อ:"), sg.Multiline(default_text=info.get("bio", ""), key="ep_bio", size=(40, 5))],
                [sg.Text("รูปภาพ (PNG/JPG):"), sg.Input(key="ep_img"), sg.FileBrowse("เลือกไฟล์", file_types=(("Image Files", "*.png;*.jpg;*.jpeg"),))],
                [sg.Button("บันทึกข้อมูล", key="save_ep"), sg.Button("ยกเลิก", key="cancel_ep")]
            ]
            
            ep_window = sg.Window("แก้ไขข้อมูลโมเดล", layout, modal=True)
            while True:
                event, values = ep_window.read()
                if event in (sg.WINDOW_CLOSED, "cancel_ep"):
                    break
                if event == "save_ep":
                    success = update_pth_metadata(pth_path, values["ep_name"], values["ep_bio"], values["ep_img"])
                    if success:
                        sg.popup("บันทึกข้อมูลลงในไฟล์ .pth สำเร็จเรียบร้อยแล้ว")
                        self.refresh_profile(pth_path)
                    else:
                        sg.popup_error("เกิดข้อผิดพลาด ไม่สามารถบันทึกข้อมูลได้")
                    break
            ep_window.close()

        def event_handler(self):
            global flag_vc
            while True:
                event, values = self.window.read()
                if event == sg.WINDOW_CLOSED:
                    self.stop_stream()
                    exit()
                    
                if event == "pth_path":
                    self.refresh_profile(values["pth_path"])
                    
                if event == "edit_profile":
                    self.open_edit_profile_window()

                if event == "reload_devices" or event == "sg_hostapi":
                    self.gui_config.sg_hostapi = values["sg_hostapi"]
                    self.update_devices(hostapi_name=values["sg_hostapi"])
                    if self.gui_config.sg_hostapi not in self.hostapis:
                        self.gui_config.sg_hostapi = self.hostapis[0]
                    self.window["sg_hostapi"].Update(values=self.hostapis)
                    self.window["sg_hostapi"].Update(value=self.gui_config.sg_hostapi)
                    if (
                        self.gui_config.sg_input_device not in self.input_devices
                        and len(self.input_devices) > 0
                    ):
                        self.gui_config.sg_input_device = self.input_devices[0]
                    self.window["sg_input_device"].Update(values=self.input_devices)
                    self.window["sg_input_device"].Update(
                        value=self.gui_config.sg_input_device
                    )
                    if self.gui_config.sg_output_device not in self.output_devices:
                        self.gui_config.sg_output_device = self.output_devices[0]
                    self.window["sg_output_device"].Update(values=self.output_devices)
                    self.window["sg_output_device"].Update(
                        value=self.gui_config.sg_output_device
                    )
                    
                if event == "start_vc" and not flag_vc:
                    if self.set_values(values) == True:
                        printt("ตรวจสอบสถานะ CUDA: %s", torch.cuda.is_available())
                        self.start_vc()
                        settings = {
                            "pth_path": values["pth_path"],
                            "index_path": values["index_path"],
                            "sg_hostapi": values["sg_hostapi"],
                            "sg_wasapi_exclusive": values["sg_wasapi_exclusive"],
                            "sg_input_device": values["sg_input_device"],
                            "sg_output_device": values["sg_output_device"],
                            "sr_type": ["sr_model", "sr_device"][
                                [
                                    values["sr_model"],
                                    values["sr_device"],
                                ].index(True)
                            ],
                            "threhold": values["threhold"],
                            "pitch": values["pitch"],
                            "rms_mix_rate": values["rms_mix_rate"],
                            "index_rate": values["index_rate"],
                            "block_time": values["block_time"],
                            "crossfade_length": values["crossfade_length"],
                            "extra_time": values["extra_time"],
                            "n_cpu": values["n_cpu"],
                            "use_jit": False,
                            "use_pv": values["use_pv"],
                            "f0method": ["pm", "harvest", "crepe", "rmvpe", "fcpe"][
                                [
                                    values["pm"],
                                    values["harvest"],
                                    values["crepe"],
                                    values["rmvpe"],
                                    values["fcpe"],
                                ].index(True)
                            ],
                        }
                        with open("configs/inuse/config.json", "w", encoding="utf-8") as j:
                            json.dump(settings, j, ensure_ascii=False)
                        if self.stream is not None:
                            self.delay_time = (
                                self.stream.latency[-1]
                                + values["block_time"]
                                + values["crossfade_length"]
                                + 0.01
                            )
                        if values["I_noise_reduce"]:
                            self.delay_time += min(values["crossfade_length"], 0.04)
                        self.window["sr_stream"].update(self.gui_config.samplerate)
                        self.window["delay_time"].update(
                            int(np.round(self.delay_time * 1000))
                        )
                        
                if event == "threhold":
                    self.gui_config.threhold = values["threhold"]
                elif event == "pitch":
                    self.gui_config.pitch = values["pitch"]
                    if hasattr(self, "rvc"):
                        self.rvc.change_key(values["pitch"])
                elif event == "formant":
                    self.gui_config.formant = values["formant"]
                    if hasattr(self, "rvc"):
                        self.rvc.change_formant(values["formant"])
                elif event == "index_rate":
                    self.gui_config.index_rate = values["index_rate"]
                    if hasattr(self, "rvc"):
                        self.rvc.change_index_rate(values["index_rate"])
                elif event == "rms_mix_rate":
                    self.gui_config.rms_mix_rate = values["rms_mix_rate"]
                elif event in ["pm", "harvest", "crepe", "rmvpe", "fcpe"]:
                    self.gui_config.f0method = event
                elif event == "I_noise_reduce":
                    self.gui_config.I_noise_reduce = values["I_noise_reduce"]
                    if self.stream is not None:
                        self.delay_time += (
                            1 if values["I_noise_reduce"] else -1
                        ) * min(values["crossfade_length"], 0.04)
                        self.window["delay_time"].update(
                            int(np.round(self.delay_time * 1000))
                        )
                elif event == "O_noise_reduce":
                    self.gui_config.O_noise_reduce = values["O_noise_reduce"]
                elif event == "use_pv":
                    self.gui_config.use_pv = values["use_pv"]
                elif event in ["vc", "im"]:
                    self.function = event
                elif event == "stop_vc" or event != "start_vc":
                    self.stop_stream()

        def set_values(self, values):
            if len(values["pth_path"].strip()) == 0:
                sg.popup("กรุณาเลือกไฟล์ .pth ก่อนเริ่มระบบ")
                return False
            if len(values["index_path"].strip()) == 0:
                sg.popup("กรุณาเลือกไฟล์ .index ด้วยครับ")
                return False
            
            self.set_devices(values["sg_input_device"], values["sg_output_device"])
            self.config.use_jit = False 
            self.gui_config.sg_hostapi = values["sg_hostapi"]
            self.gui_config.sg_wasapi_exclusive = values["sg_wasapi_exclusive"]
            self.gui_config.sg_input_device = values["sg_input_device"]
            self.gui_config.sg_output_device = values["sg_output_device"]
            self.gui_config.pth_path = values["pth_path"]
            self.gui_config.index_path = values["index_path"]
            self.gui_config.sr_type = ["sr_model", "sr_device"][
                [
                    values["sr_model"],
                    values["sr_device"],
                ].index(True)
            ]
            self.gui_config.threhold = values["threhold"]
            self.gui_config.pitch = values["pitch"]
            self.gui_config.formant = values["formant"]
            self.gui_config.block_time = values["block_time"]
            self.gui_config.crossfade_time = values["crossfade_length"]
            self.gui_config.extra_time = values["extra_time"]
            self.gui_config.I_noise_reduce = values["I_noise_reduce"]
            self.gui_config.O_noise_reduce = values["O_noise_reduce"]
            self.gui_config.use_pv = values["use_pv"]
            self.gui_config.rms_mix_rate = values["rms_mix_rate"]
            self.gui_config.index_rate = values["index_rate"]
            self.gui_config.n_cpu = values["n_cpu"]
            self.gui_config.f0method = ["pm", "harvest", "crepe", "rmvpe", "fcpe"][
                [
                    values["pm"],
                    values["harvest"],
                    values["crepe"],
                    values["rmvpe"],
                    values["fcpe"],
                ].index(True)
            ]
            return True

        def start_vc(self):
            torch.cuda.empty_cache()
            self.rvc = rvc_for_realtime.RVC(
                self.gui_config.pitch,
                self.gui_config.formant,
                self.gui_config.pth_path,
                self.gui_config.index_path,
                self.gui_config.index_rate,
                self.gui_config.n_cpu,
                inp_q,
                opt_q,
                self.config,
                self.rvc if hasattr(self, "rvc") else None,
            )
            self.gui_config.samplerate = (
                self.rvc.tgt_sr
                if self.gui_config.sr_type == "sr_model"
                else self.get_device_samplerate()
            )
            self.gui_config.channels = self.get_device_channels()
            self.zc = self.gui_config.samplerate // 100
            self.block_frame = (
                int(
                    np.round(
                        self.gui_config.block_time
                        * self.gui_config.samplerate
                        / self.zc
                    )
                )
                * self.zc
            )
            self.block_frame_16k = 160 * self.block_frame // self.zc
            self.crossfade_frame = (
                int(
                    np.round(
                        self.gui_config.crossfade_time
                        * self.gui_config.samplerate
                        / self.zc
                    )
                )
                * self.zc
            )
            self.sola_buffer_frame = min(self.crossfade_frame, 4 * self.zc)
            self.sola_search_frame = self.zc
            self.extra_frame = (
                int(
                    np.round(
                        self.gui_config.extra_time
                        * self.gui_config.samplerate
                        / self.zc
                    )
                )
                * self.zc
            )
            self.input_wav: torch.Tensor = torch.zeros(
                self.extra_frame
                + self.crossfade_frame
                + self.sola_search_frame
                + self.block_frame,
                device=self.config.device,
                dtype=torch.float32,
            )
            self.input_wav_denoise: torch.Tensor = self.input_wav.clone()
            self.input_wav_res: torch.Tensor = torch.zeros(
                160 * self.input_wav.shape[0] // self.zc,
                device=self.config.device,
                dtype=torch.float32,
            )
            self.rms_buffer: np.ndarray = np.zeros(4 * self.zc, dtype="float32")
            self.sola_buffer: torch.Tensor = torch.zeros(
                self.sola_buffer_frame, device=self.config.device, dtype=torch.float32
            )
            self.nr_buffer: torch.Tensor = self.sola_buffer.clone()
            self.output_buffer: torch.Tensor = self.input_wav.clone()
            self.skip_head = self.extra_frame // self.zc
            self.return_length = (
                self.block_frame + self.sola_buffer_frame + self.sola_search_frame
            ) // self.zc
            self.fade_in_window: torch.Tensor = (
                torch.sin(
                    0.5
                    * np.pi
                    * torch.linspace(
                        0.0,
                        1.0,
                        steps=self.sola_buffer_frame,
                        device=self.config.device,
                        dtype=torch.float32,
                    )
                )
                ** 2
            )
            self.fade_out_window: torch.Tensor = 1 - self.fade_in_window
            self.resampler = tat.Resample(
                orig_freq=self.gui_config.samplerate,
                new_freq=16000,
                dtype=torch.float32,
            ).to(self.config.device)
            if self.rvc.tgt_sr != self.gui_config.samplerate:
                self.resampler2 = tat.Resample(
                    orig_freq=self.rvc.tgt_sr,
                    new_freq=self.gui_config.samplerate,
                    dtype=torch.float32,
                ).to(self.config.device)
            else:
                self.resampler2 = None
            self.tg = TorchGate(
                sr=self.gui_config.samplerate, n_fft=4 * self.zc, prop_decrease=0.9
            ).to(self.config.device)
            self.start_stream()

        def start_stream(self):
            global flag_vc
            if not flag_vc:
                flag_vc = True
                if (
                    "WASAPI" in self.gui_config.sg_hostapi
                    and self.gui_config.sg_wasapi_exclusive
                ):
                    extra_settings = sd.WasapiSettings(exclusive=True)
                else:
                    extra_settings = None
                self.stream = sd.Stream(
                    callback=self.audio_callback,
                    blocksize=self.block_frame,
                    samplerate=self.gui_config.samplerate,
                    channels=self.gui_config.channels,
                    dtype="float32",
                    extra_settings=extra_settings,
                )
                self.stream.start()

        def stop_stream(self):
            global flag_vc
            if flag_vc:
                flag_vc = False
                if self.stream is not None:
                    self.stream.abort()
                    self.stream.close()
                    self.stream = None

        def audio_callback(self, indata: np.ndarray, outdata: np.ndarray, frames, times, status):
            """
            ระบบประมวลผลเสียง (Audio Processing Callback)
            """
            global flag_vc
            try:
                start_time = time.perf_counter()
                indata = librosa.to_mono(indata.T)
                if self.gui_config.threhold > -60:
                    indata = np.append(self.rms_buffer, indata)
                    rms = librosa.feature.rms(
                        y=indata, frame_length=4 * self.zc, hop_length=self.zc
                    )[:, 2:]
                    self.rms_buffer[:] = indata[-4 * self.zc :]
                    indata = indata[2 * self.zc - self.zc // 2 :]
                    db_threhold = (
                        librosa.amplitude_to_db(rms, ref=1.0)[0] < self.gui_config.threhold
                    )
                    for i in range(db_threhold.shape[0]):
                        if db_threhold[i]:
                            indata[i * self.zc : (i + 1) * self.zc] = 0
                    indata = indata[self.zc // 2 :]
                self.input_wav[: -self.block_frame] = self.input_wav[
                    self.block_frame :
                ].clone()
                self.input_wav[-indata.shape[0] :] = torch.from_numpy(indata).to(
                    self.config.device
                )
                self.input_wav_res[: -self.block_frame_16k] = self.input_wav_res[
                    self.block_frame_16k :
                ].clone()
                
                # Input Noise Reduction
                if self.gui_config.I_noise_reduce:
                    self.input_wav_denoise[: -self.block_frame] = self.input_wav_denoise[
                        self.block_frame :
                    ].clone()
                    input_wav = self.input_wav[-self.sola_buffer_frame - self.block_frame :]
                    input_wav = self.tg(
                        input_wav.unsqueeze(0), self.input_wav.unsqueeze(0)
                    ).squeeze(0)
                    input_wav[: self.sola_buffer_frame] *= self.fade_in_window
                    input_wav[: self.sola_buffer_frame] += (
                        self.nr_buffer * self.fade_out_window
                    )
                    self.input_wav_denoise[-self.block_frame :] = input_wav[
                        : self.block_frame
                    ]
                    self.nr_buffer[:] = input_wav[self.block_frame :]
                    self.input_wav_res[-self.block_frame_16k - 160 :] = self.resampler(
                        self.input_wav_denoise[-self.block_frame - 2 * self.zc :]
                    )[160:]
                else:
                    self.input_wav_res[-160 * (indata.shape[0] // self.zc + 1) :] = (
                        self.resampler(self.input_wav[-indata.shape[0] - 2 * self.zc :])[
                            160:
                        ]
                    )
                    
                # Inference
                if self.function == "vc":
                    infer_wav = self.rvc.infer(
                        self.input_wav_res,
                        self.block_frame_16k,
                        self.skip_head,
                        self.return_length,
                        self.gui_config.f0method,
                    )
                    if self.resampler2 is not None:
                        infer_wav = self.resampler2(infer_wav)
                elif self.gui_config.I_noise_reduce:
                    infer_wav = self.input_wav_denoise[self.extra_frame :].clone()
                else:
                    infer_wav = self.input_wav[self.extra_frame :].clone()
                    
                # Output Noise Reduction
                if self.gui_config.O_noise_reduce and self.function == "vc":
                    self.output_buffer[: -self.block_frame] = self.output_buffer[
                        self.block_frame :
                    ].clone()
                    self.output_buffer[-self.block_frame :] = infer_wav[-self.block_frame :]
                    infer_wav = self.tg(
                        infer_wav.unsqueeze(0), self.output_buffer.unsqueeze(0)
                    ).squeeze(0)
                    
                # Volume Envelope Mixing
                if self.gui_config.rms_mix_rate < 1 and self.function == "vc":
                    if self.gui_config.I_noise_reduce:
                        input_wav = self.input_wav_denoise[self.extra_frame :]
                    else:
                        input_wav = self.input_wav[self.extra_frame :]
                    rms1 = librosa.feature.rms(
                        y=input_wav[: infer_wav.shape[0]].cpu().numpy(),
                        frame_length=4 * self.zc,
                        hop_length=self.zc,
                    )
                    rms1 = torch.from_numpy(rms1).to(self.config.device)
                    rms1 = F.interpolate(
                        rms1.unsqueeze(0),
                        size=infer_wav.shape[0] + 1,
                        mode="linear",
                        align_corners=True,
                    )[0, 0, :-1]
                    rms2 = librosa.feature.rms(
                        y=infer_wav[:].cpu().numpy(),
                        frame_length=4 * self.zc,
                        hop_length=self.zc,
                    )
                    rms2 = torch.from_numpy(rms2).to(self.config.device)
                    rms2 = F.interpolate(
                        rms2.unsqueeze(0),
                        size=infer_wav.shape[0] + 1,
                        mode="linear",
                        align_corners=True,
                    )[0, 0, :-1]
                    rms2 = torch.max(rms2, torch.zeros_like(rms2) + 1e-3)
                    infer_wav *= torch.pow(
                        rms1 / rms2, torch.tensor(1 - self.gui_config.rms_mix_rate)
                    )
                    
                # SOLA Algorithm
                conv_input = infer_wav[
                    None, None, : self.sola_buffer_frame + self.sola_search_frame
                ]
                cor_nom = F.conv1d(conv_input, self.sola_buffer[None, None, :])
                cor_den = torch.sqrt(
                    F.conv1d(
                        conv_input**2,
                        torch.ones(1, 1, self.sola_buffer_frame, device=self.config.device),
                    )
                    + 1e-8
                )
                if sys.platform == "darwin":
                    _, sola_offset = torch.max(cor_nom[0, 0] / cor_den[0, 0])
                    sola_offset = sola_offset.item()
                else:
                    sola_offset = torch.argmax(cor_nom[0, 0] / cor_den[0, 0])
                    
                infer_wav = infer_wav[sola_offset:]
                if "privateuseone" in str(self.config.device) or not self.gui_config.use_pv:
                    infer_wav[: self.sola_buffer_frame] *= self.fade_in_window
                    infer_wav[: self.sola_buffer_frame] += (
                        self.sola_buffer * self.fade_out_window
                    )
                else:
                    infer_wav[: self.sola_buffer_frame] = phase_vocoder(
                        self.sola_buffer,
                        infer_wav[: self.sola_buffer_frame],
                        self.fade_out_window,
                        self.fade_in_window,
                    )
                self.sola_buffer[:] = infer_wav[
                    self.block_frame : self.block_frame + self.sola_buffer_frame
                ]
                outdata[:] = (
                    infer_wav[: self.block_frame]
                    .repeat(self.gui_config.channels, 1)
                    .t()
                    .cpu()
                    .numpy()
                )
                total_time = time.perf_counter() - start_time
                if flag_vc:
                    self.window.write_event_value("update_infer_time", int(total_time * 1000))
            except Exception as e:
                # ป้องกันไม่ให้โปรแกรมล่มเมื่อเกิดข้อผิดพลาดในการคำนวณเสียงชั่วขณะ
                printt("เกิดข้อผิดพลาดในการประมวลผลเสียง: %s", str(e))
                outdata.fill(0)

        def update_devices(self, hostapi_name=None):
            global flag_vc
            flag_vc = False
            sd._terminate()
            sd._initialize()
            devices = sd.query_devices()
            hostapis = sd.query_hostapis()
            for hostapi in hostapis:
                for device_idx in hostapi["devices"]:
                    devices[device_idx]["hostapi_name"] = hostapi["name"]
            self.hostapis = [hostapi["name"] for hostapi in hostapis]
            if hostapi_name not in self.hostapis:
                hostapi_name = self.hostapis[0]
            self.input_devices = [
                d["name"]
                for d in devices
                if d["max_input_channels"] > 0 and d["hostapi_name"] == hostapi_name
            ]
            self.output_devices = [
                d["name"]
                for d in devices
                if d["max_output_channels"] > 0 and d["hostapi_name"] == hostapi_name
            ]
            self.input_devices_indices = [
                d["index"] if "index" in d else d["name"]
                for d in devices
                if d["max_input_channels"] > 0 and d["hostapi_name"] == hostapi_name
            ]
            self.output_devices_indices = [
                d["index"] if "index" in d else d["name"]
                for d in devices
                if d["max_output_channels"] > 0 and d["hostapi_name"] == hostapi_name
            ]

        def set_devices(self, input_device, output_device):
            sd.default.device[0] = self.input_devices_indices[
                self.input_devices.index(input_device)
            ]
            sd.default.device[1] = self.output_devices_indices[
                self.output_devices.index(output_device)
            ]
            printt("อุปกรณ์รับเสียงที่เลือก: %s:%s", str(sd.default.device[0]), input_device)
            printt("อุปกรณ์ส่งเสียงที่เลือก: %s:%s", str(sd.default.device[1]), output_device)

        def get_device_samplerate(self):
            return int(
                sd.query_devices(device=sd.default.device[0])["default_samplerate"]
            )

        def get_device_channels(self):
            max_input_channels = sd.query_devices(device=sd.default.device[0])[
                "max_input_channels"
            ]
            max_output_channels = sd.query_devices(device=sd.default.device[1])[
                "max_output_channels"
            ]
            return min(max_input_channels, max_output_channels, 2)

    gui = GUI()