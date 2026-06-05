# Made And Checked By DELTA SYNTH & Gemini AI
# Original Developers: RVC Project

import argparse
import os
import sys
import json
import shutil
from multiprocessing import cpu_count

import torch

try:
    import intel_extension_for_pytorch as ipex  # pylint: disable=import-error, unused-import

    if torch.xpu.is_available():
        from infer.modules.ipex import ipex_init

        ipex_init()
except Exception:  # pylint: disable=broad-exception-caught
    pass
import logging

logger = logging.getLogger(__name__)

version_config_list = [
    "v1/32k.json",
    "v1/40k.json",
    "v1/48k.json",
    "v2/48k.json",
    "v2/32k.json",
]


def singleton_variable(func):
    def wrapper(*args, **kwargs):
        if not wrapper.instance:
            wrapper.instance = func(*args, **kwargs)
        return wrapper.instance

    wrapper.instance = None
    return wrapper


@singleton_variable
class Config:
    def __init__(self):
        self.device = "cuda:0"
        self.is_half = True
        self.use_jit = False
        self.n_cpu = 0
        self.gpu_name = None
        self.json_config = self.load_config_json()
        self.gpu_mem = None
        (
            self.python_cmd,
            self.listen_port,
            self.iscolab,
            self.noparallel,
            self.noautoopen,
            self.dml,
        ) = self.arg_parse()
        self.instead = ""
        self.preprocess_per = 3.7
        self.x_pad, self.x_query, self.x_center, self.x_max = self.device_config()

    @staticmethod
    def load_config_json() -> dict:
        d = {}
        for config_file in version_config_list:
            # เพิกเฉยไฟล์ที่ไม่ใช่นามสกุล .json หรือไฟล์ที่ไม่เกี่ยวข้อง
            if not config_file.endswith(".json"):
                continue
                
            source_p = f"configs/{config_file}"
            target_p = f"configs/inuse/{config_file}"
            
            # เพิกเฉยการคัดลอกหากไม่มีไฟล์ต้นฉบับอยู่จริง
            if not os.path.exists(source_p):
                logger.warning(f"ไม่พบไฟล์ต้นฉบับ ข้ามการดำเนินการสำหรับ: {config_file}")
                continue
                
            if not os.path.exists(target_p):
                os.makedirs(os.path.dirname(target_p), exist_ok=True)
                shutil.copy(source_p, target_p)
                
            with open(target_p, "r", encoding="utf-8") as f:
                d[config_file] = json.load(f)
        return d

    @staticmethod
    def arg_parse() -> tuple:
        exe = sys.executable or "python"
        parser = argparse.ArgumentParser()
        parser.add_argument("--port", type=int, default=7865, help="Listen port")
        parser.add_argument("--pycmd", type=str, default=exe, help="Python command")
        parser.add_argument("--colab", action="store_true", help="Launch in colab")
        parser.add_argument(
            "--noparallel", action="store_true", help="Disable parallel processing"
        )
        parser.add_argument(
            "--noautoopen",
            action="store_true",
            help="Do not open in browser automatically",
        )
        parser.add_argument(
            "--dml",
            action="store_true",
            help="torch_dml",
        )
        cmd_opts = parser.parse_args()

        cmd_opts.port = cmd_opts.port if 0 <= cmd_opts.port <= 65535 else 7865

        return (
            cmd_opts.pycmd,
            cmd_opts.port,
            cmd_opts.colab,
            cmd_opts.noparallel,
            cmd_opts.noautoopen,
            cmd_opts.dml,
        )

    @staticmethod
    def has_mps() -> bool:
        if not torch.backends.mps.is_available():
            return False
        try:
            torch.zeros(1).to(torch.device("mps"))
            return True
        except Exception:
            return False

    @staticmethod
    def has_xpu() -> bool:
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return True
        else:
            return False

    def use_fp32_config(self):
        for config_file in version_config_list:
            if config_file not in self.json_config:
                continue
            self.json_config[config_file]["train"]["fp16_run"] = False
            target_p = f"configs/inuse/{config_file}"
            with open(target_p, "r", encoding="utf-8") as f:
                strr = f.read().replace("true", "false")
            with open(target_p, "w", encoding="utf-8") as f:
                f.write(strr)
            logger.info("เขียนทับไฟล์การตั้งค่า: " + config_file)
        self.preprocess_per = 3.0
        logger.info("เขียนทับค่าอัตราส่วนการประมวลผลล่วงหน้าเป็น %d" % (self.preprocess_per))

    def device_config(self) -> tuple:
        if torch.cuda.is_available():
            if self.has_xpu():
                self.device = self.instead = "xpu:0"
                self.is_half = True
            i_device = int(self.device.split(":")[-1])
            self.gpu_name = torch.cuda.get_device_name(i_device)
            if (
                ("16" in self.gpu_name and "V100" not in self.gpu_name.upper())
                or "P40" in self.gpu_name.upper()
                or "P10" in self.gpu_name.upper()
                or "1060" in self.gpu_name
                or "1070" in self.gpu_name
                or "1080" in self.gpu_name
            ):
                logger.info("ตรวจพบการ์ดจอ %s, บังคับใช้การคำนวณแบบ fp32", self.gpu_name)
                self.is_half = False
                self.use_fp32_config()
            else:
                logger.info("ตรวจพบการ์ดจอ %s", self.gpu_name)
            self.gpu_mem = int(
                torch.cuda.get_device_properties(i_device).total_memory
                / 1024
                / 1024
                / 1024
                + 0.4
            )
            if self.gpu_mem <= 4:
                self.preprocess_per = 3.0
        elif self.has_mps():
            logger.info("ไม่พบการ์ดจอ Nvidia ที่รองรับการทำงาน")
            self.device = self.instead = "mps"
            self.is_half = False
            self.use_fp32_config()
        else:
            logger.info("ไม่พบการ์ดจอ Nvidia ที่รองรับการทำงาน")
            self.device = self.instead = "cpu"
            self.is_half = False
            self.use_fp32_config()

        if self.n_cpu == 0:
            self.n_cpu = cpu_count()

        if self.is_half:
            # การตั้งค่าสำหรับ VRAM 6G ขึ้นไป
            x_pad = 3
            x_query = 10
            x_center = 60
            x_max = 65
        else:
            # การตั้งค่าสำหรับ VRAM 5G
            x_pad = 1
            x_query = 6
            x_center = 38
            x_max = 41

        if self.gpu_mem is not None and self.gpu_mem <= 4:
            x_pad = 1
            x_query = 5
            x_center = 30
            x_max = 32
            
        if not self.dml and not torch.cuda.is_available() and not self.has_mps():
            try:
                import torch_directml
                self.dml = True
                logger.info("ตรวจพบระบบรองรับการประมวลผล (Auto-detected GPU support), เปิดใช้งาน DirectML อัตโนมัติสำหรับ AMD/Intel")
            except Exception:
                pass

        if self.dml:
            logger.info("สลับไปใช้งานระบบ DirectML แทน")
            if (
                os.path.exists(
                    r"runtime\Lib\site-packages\onnxruntime\capi\DirectML.dll"
                )
                == False
            ):
                try:
                    os.rename(
                        r"runtime\Lib\site-packages\onnxruntime",
                        r"runtime\Lib\site-packages\onnxruntime-cuda",
                    )
                except:
                    pass
                try:
                    os.rename(
                        r"runtime\Lib\site-packages\onnxruntime-dml",
                        r"runtime\Lib\site-packages\onnxruntime",
                    )
                except:
                    pass
            import torch_directml

            self.device = torch_directml.device(torch_directml.default_device())
            self.is_half = False
        else:
            if self.instead:
                logger.info(f"สลับไปใช้งาน {self.instead} แทน")
            if (
                os.path.exists(
                    r"runtime\Lib\site-packages\onnxruntime\capi\onnxruntime_providers_cuda.dll"
                )
                == False
            ):
                try:
                    os.rename(
                        r"runtime\Lib\site-packages\onnxruntime",
                        r"runtime\Lib\site-packages\onnxruntime-dml",
                    )
                except:
                    pass
                try:
                    os.rename(
                        r"runtime\Lib\site-packages\onnxruntime-cuda",
                        r"runtime\Lib\site-packages\onnxruntime",
                    )
                except:
                    pass
                    
        logger.info(
            "ระบบทศนิยมแบบ Half-precision: %s, อุปกรณ์ที่ทำงานอยู่: %s"
            % (self.is_half, self.device)
        )
        return x_pad, x_query, x_center, x_max