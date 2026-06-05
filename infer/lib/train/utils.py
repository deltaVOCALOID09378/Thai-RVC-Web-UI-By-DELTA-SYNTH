import argparse
import glob
import json
import logging
import os
import subprocess
import sys
import shutil

import numpy as np
import torch
from scipy.io.wavfile import read

MATPLOTLIB_FLAG = False

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging


def load_checkpoint_d(checkpoint_path, combd, sbd, optimizer=None, load_opt=1):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")

    ##################
    def go(model, bkey):
        saved_state_dict = checkpoint_dict[bkey]
        if hasattr(model, "module"):
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()
        new_state_dict = {}
        for k, v in state_dict.items():  # รูปร่าง (Shape) ที่โมเดลต้องการ
            try:
                new_state_dict[k] = saved_state_dict[k]
                if saved_state_dict[k].shape != state_dict[k].shape:
                    logger.warning(
                        "shape-%s-mismatch. need: %s, get: %s",
                        k,
                        state_dict[k].shape,
                        saved_state_dict[k].shape,
                    )
                    raise KeyError
            except:
                # logger.info(traceback.format_exc())
                logger.info("%s ไม่พบในจุดตรวจสอบ (is not in the checkpoint)", k)  # ส่วนที่ขาดหายไปใน pretrain
                new_state_dict[k] = v  # ค่าสุ่มเริ่มต้นของโมเดล (Random values)
        if hasattr(model, "module"):
            model.module.load_state_dict(new_state_dict, strict=False)
        else:
            model.load_state_dict(new_state_dict, strict=False)
        return model

    go(combd, "combd")
    model = go(sbd, "sbd")
    #############
    logger.info("โหลดค่าน้ำหนักโมเดลสำเร็จ (Loaded model weights)")

    iteration = checkpoint_dict["iteration"]
    learning_rate = checkpoint_dict["learning_rate"]
    if (
        optimizer is not None and load_opt == 1
    ):  ### หากโหลดไม่ได้ หรือเป็นค่าว่าง ให้กำหนดค่าเริ่มต้นใหม่
        optimizer.load_state_dict(checkpoint_dict["optimizer"])
        
    logger.info("โหลดจุดตรวจสอบ (Loaded checkpoint) '{}' (รอบ/epoch {})".format(checkpoint_path, iteration))
    return model, optimizer, learning_rate, iteration


def load_checkpoint(checkpoint_path, model, optimizer=None, load_opt=1):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")

    saved_state_dict = checkpoint_dict["model"]
    if hasattr(model, "module"):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    new_state_dict = {}
    for k, v in state_dict.items():  # รูปร่าง (Shape) ที่โมเดลต้องการ
        try:
            new_state_dict[k] = saved_state_dict[k]
            if saved_state_dict[k].shape != state_dict[k].shape:
                logger.warning(
                    "รูปร่างไม่ตรงกัน (shape-%s-mismatch)|ต้องการ (need)-%s|ได้รับ (get)-%s",
                    k,
                    state_dict[k].shape,
                    saved_state_dict[k].shape,
                )
                raise KeyError
        except:
            logger.info("%s ไม่พบในจุดตรวจสอบ (is not in the checkpoint)", k)  # ส่วนที่ขาดหายไปใน pretrain
            new_state_dict[k] = v  # ค่าสุ่มเริ่มต้นของโมเดล
    if hasattr(model, "module"):
        model.module.load_state_dict(new_state_dict, strict=False)
    else:
        model.load_state_dict(new_state_dict, strict=False)
    logger.info("โหลดค่าน้ำหนักโมเดลสำเร็จ (Loaded model weights)")

    iteration = checkpoint_dict["iteration"]
    learning_rate = checkpoint_dict["learning_rate"]
    if (
        optimizer is not None and load_opt == 1
    ):  
        optimizer.load_state_dict(checkpoint_dict["optimizer"])
        
    logger.info("โหลดจุดตรวจสอบ (Loaded checkpoint) '{}' (รอบ/epoch {})".format(checkpoint_path, iteration))
    return model, optimizer, learning_rate, iteration


def save_checkpoint(model, optimizer, learning_rate, iteration, checkpoint_path):
    logger.info(
        "กำลังบันทึกสถานะโมเดลและตัวเพิ่มประสิทธิภาพที่รอบ (Saving model and optimizer state at epoch) {} ไปยัง {}".format(
            iteration, checkpoint_path
        )
    )
    if hasattr(model, "module"):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    torch.save(
        {
            "model": state_dict,
            "iteration": iteration,
            "optimizer": optimizer.state_dict(),
            "learning_rate": learning_rate,
        },
        checkpoint_path,
    )


def save_checkpoint_d(combd, sbd, optimizer, learning_rate, iteration, checkpoint_path):
    logger.info(
        "กำลังบันทึกสถานะโมเดลและตัวเพิ่มประสิทธิภาพที่รอบ (Saving model and optimizer state at epoch) {} ไปยัง {}".format(
            iteration, checkpoint_path
        )
    )
    if hasattr(combd, "module"):
        state_dict_combd = combd.module.state_dict()
    else:
        state_dict_combd = combd.state_dict()
    if hasattr(sbd, "module"):
        state_dict_sbd = sbd.module.state_dict()
    else:
        state_dict_sbd = sbd.state_dict()
    torch.save(
        {
            "combd": state_dict_combd,
            "sbd": state_dict_sbd,
            "iteration": iteration,
            "optimizer": optimizer.state_dict(),
            "learning_rate": learning_rate,
        },
        checkpoint_path,
    )


def summarize(
    writer,
    global_step,
    scalars={},
    histograms={},
    images={},
    audios={},
    audio_sampling_rate=22050,
):
    for k, v in scalars.items():
        writer.add_scalar(k, v, global_step)
    for k, v in histograms.items():
        writer.add_histogram(k, v, global_step)
    for k, v in images.items():
        writer.add_image(k, v, global_step, dataformats="HWC")
    for k, v in audios.items():
        writer.add_audio(k, v, global_step, audio_sampling_rate)


def latest_checkpoint_path(dir_path, regex="G_*.pth"):
    f_list = glob.glob(os.path.join(dir_path, regex))
    f_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
    x = f_list[-1]
    logger.debug(x)
    return x


def plot_spectrogram_to_numpy(spectrogram):
    global MATPLOTLIB_FLAG
    if not MATPLOTLIB_FLAG:
        import matplotlib

        matplotlib.use("Agg")
        MATPLOTLIB_FLAG = True
        mpl_logger = logging.getLogger("matplotlib")
        mpl_logger.setLevel(logging.WARNING)
    import matplotlib.pylab as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data


def plot_alignment_to_numpy(alignment, info=None):
    global MATPLOTLIB_FLAG
    if not MATPLOTLIB_FLAG:
        import matplotlib

        matplotlib.use("Agg")
        MATPLOTLIB_FLAG = True
        mpl_logger = logging.getLogger("matplotlib")
        mpl_logger.setLevel(logging.WARNING)
    import matplotlib.pylab as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(
        alignment.transpose(), aspect="auto", origin="lower", interpolation="none"
    )
    fig.colorbar(im, ax=ax)
    xlabel = "Decoder timestep"
    if info is not None:
        xlabel += "\n\n" + info
    plt.xlabel(xlabel)
    plt.ylabel("Encoder timestep")
    plt.tight_layout()

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data


def load_wav_to_torch(full_path):
    sampling_rate, data = read(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def load_filepaths_and_text(filename, split="|"):
    with open(filename, encoding="utf-8") as f:
        filepaths_and_text = [line.strip().split(split) for line in f]
    return filepaths_and_text


def get_hparams(init=True):
    """
    รายการตั้งค่า (Parameters Configuration)
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-se",
        "--save_every_epoch",
        type=int,
        required=True,
        help="ความถี่ในการบันทึกจุดตรวจสอบ (checkpoint save frequency in epoch)",
    )
    parser.add_argument(
        "-te", "--total_epoch", type=int, required=True, help="จำนวนรอบทั้งหมด (total_epoch)"
    )
    parser.add_argument(
        "-pg", "--pretrainG", type=str, default="", help="เส้นทางโมเดลสร้างเสียงพื้นฐาน (Pretrained Generator path)"
    )
    parser.add_argument(
        "-pd", "--pretrainD", type=str, default="", help="เส้นทางโมเดลตรวจสอบพื้นฐาน (Pretrained Discriminator path)"
    )
    parser.add_argument("-g", "--gpus", type=str, default="0", help="รหัส GPU คั่นด้วย - (split by -)")
    parser.add_argument(
        "-bs", "--batch_size", type=int, required=True, help="ขนาดของแต่ละชุดข้อมูล (batch size)"
    )
    parser.add_argument(
        "-e", "--experiment_dir", type=str, required=True, help="โฟลเดอร์ทดลอง (experiment dir)"
    )
    parser.add_argument(
        "-sr", "--sample_rate", type=str, required=True, help="อัตราสุ่มตัวอย่าง (sample rate, 32k/40k/48k)"
    )
    parser.add_argument(
        "-sw",
        "--save_every_weights",
        type=str,
        default="0",
        help="บันทึกไฟล์น้ำหนักโมเดลขณะบันทึกจุดตรวจสอบ (save the extracted model in weights directory when saving checkpoints)",
    )
    parser.add_argument(
        "-v", "--version", type=str, required=True, help="เวอร์ชันของโมเดล (model version)"
    )
    parser.add_argument(
        "-f0",
        "--if_f0",
        type=int,
        required=True,
        help="ใช้ F0 เป็นหนึ่งในข้อมูลป้อนเข้าโมเดลหรือไม่ 1 หรือ 0 (use f0 as one of the inputs of the model)",
    )
    parser.add_argument(
        "-l",
        "--if_latest",
        type=int,
        required=True,
        help="บันทึกเฉพาะไฟล์ G/D ล่าสุดหรือไม่ 1 หรือ 0 (if only save the latest G/D pth file)",
    )
    parser.add_argument(
        "-c",
        "--if_cache_data_in_gpu",
        type=int,
        required=True,
        help="แคชข้อมูลชุดฝึกลงในหน่วยความจำ GPU หรือไม่ 1 หรือ 0 (if caching the dataset in GPU memory)",
    )

    args = parser.parse_args()
    name = args.experiment_dir
    experiment_dir = os.path.join("./logs", args.experiment_dir)

    config_save_path = os.path.join(experiment_dir, "config.json")
    # เพิ่ม encoding="utf-8" เพื่อป้องกันปัญหาไฟล์ JSON มีตัวอักษรพิเศษ
    with open(config_save_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    hparams = HParams(**config)
    hparams.model_dir = hparams.experiment_dir = experiment_dir
    hparams.save_every_epoch = args.save_every_epoch
    hparams.name = name
    hparams.total_epoch = args.total_epoch
    hparams.pretrainG = args.pretrainG
    hparams.pretrainD = args.pretrainD
    hparams.version = args.version
    hparams.gpus = args.gpus
    hparams.train.batch_size = args.batch_size
    hparams.sample_rate = args.sample_rate
    hparams.if_f0 = args.if_f0
    hparams.if_latest = args.if_latest
    hparams.save_every_weights = args.save_every_weights
    hparams.if_cache_data_in_gpu = args.if_cache_data_in_gpu
    hparams.data.training_files = "%s/filelist.txt" % experiment_dir
    return hparams


def get_hparams_from_dir(model_dir):
    config_save_path = os.path.join(model_dir, "config.json")
    with open(config_save_path, "r", encoding="utf-8") as f:
        data = f.read()
    config = json.loads(data)

    hparams = HParams(**config)
    hparams.model_dir = model_dir
    return hparams


def get_hparams_from_file(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        data = f.read()
    config = json.loads(data)

    hparams = HParams(**config)
    return hparams


def check_git_hash(model_dir):
    source_dir = os.path.dirname(os.path.realpath(__file__))
    if not os.path.exists(os.path.join(source_dir, ".git")):
        logger.warning(
            "{} ไม่ใช่ที่เก็บ git ระบบจะข้ามการตรวจสอบค่าแฮช (is not a git repository, therefore hash value comparison will be ignored.)".format(
                source_dir
            )
        )
        return

    cur_hash = subprocess.getoutput("git rev-parse HEAD")

    path = os.path.join(model_dir, "githash")
    if os.path.exists(path):
        saved_hash = open(path, "r", encoding="utf-8").read()
        if saved_hash != cur_hash:
            logger.warning(
                "ค่า git hash ไม่ตรงกัน (git hash values are different). {}(ที่บันทึกไว้/saved) != {}(ปัจจุบัน/current)".format(
                    saved_hash[:8], cur_hash[:8]
                )
            )
    else:
        open(path, "w", encoding="utf-8").write(cur_hash)


def get_logger(model_dir, filename="train.log"):
    global logger
    logger = logging.getLogger(os.path.basename(model_dir))
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
        
    # บังคับการเข้ารหัส UTF-8 เพื่อป้องกัน Error เวลาพิมพ์ภาษาไทยลงใน train.log
    h = logging.FileHandler(os.path.join(model_dir, filename), encoding="utf-8")
    h.setLevel(logging.DEBUG)
    h.setFormatter(formatter)
    logger.addHandler(h)
    
    # เพิ่ม StreamHandler ให้แสดงผลบน Console ได้อย่างถูกต้อง
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.DEBUG)
    
    if not logger.handlers:
        logger.addHandler(console_handler)
        
    return logger


class HParams:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if type(v) == dict:
                v = HParams(**v)
            self[k] = v

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()