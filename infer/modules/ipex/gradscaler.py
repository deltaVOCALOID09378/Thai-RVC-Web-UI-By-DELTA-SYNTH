# Version 1.1
# Made And Checked By DELTA SYNTH & Gemini AI -> Original IPEX Contributors

from collections import defaultdict
import torch
import intel_extension_for_pytorch as ipex  # pylint: disable=import-error, unused-import
import intel_extension_for_pytorch._C as core  # pylint: disable=import-error, unused-import

# pylint: disable=protected-access, missing-function-docstring, line-too-long

OptState = ipex.cpu.autocast._grad_scaler.OptState
_MultiDeviceReplicator = ipex.cpu.autocast._grad_scaler._MultiDeviceReplicator
_refresh_per_optimizer_state = (
    ipex.cpu.autocast._grad_scaler._refresh_per_optimizer_state
)


def _unscale_grads_(
    self, optimizer, inv_scale, found_inf, allow_fp16
):  # pylint: disable=unused-argument
    """
    ฟังก์ชันภายในสำหรับทำการหาร (Unscale) ค่า Gradient ของพารามิเตอร์ทั้งหมด
    พร้อมทั้งตรวจสอบหาค่าอนันต์ (Inf) หรือค่าที่ไม่ใช่ตัวเลข (NaN)
    """
    per_device_inv_scale = _MultiDeviceReplicator(inv_scale)
    per_device_found_inf = _MultiDeviceReplicator(found_inf)

    # เพื่อเตรียมการสำหรับ _amp_foreach_non_finite_check_and_unscale_ เราจะแยก Gradient ตามอุปกรณ์และชนิดข้อมูล (Dtype)
    # เนื่องจากอาจมี Gradient นับร้อยรายการ เราจึงต้องการวนลูปตรวจสอบเพียงครั้งเดียว
    per_device_and_dtype_grads = defaultdict(lambda: defaultdict(list))  # type: ignore[var-annotated]
    
    # ซิงค์ Gradient เข้ากับ Master Weight หากมีการกำหนดไว้
    if hasattr(optimizer, "sync_grad"):
        optimizer.sync_grad()
        
    with torch.no_grad():
        for group in optimizer.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue
                if (not allow_fp16) and param.grad.dtype == torch.float16:
                    raise ValueError("ไม่อนุญาตให้ทำการ Unscale ข้อมูล Gradient ชนิด FP16")
                
                if param.grad.is_sparse:
                    # หาก is_coalesced() == False หมายความว่า Sparse Grad มีค่าดัชนี (Indices) ที่ซ้ำซาก
                    # การเรียกใช้ coalesce() จะช่วยรวมดัชนีที่ซ้ำกันและบวกค่าที่มีดัชนีเดียวกันเข้าด้วยกัน
                    # สำหรับค่า FP16 ที่ถูกสเกลแล้ว มีโอกาสสูงที่การรวมค่าจะทำให้เกิดการล้นของข้อมูล (Overflow)
                    # ดังนั้นเราจึงควรตรวจสอบที่ _values() หลังจากรวมค่าแล้ว
                    if param.grad.dtype is torch.float16:
                        param.grad = param.grad.coalesce()
                    to_unscale = param.grad._values()
                else:
                    to_unscale = param.grad

                # ย้ายข้อมูลไปยัง CPU ชั่วคราวเพื่อความเสถียรในการคำนวณของ IPEX
                to_unscale = to_unscale.to("cpu")
                per_device_and_dtype_grads[to_unscale.device][to_unscale.dtype].append(
                    to_unscale
                )

        for _, per_dtype_grads in per_device_and_dtype_grads.items():
            for grads in per_dtype_grads.values():
                core._amp_foreach_non_finite_check_and_unscale_(
                    grads,
                    per_device_found_inf.get("cpu"),
                    per_device_inv_scale.get("cpu"),
                )

    return per_device_found_inf._per_device_tensors


def unscale_(self, optimizer):
    """
    ทำการหาร (Unscale) ค่า Tensor ของ Gradient ใน Optimizer ด้วยค่า Scale Factor
    
    ฟังก์ชัน :meth:`unscale_` เป็นทางเลือกสำหรับการใช้งานในกรณีที่คุณต้องการ
    ปรับแต่งหรือตรวจสอบ Gradient ก่อนที่จะเรียกใช้ :meth:`step`
    หากไม่ได้เรียกใช้ :meth:`unscale_` อย่างชัดเจน ระบบจะทำการหาร Gradient ให้โดยอัตโนมัติระหว่างการ :meth:`step`
    
    คำเตือน:
        - ควรเรียกใช้ :meth:`unscale_` เพียงครั้งเดียวต่อหนึ่ง Optimizer ในแต่ละรอบการ :meth:`step`
          และการเรียกซ้ำจะทำให้เกิดข้อผิดพลาด RuntimeError
    """
    if not self._enabled:
        return

    self._check_scale_growth_tracker("unscale_")

    optimizer_state = self._per_optimizer_states[id(optimizer)]

    if optimizer_state["stage"] is OptState.UNSCALED:  # pylint: disable=no-else-raise
        raise RuntimeError(
            "มีการเรียกใช้คำสั่ง unscale_() กับ Optimizer นี้ไปแล้วตั้งแต่การ update() ครั้งล่าสุด"
        )
    elif optimizer_state["stage"] is OptState.STEPPED:
        raise RuntimeError("ไม่สามารถเรียกใช้คำสั่ง unscale_() หลังจากเรียกใช้ step() ไปแล้วได้")

    # การหารด้วย FP32 อาจมีความคลาดเคลื่อนในบางการตั้งค่าคอมไพล์ เราจึงใช้การหาค่าส่วนกลับ (Reciprocal) ในรูปแบบ FP64 แทน
    assert self._scale is not None
    inv_scale = (
        self._scale.to("cpu").double().reciprocal().float().to(self._scale.device)
    )
    found_inf = torch.full((1,), 0.0, dtype=torch.float32, device=self._scale.device)

    optimizer_state["found_inf_per_device"] = self._unscale_grads_(
        optimizer, inv_scale, found_inf, False
    )
    optimizer_state["stage"] = OptState.UNSCALED


def update(self, new_scale=None):
    """
    อัปเดตค่า Scale Factor สำหรับรอบการทำงานถัดไป
    
    หากมีการข้ามขั้นตอน (Skipped Step) ของ Optimizer ค่า Scale จะถูกคูณด้วย ``backoff_factor`` เพื่อลดทอนขนาดลง
    แต่หากการทำงานดำเนินไปอย่างราบรื่นติดต่อกันตามจำนวน ``growth_interval`` ค่า Scale จะถูกคูณด้วย ``growth_factor`` เพื่อเพิ่มขนาดขึ้น
    """
    if not self._enabled:
        return

    _scale, _growth_tracker = self._check_scale_growth_tracker("update")

    if new_scale is not None:
        # ยอมรับค่า Scale ใหม่ที่ผู้ใช้กำหนดเอง
        if isinstance(new_scale, float):
            self._scale.fill_(new_scale)  # type: ignore[union-attr]
        else:
            reason = "new_scale จะต้องเป็นตัวเลขทศนิยม (Float) หรือ torch.FloatTensor ขนาด 1 อีลีเมนต์ ที่มี requires_grad=False"
            assert isinstance(new_scale, torch.FloatTensor), reason  # type: ignore[attr-defined]
            assert new_scale.numel() == 1, reason
            assert new_scale.requires_grad is False, reason
            self._scale.copy_(new_scale)  # type: ignore[union-attr]
    else:
        # รวบรวมข้อมูล Inf/NaN ที่ได้จาก Optimizers เพื่อนำมาอัปเดตค่า Scale
        found_infs = [
            found_inf.to(device="cpu", non_blocking=True)
            for state in self._per_optimizer_states.values()
            for found_inf in state["found_inf_per_device"].values()
        ]

        assert len(found_infs) > 0, "ไม่พบการบันทึกข้อมูลการตรวจสอบ Inf ก่อนที่จะทำการอัปเดต"

        found_inf_combined = found_infs[0]
        if len(found_infs) > 1:
            for i in range(1, len(found_infs)):
                found_inf_combined += found_infs[i]

        to_device = _scale.device
        # ย้ายข้อมูลมาประมวลผลบน CPU ชั่วคราวเพื่อความแม่นยำและหลีกเลี่ยงข้อผิดพลาดจากฮาร์ดแวร์
        _scale = _scale.to("cpu")
        _growth_tracker = _growth_tracker.to("cpu")

        core._amp_update_scale_(
            _scale,
            _growth_tracker,
            found_inf_combined,
            self._growth_factor,
            self._backoff_factor,
            self._growth_interval,
        )

        _scale = _scale.to(to_device)
        _growth_tracker = _growth_tracker.to(to_device)
        
    # ล้างข้อมูลสถานะที่เก็บรวบรวมมาจาก Optimizers ในรอบการทำงานนี้ เพื่อเตรียมพร้อมสำหรับรอบถัดไป
    self._per_optimizer_states = defaultdict(_refresh_per_optimizer_state)


def gradscaler_init():
    """
    ฟังก์ชันสำหรับเริ่มต้นและติดตั้ง GradScaler ที่ปรับแต่งแล้วเข้าสู่ระบบของ PyTorch สำหรับ XPU
    """
    torch.xpu.amp.GradScaler = ipex.cpu.autocast._grad_scaler.GradScaler
    torch.xpu.amp.GradScaler._unscale_grads_ = _unscale_grads_
    torch.xpu.amp.GradScaler.unscale_ = unscale_
    torch.xpu.amp.GradScaler.update = update
    return torch.xpu.amp.GradScaler