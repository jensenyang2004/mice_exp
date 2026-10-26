import enum

from .attention_processor_base import Flux2AttnProcessor, Flux2ParallelSelfAttnProcessor
from .attention_processor_APITASM_kernel_nonlap import Flux2APITASMAttnProcessorKernelNonLap, Flux2ParallelSelfAttnProcessorAPITASMKernelNonLap


class AttentionSetting(enum.Enum):
    FULL = 'full'
    APITASMkernelNonLap = 'apitasmkernelnonlap'
    APITASMkernelNonLapStrict = 'apitasmkernelnonlapstrict'
    APITA = 'apita'  # IDAttn baseline: binary hard mask, no kernel falloff


def get_attention_processors(attention_setting: AttentionSetting, sigma_scale: float = 0.6, kernel_size: int = 11, temperature: float = 3.0):
    if attention_setting == AttentionSetting.FULL:
        return Flux2AttnProcessor(), Flux2ParallelSelfAttnProcessor()
    elif attention_setting == AttentionSetting.APITASMkernelNonLap:
        return Flux2APITASMAttnProcessorKernelNonLap(kernel_size=kernel_size, temperature=temperature), Flux2ParallelSelfAttnProcessorAPITASMKernelNonLap(kernel_size=kernel_size, temperature=temperature)
    elif attention_setting == AttentionSetting.APITASMkernelNonLapStrict:
        return Flux2APITASMAttnProcessorKernelNonLap(kernel_size=kernel_size, temperature=temperature, strict=True), Flux2ParallelSelfAttnProcessorAPITASMKernelNonLap(kernel_size=kernel_size, temperature=temperature, strict=True)
    elif attention_setting == AttentionSetting.APITA:
        return Flux2APITASMAttnProcessorKernelNonLap(binary=True), Flux2ParallelSelfAttnProcessorAPITASMKernelNonLap(binary=True)
    else:
        raise NotImplementedError(f"Attention setting {attention_setting.value} is not supported")
