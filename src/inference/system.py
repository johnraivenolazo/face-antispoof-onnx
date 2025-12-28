import platform
import onnxruntime as ort
from typing import Optional

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import cpuinfo

    HAS_CPUINFO = True
except ImportError:
    HAS_CPUINFO = False

try:
    import GPUtil

    HAS_GPUTIL = True
except ImportError:
    HAS_GPUTIL = False


def get_cpu_info() -> str:
    cpu_name = None
    cpu_freq_mhz = None
    cpu_cores = None
    cpu_threads = None

    if HAS_CPUINFO:
        try:
            info = cpuinfo.get_cpu_info()
            cpu_name = (
                info.get("brand_raw") or info.get("brand") or info.get("model name")
            )
            if cpu_name:
                cpu_name = cpu_name.strip()
        except Exception:
            pass

    if HAS_PSUTIL:
        try:
            cpu_freq = psutil.cpu_freq()
            cpu_cores = psutil.cpu_count(logical=False)
            cpu_threads = psutil.cpu_count(logical=True)
            if cpu_freq and cpu_freq.current:
                cpu_freq_mhz = cpu_freq.current
        except Exception:
            pass

    if not cpu_name:
        cpu_name = platform.processor() or "Unknown CPU"

    parts = []
    if cpu_name:
        parts.append(cpu_name)

    if cpu_cores and cpu_threads:
        if cpu_cores == cpu_threads:
            parts.append(f"{cpu_cores} cores")
        else:
            parts.append(f"{cpu_cores}C/{cpu_threads}T")

    if cpu_freq_mhz:
        if cpu_freq_mhz >= 1000:
            parts.append(f"{cpu_freq_mhz/1000:.2f} GHz")
        else:
            parts.append(f"{cpu_freq_mhz:.0f} MHz")

    return " | ".join(parts) if parts else "Unknown CPU"


def get_gpu_info() -> Optional[str]:
    if HAS_GPUTIL:
        try:
            gpus = GPUtil.getGPUs()
            if gpus and len(gpus) > 0:
                gpu = gpus[0]
                return gpu.name.strip()
        except Exception:
            pass

    try:
        import subprocess

        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().split("\n")[0]
    except Exception:
        pass

    return None


def get_execution_provider_name(session: ort.InferenceSession) -> str:
    try:
        providers = session.get_providers()
        if "CUDAExecutionProvider" in providers:
            return "CUDA"
        elif "CPUExecutionProvider" in providers:
            return "CPU"
        elif providers:
            return providers[0].replace("ExecutionProvider", "")
        return "Unknown"
    except Exception:
        return "Unknown"
