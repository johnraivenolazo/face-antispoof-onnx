import onnxruntime as ort
from typing import Tuple, Optional
from pathlib import Path


def load_model(model_path: str) -> Tuple[Optional[ort.InferenceSession], Optional[str]]:
    if not Path(model_path).exists():
        return None, None

    try:
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        available_providers = ort.get_available_providers()
        preferred_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        providers = [p for p in preferred_providers if p in available_providers]

        if not providers:
            providers = available_providers

        ort_session = ort.InferenceSession(
            model_path, sess_options=sess_options, providers=providers
        )
        input_name = ort_session.get_inputs()[0].name
        return ort_session, input_name
    except Exception:
        return None, None
