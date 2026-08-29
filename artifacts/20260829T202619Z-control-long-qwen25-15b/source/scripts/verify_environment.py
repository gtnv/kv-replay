import json
import platform

import torch
import transformers

if torch.__version__.split("+")[0] != "2.7.0":
    raise RuntimeError(f"expected torch 2.7.0, found {torch.__version__}")
if transformers.__version__ != "4.57.6":
    raise RuntimeError(f"expected transformers 4.57.6, found {transformers.__version__}")
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable")
if torch.cuda.get_device_capability() != (9, 0):
    raise RuntimeError(
        f"expected H100 compute capability 9.0, found {torch.cuda.get_device_capability()}"
    )

print(
    json.dumps(
        {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "capability": list(torch.cuda.get_device_capability()),
        },
        indent=2,
        sort_keys=True,
    )
)
