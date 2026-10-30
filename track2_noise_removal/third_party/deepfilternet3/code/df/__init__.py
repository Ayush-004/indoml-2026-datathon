# [IndoML-T2 change] Reduced package init. The upstream df/__init__.py (deepfilternet 0.5.6) also imports
# df.enhance and df.version; df.enhance imports df.io, which needs torchaudio.backend.common (removed in new
# torchaudio). Inference here needs only config, model, modules, multiframe, utils and deepfilternet3.
from .config import config  # noqa: F401
