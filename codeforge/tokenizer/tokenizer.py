"""Public re-export of the tokenizer under the CodeForge-facing name.

All model, data-pipeline, and inference code imports::

    from codeforge.tokenizer.tokenizer import CodeForgeTokenizer

This module aliases :class:`TCFPTokenizer` so that the internal TCFP
implementation detail stays encapsulated while the rest of the codebase
uses the project-level name.
"""

from .wrapper import TCFPTokenizer

# Public alias used across the codebase
CodeForgeTokenizer = TCFPTokenizer

__all__ = ["CodeForgeTokenizer"]
