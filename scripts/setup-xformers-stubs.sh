#!/bin/bash
# Create minimal xformers stubs so audiocraft runs on macOS (no xformers wheel).
# Run once after installing audiocraft.
VENV=${1:-.venv}
SITE="$VENV/lib/python3.11/site-packages"
mkdir -p "$SITE/xformers/ops" "$SITE/xformers/profiler" "$SITE/xformers/checkpoint_fairinternal"
cat > "$SITE/xformers/__init__.py" << 'EOF'
pass
EOF
cat > "$SITE/xformers/ops/__init__.py" << 'EOF'
"""Minimal xformers.ops stubs — fall back to native PyTorch attention."""
import torch.nn.functional as F
class LowerTriangularMask: pass
def unbind(tensor, dim): return tensor.unbind(dim)
def memory_efficient_attention(query, key, value, attn_bias=None, p=0.0, scale=None, op=None):
    if scale is None: scale = query.shape[-1] ** -0.5
    return F.scaled_dot_product_attention(query, key, value, attn_mask=attn_bias, dropout_p=p, scale=scale)
EOF
cat > "$SITE/xformers/profiler/__init__.py" << 'EOF'
class _Profiler: _CURRENT_PROFILER = None
def profile(*args, **kwargs):
    if len(args) == 1 and callable(args[0]): return args[0]
    def decorator(fn): return fn
    return decorator
class profiler: _Profiler = _Profiler
EOF
cat > "$SITE/xformers/checkpoint_fairinternal/__init__.py" << 'EOF'
import torch.utils.checkpoint
checkpoint = torch.utils.checkpoint.checkpoint
def _get_default_policy(): return None
EOF
echo "xformers stubs installed"
