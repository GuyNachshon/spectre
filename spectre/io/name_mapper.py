"""Role-based parameter mapping with rulesets."""

import re
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml


class ParameterRole(Enum):
    """Parameter role enumeration."""
    ATTN_IN = "ATTN_IN"  # QKV input projections
    ATTN_OUT = "ATTN_OUT"  # Attention output projections
    MLP_FC = "MLP_FC"  # Feed-forward W1 (or gate/up)
    MLP_OUT = "MLP_OUT"  # Feed-forward W2 (or down)
    LN_GAMMA = "LN_GAMMA"  # LayerNorm γ
    LN_BETA = "LN_BETA"  # LayerNorm β
    EMB_TOK = "EMB_TOK"  # Token embedding
    EMB_POS = "EMB_POS"  # Positional embedding
    FINAL_NORM = "FINAL_NORM"  # Last normalization
    OTHER = "OTHER"  # Everything else


class NameMapper:
    """Maps parameter names to roles based on rulesets."""
    
    def __init__(self, ruleset: Optional[Dict] = None, ruleset_path: Optional[Path] = None):
        """
        Initialize name mapper with ruleset.
        
        Args:
            ruleset: Dictionary with ruleset configuration
            ruleset_path: Path to YAML ruleset file
        """
        if ruleset_path:
            with open(ruleset_path, "r") as f:
                self.ruleset = yaml.safe_load(f)
        elif ruleset:
            self.ruleset = ruleset
        else:
            # Default to GPT-like ruleset
            self.ruleset = self._get_default_ruleset()
        
        self._compile_patterns()
    
    def _get_default_ruleset(self) -> Dict:
        """Get default GPT-like ruleset."""
        return {
            "name": "gpt_like",
            "patterns": {
                "ATTN_IN": [
                    r"\.attn\.c_attn\.(weight|bias)",
                    r"\.attn\.qkv\.(weight|bias)",
                    r"\.attention\.query_key_value\.(weight|bias)",
                ],
                "ATTN_OUT": [
                    r"\.attn\.c_proj\.(weight|bias)",
                    r"\.attn\.out_proj\.(weight|bias)",
                    r"\.attention\.dense\.(weight|bias)",
                ],
                "MLP_FC": [
                    r"\.mlp\.c_fc\.(weight|bias)",
                    r"\.mlp\.gate_proj\.(weight|bias)",
                    r"\.mlp\.up_proj\.(weight|bias)",
                    r"\.feed_forward\.dense_h_to_4h\.(weight|bias)",
                ],
                "MLP_OUT": [
                    r"\.mlp\.c_proj\.(weight|bias)",
                    r"\.mlp\.down_proj\.(weight|bias)",
                    r"\.feed_forward\.dense_4h_to_h\.(weight|bias)",
                ],
                "LN_GAMMA": [
                    r"\.ln_[12]\.(weight|gamma)",
                    r"\.layer_norm\.(weight|gamma)",
                    r"\.input_layernorm\.(weight|gamma)",
                    r"\.post_attention_layernorm\.(weight|gamma)",
                ],
                "LN_BETA": [
                    r"\.ln_[12]\.bias",
                    r"\.layer_norm\.bias",
                    r"\.input_layernorm\.bias",
                    r"\.post_attention_layernorm\.bias",
                ],
                "EMB_TOK": [
                    r"\.wte\.(weight|embedding)",
                    r"\.token_embeddings\.(weight|embedding)",
                    r"\.embed_tokens\.(weight|embedding)",
                ],
                "EMB_POS": [
                    r"\.wpe\.(weight|embedding)",
                    r"\.position_embeddings\.(weight|embedding)",
                    r"\.embed_positions\.(weight|embedding)",
                ],
                "FINAL_NORM": [
                    r"\.ln_f\.(weight|gamma)",
                    r"\.final_layer_norm\.(weight|gamma)",
                    r"\.layer_norm\.(weight|gamma)$",  # Final layer norm
                ],
            },
        }
    
    def _compile_patterns(self):
        """Compile regex patterns for faster matching."""
        self.compiled_patterns = {}
        for role_name, patterns in self.ruleset.get("patterns", {}).items():
            try:
                role = ParameterRole[role_name]
            except KeyError:
                continue
            
            self.compiled_patterns[role] = [
                re.compile(pattern) for pattern in patterns
            ]
    
    def map_name(
        self, 
        name: str, 
        shape: Tuple[int, ...], 
        layer_idx: Optional[int] = None
    ) -> ParameterRole:
        """
        Map parameter name to role.
        
        Args:
            name: Parameter name
            shape: Parameter shape
            layer_idx: Optional layer index for heuristics
            
        Returns:
            ParameterRole enum value
        """
        # Try pattern matching first
        for role, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                if pattern.search(name):
                    return role
        
        # Fall back to shape-based heuristics
        return self._infer_role_from_shape(name, shape, layer_idx)
    
    def _infer_role_from_shape(
        self, 
        name: str, 
        shape: Tuple[int, ...], 
        layer_idx: Optional[int] = None
    ) -> ParameterRole:
        """
        Infer role from shape and position heuristics.
        
        Args:
            name: Parameter name
            shape: Parameter shape
            layer_idx: Optional layer index
            
        Returns:
            Inferred ParameterRole
        """
        # Embedding heuristics
        if len(shape) == 2 and "embed" in name.lower():
            if "pos" in name.lower():
                return ParameterRole.EMB_POS
            return ParameterRole.EMB_TOK
        
        # LayerNorm heuristics (1D)
        if len(shape) == 1:
            if "norm" in name.lower() or "ln" in name.lower():
                if "gamma" in name.lower() or "weight" in name.lower():
                    return ParameterRole.LN_GAMMA
                return ParameterRole.LN_BETA
            return ParameterRole.OTHER
        
        # 2D matrix heuristics
        if len(shape) == 2:
            # Large matrices are likely attention or MLP
            if shape[0] > 1000 or shape[1] > 1000:
                if "attn" in name.lower() or "attention" in name.lower():
                    if "out" in name.lower() or "proj" in name.lower():
                        return ParameterRole.ATTN_OUT
                    return ParameterRole.ATTN_IN
                if "mlp" in name.lower() or "ffn" in name.lower():
                    if "out" in name.lower() or "down" in name.lower():
                        return ParameterRole.MLP_OUT
                    return ParameterRole.MLP_FC
        
        return ParameterRole.OTHER
    
    def get_layer_index(self, name: str) -> Optional[int]:
        """
        Extract layer index from parameter name.
        
        Args:
            name: Parameter name
            
        Returns:
            Layer index or None if not found
        """
        # Common patterns: .h.0., .layers.0., .transformer.h.0.
        patterns = [
            r"\.h\.(\d+)\.",
            r"\.layers\.(\d+)\.",
            r"\.transformer\.h\.(\d+)\.",
            r"\.encoder\.layer\.(\d+)\.",
            r"\[(\d+)\]",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, name)
            if match:
                return int(match.group(1))
        
        return None


# Predefined rulesets
RULESETS = {
    "gpt_like": {
        "name": "gpt_like",
        "patterns": {
            "ATTN_IN": [
                r"\.attn\.c_attn\.(weight|bias)",
                r"\.attention\.query_key_value\.(weight|bias)",
            ],
            "ATTN_OUT": [
                r"\.attn\.c_proj\.(weight|bias)",
                r"\.attention\.dense\.(weight|bias)",
            ],
            "MLP_FC": [
                r"\.mlp\.c_fc\.(weight|bias)",
                r"\.feed_forward\.dense_h_to_4h\.(weight|bias)",
            ],
            "MLP_OUT": [
                r"\.mlp\.c_proj\.(weight|bias)",
                r"\.feed_forward\.dense_4h_to_h\.(weight|bias)",
            ],
            "LN_GAMMA": [
                r"\.ln_[12]\.(weight|gamma)",
                r"\.input_layernorm\.(weight|gamma)",
            ],
            "LN_BETA": [
                r"\.ln_[12]\.bias",
                r"\.input_layernorm\.bias",
            ],
            "EMB_TOK": [
                r"\.wte\.(weight|embedding)",
                r"\.token_embeddings\.(weight|embedding)",
            ],
            "EMB_POS": [
                r"\.wpe\.(weight|embedding)",
                r"\.position_embeddings\.(weight|embedding)",
            ],
            "FINAL_NORM": [
                r"\.ln_f\.(weight|gamma)",
                r"\.final_layer_norm\.(weight|gamma)",
            ],
        },
    },
    "llama_like": {
        "name": "llama_like",
        "patterns": {
            "ATTN_IN": [
                r"\.self_attn\.q_proj\.(weight|bias)",
                r"\.self_attn\.k_proj\.(weight|bias)",
                r"\.self_attn\.v_proj\.(weight|bias)",
            ],
            "ATTN_OUT": [
                r"\.self_attn\.o_proj\.(weight|bias)",
            ],
            "MLP_FC": [
                r"\.mlp\.gate_proj\.(weight|bias)",
                r"\.mlp\.up_proj\.(weight|bias)",
            ],
            "MLP_OUT": [
                r"\.mlp\.down_proj\.(weight|bias)",
            ],
            "LN_GAMMA": [
                r"\.input_layernorm\.(weight|gamma)",
                r"\.post_attention_layernorm\.(weight|gamma)",
            ],
            "LN_BETA": [
                r"\.input_layernorm\.bias",
                r"\.post_attention_layernorm\.bias",
            ],
            "EMB_TOK": [
                r"\.embed_tokens\.(weight|embedding)",
            ],
            "FINAL_NORM": [
                r"\.norm\.(weight|gamma)",
            ],
        },
    },
    "mistral_like": {
        "name": "mistral_like",
        "patterns": {
            "ATTN_IN": [
                r"\.self_attn\.q_proj\.(weight|bias)",
                r"\.self_attn\.k_proj\.(weight|bias)",
                r"\.self_attn\.v_proj\.(weight|bias)",
            ],
            "ATTN_OUT": [
                r"\.self_attn\.o_proj\.(weight|bias)",
            ],
            "MLP_FC": [
                r"\.mlp\.gate_proj\.(weight|bias)",
                r"\.mlp\.up_proj\.(weight|bias)",
            ],
            "MLP_OUT": [
                r"\.mlp\.down_proj\.(weight|bias)",
            ],
            "LN_GAMMA": [
                r"\.input_layernorm\.(weight|gamma)",
                r"\.post_attention_layernorm\.(weight|gamma)",
            ],
            "EMB_TOK": [
                r"\.embed_tokens\.(weight|embedding)",
            ],
            "FINAL_NORM": [
                r"\.norm\.(weight|gamma)",
            ],
        },
    },
    "phi_like": {
        "name": "phi_like",
        "patterns": {
            "ATTN_IN": [
                r"\.mixer\.Wqkv\.(weight|bias)",
            ],
            "ATTN_OUT": [
                r"\.mixer\.out_proj\.(weight|bias)",
            ],
            "MLP_FC": [
                r"\.mlp\.fc1\.(weight|bias)",
            ],
            "MLP_OUT": [
                r"\.mlp\.fc2\.(weight|bias)",
            ],
            "LN_GAMMA": [
                r"\.ln\.(weight|gamma)",
            ],
            "EMB_TOK": [
                r"\.embd\.wte\.(weight|embedding)",
            ],
            "FINAL_NORM": [
                r"\.final_ln\.(weight|gamma)",
            ],
        },
    },
}


def get_ruleset(name: str) -> Dict:
    """
    Get predefined ruleset by name.
    
    Args:
        name: Ruleset name (gpt_like, llama_like, mistral_like, phi_like)
        
    Returns:
        Ruleset dictionary
    """
    if name not in RULESETS:
        raise ValueError(f"Unknown ruleset: {name}. Available: {list(RULESETS.keys())}")
    return RULESETS[name]

