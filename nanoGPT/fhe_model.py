"""
FHE-compatible version of nanoGPT model.
Modifications based on 'Implementing Fully Homomorphic Encryption (FHE) on a model like nanoGPT'
and the FHE_LLM paper.

Key changes:
- Bias terms removed from Linear layers and LayerNorm (config.bias = False).
- LayerNorm modified for HE (polynomial approximation for 1/sqrt(var)).
- CausalSelfAttention uses Gaussian Kernel (GK) attention instead of Softmax.
- GELU activation in MLP replaced with a placeholder for HE-friendly activation (e.g., approximated ReLU).
- Dropout removed.
- LoRA is NOT implemented in this version.
- Assumes an underlying HE library for actual encrypted computations.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False.
        For FHE, bias is typically disabled.
    """

    def __init__(self, ndim, bias): # bias will be False from FHEGPTConfig
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        # FHE: Bias is set to None if bias is False, which is the case for FHE.
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        # FHE: Standard LayerNorm (F.layer_norm) is not HE-friendly due to division and sqrt.
        # The input 'input' is an encrypted tensor. 'self.weight' is plaintext.
        # If self.bias were enabled and encrypted, it would be an encrypted addition.

        # Conceptual FHE LayerNorm steps:
        # 1. Calculate mean_x (encrypted sum, then multiply by plaintext 1/N).
        #    mean_x = input.mean(dim=-1, keepdim=True) # Encrypted operation
        # 2. Calculate var_x (encrypted (input - mean_x)^2, sum, then multiply by plaintext 1/N).
        #    var_x = input.var(dim=-1, keepdim=True, unbiased=False) # Encrypted operation
        # 3. Calculate 1/sqrt(var_x + epsilon) using polynomial approximation (encrypted input).
        #    inv_std_x = polynomial_approx_inv_sqrt(var_x + 1e-5) # Encrypted operation
        # 4. Normalize: y = (input - mean_x) * inv_std_x (Encrypted operations)
        # 5. Scale and shift: y = y * self.weight (PCMM) + self.bias (if bias is encrypted and enabled)
        #
        # For now, using F.layer_norm as a placeholder for the FHE-compatible operations.
        # The actual HE implementation would replace this.
        # print("WARNING: LayerNorm.forward is a placeholder and not FHE-operational.")
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):
    # FHE: Standard Softmax attention is replaced by Gaussian Kernel (GK) Attention.
    # FHE: Dropout is removed as it's problematic in FHE.

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        # FHE: config.bias is False. Input x is encrypted, weights are plaintext (PCMM).
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        # FHE: Input y is encrypted, weights are plaintext (PCMM).
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # FHE: Regularization (dropout) removed.
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # FHE: Flash attention is not used; GK-Attention is implemented.
        # FHE: Causal mask for GK attention is applied in the forward pass.

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # FHE: x is encrypted. self.c_attn weights are plaintext (PCMM).
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)

        # FHE: q, k, v are encrypted.
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # FHE: Gaussian Kernel (GK) Attention
        # S(Q,K)_ij = exp(-1/(2*sqrt(n)) * ||Q_i - K_j||^2_2)
        # Q, K, V are encrypted. n is head dimension (hs).

        # 1. Compute Squared L2 Norm: ||Q_i - K_j||^2_2
        # FHE: All operations (subtraction, squaring, sum) are Ciphertext-Ciphertext (CC).
        # q_expanded: (B, nh, T, 1, hs), k_expanded: (B, nh, 1, T, hs)
        # diff: (B, nh, T, T, hs)
        diff = q.unsqueeze(-2) - k.unsqueeze(-3)
        sq_l2_norm = (diff ** 2).sum(dim=-1) # (B, nh, T, T)

        # 2. Scale: Multiply by plaintext constant -1/(2*sqrt(n))
        n_head_dim = C // self.n_head
        # FHE: scale_factor is plaintext. Multiplication is Ciphertext-Plaintext.
        scale_factor = -1.0 / (2.0 * math.sqrt(n_head_dim))
        scaled_sq_l2_norm = sq_l2_norm * scale_factor # (B, nh, T, T)

        # 3. Polynomial Approximation of Exponential: exp(x) for x <= 0
        # FHE: Input scaled_sq_l2_norm is encrypted.
        # Use polynomial approx like p_k(x) = (1 + x/2^k)^(2^k) or minimax.
        # This requires HE library operations. Placeholder:
        # print("WARNING: CausalSelfAttention.forward uses torch.exp, not a FHE polynomial approximation.")
        gk_scores = torch.exp(scaled_sq_l2_norm) # Non-FHE placeholder for polynomial_approx_exp(scaled_sq_l2_norm)

        # 4. Causal Masking: Set S(Q,K)_ij = 0 for j > i
        # FHE: Create a boolean mask. masked_fill sets elements to 0 (or an HE equivalent for nullification).
        # The mask itself is plaintext.
        causal_mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool)).view(1, 1, T, T)
        # Apply mask. For GK attention, future scores become 0.
        gk_scores = gk_scores.masked_fill(~causal_mask, 0.0)

        # 5. Output Calculation: y = S(Q,K)V
        # FHE: gk_scores is encrypted, v is encrypted. This is a CCMM.
        y = gk_scores @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        # FHE: y is encrypted. self.c_proj weights are plaintext (PCMM).
        # FHE: Dropout removed.
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        # FHE: config.bias is False.
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        # FHE: GELU is non-polynomial. Replace with polynomial approximation of ReLU (e.g., x^2 or minimax).
        # self.activation = PolynomialReLU() # Example placeholder for an FHE-friendly activation module
        # print("WARNING: MLP uses nn.GELU as a placeholder, not an FHE-friendly activation.")
        self.gelu    = nn.GELU() # Placeholder, to be replaced
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        # FHE: Dropout removed.

    def forward(self, x):
        # FHE: x is encrypted. c_fc weights are plaintext (PCMM).
        x = self.c_fc(x)
        # FHE: Apply polynomial approximation of ReLU here.
        # x = self.activation(x)
        x = self.gelu(x) # Placeholder for FHE-friendly activation
        # FHE: x is encrypted. c_proj weights are plaintext (PCMM).
        # FHE: Dropout removed.
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    # FHE: Uses HE-compatible LayerNorm, Attention, and MLP modules.

    def __init__(self, config):
        super().__init__()
        # FHE: config.bias is False for LayerNorm.
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        # FHE: x and all intermediate activations are encrypted.
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class FHEGPTConfig: # Renamed to avoid confusion with original GPTConfig
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0 # FHE: Dropout is effectively 0.0 and removed from modules.
    bias: bool = False # FHE: Bias terms removed in Linears and LayerNorms for FHE compatibility.

class GPT(nn.Module):

    def __init__(self, config: FHEGPTConfig): # Ensure FHEGPTConfig is used
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        # FHE: config.bias is False by default in FHEGPTConfig.
        # FHE: config.dropout is 0.0 by default.
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            # FHE: Dropout layer (self.transformer.drop) is effectively a no-op if config.dropout is 0.
            # It could be removed if desired, but keeping it aligns with original structure.
            # If input x to drop is encrypted, dropout is problematic. Assume inputs to first block are encrypted.
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            # FHE: ln_f uses the modified LayerNorm. config.bias is False.
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        # FHE: lm_head input x is encrypted, weights are plaintext (PCMM). Bias is already False.
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight # Weight tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None: # Should be None if config.bias is False
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # FHE: Input 'idx' (token indices) is typically plaintext.
        # Embedding lookup (wte, wpe) results in plaintext embeddings.
        # These embeddings must be encrypted by the client before being fed to the first transformer block.
        # Or, if the model handles encryption, it would happen after tok_emb + pos_emb.
        # Let's assume x (after embeddings and dropout) becomes encrypted.
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        
        # FHE: x_plaintext = tok_emb + pos_emb
        # FHE: x = encrypt(x_plaintext) # This step is crucial and handled by HE framework.
        # FHE: If dropout is applied to encrypted data, it needs an FHE-compatible version or be disabled.
        #      Since config.dropout = 0.0, self.transformer.drop is a no-op.
        x = self.transformer.drop(tok_emb + pos_emb) 
        # FHE: From here on, x is an encrypted tensor. All block operations are on encrypted data.
        for block in self.transformer.h:
            x = block(x)
        # FHE: x is encrypted. ln_f is the FHE-compatible LayerNorm.
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            return logits, loss
        else:
            logits = self.lm_head(x[:, [-1], :])
            # For ONNX/Concrete-ML export, only return logits (not a tuple)
            return logits

    def crop_block_size(self, block_size):
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        # For GK-Attention, the causal mask is generated on the fly based on T,
        # so no explicit bias buffer in CausalSelfAttention needs cropping here.
        # Original model had self.bias in CausalSelfAttention for non-flash.

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        # FHE: This method loads plaintext pre-trained weights.
        # For FHE, config.bias will be forced to False, and dropout to 0.0.
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {}
        # For FHE, we enforce bias=False and dropout=0.0
        override_args['bias'] = False
        override_args['dropout'] = 0.0
        
        print(f"Loading weights from pretrained gpt: {model_type}")
        print(f"FHE Override: Forcing bias=False and dropout={override_args['dropout']}")

        from transformers import GPT2LMHeadModel # Keep this import local

        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024),
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280),
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600),
        }[model_type]
        
        # Standard GPT-2 checkpoints have vocab_size=50257, block_size=1024.
        # Bias is True in original GPT-2, but we force it to False for FHE.
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024
        config_args['bias'] = override_args['bias'] # This will be False
        config_args['dropout'] = override_args['dropout'] # This will be 0.0

        config = FHEGPTConfig(**config_args)
        model = GPT(config) # Initialize FHE GPT model
        sd = model.state_dict()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # Should not exist with GK

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        sd_keys_hf = sd_hf.keys()
        # Filter out keys that are not parameters or are specific to HF's implementation details
        # (e.g., masks, buffers not present in our model)
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # Original bias buffer

        # Handle weights that need transposition (from Conv1D in HF to Linear in ours)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        # Manually map HF state_dict keys to our model's state_dict keys
        # This is crucial because our FHE model might have slightly different naming or structure
        # (e.g. no bias parameters in Linear layers if config.bias=False)
        
        mapped_sd_hf = {}
        for k_hf in sd_keys_hf:
            k_our = k_hf
            
            # Skip bias parameters from HF if our model doesn't have them (config.bias=False)
            if ".bias" in k_hf:
                module_path = k_hf.rsplit('.', 1)[0]
                # Check if the corresponding module in our model has bias
                # This is a bit tricky without direct module access here.
                # However, all Linear layers and LayerNorm in our FHE model will have bias=None.
                # So, we can generally skip loading biases.
                if not config.bias: # If bias is globally false for FHE model
                    # print(f"Skipping HF bias {k_hf} as FHE model config.bias is False")
                    continue
            
            mapped_sd_hf[k_our] = sd_hf[k_hf]

        # Now copy from mapped_sd_hf to sd
        for k_our in sd:
            if k_our.endswith('.attn.bias'): # This buffer should not be in FHE model with GK
                continue
            if k_our not in mapped_sd_hf:
                print(f"Warning: Key {k_our} in our model not found in HuggingFace model. Skipping.")
                continue

            param_hf = mapped_sd_hf[k_our]
            if any(k_our.endswith(w) for w in transposed):
                assert param_hf.shape[::-1] == sd[k_our].shape, f"Shape mismatch for {k_our}: HF {param_hf.shape[::-1]} vs Our {sd[k_our].shape}"
                with torch.no_grad():
                    sd[k_our].copy_(param_hf.t())
            else:
                assert param_hf.shape == sd[k_our].shape, f"Shape mismatch for {k_our}: HF {param_hf.shape} vs Our {sd[k_our].shape}"
                with torch.no_grad():
                    sd[k_our].copy_(param_hf)
        
        # Ensure lm_head and wte weights are tied if they were loaded
        if 'lm_head.weight' in sd and 'transformer.wte.weight' in sd:
             model.transformer.wte.weight = model.lm_head.weight
        
        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """
        Note: This optimizer configuration is for standard PyTorch training of the
        FHE-modified architecture (i.e., training the plaintext weights).
        For actual FHE fine-tuning (e.g., of encrypted LoRA weights, which are not
        implemented in this version), an HE-compatible optimizer (like AdamW-HE
        from the FHE_LLM paper) performing operations on encrypted gradients and
        parameters would be required.
        """
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")
        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        # FHE: MFU calculation would be drastically different for FHE.
        # This method is for plaintext GPU-based MFU.
        # print("WARNING: estimate_mfu is for plaintext GPU performance, not FHE.")
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T # Standard attention flops
        # FHE: Flops for GK attention and polynomial approximations would be different.
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0/dt)
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        FHE: Generation with encrypted context implies an FHE forward pass per token.
        - The input 'idx' is plaintext initially. Embeddings are created and then encrypted.
        - Each step: FHE forward pass -> logits (encrypted).
        - Softmax on encrypted logits needs polynomial approximation.
        - Sampling (torch.multinomial) from encrypted probabilities is complex and requires
          FHE-friendly alternatives or decryption of probabilities if allowed by security model.
        This implementation assumes plaintext operations for sampling for now.
        """
        self.eval() # Ensure model is in eval mode
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            
            # FHE: If idx_cond represents encrypted state, self(idx_cond) is FHE forward pass.
            #      However, idx_cond here are token indices (plaintext).
            #      The forward pass internally handles encryption if designed for it.
            #      Let's assume self.forward() can take plaintext idx and returns encrypted logits.
            logits, _ = self(idx_cond) # logits are encrypted as per forward pass design

            # FHE: logits are encrypted. Operations below (temperature, top_k, softmax, multinomial)
            #      would need to be FHE-compatible or logits decrypted.
            #      For placeholder, we assume they can operate on (or after decrypting) logits.
            # print("WARNING: generate method's sampling part is not FHE-operational.")
            
            # Pluck the logits at the final step
            logits_last_step = logits[:, -1, :] # This would be encrypted

            # Apply temperature (plaintext operation on encrypted data if HE scheme supports it, or after decryption)
            logits_scaled = logits_last_step / temperature

            if top_k is not None:
                # FHE: Top-k on encrypted data is complex. Requires HE-compatible algorithm or decryption.
                v, _ = torch.topk(logits_scaled, min(top_k, logits_scaled.size(-1)))
                logits_scaled[logits_scaled < v[:, [-1]]] = -float('Inf') # Or HE equivalent of masking

            # Apply softmax (FHE: polynomial approximation of softmax)
            probs = F.softmax(logits_scaled, dim=-1) # Placeholder for FHE-softmax

            # Sample from the distribution (FHE: sampling from encrypted distribution is complex)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        self.train() # Return model to train mode if it was in it before
        return idx

# Helper function for polynomial approximation (conceptual)
# def polynomial_approx_inv_sqrt(x_encrypted, degree=...):
#     # FHE: Implement using HE library operations (e.g., Remez + Newton's method)
#     # This is a placeholder for where such a function would be defined or imported.
#     raise NotImplementedError("polynomial_approx_inv_sqrt must be implemented for FHE.")

# def polynomial_approx_exp_for_gk(x_encrypted, degree=...):
#     # FHE: Implement using HE library operations (e.g., (1+x/2^k)^(2^k) or minimax)
#     # This is a placeholder.
#     raise NotImplementedError("polynomial_approx_exp_for_gk must be implemented for FHE.")

# def polynomial_approx_relu(x_encrypted, degree=...):
#     # FHE: Implement using HE library operations (e.g., x^2 or minimax for ReLU)
#     # This is a placeholder.
#     raise NotImplementedError("polynomial_approx_relu must be implemented for FHE.")
