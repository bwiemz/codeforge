"""Code generation / inference engine.

Supports autoregressive sampling with KV-cache, repetition penalty,
Fill-in-the-Middle (FIM), chat mode, and batched generation.
"""

import sys
from pathlib import Path
from typing import Iterator, Optional

import torch
import torch.nn.functional as F

from ..model.config import ModelConfig
from ..model.transformer import CodeForgeModel
from ..tokenizer.tokenizer import CodeForgeTokenizer


class CodeForgeGenerator:
    """Autoregressive text generator with KV-cache, sampling, and FIM support."""

    def __init__(
        self,
        model: CodeForgeModel,
        tokenizer: CodeForgeTokenizer,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device).eval()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        tokenizer_path: str | Path,
        device: Optional[torch.device] = None,
    ) -> "CodeForgeGenerator":
        """Load a generator from a training checkpoint."""
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        config = ModelConfig(**ckpt["model_config"])
        model = CodeForgeModel(config)
        model.load_state_dict(ckpt["model_state_dict"])

        tokenizer = CodeForgeTokenizer(tokenizer_path)
        return cls(model, tokenizer, device)

    def _apply_repetition_penalty(
        self,
        logits: torch.Tensor,
        generated_ids: list[int],
        penalty: float = 1.2,
        frequency_penalty: float = 0.0,
    ) -> torch.Tensor:
        """Apply repetition and frequency penalties to logits.

        Args:
            logits: Raw logits for the next token
            generated_ids: Previously generated token IDs
            penalty: Multiplicative penalty for seen tokens (>1 = less repetition)
            frequency_penalty: Additive penalty scaled by token frequency
        """
        if penalty == 1.0 and frequency_penalty == 0.0:
            return logits

        if not generated_ids:
            return logits

        # Count token frequencies
        token_counts = {}
        for tid in generated_ids:
            token_counts[tid] = token_counts.get(tid, 0) + 1

        for tid, count in token_counts.items():
            # Multiplicative penalty
            if logits[tid] > 0:
                logits[tid] /= penalty
            else:
                logits[tid] *= penalty

            # Frequency-based additive penalty
            logits[tid] -= frequency_penalty * count

        return logits

    def _sample_token(
        self,
        logits: torch.Tensor,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
    ) -> int:
        """Sample a single token from logits with temperature, top-k, and top-p."""
        if temperature <= 0:
            return logits.argmax(dim=-1).item()

        logits = logits / temperature

        # Top-k filtering
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))
            kth_values = torch.topk(logits, top_k).values[..., -1:]
            logits = torch.where(logits < kth_values, float("-inf"), logits)

        # Top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
            sorted_logits[mask] = float("-inf")
            logits = sorted_logits.scatter(-1, sorted_indices, sorted_logits)

        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).item()

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        stop_tokens: Optional[list[int]] = None,
        repetition_penalty: float = 1.0,
        frequency_penalty: float = 0.0,
    ) -> str:
        """Generate text from a prompt."""
        tokens = list(self.stream_generate(
            prompt, max_tokens, temperature, top_k, top_p,
            stop_tokens, repetition_penalty, frequency_penalty,
        ))
        return "".join(tokens)

    @torch.no_grad()
    def stream_generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        stop_tokens: Optional[list[int]] = None,
        repetition_penalty: float = 1.0,
        frequency_penalty: float = 0.0,
    ) -> Iterator[str]:
        """Generate text token by token, yielding decoded strings."""
        if stop_tokens is None:
            stop_tokens = [self.tokenizer.EOS_ID]

        input_ids = self.tokenizer.encode(prompt)
        tokens = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        logits, _, kv_caches = self.model(tokens, kv_caches=None, start_pos=0)

        yield prompt

        generated_ids = []
        pos = tokens.shape[1]

        for _ in range(max_tokens):
            current_logits = logits[0, -1, :].clone()

            # Apply repetition penalty
            current_logits = self._apply_repetition_penalty(
                current_logits, generated_ids, repetition_penalty, frequency_penalty
            )

            next_token_id = self._sample_token(current_logits, temperature, top_k, top_p)

            if next_token_id in stop_tokens:
                break

            generated_ids.append(next_token_id)
            new_text = self.tokenizer.decode([next_token_id], skip_special_tokens=True)
            yield new_text

            new_token = torch.tensor([[next_token_id]], dtype=torch.long, device=self.device)
            logits, _, kv_caches = self.model(new_token, kv_caches=kv_caches, start_pos=pos)
            pos += 1

    @torch.no_grad()
    def generate_fim(
        self,
        prefix: str,
        suffix: str,
        max_tokens: int = 256,
        temperature: float = 0.6,
        top_k: int = 40,
        top_p: float = 0.95,
    ) -> str:
        """Fill-in-the-Middle generation."""
        input_ids = self.tokenizer.encode_fim(prefix, suffix)
        tokens = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        logits, _, kv_caches = self.model(tokens, kv_caches=None, start_pos=0)

        stop_tokens = [self.tokenizer.EOS_ID, self.tokenizer.ENDOFCODE_ID]
        generated = []
        pos = tokens.shape[1]

        for _ in range(max_tokens):
            next_token_id = self._sample_token(logits[0, -1, :], temperature, top_k, top_p)

            if next_token_id in stop_tokens:
                break

            generated.append(next_token_id)
            new_token = torch.tensor([[next_token_id]], dtype=torch.long, device=self.device)
            logits, _, kv_caches = self.model(new_token, kv_caches=kv_caches, start_pos=pos)
            pos += 1

        return self.tokenizer.decode(generated, skip_special_tokens=True)

    @torch.no_grad()
    def generate_chat(
        self,
        conversation,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.95,
        repetition_penalty: float = 1.1,
    ) -> str:
        """Generate a response in chat mode.

        Args:
            conversation: A Conversation object from chat_template
            max_tokens: Max tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated assistant response text
        """
        # Format conversation with generation prompt
        prompt = conversation.format(add_generation_prompt=True)

        # Generate with end token as stop
        end_token_text = "<|end|>"
        full_output = self.generate(
            prompt, max_tokens, temperature, top_k, top_p,
            repetition_penalty=repetition_penalty,
        )

        # Extract just the assistant response
        response = full_output[len(prompt):]

        # Trim at end token if present
        if end_token_text in response:
            response = response[: response.index(end_token_text)]

        return response.strip()

    @torch.no_grad()
    def generate_batch(
        self,
        prompts: list[str],
        max_tokens: int = 256,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
    ) -> list[str]:
        """Generate completions for multiple prompts.

        Pads shorter prompts and processes in parallel for efficiency.
        """
        # Encode all prompts
        all_ids = [self.tokenizer.encode(p) for p in prompts]
        max_prompt_len = max(len(ids) for ids in all_ids)

        # Left-pad to align
        padded = []
        for ids in all_ids:
            padding = [self.tokenizer.PAD_ID] * (max_prompt_len - len(ids))
            padded.append(padding + ids)

        tokens = torch.tensor(padded, dtype=torch.long, device=self.device)

        # Process all prompts together
        logits, _, kv_caches = self.model(tokens, kv_caches=None, start_pos=0)

        # Generate token by token for all prompts
        results = [list(ids) for ids in all_ids]
        active = list(range(len(prompts)))
        pos = max_prompt_len

        for _ in range(max_tokens):
            if not active:
                break

            next_tokens = []
            for batch_idx in range(len(prompts)):
                if batch_idx in active:
                    token_id = self._sample_token(
                        logits[batch_idx, -1, :], temperature, top_k, top_p
                    )
                    if token_id == self.tokenizer.EOS_ID:
                        active.remove(batch_idx)
                        next_tokens.append(self.tokenizer.PAD_ID)
                    else:
                        results[batch_idx].append(token_id)
                        next_tokens.append(token_id)
                else:
                    next_tokens.append(self.tokenizer.PAD_ID)

            new_tokens = torch.tensor([next_tokens], dtype=torch.long, device=self.device).T.unsqueeze(-1)
            new_tokens = new_tokens.squeeze(-1)
            logits, _, kv_caches = self.model(new_tokens, kv_caches=kv_caches, start_pos=pos)
            pos += 1

        # Decode results
        return [
            self.tokenizer.decode(ids, skip_special_tokens=True)
            for ids in results
        ]
