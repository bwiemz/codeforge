"""Direct Preference Optimization (DPO) trainer.

Trains the model to prefer "chosen" responses over "rejected" responses
without needing a separate reward model. This aligns the model to produce
higher-quality, more helpful code.

Reference: "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"
(Rafailov et al., 2023)
"""

import json
import time
from pathlib import Path
from typing import Iterator, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
from tqdm import tqdm

from ..model.transformer import CodeForgeModel
from ..tokenizer.tokenizer import CodeForgeTokenizer
from .chat_template import Conversation, Message
from .scheduler import get_cosine_schedule_with_warmup


class DPODataset(IterableDataset):
    """Dataset for DPO training.

    Expects JSONL files with format:
    {
        "prompt": "Write a function...",
        "chosen": "def good_solution()...",
        "rejected": "def bad_solution()..."
    }
    """

    def __init__(
        self,
        data_paths: list[str | Path],
        tokenizer: CodeForgeTokenizer,
        max_seq_len: int = 2048,
    ):
        self.data_paths = [Path(p) for p in data_paths]
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def _tokenize_pair(self, prompt: str, response: str) -> dict[str, torch.Tensor]:
        """Tokenize a prompt+response pair."""
        full_text = prompt + response
        ids = self.tokenizer.encode(full_text, add_special_tokens=True)
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=True)

        if len(ids) > self.max_seq_len:
            ids = ids[: self.max_seq_len]

        input_ids = torch.tensor(ids[:-1], dtype=torch.long)
        targets = torch.tensor(ids[1:], dtype=torch.long)

        # Only compute loss on the response portion
        response_start = max(0, len(prompt_ids) - 1)
        labels = targets.clone()
        labels[:response_start] = -1

        return {"input_ids": input_ids, "labels": labels}

    def __iter__(self) -> Iterator[dict]:
        for path in self.data_paths:
            if not path.exists():
                continue
            with open(path, encoding="utf-8") as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                    except json.JSONDecodeError:
                        continue

                    prompt = data.get("prompt", "")
                    chosen = data.get("chosen", "")
                    rejected = data.get("rejected", "")

                    if not (prompt and chosen and rejected):
                        continue

                    chosen_data = self._tokenize_pair(prompt, chosen)
                    rejected_data = self._tokenize_pair(prompt, rejected)

                    yield {
                        "chosen_input_ids": chosen_data["input_ids"],
                        "chosen_labels": chosen_data["labels"],
                        "rejected_input_ids": rejected_data["input_ids"],
                        "rejected_labels": rejected_data["labels"],
                    }


def _get_logprobs(
    model: CodeForgeModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Compute per-token log probabilities for labeled tokens."""
    logits, _, _ = model(input_ids)
    log_probs = F.log_softmax(logits, dim=-1)

    # Gather log probs for target tokens
    per_token_logprobs = torch.gather(
        log_probs, dim=-1, index=labels.unsqueeze(-1)
    ).squeeze(-1)

    # Mask out non-response tokens (labels == -1)
    mask = labels != -1
    per_token_logprobs = per_token_logprobs * mask.float()

    # Sum log probs per sequence
    return per_token_logprobs.sum(dim=-1)


class DPOTrainer:
    """Direct Preference Optimization trainer.

    Optimizes: L_DPO = -log(sigmoid(beta * (log_pi(chosen) - log_pi(rejected)
                                             - log_ref(chosen) + log_ref(rejected))))
    """

    def __init__(
        self,
        model: CodeForgeModel,
        ref_model: CodeForgeModel,
        dataset: DPODataset,
        beta: float = 0.1,
        batch_size: int = 2,
        learning_rate: float = 5e-7,
        max_steps: int = 2000,
        warmup_steps: int = 50,
        gradient_accumulation_steps: int = 4,
        max_grad_norm: float = 1.0,
        checkpoint_dir: str = "checkpoints/dpo",
        checkpoint_every: int = 500,
        log_every: int = 10,
        precision: str = "bf16",
    ):
        self.model = model
        self.ref_model = ref_model
        self.dataset = dataset
        self.beta = beta
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_every = checkpoint_every
        self.log_every = log_every

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.ref_model = self.ref_model.to(self.device).eval()

        # Freeze reference model
        for param in self.ref_model.parameters():
            param.requires_grad = False

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, betas=(0.9, 0.95)
        )
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer, warmup_steps, max_steps
        )

        self.use_amp = precision != "fp32" and self.device.type == "cuda"
        self.amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
        self.scaler = torch.amp.GradScaler("cuda", enabled=(precision == "fp16"))

        self.global_step = 0

    def _compute_dpo_loss(self, batch: dict) -> tuple[torch.Tensor, dict]:
        """Compute DPO loss for a batch."""
        chosen_ids = batch["chosen_input_ids"].to(self.device)
        chosen_labels = batch["chosen_labels"].to(self.device)
        rejected_ids = batch["rejected_input_ids"].to(self.device)
        rejected_labels = batch["rejected_labels"].to(self.device)

        # Policy model log probs
        pi_chosen = _get_logprobs(self.model, chosen_ids, chosen_labels)
        pi_rejected = _get_logprobs(self.model, rejected_ids, rejected_labels)

        # Reference model log probs (no gradient)
        with torch.no_grad():
            ref_chosen = _get_logprobs(self.ref_model, chosen_ids, chosen_labels)
            ref_rejected = _get_logprobs(self.ref_model, rejected_ids, rejected_labels)

        # DPO loss
        logits = self.beta * (
            (pi_chosen - ref_chosen) - (pi_rejected - ref_rejected)
        )
        loss = -F.logsigmoid(logits).mean()

        # Metrics
        with torch.no_grad():
            reward_chosen = self.beta * (pi_chosen - ref_chosen)
            reward_rejected = self.beta * (pi_rejected - ref_rejected)
            accuracy = (reward_chosen > reward_rejected).float().mean()

        metrics = {
            "loss": loss.item(),
            "accuracy": accuracy.item(),
            "reward_chosen": reward_chosen.mean().item(),
            "reward_rejected": reward_rejected.mean().item(),
            "reward_margin": (reward_chosen - reward_rejected).mean().item(),
        }

        return loss, metrics

    def _save_checkpoint(self) -> None:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self.checkpoint_dir / f"dpo_step_{self.global_step}.pt"
        torch.save({
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "model_config": self.model.config.__dict__,
        }, path)
        print(f"DPO checkpoint saved: {path}")

    def train(self) -> None:
        """Run DPO training loop."""
        self.model.train()
        loader = DataLoader(self.dataset, batch_size=self.batch_size)

        accumulation_loss = 0.0
        pbar = tqdm(total=self.max_steps, desc="DPO Training")

        for micro_step, batch in enumerate(loader):
            if self.global_step >= self.max_steps:
                break

            with torch.amp.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                loss, metrics = self._compute_dpo_loss(batch)
                loss = loss / self.gradient_accumulation_steps

            self.scaler.scale(loss).backward()
            accumulation_loss += metrics["loss"]

            if (micro_step + 1) % self.gradient_accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.scheduler.step()
                self.global_step += 1

                if self.global_step % self.log_every == 0:
                    pbar.set_postfix(
                        loss=f"{accumulation_loss/self.gradient_accumulation_steps:.4f}",
                        acc=f"{metrics['accuracy']:.2f}",
                        margin=f"{metrics['reward_margin']:.3f}",
                    )
                    pbar.update(self.log_every)

                accumulation_loss = 0.0

                if self.global_step % self.checkpoint_every == 0:
                    self._save_checkpoint()

        pbar.close()
        self._save_checkpoint()
        print(f"DPO complete. {self.global_step} steps.")
