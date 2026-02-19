"""Supervised Fine-Tuning (SFT) trainer.

Trains on instruction-response pairs with loss only computed
on assistant response tokens (instruction tokens are masked out).
"""

import json
import time
from collections.abc import Iterator
from pathlib import Path

import torch
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm

from ..model.transformer import CodeForgeModel
from ..tokenizer.tokenizer import CodeForgeTokenizer
from .chat_template import Conversation, Message
from .scheduler import get_cosine_schedule_with_warmup


class SFTDataset(IterableDataset):
    """Dataset for supervised fine-tuning on instruction-response pairs.

    Supports:
    - Local JSONL files: {"instruction": "...", "response": "..."}
      or multi-turn: {"messages": [{"role": "user", "content": "..."}, ...]}
    - HuggingFace datasets: Magicoder, Code Alpaca, ShareGPT, etc.
    """

    def __init__(
        self,
        data_paths: list[str | Path] | None = None,
        tokenizer: CodeForgeTokenizer = None,
        max_seq_len: int = 2048,
        hf_dataset: str | None = None,
        hf_split: str = "train",
    ):
        self.data_paths = [Path(p) for p in (data_paths or [])]
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.hf_dataset = hf_dataset
        self.hf_split = hf_split

    def _load_conversations(self) -> Iterator[Conversation]:
        """Load and parse conversation data from local files and HF datasets."""
        # Local JSONL files
        for path in self.data_paths:
            if not path.exists():
                print(f"Warning: SFT data file not found: {path}")
                continue

            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    conv = self._parse_sample(data)
                    if conv:
                        yield conv

        # HuggingFace dataset
        if self.hf_dataset:
            yield from self._load_hf_conversations()

    def _parse_sample(self, data: dict) -> Conversation | None:
        """Parse a single sample dict into a Conversation."""
        conv = Conversation()

        if "messages" in data:
            for msg in data["messages"]:
                conv.messages.append(
                    Message(role=msg["role"], content=msg["content"])
                )
        elif "conversations" in data:
            # ShareGPT format
            for msg in data["conversations"]:
                role = msg.get("from", msg.get("role", ""))
                content = msg.get("value", msg.get("content", ""))
                if role in ("human", "user"):
                    conv.add_user(content)
                elif role in ("gpt", "assistant"):
                    conv.add_assistant(content)
        elif "instruction" in data and "response" in data:
            conv.add_user(data["instruction"])
            conv.add_assistant(data["response"])
        elif "problem" in data and "solution" in data:
            # Magicoder OSS-Instruct format
            conv.add_user(data["problem"])
            conv.add_assistant(data["solution"])
        else:
            return None

        if data.get("system"):
            conv.system_prompt = data["system"]

        return conv if conv.messages else None

    def _load_hf_conversations(self) -> Iterator[Conversation]:
        """Load conversations from a HuggingFace dataset."""
        from datasets import load_dataset

        print(f"Loading SFT dataset from HuggingFace: {self.hf_dataset}")
        ds = load_dataset(self.hf_dataset, split=self.hf_split, streaming=True)

        count = 0
        for sample in ds:
            conv = self._parse_sample(sample)
            if conv:
                count += 1
                if count % 5000 == 0:
                    print(f"  Loaded {count:,} SFT conversations...")
                yield conv

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        for conv in self._load_conversations():
            sft_data = conv.format_for_sft(self.tokenizer)
            ids = sft_data["input_ids"]
            mask = sft_data["loss_mask"]

            # Truncate to max_seq_len
            if len(ids) > self.max_seq_len:
                ids = ids[: self.max_seq_len]
                mask = mask[: self.max_seq_len]

            if len(ids) < 4:  # Skip very short sequences
                continue

            # Create input/target pairs
            input_ids = torch.tensor(ids[:-1], dtype=torch.long)
            targets = torch.tensor(ids[1:], dtype=torch.long)
            loss_mask = torch.tensor(mask[1:], dtype=torch.bool)

            # Mask targets where loss shouldn't be computed
            targets[~loss_mask] = -1  # -1 is ignored by cross_entropy

            yield {
                "input_ids": input_ids,
                "targets": targets,
            }


class SFTTrainer:
    """Supervised Fine-Tuning trainer with instruction masking."""

    def __init__(
        self,
        model: CodeForgeModel,
        dataset: SFTDataset,
        batch_size: int = 4,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        max_steps: int = 5000,
        warmup_steps: int = 100,
        max_grad_norm: float = 1.0,
        gradient_accumulation_steps: int = 4,
        checkpoint_dir: str = "checkpoints/sft",
        checkpoint_every: int = 1000,
        log_every: int = 10,
        precision: str = "bf16",
        embed_lr_ratio: float = 1.0,
        scheduler_type: str = "cosine",
        stable_steps: int = 0,
        decay_steps: int = 0,
        min_lr_ratio: float = 0.0,
        decay_type: str = "cosine",
    ):
        self.model = model
        self.dataset = dataset
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps
        self.max_grad_norm = max_grad_norm
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_every = checkpoint_every
        self.log_every = log_every

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        # Optimizer: 3 param groups (decay / no-decay / embedding)
        from .optimizer import build_param_groups

        param_groups = build_param_groups(
            model, learning_rate, weight_decay, embed_lr_ratio,
        )
        use_fused = self.device.type == "cuda"
        self.optimizer = torch.optim.AdamW(
            param_groups, lr=learning_rate, betas=(0.9, 0.95),
            eps=1e-8, fused=use_fused,
        )

        # LR scheduler
        if scheduler_type == "wsd":
            from .scheduler import get_wsd_schedule

            decay_len = decay_steps or (max_steps - warmup_steps - stable_steps)
            if decay_len <= 0:
                raise ValueError(
                    f"WSD decay_steps resolved to {decay_len} "
                    f"(max_steps={max_steps}, warmup={warmup_steps}, "
                    f"stable={stable_steps}). Check your config."
                )
            self.scheduler = get_wsd_schedule(
                self.optimizer, warmup_steps, stable_steps,
                decay_len, min_lr_ratio, decay_type,
            )
        else:
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer, warmup_steps, max_steps,
                min_lr_ratio=min_lr_ratio,
            )

        self.use_amp = precision != "fp32" and self.device.type == "cuda"
        self.amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
        self.scaler = torch.amp.GradScaler("cuda", enabled=(precision == "fp16"))

        self.global_step = 0

    def _save_checkpoint(self) -> None:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self.checkpoint_dir / f"sft_step_{self.global_step}.pt"
        torch.save({
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "model_config": self.model.config.__dict__,
        }, path)
        print(f"SFT checkpoint saved: {path}")

    def train(self) -> None:
        """Run SFT training loop."""
        self.model.train()
        loader = DataLoader(self.dataset, batch_size=self.batch_size)

        accumulation_loss = 0.0
        time.time()
        pbar = tqdm(total=self.max_steps, desc="SFT Training")

        for micro_step, batch in enumerate(loader):
            if self.global_step >= self.max_steps:
                break

            input_ids = batch["input_ids"].to(self.device)
            targets = batch["targets"].to(self.device)

            with torch.amp.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                _, loss, _ = self.model(input_ids, targets)
                loss = loss / self.gradient_accumulation_steps

            self.scaler.scale(loss).backward()
            accumulation_loss += loss.item()

            if (micro_step + 1) % self.gradient_accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.scheduler.step()
                self.global_step += 1

                if self.global_step % self.log_every == 0:
                    lr = self.scheduler.get_last_lr()[0]
                    pbar.set_postfix(loss=f"{accumulation_loss:.4f}", lr=f"{lr:.2e}")
                    pbar.update(self.log_every)
                    time.time()

                accumulation_loss = 0.0

                if self.global_step % self.checkpoint_every == 0:
                    self._save_checkpoint()

        pbar.close()
        self._save_checkpoint()
        print(f"SFT complete. {self.global_step} steps.")
