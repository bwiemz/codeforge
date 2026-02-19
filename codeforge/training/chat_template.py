"""Chat/instruction template system for CodeForge.

Defines the conversation format used for supervised fine-tuning (SFT)
and inference. Uses a ChatML-inspired format.
"""

from dataclasses import dataclass, field
from typing import Optional


# Special tokens for chat format
SYSTEM_TOKEN = "<|system|>"
USER_TOKEN = "<|user|>"
ASSISTANT_TOKEN = "<|assistant|>"
END_TOKEN = "<|end|>"

# All chat-specific special tokens
CHAT_SPECIAL_TOKENS = [
    SYSTEM_TOKEN,
    USER_TOKEN,
    ASSISTANT_TOKEN,
    END_TOKEN,
]

DEFAULT_SYSTEM_PROMPT = (
    "You are CodeForge, an expert coding assistant. You write clean, "
    "efficient, well-documented code. You explain your reasoning clearly "
    "and consider edge cases."
)


@dataclass
class Message:
    role: str       # "system", "user", or "assistant"
    content: str


@dataclass
class Conversation:
    messages: list[Message] = field(default_factory=list)
    system_prompt: str = DEFAULT_SYSTEM_PROMPT

    def add_system(self, content: str) -> "Conversation":
        self.system_prompt = content
        return self

    def add_user(self, content: str) -> "Conversation":
        self.messages.append(Message(role="user", content=content))
        return self

    def add_assistant(self, content: str) -> "Conversation":
        self.messages.append(Message(role="assistant", content=content))
        return self

    def format(self, add_generation_prompt: bool = False) -> str:
        """Format the conversation into the chat template string.

        Args:
            add_generation_prompt: If True, append the assistant token at the end
                (used during inference to prompt the model to generate)
        """
        parts = [f"{SYSTEM_TOKEN}\n{self.system_prompt}{END_TOKEN}"]

        for msg in self.messages:
            if msg.role == "user":
                parts.append(f"{USER_TOKEN}\n{msg.content}{END_TOKEN}")
            elif msg.role == "assistant":
                parts.append(f"{ASSISTANT_TOKEN}\n{msg.content}{END_TOKEN}")

        if add_generation_prompt:
            parts.append(f"{ASSISTANT_TOKEN}\n")

        return "\n".join(parts)

    def format_for_sft(self, tokenizer) -> dict:
        """Format for SFT training with loss masking.

        Returns token IDs and a mask indicating which tokens should
        contribute to the loss (only assistant responses).
        """
        full_text = self.format()
        all_ids = tokenizer.encode(full_text, add_special_tokens=False)

        # Build loss mask: True for assistant tokens, False for everything else
        # We need to identify which token ranges correspond to assistant responses
        loss_mask = [False] * len(all_ids)

        # Find assistant response boundaries
        formatted = self.format()
        pos = 0
        for msg in self.messages:
            if msg.role == "assistant":
                # Find this assistant response in the formatted text
                marker = f"{ASSISTANT_TOKEN}\n{msg.content}{END_TOKEN}"
                start_pos = formatted.find(marker, pos)
                if start_pos == -1:
                    continue

                # The content starts after "<|assistant|>\n"
                content_start = start_pos + len(ASSISTANT_TOKEN) + 1
                content_end = content_start + len(msg.content)

                # Convert character positions to token positions (approximate)
                prefix = formatted[:content_start]
                prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
                content_ids = tokenizer.encode(
                    formatted[:content_end], add_special_tokens=False
                )

                # Mark assistant content tokens for loss
                for idx in range(len(prefix_ids), min(len(content_ids), len(loss_mask))):
                    loss_mask[idx] = True

                pos = content_end

        return {
            "input_ids": all_ids,
            "loss_mask": loss_mask,
        }


def format_instruction_pair(
    instruction: str,
    response: str,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> str:
    """Format a simple instruction-response pair for SFT."""
    conv = Conversation(system_prompt=system_prompt)
    conv.add_user(instruction)
    conv.add_assistant(response)
    return conv.format()


def format_coding_task(
    task: str,
    code: str,
    explanation: Optional[str] = None,
    language: str = "python",
) -> str:
    """Format a coding task for SFT training."""
    response_parts = [f"```{language}\n{code}\n```"]
    if explanation:
        response_parts.append(f"\n{explanation}")

    return format_instruction_pair(task, "\n".join(response_parts))
