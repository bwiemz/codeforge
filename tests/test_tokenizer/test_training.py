"""End-to-end training pipeline tests using a tiny corpus."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

# Skip entire module if tokenizers is not installed
tokenizers = pytest.importorskip("tokenizers")

from codeforge.tokenizer.config import TokenizerConfig
from codeforge.tokenizer.constants import SPECIAL_TOKENS, VOCAB_SIZE
from codeforge.tokenizer.training import build_tokenizer_pipeline, train_tokenizer

# A small but representative code corpus for testing
TINY_CORPUS = """\
def hello_world():
    print("Hello, world!")
    return 42

class Calculator:
    def __init__(self):
        self.result = 0

    def add(self, x, y):
        self.result = x + y
        return self.result

    def multiply(self, x, y):
        self.result = x * y
        return self.result

# Constants
MAX_VALUE = 9999
PI = 3.14159
HEX_COLOR = 0xDEADBEEF

for i in range(100):
    if i % 2 == 0:
        print(f"Even: {i}")
    else:
        print(f"Odd: {i}")

function greet(name) {
    const message = `Hello, ${name}!`;
    console.log(message);
    return message;
}

const arr = [1, 2, 3, 4, 5];
const doubled = arr.map(x => x * 2);
const filtered = arr.filter(x => x >= 3);

fn main() {
    let mut count = 0;
    for i in 0..10 {
        count += i;
        println!("Count: {}", count);
    }
}

impl Display for MyStruct {
    fn fmt(&self, f: &mut Formatter) -> Result {
        write!(f, "value: {}", self.value)
    }
}

func processItems(items []string) error {
    for _, item := range items {
        if err := validate(item); err != nil {
            return fmt.Errorf("invalid: %w", err)
        }
    }
    return nil
}

// Common operators and patterns
int result = (a == b) ? 1 : 0;
bool check = (x != y) && (x >= 0);
auto ptr = std::make_unique<int>(42);
template<typename T>
void swap(T& a, T& b) {
    T temp = a;
    a = b;
    b = temp;
}
"""


@pytest.fixture()
def tiny_corpus_file(tmp_path: Path) -> Path:
    """Write the tiny corpus to a temp file and return its path."""
    corpus_path = tmp_path / "corpus.txt"
    # Repeat corpus to give BPE enough data for meaningful merges
    corpus_path.write_text(TINY_CORPUS * 50, encoding="utf-8")
    return corpus_path


class TestBuildPipeline:
    def test_builds_without_error(self) -> None:
        config = TokenizerConfig()
        tok, trainer = build_tokenizer_pipeline(config)
        assert tok is not None
        assert trainer is not None


class TestTrainTokenizer:
    def test_raises_without_corpus(self) -> None:
        config = TokenizerConfig(output_dir=Path(tempfile.mkdtemp()))
        with pytest.raises(ValueError, match="No corpus files"):
            train_tokenizer(config)

    def test_full_pipeline(self, tiny_corpus_file: Path, tmp_path: Path) -> None:
        """Train on tiny corpus and verify key properties."""
        config = TokenizerConfig(
            output_dir=tmp_path / "output",
            corpus_paths=[tiny_corpus_file],
        )

        tok = train_tokenizer(config)

        # Vocab size must be exactly 49,152
        assert tok.get_vocab_size() == VOCAB_SIZE

        # Special tokens at correct IDs
        vocab = tok.get_vocab()
        for token_str, expected_id in SPECIAL_TOKENS.items():
            assert vocab.get(token_str) == expected_id, (
                f"{token_str} at {vocab.get(token_str)}, expected {expected_id}"
            )

        # Output file exists
        assert (tmp_path / "output" / "tcfp_tokenizer.json").exists()

        # Post-processor configured: encode with add_special_tokens wraps BOS/EOS
        encoded = tok.encode("test", add_special_tokens=True)
        assert encoded.ids[0] == SPECIAL_TOKENS["<|bos|>"]
        assert encoded.ids[-1] == SPECIAL_TOKENS["<|eos|>"]

    def test_roundtrip_fidelity(self, tiny_corpus_file: Path, tmp_path: Path) -> None:
        """Verify encode → decode roundtrip on training data."""
        config = TokenizerConfig(
            output_dir=tmp_path / "output",
            corpus_paths=[tiny_corpus_file],
        )
        tok = train_tokenizer(config)

        samples = [
            "def hello():",
            "    return 42",
            "const x = 123;",
            "fn main() {",
            "func foo() error {",
            "x == y",
            "a != b",
            "0xDEADBEEF",
        ]

        for sample in samples:
            # No BOS/EOS wrapping — pure content roundtrip.
            # skip_special_tokens not needed since no specials are in ids.
            encoded = tok.encode(sample, add_special_tokens=False)
            decoded = tok.decode(encoded.ids)
            assert decoded == sample, f"Roundtrip failed: {sample!r} -> {decoded!r}"
