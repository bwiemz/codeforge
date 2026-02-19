"""End-to-end example: train a TCFP-aligned tokenizer and validate it.

Usage::

    python examples/train_tokenizer.py --corpus data/*.txt --output tokenizer_output

If no corpus files are provided, a small built-in sample is used for
demonstration purposes (not suitable for production).
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from codeforge.tokenizer.alignment import check_alignment
from codeforge.tokenizer.config import TokenizerConfig
from codeforge.tokenizer.constants import VOCAB_SIZE
from codeforge.tokenizer.training import train_tokenizer
from codeforge.tokenizer.validation import validate_tokenizer
from codeforge.tokenizer.wrapper import TCFPTokenizer

# ── Built-in demo corpus (multi-language code + NL) ────────────────
DEMO_CORPUS = """\
# Python
def fibonacci(n: int) -> int:
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

class DataProcessor:
    def __init__(self, data: list[float]):
        self.data = data
        self.result = 0.0

    def compute_mean(self) -> float:
        self.result = sum(self.data) / len(self.data)
        return self.result

    async def fetch_data(self, url: str) -> dict:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                return await response.json()

for i in range(1000):
    if i % 15 == 0:
        print("FizzBuzz")
    elif i % 3 == 0:
        print("Fizz")
    elif i % 5 == 0:
        print("Buzz")

// JavaScript / TypeScript
function mergeSort(arr) {
    if (arr.length <= 1) return arr;
    const mid = Math.floor(arr.length / 2);
    const left = mergeSort(arr.slice(0, mid));
    const right = mergeSort(arr.slice(mid));
    return merge(left, right);
}

const fetchData = async (url: string): Promise<Response> => {
    const response = await fetch(url);
    if (!response.ok) {
        throw new Error(`HTTP error: ${response.status}`);
    }
    return response.json();
};

interface UserConfig {
    name: string;
    email: string;
    preferences: Record<string, boolean>;
}

// Rust
fn binary_search<T: Ord>(slice: &[T], target: &T) -> Option<usize> {
    let mut low = 0;
    let mut high = slice.len();
    while low < high {
        let mid = low + (high - low) / 2;
        match slice[mid].cmp(target) {
            Ordering::Less => low = mid + 1,
            Ordering::Greater => high = mid,
            Ordering::Equal => return Some(mid),
        }
    }
    None
}

impl Display for Matrix {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        for row in &self.data {
            writeln!(f, "{:?}", row)?;
        }
        Ok(())
    }
}

// Go
func httpHandler(w http.ResponseWriter, r *http.Request) {
    ctx := r.Context()
    data, err := fetchFromDB(ctx, r.URL.Query().Get("id"))
    if err != nil {
        http.Error(w, err.Error(), http.StatusInternalServerError)
        return
    }
    json.NewEncoder(w).Encode(data)
}

// Java
public class HashMap<K, V> {
    private Entry<K, V>[] buckets;
    private int size;

    public V get(K key) {
        int index = hash(key) % buckets.length;
        Entry<K, V> entry = buckets[index];
        while (entry != null) {
            if (entry.key.equals(key)) {
                return entry.value;
            }
            entry = entry.next;
        }
        return null;
    }
}

// C++
template<typename T>
class SmartPointer {
    T* ptr;
public:
    explicit SmartPointer(T* p = nullptr) : ptr(p) {}
    ~SmartPointer() { delete ptr; }
    T& operator*() const { return *ptr; }
    T* operator->() const { return ptr; }
};

std::vector<int> result = std::transform(
    data.begin(), data.end(),
    std::back_inserter(output),
    [](int x) { return x * 2; }
);

// C
typedef struct {
    int width;
    int height;
    unsigned char* pixels;
} Image;

void* memory_pool_alloc(MemoryPool* pool, size_t size) {
    if (pool->offset + size > pool->capacity) {
        return NULL;
    }
    void* ptr = pool->buffer + pool->offset;
    pool->offset += size;
    return ptr;
}

/* Natural language documentation */
The TCFP (Tensor Core Floating Point) format enables FP8 training
with near-BF16 convergence quality.  It decomposes each weight into
two FP8 components (W_hi + W_lo) and executes dual GEMMs on tensor
cores, achieving 19% faster training than BF16 baselines.

Key design parameters:
- Vocabulary size: 49,152 (384 x 128, aligned to all tile sizes)
- Block size: 32 (for per-block FP8 scaling)
- Tile size: 128 (primary BLOCK_N in Triton autotune configs)
- Hidden dimensions: must be divisible by 128 for peak throughput

Port numbers like 8080, 3000, 443, and 5432 should tokenize as
individual digits: 8, 0, 8, 0 for consistent position-independent
representation.  Version numbers like 3.14.159 and dates like
2024-01-15 also benefit from individual digit splitting.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a TCFP-aligned code tokenizer",
    )
    parser.add_argument(
        "--corpus",
        nargs="*",
        type=Path,
        default=None,
        help="Corpus files to train on (default: built-in demo)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tokenizer_output"),
        help="Output directory (default: tokenizer_output)",
    )
    parser.add_argument(
        "--min-frequency",
        type=int,
        default=2,
        help="Minimum token frequency for BPE (default: 2)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ── 1. Alignment check ─────────────────────────────────────────
    report = check_alignment(VOCAB_SIZE)
    print("=== FP8 Tensor Core Alignment ===")
    print(report.summary())
    print()

    # ── 2. Prepare corpus ──────────────────────────────────────────
    if args.corpus:
        corpus_files = args.corpus
        print(f"Training on {len(corpus_files)} corpus file(s)")
    else:
        print("No corpus provided — using built-in demo corpus")
        tmp_dir = Path(tempfile.mkdtemp())
        tmp_file = tmp_dir / "demo_corpus.txt"
        # Repeat to give BPE enough data for meaningful merges
        tmp_file.write_text(DEMO_CORPUS * 100, encoding="utf-8")
        corpus_files = [tmp_file]

    # ── 3. Train ───────────────────────────────────────────────────
    config = TokenizerConfig(
        output_dir=args.output,
        corpus_paths=corpus_files,
        min_frequency=args.min_frequency,
    )

    print(f"\nTraining {VOCAB_SIZE}-vocab BPE tokenizer...")
    tok = train_tokenizer(config)
    print(f"Saved to {config.output_path}")

    # ── 4. Validate ────────────────────────────────────────────────
    print("\n=== Post-Training Validation ===")

    # Roundtrip samples
    roundtrip_samples = [
        "def hello_world():\n    print('hello')\n",
        "const x = 42;\nconst y = x * 2;",
        "fn main() {\n    let x = 0xDEAD;\n}",
        "func foo() error { return nil }",
        "    \t\n  ",  # whitespace
        "a == b && c != d || e >= f",
        "port 8080 and version 3.14.159",
    ]

    # Validation texts by language
    validation_texts = {
        "python": (
            "def fibonacci(n):\n    if n <= 1:\n"
            "        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n"
        ),
        "javascript": (
            "function greet(name) {\n    return `Hello, ${name}!`;\n}\n"
        ),
        "rust": (
            "fn main() {\n    let mut x = 0;\n"
            "    for i in 0..10 { x += i; }\n"
            '    println!("{}", x);\n}\n'
        ),
        "go": (
            "func main() {\n"
            '    items := []string{"a", "b"}\n'
            "    for _, v := range items { fmt.Println(v) }\n}\n"
        ),
    }

    val_report = validate_tokenizer(
        tok,
        validation_texts=validation_texts,
        roundtrip_samples=roundtrip_samples,
    )
    print(val_report.summary())

    # ── 5. Quick demo ──────────────────────────────────────────────
    print("\n=== Quick Demo ===")
    wrapper = TCFPTokenizer(config.output_path)
    print(f"vocab_size:  {wrapper.vocab_size}")
    print(f"pad_id:      {wrapper.pad_id}")
    print(f"bos_id:      {wrapper.bos_id}")
    print(f"eos_id:      {wrapper.eos_id}")
    print(f"fim_prefix:  {wrapper.fim_prefix_id}")
    print(f"im_start:    {wrapper.im_start_id}")

    demo_text = "def add(x, y):\n    return x + y"
    ids = wrapper.encode(demo_text)
    print(f"\nText:    {demo_text!r}")
    print(f"Tokens:  {ids}")
    print(f"Decoded: {wrapper.decode(ids)!r}")

    # Digit splitting demo
    digit_text = "port 8080"
    digit_ids = wrapper.encode(digit_text)
    digit_tokens = [wrapper.inner.id_to_token(i) for i in digit_ids]
    print(f"\nDigit splitting: {digit_text!r}")
    print(f"Tokens: {digit_tokens}")


if __name__ == "__main__":
    main()
