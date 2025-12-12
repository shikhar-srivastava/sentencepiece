# Build and Usage Guide for Rényi Entropy-Constrained Unigram Tokenizer

## ✅ Build Status: SUCCESS

The implementation has been successfully built and tested!

## Build Process

### 1. Setup Environment

```bash
conda activate tok
```

### 2. Regenerate Proto Files (if modified)

If you modify `src/sentencepiece_model.proto`, you need to regenerate the C++ files:

```bash
# Download protoc 3.14.0 (matches bundled protobuf-lite)
cd /tmp
wget https://github.com/protocolbuffers/protobuf/releases/download/v3.14.0/protoc-3.14.0-linux-x86_64.zip
unzip protoc-3.14.0-linux-x86_64.zip -d protoc-3.14

# Regenerate proto files
cd /localdisk/ssrivas9/sentencepiece/src
/tmp/protoc-3.14/bin/protoc --cpp_out=lite:builtin_pb sentencepiece_model.proto
```

### 3. Build

```bash
cd /localdisk/ssrivas9/sentencepiece
conda activate tok
mkdir -p build && cd build
cmake ..  # Uses internal protobuf (standard repo method)
make -j$(nproc)
```

### 4. Verify Build

```bash
./src/spm_train --help | grep -i entropy
```

You should see the new entropy-related flags.

## Usage Examples

### Basic Training with Entropy Constraint

```bash
./build/src/spm_train \
  --input=your_corpus.txt \
  --model_prefix=model \
  --vocab_size=8000 \
  --model_type=unigram \
  --target_renyi_entropy=10.5 \
  --renyi_alpha=2.0 \
  --entropy_distribution_type=EMPIRICAL_FREQUENCIES \
  --entropy_optimization_mode=ENTROPY_BOTH \
  --entropy_tolerance=0.01
```

### Parameters Explained

- **`--target_renyi_entropy`**: Target entropy value (0 = disabled)
- **`--renyi_alpha`**: 
  - `1.0` = Shannon entropy
  - `2.0` = Collision entropy (recommended, default)
  - `0.0` = Hartley entropy (log of vocab size)
- **`--entropy_distribution_type`**:
  - `MODEL_PROBABILITIES`: Uses learned log-probabilities (faster)
  - `EMPIRICAL_FREQUENCIES`: Uses actual corpus frequencies (more accurate, recommended)
- **`--entropy_optimization_mode`**:
  - `ENTROPY_DISABLED`: No entropy constraint (default)
  - `ENTROPY_PRUNING_CONSTRAINT`: Adjust pruning based on entropy
  - `ENTROPY_STOPPING_CRITERION`: Stop when target entropy reached
  - `ENTROPY_BOTH`: Use both strategies (recommended)
- **`--entropy_tolerance`**: Relative tolerance (e.g., 0.01 = ±1%)

### Example Output

During training, you'll see logs like:

```
LOG(INFO) Entropy-based optimization enabled. Target entropy: 10.5 (alpha=2)
LOG(INFO) EM iteration=0 current_entropy=11.2 target_entropy=10.5
LOG(INFO) Current entropy: 11.2 Target entropy: 10.5 Tolerance: 0.01
LOG(INFO) Entropy above target (ratio=1.06667), keeping fewer pieces: 7500
LOG(INFO) EM iteration=1 current_entropy=10.48 target_entropy=10.5
LOG(INFO) Training completed. Final entropy: 10.48 (target: 10.5)
```

## Test Results

Successfully trained a model with:
- ✅ Entropy calculation working
- ✅ Entropy-guided pruning adjusting piece count
- ✅ Tolerance being respected
- ✅ Training completing successfully

## Quick Test

```bash
# Create test corpus
cat > test_corpus.txt << 'EOF'
hello world this is a test
sentence piece tokenizer training
machine learning natural language processing
EOF

# Train with entropy constraint
./build/src/spm_train \
  --input=test_corpus.txt \
  --model_prefix=test_model \
  --vocab_size=30 \
  --model_type=unigram \
  --target_renyi_entropy=3.5 \
  --renyi_alpha=2.0 \
  --entropy_distribution_type=EMPIRICAL_FREQUENCIES \
  --entropy_optimization_mode=ENTROPY_BOTH \
  --entropy_tolerance=0.05
```

## Notes

1. **Proto File Regeneration**: If you modify `.proto` files, you must regenerate with protoc 3.14.0 to match the bundled protobuf-lite runtime.

2. **Standard Build Method**: The repository uses internal protobuf by default (`SPM_PROTOBUF_PROVIDER=internal`), which is what we're using.

3. **Entropy Convergence**: For small corpora, the target entropy might not be achievable. The system will try to get as close as possible within the tolerance.

4. **Performance**: `EMPIRICAL_FREQUENCIES` is more accurate but slower than `MODEL_PROBABILITIES` as it requires tokenizing the corpus.

## Troubleshooting

### Build Errors

If you see protobuf-related errors:
1. Make sure proto files are regenerated with protoc 3.14.0
2. Use standard build method (internal protobuf)
3. Clean build: `rm -rf build && mkdir build && cd build && cmake .. && make`

### Training Errors

- **"Vocabulary size too high"**: Reduce `--vocab_size` for small corpora
- **Entropy not converging**: Try adjusting `--entropy_tolerance` or use `ENTROPY_BOTH` mode

## Files Modified

- `src/sentencepiece_model.proto` - Added entropy parameters
- `src/unigram_model_trainer.h` - Added entropy calculation methods
- `src/unigram_model_trainer.cc` - Implemented entropy-guided training
- `src/spm_train_main.cc` - Added CLI flags
- `src/spec_parser.h` - Added parameter parsing
- `src/unigram_model_trainer_test.cc` - Added tests

## Next Steps

1. Train on your actual corpus with desired entropy target
2. Experiment with different α values (1.0, 2.0, etc.)
3. Compare MODEL_PROBABILITIES vs EMPIRICAL_FREQUENCIES
4. Adjust tolerance for faster/more precise convergence

