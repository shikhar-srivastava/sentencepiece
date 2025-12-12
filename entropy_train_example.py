#!/usr/bin/env python
"""
Example: Training SentencePiece Unigram with Rényi Entropy Constraints

This is equivalent to the command line:
./build/src/spm_train \
  --input=test_corpus.txt \
  --model_prefix=test_model \
  --vocab_size=40 \
  --model_type=unigram \
  --target_renyi_entropy=3.5 \
  --renyi_alpha=2.0 \
  --entropy_distribution_type=EMPIRICAL_FREQUENCIES \
  --entropy_optimization_mode=ENTROPY_BOTH \
  --entropy_tolerance=0.005
"""

import sentencepiece as spm

# Train with Rényi entropy constraints
spm.SentencePieceTrainer.train(
    # Standard parameters
    input='test_corpus.txt',
    model_prefix='test_model_py',
    vocab_size=40,
    model_type='unigram',
    
    # Rényi entropy parameters
    target_renyi_entropy=3.5,                           # Target entropy value (0 = disabled)
    renyi_alpha=2.0,                                     # Alpha: 0=Hartley, 1=Shannon, 2=Collision
    entropy_distribution_type='EMPIRICAL_FREQUENCIES',  # or 'MODEL_PROBABILITIES'
    entropy_optimization_mode='ENTROPY_BOTH',           # or 'ENTROPY_DISABLED', 'ENTROPY_PRUNING_CONSTRAINT', 'ENTROPY_STOPPING_CRITERION'
    entropy_tolerance=0.005                             # ±0.5% tolerance
)

print("✓ Training completed!")
print("✓ Model saved as test_model_py.model")

# Load and use the trained model
sp = spm.SentencePieceProcessor(model_file='test_model_py.model')
text = "hello world this is a test"
encoded = sp.encode(text, out_type=str)
print(f"\nExample encoding of '{text}':")
print(f"Tokens: {encoded}")


