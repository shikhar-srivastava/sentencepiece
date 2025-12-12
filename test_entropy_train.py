import sentencepiece as spm

# Train with entropy constraints - equivalent to command line
spm.SentencePieceTrainer.train(
    input='test_corpus.txt',
    model_prefix='test_model_py',
    vocab_size=40,
    model_type='unigram',
    target_renyi_entropy=3.5,
    renyi_alpha=2.0,
    entropy_distribution_type='EMPIRICAL_FREQUENCIES',
    entropy_optimization_mode='ENTROPY_BOTH',
    entropy_tolerance=0.005
)

print("Training completed! Model saved as test_model_py.model")
