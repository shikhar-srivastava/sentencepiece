// Copyright 2016 Google Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.!

#include "unigram_model_trainer.h"

#include <string>
#include <vector>

#include "filesystem.h"
#include "sentencepiece_model.pb.h"
#include "sentencepiece_processor.h"
#include "sentencepiece_trainer.h"
#include "testharness.h"
#include "third_party/absl/strings/str_cat.h"
#include "third_party/absl/strings/str_join.h"
#include "util.h"

namespace sentencepiece {
namespace unigram {

// Space symbol
#define WS "\xe2\x96\x81"

TEST(UnigramTrainerTest, TrainerModelTest) {
  TrainerSpec trainer_spec;
  NormalizerSpec normalizer_spec;
  const TrainerModel model(trainer_spec, normalizer_spec);
  EXPECT_EQ(EncodeResult(), model.Encode("test"));
}

struct TrainerResult {
  std::string sentence_pieces;
  std::vector<std::pair<std::string, float>> seed_pieces_and_probs;
};

TrainerResult RunTrainer(const std::vector<std::string>& input, int size,
                         const bool use_dp = false, const float dp_noise = 0.0,
                         const uint32_t dp_clip = 0) {
  const std::string input_file =
      util::JoinPath(::testing::TempDir(), "input");
  const std::string model_prefix =
      util::JoinPath(::testing::TempDir(), "model");
  {
    auto output = filesystem::NewWritableFile(input_file);
    for (const auto& line : input) {
      output->WriteLine(line);
    }
  }

  TrainerSpec trainer_spec;
  trainer_spec.set_input_format("tsv");
  trainer_spec.set_model_type(TrainerSpec::UNIGRAM);
  trainer_spec.add_input(input_file);
  trainer_spec.set_vocab_size(size - 3);  // remove <unk>, <s>, </s>
  trainer_spec.set_model_prefix(model_prefix);

  trainer_spec.set_enable_differential_privacy(use_dp);
  trainer_spec.set_differential_privacy_noise_level(dp_noise);
  trainer_spec.set_differential_privacy_clipping_threshold(dp_clip);

  NormalizerSpec normalizer_spec;
  normalizer_spec.set_name("identity");
  normalizer_spec.set_add_dummy_prefix(false);

  NormalizerSpec denormalizer_spec;

  std::vector<std::pair<std::string, float>> seed_pieces;

  {
    Trainer trainer(trainer_spec, normalizer_spec, denormalizer_spec);
    EXPECT_OK(trainer.LoadSentences());
    TrainerModel::SentencePieces res = trainer.MakeSeedSentencePieces();

    for (const auto& piece : res) {
      seed_pieces.emplace_back(piece.first, piece.second);
    }
  }

  std::vector<std::string> pieces;

  {
    Trainer trainer(trainer_spec, normalizer_spec, denormalizer_spec);
    EXPECT_TRUE(trainer.Train().ok());

    SentencePieceProcessor processor;
    EXPECT_TRUE(processor.Load(model_prefix + ".model").ok());

    const auto& model = processor.model_proto();

    // remove <unk>, <s>, </s>
    for (int i = 3; i < model.pieces_size(); ++i) {
      pieces.emplace_back(model.pieces(i).piece());
    }
  }

  TrainerResult res;
  res.seed_pieces_and_probs = seed_pieces;
  std::sort(pieces.begin(), pieces.end());
  res.sentence_pieces = absl::StrJoin(pieces, " ");
  return res;
}

TEST(UnigramTrainerTest, BasicTest) {
  const auto& res = RunTrainer(
      {"magnanimity \t 5", "Pineapple \t 6", "i have an apple and a pen \t 1",
       "Overly \t 6", "Available \t 3"},
      30);

  // Check seed pieces.
  EXPECT_EQ(27, res.seed_pieces_and_probs.size());

  // Check final pieces.
  EXPECT_EQ("A O P a an apple b d e g h i l le m n p r t v ve y ▁ ▁an",
            res.sentence_pieces);
}

TEST(UnigramTrainerTest, BasicDPTest) {
  // no noise, clipping.
  {
    const auto& res = RunTrainer(
        {"magnanimity \t 5", "Pineapple \t 6", "i have an apple and a pen \t 1",
         "Overly \t 6", "Available \t 5"},
        22, true /*use_dp*/, 0 /*dp_noise*/, 4 /*dp_clipping*/);

    // Got 16 instead of 27 seeds.
    EXPECT_EQ(16, res.seed_pieces_and_probs.size());

    // And they are equiv to if the last sentence was not there.
    const auto& res_nodp = RunTrainer(
        {"magnanimity \t 5", "Pineapple \t 6", "Overly \t 6", "Available \t 5"},
        22);

    EXPECT_EQ(res.seed_pieces_and_probs, res_nodp.seed_pieces_and_probs);

    // Check final pieces.
    EXPECT_EQ(res.sentence_pieces, res_nodp.sentence_pieces);
  }
}

namespace {

static constexpr char kTestInputData[] = "wagahaiwa_nekodearu.txt";

TEST(UnigramTrainerTest, EndToEndTest) {
  const std::string input =
      util::JoinPath(::testing::SrcDir(), kTestInputData);

  ASSERT_TRUE(
      SentencePieceTrainer::Train(
          absl::StrCat(
              "--model_prefix=",
              util::JoinPath(::testing::TempDir(), "tmp_model"),
              " --input=", input,
              " --vocab_size=8000 --normalization_rule_name=identity",
              " --model_type=unigram --user_defined_symbols=<user>",
              " --control_symbols=<ctrl> --max_sentence_length=2048"))
          .ok());

  SentencePieceProcessor sp;
  EXPECT_TRUE(sp.Load(util::JoinPath(::testing::TempDir(),
                                     "tmp_model.model"))
                  .ok());
  EXPECT_EQ(8000, sp.GetPieceSize());

  const int cid = sp.PieceToId("<ctrl>");
  const int uid = sp.PieceToId("<user>");
  EXPECT_TRUE(sp.IsControl(cid));
  EXPECT_FALSE(sp.IsUnknown(uid));

  std::vector<std::string> tok;

  EXPECT_TRUE(sp.Encode("", &tok).ok());
  EXPECT_TRUE(tok.empty());

  EXPECT_TRUE(sp.Encode("吾輩《わがはい》は猫である。名前はまだ無い。"
                        "どこで生れたかとんと見当《けんとう》がつかぬ。"
                        "何でも薄暗いじめじめした所でニャーニャー泣いていた事だ"
                        "けは記憶している"
                        "。",
                        &tok)
                  .ok());
  // TODO(taku): Temporally disable this test on Windows.
#ifndef OS_WIN
  LOG(INFO) << "[" << absl::StrJoin(tok, " ") << std::endl;
  EXPECT_EQ(
      WS
      " 吾輩 《 わが はい 》 は猫である 。 名前はまだ 無 い 。 どこ で 生 れた "
      "か とん と 見当 《 けん とう 》 が つか ぬ 。 何でも 薄 暗 い じめ じめ "
      "した 所で ニャーニャー 泣 い ていた 事 だけは 記憶 している 。",
      absl::StrJoin(tok, " "));
#endif
}

// Test Renyi entropy calculation
TEST(UnigramTrainerTest, ComputeRenyiEntropyTest) {
  TrainerSpec trainer_spec;
  trainer_spec.set_model_type(TrainerSpec::UNIGRAM);
  NormalizerSpec normalizer_spec;
  NormalizerSpec denormalizer_spec;
  
  Trainer trainer(trainer_spec, normalizer_spec, denormalizer_spec);
  
  // Test uniform distribution
  {
    std::vector<float> uniform_probs = {0.25, 0.25, 0.25, 0.25};
    
    // Shannon entropy (alpha=1) for uniform distribution of size 4
    // H = -sum(p * log(p)) = -4 * (0.25 * log(0.25)) = log(4) ≈ 1.386
    float entropy_shannon = trainer.ComputeRenyiEntropy(uniform_probs, 1.0);
    EXPECT_NEAR(entropy_shannon, std::log(4.0), 0.01);
    
    // Collision entropy (alpha=2) for uniform distribution
    // H = -log(sum(p^2)) = -log(4 * 0.25^2) = -log(0.25) = log(4)
    float entropy_collision = trainer.ComputeRenyiEntropy(uniform_probs, 2.0);
    EXPECT_NEAR(entropy_collision, std::log(4.0), 0.01);
    
    // Hartley entropy (alpha=0) should be log(vocab_size)
    float entropy_hartley = trainer.ComputeRenyiEntropy(uniform_probs, 0.0);
    EXPECT_NEAR(entropy_hartley, std::log(4.0), 0.01);
  }
  
  // Test non-uniform distribution
  {
    std::vector<float> skewed_probs = {0.5, 0.3, 0.15, 0.05};
    
    // Shannon entropy should be less than uniform case
    float entropy_shannon = trainer.ComputeRenyiEntropy(skewed_probs, 1.0);
    EXPECT_LT(entropy_shannon, std::log(4.0));
    EXPECT_GT(entropy_shannon, 0.0);
    
    // Expected: -0.5*log(0.5) - 0.3*log(0.3) - 0.15*log(0.15) - 0.05*log(0.05)
    float expected_shannon = -(0.5 * std::log(0.5) + 
                                0.3 * std::log(0.3) + 
                                0.15 * std::log(0.15) + 
                                0.05 * std::log(0.05));
    EXPECT_NEAR(entropy_shannon, expected_shannon, 0.01);
    
    // Collision entropy
    float entropy_collision = trainer.ComputeRenyiEntropy(skewed_probs, 2.0);
    float expected_collision = -std::log(0.5*0.5 + 0.3*0.3 + 0.15*0.15 + 0.05*0.05);
    EXPECT_NEAR(entropy_collision, expected_collision, 0.01);
  }
  
  // Test edge cases
  {
    // Empty distribution
    std::vector<float> empty_probs;
    float entropy_empty = trainer.ComputeRenyiEntropy(empty_probs, 2.0);
    EXPECT_EQ(entropy_empty, 0.0);
    
    // Distribution with zeros
    std::vector<float> probs_with_zeros = {0.5, 0.0, 0.5, 0.0};
    float entropy = trainer.ComputeRenyiEntropy(probs_with_zeros, 2.0);
    // Should effectively be a distribution of size 2
    EXPECT_NEAR(entropy, std::log(2.0), 0.01);
  }
}

// Test entropy target checking
TEST(UnigramTrainerTest, IsEntropyTargetMetTest) {
  TrainerSpec trainer_spec;
  trainer_spec.set_model_type(TrainerSpec::UNIGRAM);
  trainer_spec.set_target_renyi_entropy(10.0);
  trainer_spec.set_entropy_tolerance(0.01);  // 1% tolerance
  
  NormalizerSpec normalizer_spec;
  NormalizerSpec denormalizer_spec;
  
  Trainer trainer(trainer_spec, normalizer_spec, denormalizer_spec);
  
  // Within tolerance (10.0 ± 1%)
  EXPECT_TRUE(trainer.IsEntropyTargetMet(10.0));
  EXPECT_TRUE(trainer.IsEntropyTargetMet(10.05));
  EXPECT_TRUE(trainer.IsEntropyTargetMet(9.95));
  
  // Outside tolerance
  EXPECT_FALSE(trainer.IsEntropyTargetMet(10.2));
  EXPECT_FALSE(trainer.IsEntropyTargetMet(9.8));
  EXPECT_FALSE(trainer.IsEntropyTargetMet(11.0));
  EXPECT_FALSE(trainer.IsEntropyTargetMet(9.0));
}

// Integration test: training with entropy constraint
TEST(UnigramTrainerTest, TrainWithEntropyConstraintTest) {
  const std::string input_file =
      util::JoinPath(::testing::TempDir(), "entropy_input");
  const std::string model_prefix =
      util::JoinPath(::testing::TempDir(), "entropy_model");
  
  // Create test corpus
  {
    auto output = filesystem::NewWritableFile(input_file);
    for (int i = 0; i < 100; ++i) {
      output->WriteLine("hello world\t100");
      output->WriteLine("this is a test\t100");
      output->WriteLine("sentence piece tokenizer\t100");
    }
  }
  
  TrainerSpec trainer_spec;
  trainer_spec.set_input_format("tsv");
  trainer_spec.set_model_type(TrainerSpec::UNIGRAM);
  trainer_spec.add_input(input_file);
  trainer_spec.set_vocab_size(50);
  trainer_spec.set_model_prefix(model_prefix);
  
  // Set entropy parameters
  trainer_spec.set_target_renyi_entropy(3.5);  // Target entropy
  trainer_spec.set_renyi_alpha(2.0);  // Collision entropy
  trainer_spec.set_entropy_distribution_type(TrainerSpec::EMPIRICAL_FREQUENCIES);
  trainer_spec.set_entropy_optimization_mode(TrainerSpec::ENTROPY_BOTH);
  trainer_spec.set_entropy_tolerance(0.05);  // 5% tolerance for test
  
  NormalizerSpec normalizer_spec;
  normalizer_spec.set_name("identity");
  normalizer_spec.set_add_dummy_prefix(false);
  
  NormalizerSpec denormalizer_spec;
  
  // Train the model
  Trainer trainer(trainer_spec, normalizer_spec, denormalizer_spec);
  EXPECT_TRUE(trainer.Train().ok());
  
  // Verify model was created
  SentencePieceProcessor processor;
  EXPECT_TRUE(processor.Load(model_prefix + ".model").ok());
  
  // The model should have been trained successfully
  // We don't test exact entropy here as it depends on the corpus,
  // but we verify the training completed without errors
  EXPECT_GT(processor.GetPieceSize(), 0);
}

// Test different optimization modes
TEST(UnigramTrainerTest, EntropyOptimizationModesTest) {
  const std::string input_file =
      util::JoinPath(::testing::TempDir(), "modes_input");
  
  // Create test corpus
  {
    auto output = filesystem::NewWritableFile(input_file);
    for (int i = 0; i < 50; ++i) {
      output->WriteLine("test sentence\t100");
    }
  }
  
  // Test each mode doesn't crash
  std::vector<TrainerSpec::EntropyOptimizationMode> modes = {
      TrainerSpec::ENTROPY_DISABLED,
      TrainerSpec::ENTROPY_PRUNING_CONSTRAINT,
      TrainerSpec::ENTROPY_STOPPING_CRITERION,
      TrainerSpec::ENTROPY_BOTH
  };
  
  for (const auto& mode : modes) {
    const std::string model_prefix =
        util::JoinPath(::testing::TempDir(), 
                      absl::StrCat("mode_model_", static_cast<int>(mode)));
    
    TrainerSpec trainer_spec;
    trainer_spec.set_input_format("tsv");
    trainer_spec.set_model_type(TrainerSpec::UNIGRAM);
    trainer_spec.add_input(input_file);
    trainer_spec.set_vocab_size(30);
    trainer_spec.set_model_prefix(model_prefix);
    trainer_spec.set_target_renyi_entropy(3.0);
    trainer_spec.set_entropy_optimization_mode(mode);
    
    NormalizerSpec normalizer_spec;
    normalizer_spec.set_name("identity");
    normalizer_spec.set_add_dummy_prefix(false);
    
    NormalizerSpec denormalizer_spec;
    
    Trainer trainer(trainer_spec, normalizer_spec, denormalizer_spec);
    EXPECT_TRUE(trainer.Train().ok());
  }
}

}  // namespace
}  // namespace unigram
}  // namespace sentencepiece
