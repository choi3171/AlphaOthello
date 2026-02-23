#include "search.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {

struct AtomicSearchProfile {
  std::atomic<uint64_t> games{0};
  std::atomic<uint64_t> game_total_ns{0};
  std::atomic<uint64_t> mcts_calls{0};
  std::atomic<uint64_t> mcts_total_ns{0};
  std::atomic<uint64_t> infer_calls{0};
  std::atomic<uint64_t> infer_total_ns{0};
};

class ScopedAddNs {
 public:
  explicit ScopedAddNs(std::atomic<uint64_t>* target) : target_(target), start_(std::chrono::steady_clock::now()) {}

  ~ScopedAddNs() {
    if (target_ == nullptr) {
      return;
    }
    const auto end = std::chrono::steady_clock::now();
    const auto ns = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start_).count());
    target_->fetch_add(ns, std::memory_order_relaxed);
  }

 private:
  std::atomic<uint64_t>* target_;
  std::chrono::steady_clock::time_point start_;
};

struct Node {
  gomoku::Board state{};
  int parent = -1;
  int first_child = -1;
  int num_children = 0;
  int action_taken = -1;
  float prior = 0.0f;
  int visit_count = 0;
  float value_sum = 0.0f;
  bool terminal_known = false;
  bool terminal = false;
  float terminal_value = 0.0f;
};

inline float ucb_score(const Node& child, float cpuct, float sqrt_parent_visits) {
  const float q = (child.visit_count == 0) ? 0.0f : -(child.value_sum / static_cast<float>(child.visit_count));
  const float u = cpuct * child.prior * (sqrt_parent_visits / static_cast<float>(child.visit_count + 1));
  return q + u;
}

int select_child(int node_idx, const std::vector<Node>& tree, float cpuct) {
  int best_idx = -1;
  float best_score = -1e30f;
  const Node& parent = tree[node_idx];

  const int start = parent.first_child;
  const int end = start + parent.num_children;

  const float sqrt_parent_visits = std::sqrt(static_cast<float>(std::max(parent.visit_count, 1)));

  for (int i = start; i < end; i++) {
    const float score = ucb_score(tree[i], cpuct, sqrt_parent_visits);
    if (score > best_score) {
      best_score = score;
      best_idx = i;
    }
  }
  return best_idx;
}

bool evaluate_terminal(Node& node) {
  if (node.terminal_known) return node.terminal;
  node.terminal_known = true;
  if (node.action_taken >= 0 &&
      gomoku::check_win(node.state, node.action_taken, -1)) {
    node.terminal = true;
    node.terminal_value = -1.0f;
    return true;
  }
  if (gomoku::is_full(node.state)) {
    node.terminal = true;
    node.terminal_value = 0.0f;
    return true;
  }
  node.terminal = false;
  node.terminal_value = 0.0f;
  return false;
}

void backpropagate(int node_idx, std::vector<Node>& tree, float value) {
  while (node_idx != -1) {
    tree[node_idx].visit_count += 1;
    tree[node_idx].value_sum += value;
    value = -value;
    node_idx = tree[node_idx].parent;
  }
}

std::array<float, gomoku::kActionSize> masked_normalized_policy(const std::array<float, gomoku::kActionSize>& policy,
                                                                const gomoku::MoveList& valid_moves) {
  std::array<float, gomoku::kActionSize> out{};
  out.fill(0.0f);
  if (valid_moves.empty()) return out;
  double sum = 0.0;
  for (int a : valid_moves) {
    const float p = std::max(0.0f, policy[a]);
    out[a] = p;
    sum += p;
  }
  if (sum <= 1e-12) {
    const float uniform = 1.0f / static_cast<float>(valid_moves.size());
    for (int a : valid_moves) out[a] = uniform;
    return out;
  }
  for (int a : valid_moves) out[a] = static_cast<float>(out[a] / sum);
  return out;
}

void apply_dirichlet_noise(std::array<float, gomoku::kActionSize>& policy, const gomoku::MoveList& valid_moves,
                           float epsilon, float alpha, std::mt19937& rng) {
  if (epsilon <= 0.0f || valid_moves.empty()) return;
  std::gamma_distribution<float> gamma(alpha, 1.0f);
  std::vector<float> noise(valid_moves.size(), 0.0f);
  float sum = 0.0f;
  for (size_t i = 0; i < valid_moves.size(); i++) {
    noise[i] = gamma(rng);
    sum += noise[i];
  }
  if (sum <= 1e-12f) return;
  for (size_t i = 0; i < valid_moves.size(); i++) noise[i] /= sum;
  for (size_t i = 0; i < valid_moves.size(); i++) {
    const int a = valid_moves[i];
    policy[a] = (1.0f - epsilon) * policy[a] + epsilon * noise[i];
  }
}

void expand_batch(int node_idx, std::vector<Node>& tree, const std::array<float, gomoku::kActionSize>& policy) {
  const auto valid = gomoku::valid_moves(tree[node_idx].state);
  tree[node_idx].first_child = static_cast<int>(tree.size());
  int child_count = 0;
  for (int action : valid) {
    if (policy[action] <= 0.0f) continue;
    gomoku::Board child_state = tree[node_idx].state;
    gomoku::apply_move(child_state, action, 1);
    child_state = gomoku::flipped_perspective(child_state);

    tree.emplace_back();
    Node& child = tree.back();
    child.state = child_state;
    child.parent = node_idx;
    child.action_taken = action;
    child.prior = policy[action];
    child_count++;
  }
  tree[node_idx].num_children = child_count;
}

std::vector<std::pair<std::array<float, gomoku::kActionSize>, float>> infer_batch_with_profile(
    OnnxInfer& infer, const std::vector<gomoku::Board>& states, AtomicSearchProfile* profile) {
  if (states.empty()) return {};
  const auto infer_start = std::chrono::steady_clock::now();
  auto results = infer.infer_batch(states);
  if (profile != nullptr) {
    const auto infer_end = std::chrono::steady_clock::now();
    const auto ns =
        static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(infer_end - infer_start).count());
    profile->infer_total_ns.fetch_add(ns, std::memory_order_relaxed);
    profile->infer_calls.fetch_add(static_cast<uint64_t>(states.size()), std::memory_order_relaxed);
  }
  return results;
}

std::vector<std::array<float, gomoku::kActionSize>> run_mcts_batch(
    OnnxInfer& infer, const std::vector<gomoku::Board>& canonical_roots, const SearchParams& params, std::mt19937& rng,
    AtomicSearchProfile* profile, std::vector<float>* average_leaf_depths, std::vector<float>* max_leaf_depths,
    std::vector<Node>& tree) {
  const size_t num_roots = canonical_roots.size();
  std::vector<std::array<float, gomoku::kActionSize>> all_action_probs(num_roots);
  if (num_roots == 0) return all_action_probs;

  ScopedAddNs mcts_timer(profile ? &profile->mcts_total_ns : nullptr);
  if (profile != nullptr) {
    profile->mcts_calls.fetch_add(static_cast<uint64_t>(num_roots), std::memory_order_relaxed);
  }

  tree.clear();
  std::vector<int> root_indices(num_roots);

  for (size_t i = 0; i < num_roots; ++i) {
    Node root;
    root.state = canonical_roots[i];
    root.visit_count = 1;
    root_indices[i] = static_cast<int>(tree.size());
    tree.push_back(root);
  }

  const auto root_evals = infer_batch_with_profile(infer, canonical_roots, profile);
  for (size_t i = 0; i < num_roots; ++i) {
    auto policy = masked_normalized_policy(root_evals[i].first, gomoku::valid_moves(tree[root_indices[i]].state));
    apply_dirichlet_noise(policy, gomoku::valid_moves(tree[root_indices[i]].state), params.dirichlet_epsilon,
                          params.dirichlet_alpha, rng);
    policy = masked_normalized_policy(policy, gomoku::valid_moves(tree[root_indices[i]].state));
    expand_batch(root_indices[i], tree, policy);
  }

  std::vector<int> depth_sum(num_roots, 0);
  std::vector<int> depth_max(num_roots, 0);
  std::vector<int> depth_count(num_roots, 0);
  std::vector<int> curr_leaves(num_roots);
  std::vector<gomoku::Board> states_to_infer;
  std::vector<int> game_indices_to_infer;
  states_to_infer.reserve(num_roots);
  game_indices_to_infer.reserve(num_roots);

  for (int s = 0; s < params.num_searches; ++s) {
    states_to_infer.clear();
    game_indices_to_infer.clear();
    for (size_t i = 0; i < num_roots; ++i) {
      int curr = root_indices[i];
      int depth = 0;
      while (tree[curr].num_children > 0) {
        curr = select_child(curr, tree, params.cpuct);
        depth++;
      }
      curr_leaves[i] = curr;
      depth_sum[i] += depth;
      depth_max[i] = std::max(depth_max[i], depth);
      depth_count[i]++;

      if (!evaluate_terminal(tree[curr])) {
        states_to_infer.push_back(tree[curr].state);
        game_indices_to_infer.push_back(static_cast<int>(i));
      } else {
        backpropagate(curr, tree, tree[curr].terminal_value);
      }
    }

    if (!states_to_infer.empty()) {
      auto evals = infer_batch_with_profile(infer, states_to_infer, profile);
      for (size_t j = 0; j < evals.size(); ++j) {
        int game_idx = game_indices_to_infer[j];
        int leaf_idx = curr_leaves[game_idx];
        auto policy = masked_normalized_policy(evals[j].first, gomoku::valid_moves(tree[leaf_idx].state));
        expand_batch(leaf_idx, tree, policy);
        backpropagate(leaf_idx, tree, evals[j].second);
      }
    }
  }

  for (size_t i = 0; i < num_roots; ++i) {
    int root_idx = root_indices[i];
    float sum_visits = 0;
    int start = tree[root_idx].first_child;
    int end = start + tree[root_idx].num_children;
    for (int j = start; j < end; ++j) {
      all_action_probs[i][tree[j].action_taken] = static_cast<float>(tree[j].visit_count);
      sum_visits += all_action_probs[i][tree[j].action_taken];
    }
    for (int a = 0; a < gomoku::kActionSize; ++a) {
      all_action_probs[i][a] /= (sum_visits + 1e-9f);
    }
  }

  if (average_leaf_depths) {
    average_leaf_depths->assign(num_roots, 0.0f);
    for (size_t i = 0; i < num_roots; ++i) {
      if (depth_count[i] > 0) (*average_leaf_depths)[i] = (float)depth_sum[i] / depth_count[i];
    }
  }
  if (max_leaf_depths) {
    max_leaf_depths->assign(num_roots, 0.0f);
    for (size_t i = 0; i < num_roots; ++i) {
      (*max_leaf_depths)[i] = (float)depth_max[i];
    }
  }

  return std::move(all_action_probs);
}

int sample_action(const std::array<float, gomoku::kActionSize>& probs, const gomoku::MoveList& valid_moves,
                  float temperature, std::mt19937& rng) {
  if (valid_moves.empty()) {
    return 0;
  }

  if (temperature <= 1e-6f) {
    int best_action = valid_moves[0];
    float best_prob = probs[best_action];
    for (int a : valid_moves) {
      if (probs[a] > best_prob) {
        best_prob = probs[a];
        best_action = a;
      }
    }
    return best_action;
  }

  std::vector<double> weights(valid_moves.size(), 0.0);
  double sum = 0.0;
  const double inv_temp = 1.0 / static_cast<double>(temperature);
  for (size_t i = 0; i < valid_moves.size(); i++) {
    const double w = std::pow(std::max(0.0f, probs[valid_moves[i]]), inv_temp);
    weights[i] = w;
    sum += w;
  }
  if (sum <= 1e-12) {
    std::uniform_int_distribution<int> dist(
        0, static_cast<int>(valid_moves.size()) - 1);
    return valid_moves[dist(rng)];
  }
  for (double& w : weights) {
    w /= sum;
  }
  std::discrete_distribution<int> dist(weights.begin(), weights.end());
  return valid_moves[dist(rng)];
}

float scheduled_temperature(const SearchParams& params, int move_number) {
  const double base = static_cast<double>(params.temperature);
  const double early = static_cast<double>(params.temperature_early);
  const double halflife = static_cast<double>(params.temperature_halflife);
  if (halflife <= 1e-9) {
    return static_cast<float>(std::max(0.0, base));
  }

  const int clamped_move = std::max(0, move_number);
  const double board_area = static_cast<double>(gomoku::kActionSize);
  const double board_scale = 19.0 / std::sqrt(board_area);
  const double halflives = (static_cast<double>(clamped_move) / halflife) * board_scale;
  const double scheduled = base + (early - base) * std::pow(0.5, halflives);
  return static_cast<float>(std::max(0.0, scheduled));
}

struct HistStep {
  gomoku::Board canonical{};
  std::array<float, gomoku::kActionSize> policy{};
  int8_t player = 1;
};

struct GameResult {
  std::vector<TrainingRow> rows;
  int winner = 0;
  gomoku::Board final_state{};
  std::vector<float> average_depth;
  std::vector<float> max_depth;
};

struct ActiveGame {
  gomoku::Board board = gomoku::initial_board();
  int8_t player = 1;
  std::vector<HistStep> hist;
  std::vector<float> average_depth;
  std::vector<float> max_depth;
  std::chrono::steady_clock::time_point start_time =
      std::chrono::steady_clock::now();
};

GameResult finalize_game(ActiveGame&& game, int winner) {
  std::vector<TrainingRow> rows;
  rows.reserve(game.hist.size());
  for (const HistStep& h : game.hist) {
    TrainingRow row;
    row.state = h.canonical;
    row.policy = h.policy;
    if (winner == 0) {
      row.value = 0.0f;
    } else {
      row.value = (winner == h.player) ? 1.0f : -1.0f;
    }
    rows.push_back(row);
  }

  GameResult result;
  result.rows = std::move(rows);
  result.winner = winner;
  result.final_state = game.board;
  result.average_depth = std::move(game.average_depth);
  result.max_depth = std::move(game.max_depth);
  return result;
}

}  // namespace

SelfplayResult run_selfplay_games(OnnxInfer& infer, const SearchParams& params, int num_games, int num_threads,
                                  uint64_t seed) {
  SelfplayResult result;
  if (num_games <= 0) {
    return result;
  }
  if (num_threads <= 0) {
    num_threads = 1;
  }

  // Interpret params.parallel_games as "parallel games per worker thread".
  const int parallel_games_per_worker = std::max(1, params.parallel_games);
  const int num_workers = std::max(1, std::min(num_threads, num_games));

  std::atomic<int> next_game(0);
  AtomicSearchProfile atomic_profile;
  std::mutex result_mutex;
  result.rows.reserve(static_cast<size_t>(num_games * 40));
  result.stats.average_depth_lists.reserve(static_cast<size_t>(num_games));
  result.stats.max_depth_lists.reserve(static_cast<size_t>(num_games));
  result.stats.final_states.reserve(static_cast<size_t>(num_games));

  std::vector<std::thread> workers;
  workers.reserve(num_workers);
  for (int t = 0; t < num_workers; t++) {
    const int local_parallel_games = parallel_games_per_worker;
    workers.emplace_back([&, t, local_parallel_games]() {
      std::mt19937 rng(static_cast<uint32_t>(seed + 10007ULL * static_cast<uint64_t>(t)));

      std::vector<Node> thread_tree_pool;
      thread_tree_pool.reserve(static_cast<size_t>(params.num_searches * local_parallel_games * 20));

      std::vector<ActiveGame> active_games;
      active_games.reserve(
          static_cast<size_t>(std::max(1, local_parallel_games)));

      auto refill_active_games = [&]() {
        while (static_cast<int>(active_games.size()) < local_parallel_games) {
          const int game_idx = next_game.fetch_add(1);
          if (game_idx >= num_games) {
            break;
          }
          ActiveGame game;
          game.hist.reserve(gomoku::kActionSize);
          game.average_depth.reserve(gomoku::kActionSize);
          game.max_depth.reserve(gomoku::kActionSize);
          game.start_time = std::chrono::steady_clock::now();
          active_games.push_back(std::move(game));
        }
      };

      refill_active_games();
      while (true) {
        if (active_games.empty()) {
          break;
        }

        std::vector<gomoku::Board> canonical_states(active_games.size());
        for (size_t i = 0; i < active_games.size(); i++) {
          canonical_states[i] = gomoku::canonical_board(active_games[i].board, active_games[i].player);
        }

        std::vector<float> average_depths;
        std::vector<float> max_depths;
        const auto all_action_probs = run_mcts_batch(infer, canonical_states, params, rng, &atomic_profile,
                                                     &average_depths, &max_depths, thread_tree_pool);

        for (int i = static_cast<int>(active_games.size()) - 1; i >= 0; i--) {
          ActiveGame& game = active_games[static_cast<size_t>(i)];
          game.average_depth.push_back(average_depths[static_cast<size_t>(i)]);
          game.max_depth.push_back(max_depths[static_cast<size_t>(i)]);

          HistStep step;
          step.canonical = canonical_states[static_cast<size_t>(i)];
          step.policy = all_action_probs[static_cast<size_t>(i)];
          step.player = game.player;
          game.hist.push_back(step);

          const auto valid = gomoku::valid_moves(game.board);
          const float move_temp = scheduled_temperature(params, static_cast<int>(game.hist.size()) - 1);
          const int action = sample_action(all_action_probs[static_cast<size_t>(i)], valid, move_temp, rng);

          gomoku::apply_move(game.board, action, game.player);
          const bool win = gomoku::check_win(game.board, action, game.player);
          const bool full = gomoku::is_full(game.board);
          if (win || full) {
            const auto game_end_time = std::chrono::steady_clock::now();
            const auto game_ns = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(game_end_time - game.start_time).count());
            atomic_profile.game_total_ns.fetch_add(game_ns, std::memory_order_relaxed);
            atomic_profile.games.fetch_add(1, std::memory_order_relaxed);

            const int winner = win ? game.player : 0;
            GameResult game_result = finalize_game(std::move(game), winner);
            {
              std::lock_guard<std::mutex> lock(result_mutex);
              result.rows.insert(result.rows.end(), game_result.rows.begin(), game_result.rows.end());
              result.stats.average_depth_lists.push_back(std::move(game_result.average_depth));
              result.stats.max_depth_lists.push_back(std::move(game_result.max_depth));
              result.stats.final_states.push_back(game_result.final_state);
              if (game_result.winner == 1) {
                result.stats.win += 1;
              } else if (game_result.winner == -1) {
                result.stats.lose += 1;
              } else {
                result.stats.draw += 1;
              }
            }

            if (static_cast<size_t>(i) + 1 != active_games.size()) {
              active_games[static_cast<size_t>(i)] =
                  std::move(active_games.back());
            }
            active_games.pop_back();
          } else {
            game.player = static_cast<int8_t>(-game.player);
          }
        }
        refill_active_games();
      }
    });
  }

  for (auto& w : workers) {
    w.join();
  }
  result.profile.games = atomic_profile.games.load(std::memory_order_relaxed);
  result.profile.game_total_ns =
      atomic_profile.game_total_ns.load(std::memory_order_relaxed);
  result.profile.mcts_calls =
      atomic_profile.mcts_calls.load(std::memory_order_relaxed);
  result.profile.mcts_total_ns =
      atomic_profile.mcts_total_ns.load(std::memory_order_relaxed);
  result.profile.infer_calls =
      atomic_profile.infer_calls.load(std::memory_order_relaxed);
  result.profile.infer_total_ns =
      atomic_profile.infer_total_ns.load(std::memory_order_relaxed);
  return result;
}

void write_memory_file(const std::string& path,
                       const std::vector<TrainingRow>& rows) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) {
    throw std::runtime_error("failed to open output file: " + path);
  }

  const uint32_t count = static_cast<uint32_t>(rows.size());
  out.write(reinterpret_cast<const char*>(&count), sizeof(uint32_t));

  for (const TrainingRow& r : rows) {
    out.write(reinterpret_cast<const char*>(&r.state), sizeof(gomoku::Board));

    out.write(reinterpret_cast<const char*>(r.policy.data()),
              static_cast<std::streamsize>(Quoridor::ACTION_SIZE * sizeof(float)));
    out.write(reinterpret_cast<const char*>(&r.value), sizeof(float));
  }
}

void write_stats_file(const std::string& path, const SelfplayStats& stats) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) {
    throw std::runtime_error("failed to open stats file: " + path);
  }

  out.write(reinterpret_cast<const char*>(&stats.win), sizeof(uint32_t));
  out.write(reinterpret_cast<const char*>(&stats.draw), sizeof(uint32_t));
  out.write(reinterpret_cast<const char*>(&stats.lose), sizeof(uint32_t));

  auto write_depth_lists = [&](const std::vector<std::vector<float>>& lists) {
    const uint32_t list_count = static_cast<uint32_t>(lists.size());
    out.write(reinterpret_cast<const char*>(&list_count), sizeof(uint32_t));
    for (const auto& depth_list : lists) {
      const uint32_t len = static_cast<uint32_t>(depth_list.size());
      out.write(reinterpret_cast<const char*>(&len), sizeof(uint32_t));
      if (!depth_list.empty()) {
        out.write(reinterpret_cast<const char*>(depth_list.data()),
                  static_cast<std::streamsize>(depth_list.size() * sizeof(float)));
      }
    }
  };

  write_depth_lists(stats.average_depth_lists);
  write_depth_lists(stats.max_depth_lists);

  const uint32_t final_states_count = static_cast<uint32_t>(stats.final_states.size());
  out.write(reinterpret_cast<const char*>(&final_states_count), sizeof(uint32_t));

  for (const auto& board : stats.final_states) {
    // 4. 마지막 상태 저장 (역시 주소값 & 사용)
    out.write(reinterpret_cast<const char*>(&board), sizeof(gomoku::Board));
  }
}