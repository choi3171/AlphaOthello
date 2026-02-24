#include "game.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <stdexcept>
#include <unordered_map>

namespace game {

namespace {

std::string normalize_name(std::string s) {
  for (char& ch : s) {
    ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  }
  return s;
}

std::unordered_map<std::string, const GameSpec*>& registry() {
  static std::unordered_map<std::string, const GameSpec*> table;
  return table;
}

void ensure_state_size(const Config& cfg, const State& state) {
  (void)state;
  if (cfg.state_size > kMaxStateBytes) {
    throw std::runtime_error("state size mismatch for game: " + cfg.name);
  }
}

State alloc_state(const Config& cfg) {
  (void)cfg;
  return State{};
}

}  // namespace

void register_game_spec(const GameSpec* spec) {
  if (spec == nullptr || spec->name == nullptr) {
    throw std::runtime_error("register_game_spec: invalid spec");
  }
  registry().emplace(normalize_name(spec->name), spec);
}

Config make_config(const std::string& name) {
  std::string key = normalize_name(name);
  if (key == "gomoku10") {
    key = "gomoku";
  } else if (key == "quoridor7") {
    key = "quoridor";
  }

  const auto it = registry().find(key);
  if (it == registry().end()) {
    throw std::invalid_argument("Unsupported game: " + name);
  }
  const GameSpec* spec = it->second;
  if (spec->state_size > kMaxStateBytes) {
    throw std::runtime_error("game state too large for fixed buffer: " + std::string(spec->name));
  }
  Config cfg;
  cfg.name = spec->name;
  cfg.board_size = spec->board_size;
  cfg.action_size = spec->action_size;
  cfg.input_channels = spec->input_channels;
  cfg.state_size = spec->state_size;
  cfg.spec = spec;
  return cfg;
}

State initial_state(const Config& cfg) {
  if (cfg.spec == nullptr || cfg.spec->init_state == nullptr) {
    throw std::runtime_error("initial_state: invalid config");
  }
  State out = alloc_state(cfg);
  cfg.spec->init_state(out.bytes.data());
  return out;
}

State canonical_board(const Config& cfg, const State& state, int8_t player) {
  if (cfg.spec == nullptr || cfg.spec->canonical_board == nullptr) {
    throw std::runtime_error("canonical_board: invalid config");
  }
  ensure_state_size(cfg, state);
  State out = alloc_state(cfg);
  cfg.spec->canonical_board(state.bytes.data(), player, out.bytes.data());
  return out;
}

State flipped_perspective(const Config& cfg, const State& state) {
  if (cfg.spec == nullptr || cfg.spec->flipped_perspective == nullptr) {
    throw std::runtime_error("flipped_perspective: invalid config");
  }
  ensure_state_size(cfg, state);
  State out = alloc_state(cfg);
  cfg.spec->flipped_perspective(state.bytes.data(), out.bytes.data());
  return out;
}

void valid_moves(const Config& cfg, const State& state, std::vector<int>& out_moves) {
  if (cfg.spec == nullptr || cfg.spec->get_valid_moves == nullptr) {
    throw std::runtime_error("valid_moves: invalid config");
  }
  ensure_state_size(cfg, state);
  thread_local std::vector<int> scratch;
  if (scratch.size() < static_cast<size_t>(cfg.action_size)) {
    scratch.resize(static_cast<size_t>(cfg.action_size));
  }
  const int count = cfg.spec->get_valid_moves(state.bytes.data(), scratch.data());
  if (count < 0 || count > cfg.action_size) {
    throw std::runtime_error("valid_moves: invalid move count");
  }
  out_moves.resize(static_cast<size_t>(count));
  if (count > 0) {
    std::memcpy(out_moves.data(), scratch.data(), static_cast<size_t>(count) * sizeof(int));
  }
}

int valid_moves_count(const Config& cfg, const State& state, int* moves_out) {
  if (cfg.spec == nullptr || cfg.spec->get_valid_moves == nullptr) {
    throw std::runtime_error("valid_moves_count: invalid config");
  }
  ensure_state_size(cfg, state);
  const int count = cfg.spec->get_valid_moves(state.bytes.data(), moves_out);
  if (count < 0 || count > cfg.action_size) {
    throw std::runtime_error("valid_moves_count: invalid move count");
  }
  return count;
}

bool is_full(const Config& cfg, const State& state) {
  if (cfg.spec == nullptr || cfg.spec->is_full == nullptr) {
    throw std::runtime_error("is_full: invalid config");
  }
  ensure_state_size(cfg, state);
  return cfg.spec->is_full(state.bytes.data());
}

bool apply_move(const Config& cfg, State& state, int action, int8_t player) {
  if (cfg.spec == nullptr || cfg.spec->apply_move == nullptr) {
    throw std::runtime_error("apply_move: invalid config");
  }
  ensure_state_size(cfg, state);
  return cfg.spec->apply_move(state.bytes.data(), action, player);
}

bool check_win(const Config& cfg, const State& state, int action, int8_t player) {
  if (cfg.spec == nullptr || cfg.spec->check_win == nullptr) {
    throw std::runtime_error("check_win: invalid config");
  }
  ensure_state_size(cfg, state);
  return cfg.spec->check_win(state.bytes.data(), action, player);
}

void encode_state(const Config& cfg, const State& state, std::vector<float>& out_encoded) {
  if (cfg.spec == nullptr || cfg.spec->encode_state == nullptr) {
    throw std::runtime_error("encode_state: invalid config");
  }
  ensure_state_size(cfg, state);
  out_encoded.resize(static_cast<size_t>(encoded_state_size(cfg)));
  cfg.spec->encode_state(state.bytes.data(), out_encoded.data());
}

void to_board_plane(const Config& cfg, const State& state, std::vector<int8_t>& out_plane) {
  if (cfg.spec == nullptr || cfg.spec->to_board_plane == nullptr) {
    throw std::runtime_error("to_board_plane: invalid config");
  }
  ensure_state_size(cfg, state);
  out_plane.resize(static_cast<size_t>(cfg.board_size * cfg.board_size));
  cfg.spec->to_board_plane(state.bytes.data(), out_plane.data());
}

}  // namespace game
