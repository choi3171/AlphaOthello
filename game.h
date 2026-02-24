#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace game {

struct GameSpec {
  const char* name;
  int board_size;
  int action_size;
  int input_channels;
  size_t state_size;

  void (*init_state)(void* out_state);
  void (*canonical_board)(const void* state, int8_t player, void* out_state);
  void (*flipped_perspective)(const void* state, void* out_state);
  int (*get_valid_moves)(const void* state, int* moves_out);
  bool (*is_full)(const void* state);
  bool (*apply_move)(void* state, int action, int8_t player);
  bool (*check_win)(const void* state, int action, int8_t player);
  void (*encode_state)(const void* state, float* out_encoded);
  void (*to_board_plane)(const void* state, int8_t* out_plane);
};

struct Config {
  std::string name;
  int board_size = 0;
  int action_size = 0;
  int input_channels = 0;
  size_t state_size = 0;
  const GameSpec* spec = nullptr;
};

constexpr size_t kMaxStateBytes = 512;

struct State {
  std::array<uint8_t, kMaxStateBytes> bytes{};
};

void register_game_spec(const GameSpec* spec);
Config make_config(const std::string& name);

State initial_state(const Config& cfg);
State canonical_board(const Config& cfg, const State& state, int8_t player);
State flipped_perspective(const Config& cfg, const State& state);

void valid_moves(const Config& cfg, const State& state, std::vector<int>& out_moves);
int valid_moves_count(const Config& cfg, const State& state, int* moves_out);
bool is_full(const Config& cfg, const State& state);
bool apply_move(const Config& cfg, State& state, int action, int8_t player);
bool check_win(const Config& cfg, const State& state, int action, int8_t player);

void encode_state(const Config& cfg, const State& state, std::vector<float>& out_encoded);
void to_board_plane(const Config& cfg, const State& state, std::vector<int8_t>& out_plane);

inline int encoded_state_size(const Config& cfg) {
  return cfg.input_channels * cfg.board_size * cfg.board_size;
}

}  // namespace game

#define REGISTER_GAME_SPEC(SPEC_EXPR)                                   \
  namespace {                                                            \
  struct AutoRegisterGameSpec {                                          \
    AutoRegisterGameSpec() { game::register_game_spec(&(SPEC_EXPR)); }   \
  };                                                                     \
  static AutoRegisterGameSpec kAutoRegisterGameSpecInstance;             \
  }  // namespace
