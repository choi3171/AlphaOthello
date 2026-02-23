#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>

#ifdef _MSC_VER
#include <intrin.h>
#endif

namespace Quoridor {
using bitboard = uint64_t;

struct State {
  bitboard p_bits[2];
  bitboard walls_h;
  bitboard walls_v;
  bitboard h_block;
  bitboard v_block;
  int8_t walls_left[2];
  int8_t turn;
  int8_t operator[](int i) const { return 0; }
  int8_t operator[](size_t i) const { return 0; }
};

struct MoveList {
  const int* data;
  int _count;
  int size() const { return _count; }
  bool empty() const { return _count == 0; }
  int operator[](int i) const { return data[i]; }
  const int* begin() const { return data; }
  const int* end() const { return data + _count; }
};

struct BridgeMask {
  uint64_t h_mask;
  uint64_t v_mask;
  uint8_t base;
};

using Board = State;

constexpr int SIZE = 7;
constexpr int NUM_SQUARES = SIZE * SIZE;
constexpr int WALL_SIZE = SIZE - 1;
constexpr int WALL_CNT = WALL_SIZE * WALL_SIZE;
constexpr int ACTION_SIZE = NUM_SQUARES + 2 * WALL_CNT;
constexpr int kActionSize = ACTION_SIZE;
constexpr int kBoardSize = SIZE;
constexpr int WALLS_LEFT = 5;

State get_initial_state();
State apply_action(const State& state, int action_idx);
int get_valid_moves(State& state, int* moves_out);
bool check_win(const State& state, int p_idx);
State change_perspective(const State& state, int player);

// 1. initial_board() -> get_initial_state()
inline State initial_board() { return get_initial_state(); }

// 2. canonical_board() -> change_perspective() with player perspective
inline State canonical_board(const State& state, int8_t player) {
  if (player == 1) return state;
  return change_perspective(state, 1);
}

// 3. flipped_perspective() -> change_perspective() with player perspective
inline State flipped_perspective(const State& state) { return change_perspective(state, 1); }

// 4. valid_moves()
inline MoveList valid_moves(State& state) {
  static thread_local int scratch[ACTION_SIZE];
  int count = get_valid_moves(state, scratch);
  return {scratch, count};
}

// 5. is_full()
inline bool is_full(State& state) { return valid_moves(state).empty(); }

// 6. apply_move() -> apply_action()
inline bool apply_move(State& state, int action, int8_t player) {
  state.turn = (player == 1) ? 0 : 1;
  state = apply_action(state, action);
  return true;
}

// 7. check_win() -> overloaded check_win() with player perspective
inline bool check_win(const State& state, int action, int8_t player) {
  int p_idx = (player == 1) ? 0 : 1;
  return check_win(state, p_idx);
}

}  // namespace Quoridor

namespace gomoku = Quoridor;  // for porting