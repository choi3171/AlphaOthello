#include "game.h"

#include <cstddef>

#include "quoridor.h"

namespace {

using Board = Quoridor::Board;

const Board& as_board_const(const void* p) {
  return *reinterpret_cast<const Board*>(p);
}

Board& as_board(void* p) {
  return *reinterpret_cast<Board*>(p);
}

void init_state(void* out_state) {
  as_board(out_state) = Quoridor::initial_board();
}

void canonical_board(const void* state, int8_t player, void* out_state) {
  as_board(out_state) = Quoridor::canonical_board(as_board_const(state), player);
}

void flipped_perspective(const void* state, void* out_state) {
  as_board(out_state) = Quoridor::flipped_perspective(as_board_const(state));
}

int get_valid_moves(const void* state, int* moves_out) {
  const auto valid = Quoridor::valid_moves(as_board_const(state));
  int count = 0;
  for (int action : valid) {
    moves_out[count++] = action;
  }
  return count;
}

bool is_full(const void* state) {
  return Quoridor::is_full(as_board_const(state));
}

bool apply_move(void* state, int action, int8_t player) {
  return Quoridor::apply_move(as_board(state), action, player);
}

bool check_win(const void* state, int action, int8_t player) {
  return Quoridor::check_win(as_board_const(state), action, player);
}

void encode_state(const void* state, float* out_encoded) {
  const Board& s = as_board_const(state);
  const int area = Quoridor::kBoardSize * Quoridor::kBoardSize;
  for (int i = 0; i < area; i++) {
    out_encoded[static_cast<size_t>(i)] = Quoridor::bit_test(s.p_bits[0], i) ? 1.0f : 0.0f;
    out_encoded[static_cast<size_t>(area + i)] = Quoridor::bit_test(s.p_bits[1], i) ? 1.0f : 0.0f;
    out_encoded[static_cast<size_t>(2 * area + i)] = Quoridor::bit_test(s.h_block, i) ? 1.0f : 0.0f;
    out_encoded[static_cast<size_t>(3 * area + i)] = Quoridor::bit_test(s.v_block, i) ? 1.0f : 0.0f;
  }
  const float my_walls = static_cast<float>(s.walls_left[0]) / static_cast<float>(Quoridor::WALLS_LEFT);
  const float opp_walls = static_cast<float>(s.walls_left[1]) / static_cast<float>(Quoridor::WALLS_LEFT);
  for (int i = 0; i < area; i++) {
    out_encoded[static_cast<size_t>(4 * area + i)] = my_walls;
    out_encoded[static_cast<size_t>(5 * area + i)] = opp_walls;
  }
}

void to_board_plane(const void* state, int8_t* out_plane) {
  const Board& s = as_board_const(state);
  const int area = Quoridor::kBoardSize * Quoridor::kBoardSize;
  for (int i = 0; i < area; i++) {
    if (Quoridor::bit_test(s.p_bits[0], i)) {
      out_plane[static_cast<size_t>(i)] = 1;
    } else if (Quoridor::bit_test(s.p_bits[1], i)) {
      out_plane[static_cast<size_t>(i)] = -1;
    } else {
      out_plane[static_cast<size_t>(i)] = 0;
    }
  }
}

const game::GameSpec kSpec{
    "quoridor",
    Quoridor::kBoardSize,
    Quoridor::kActionSize,
    Quoridor::kInputChannels,
    sizeof(Board),
    &init_state,
    &canonical_board,
    &flipped_perspective,
    &get_valid_moves,
    &is_full,
    &apply_move,
    &check_win,
    &encode_state,
    &to_board_plane,
};

}  // namespace

REGISTER_GAME_SPEC(kSpec);
