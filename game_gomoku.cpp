#include "game.h"

#include <cstddef>

#include "gomoku.h"

namespace {

using Board = gomoku::Board;

const Board& as_board_const(const void* p) {
  return *reinterpret_cast<const Board*>(p);
}

Board& as_board(void* p) {
  return *reinterpret_cast<Board*>(p);
}

void init_state(void* out_state) {
  as_board(out_state) = gomoku::initial_board();
}

void canonical_board(const void* state, int8_t player, void* out_state) {
  as_board(out_state) = gomoku::canonical_board(as_board_const(state), player);
}

void flipped_perspective(const void* state, void* out_state) {
  as_board(out_state) = gomoku::flipped_perspective(as_board_const(state));
}

int get_valid_moves(const void* state, int* moves_out) {
  return gomoku::get_valid_moves(as_board_const(state), moves_out);
}

bool is_full(const void* state) {
  return gomoku::is_full(as_board_const(state));
}

bool apply_move(void* state, int action, int8_t player) {
  return gomoku::apply_move(as_board(state), action, player);
}

bool check_win(const void* state, int action, int8_t player) {
  return gomoku::check_win(as_board_const(state), action, player);
}

void encode_state(const void* state, float* out_encoded) {
  const Board& b = as_board_const(state);
  for (int i = 0; i < gomoku::kActionSize; i++) {
    const int8_t v = b[static_cast<size_t>(i)];
    out_encoded[static_cast<size_t>(i)] = (v == -1) ? 1.0f : 0.0f;
    out_encoded[static_cast<size_t>(gomoku::kActionSize + i)] = (v == 0) ? 1.0f : 0.0f;
    out_encoded[static_cast<size_t>(2 * gomoku::kActionSize + i)] = (v == 1) ? 1.0f : 0.0f;
  }
}

void to_board_plane(const void* state, int8_t* out_plane) {
  const Board& b = as_board_const(state);
  for (int i = 0; i < gomoku::kActionSize; i++) {
    out_plane[static_cast<size_t>(i)] = b[static_cast<size_t>(i)];
  }
}

const game::GameSpec kSpec{
    "gomoku",
    gomoku::kBoardSize,
    gomoku::kActionSize,
    gomoku::kInputChannels,
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
