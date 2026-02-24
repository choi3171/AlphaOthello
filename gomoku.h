#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace gomoku {
struct MoveList {
  int* ptr;
  int count;

  int* begin() const { return ptr; }
  int* end() const { return ptr + count; }
  bool empty() const { return count == 0; }
  size_t size() const { return static_cast<size_t>(count); }
  int operator[](int i) const { return ptr[i]; }
};

constexpr int kBoardSize = 10;
constexpr int kActionSize = kBoardSize * kBoardSize;
using Board = std::array<int8_t, kActionSize>;

Board initial_board();
Board canonical_board(const Board& board, int8_t player);
Board flipped_perspective(const Board& board);
MoveList valid_moves(const Board& board);
bool is_full(const Board& board);
bool apply_move(Board& board, int action, int8_t player);
bool check_win(const Board& board, int action, int8_t player);

}  // namespace gomoku
