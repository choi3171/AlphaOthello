#pragma once

#include <array>
#include <cstdint>
#include <vector>

namespace gomoku {

constexpr int kBoardSize = 10;
constexpr int kActionSize = kBoardSize * kBoardSize;
using Board = std::array<int8_t, kActionSize>;

Board initial_board();
Board canonical_board(const Board& board, int8_t player);
Board flipped_perspective(const Board& board);
std::vector<int> valid_moves(const Board& board);
bool is_full(const Board& board);
bool apply_move(Board& board, int action, int8_t player);
bool check_win(const Board& board, int action, int8_t player);

}  // namespace gomoku
