#include "gomoku.h"

namespace gomoku {

namespace {
inline int row_of(int action) { return action / kBoardSize; }
inline int col_of(int action) { return action % kBoardSize; }

int count_dir(const Board& board, int r, int c, int dr, int dc, int8_t player) {
  int count = 1;

  int nr = r + dr;
  int nc = c + dc;
  while (nr >= 0 && nr < kBoardSize && nc >= 0 && nc < kBoardSize &&
         board[nr * kBoardSize + nc] == player) {
    count++;
    nr += dr;
    nc += dc;
  }

  nr = r - dr;
  nc = c - dc;
  while (nr >= 0 && nr < kBoardSize && nc >= 0 && nc < kBoardSize &&
         board[nr * kBoardSize + nc] == player) {
    count++;
    nr -= dr;
    nc -= dc;
  }

  return count;
}
}  // namespace

Board initial_board() {
  Board b{};
  b.fill(0);
  return b;
}

Board canonical_board(const Board& board, int8_t player) {
  Board out{};
  for (int i = 0; i < kActionSize; i++) {
    out[i] = static_cast<int8_t>(board[i] * player);
  }
  return out;
}

Board flipped_perspective(const Board& board) {
  Board out{};
  for (int i = 0; i < kActionSize; i++) {
    out[i] = static_cast<int8_t>(-board[i]);
  }
  return out;
}

std::vector<int> valid_moves(const Board& board) {
  std::vector<int> moves;
  moves.reserve(kActionSize);
  for (int a = 0; a < kActionSize; a++) {
    if (board[a] == 0) {
      moves.push_back(a);
    }
  }
  return moves;
}

bool is_full(const Board& board) {
  for (int i = 0; i < kActionSize; i++) {
    if (board[i] == 0) {
      return false;
    }
  }
  return true;
}

bool apply_move(Board& board, int action, int8_t player) {
  if (action < 0 || action >= kActionSize) {
    return false;
  }
  if (board[action] != 0) {
    return false;
  }
  board[action] = player;
  return true;
}

bool check_win(const Board& board, int action, int8_t player) {
  if (action < 0 || action >= kActionSize) {
    return false;
  }
  if (board[action] != player) {
    return false;
  }

  const int r = row_of(action);
  const int c = col_of(action);
  return count_dir(board, r, c, 1, 0, player) >= 5 ||
         count_dir(board, r, c, 0, 1, player) >= 5 ||
         count_dir(board, r, c, 1, 1, player) >= 5 ||
         count_dir(board, r, c, 1, -1, player) >= 5;
}

}  // namespace gomoku
