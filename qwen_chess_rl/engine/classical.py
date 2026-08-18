"""Classical alpha-beta chess engine with quiescence search.

A self-contained, dependency-light move picker that can calculate a few plies
ahead. It exists to give the RL policy genuine tactical competence: the LLM does
one-shot next-token generation and cannot reliably avoid one-move blunders, so a
shallow-but-sound search is used as a tactical oracle / safety net.

Public entry point: :func:`choose_move`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import chess

# Centipawn material values. King has no material value (it is never captured);
# its positional worth is expressed purely through the piece-square table.
PIECE_VALUES: dict[chess.PieceType, int] = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}

# MATE_SCORE is offset by the distance-from-root (ply) so that shorter mates
# score higher than longer ones; INF stays strictly larger than any mate.
MATE_SCORE = 1_000_000
INF = 10_000_000

MOBILITY_WEIGHT = 1
SHIELD_BONUS = 12
CENTER_KING_PENALTY = -25
MAX_QUIESCENCE_PLY = 16
_NODE_CHECK_MASK = 1023

# Piece-square tables are written rank-8-first (index 0 == a8), reading like a
# board from White's side. For a White piece on python-chess square ``sq``
# (0 == a1) the table index is ``sq ^ 56`` (vertical mirror); Black pieces index
# with the raw ``sq``, which yields the correctly mirrored value.
# fmt: off
_PAWN_PST = [
    0, 0, 0, 0, 0, 0, 0, 0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
    5, 5, 10, 25, 25, 10, 5, 5,
    0, 0, 0, 20, 20, 0, 0, 0,
    5, -5, -10, 0, 0, -10, -5, 5,
    5, 10, 10, -20, -20, 10, 10, 5,
    0, 0, 0, 0, 0, 0, 0, 0,
]

_KNIGHT_PST = [
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20, 0, 0, 0, 0, -20, -40,
    -30, 0, 10, 15, 15, 10, 0, -30,
    -30, 5, 15, 20, 20, 15, 5, -30,
    -30, 0, 15, 20, 20, 15, 0, -30,
    -30, 5, 10, 15, 15, 10, 5, -30,
    -40, -20, 0, 5, 5, 0, -20, -40,
    -50, -40, -30, -30, -30, -30, -40, -50,
]

_BISHOP_PST = [
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -10, 0, 5, 10, 10, 5, 0, -10,
    -10, 5, 5, 10, 10, 5, 5, -10,
    -10, 0, 10, 10, 10, 10, 0, -10,
    -10, 10, 10, 10, 10, 10, 10, -10,
    -10, 5, 0, 0, 0, 0, 5, -10,
    -20, -10, -10, -10, -10, -10, -10, -20,
]

_ROOK_PST = [
    0, 0, 0, 0, 0, 0, 0, 0,
    5, 10, 10, 10, 10, 10, 10, 5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    0, 0, 0, 5, 5, 0, 0, 0,
]

_QUEEN_PST = [
    -20, -10, -10, -5, -5, -10, -10, -20,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -10, 0, 5, 5, 5, 5, 0, -10,
    -5, 0, 5, 5, 5, 5, 0, -5,
    0, 0, 5, 5, 5, 5, 0, -5,
    -10, 5, 5, 5, 5, 5, 0, -10,
    -10, 0, 5, 0, 0, 0, 0, -10,
    -20, -10, -10, -5, -5, -10, -10, -20,
]

# Middlegame king table: rewards a castled corner, punishes the exposed centre.
_KING_PST = [
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -20, -30, -30, -40, -40, -30, -30, -20,
    -10, -20, -20, -20, -20, -20, -20, -10,
    20, 20, 0, 0, 0, 0, 20, 20,
    20, 30, 10, 0, 0, 10, 30, 20,
]
# fmt: on

_PST: dict[chess.PieceType, list[int]] = {
    chess.PAWN: _PAWN_PST,
    chess.KNIGHT: _KNIGHT_PST,
    chess.BISHOP: _BISHOP_PST,
    chess.ROOK: _ROOK_PST,
    chess.QUEEN: _QUEEN_PST,
    chess.KING: _KING_PST,
}

# Transposition-table bound flags.
_EXACT = 0
_LOWER = 1
_UPPER = 2


class _Timeout(Exception):  # noqa: N818 - internal control-flow signal, not an error
    """Raised internally to unwind the search when the budget is exhausted."""


@dataclass
class _TTEntry:
    depth: int
    value: int
    flag: int
    move: chess.Move | None


class _SearchContext:
    """Mutable per-search state: budgets, transposition table, killer moves."""

    def __init__(self, time_limit: float | None, node_budget: int | None):
        self.deadline = time.monotonic() + time_limit if time_limit else None
        self.node_budget = node_budget
        self.nodes = 0
        self.tt: dict[tuple, _TTEntry] = {}
        self.killers: dict[int, list[chess.Move]] = {}

    def tick(self) -> None:
        self.nodes += 1
        if self.node_budget is not None and self.nodes >= self.node_budget:
            raise _Timeout
        # Wall-clock is polled sparsely — time.monotonic() per node is a
        # measurable fraction of the search cost at these node rates.
        if self.deadline is not None and (self.nodes & _NODE_CHECK_MASK) == 0:
            if time.monotonic() >= self.deadline:
                raise _Timeout


def evaluate(board: chess.Board) -> int:
    """Static evaluation in centipawns from White's perspective.

    Combines material, piece-square tables, a small mobility term, and a light
    king-safety term. Positive favours White, negative favours Black.
    """
    score = 0
    for sq, piece in board.piece_map().items():
        value = PIECE_VALUES[piece.piece_type]
        pst = _PST[piece.piece_type]
        if piece.color == chess.WHITE:
            score += value + pst[sq ^ 56]
        else:
            score -= value + pst[sq]
    score += _mobility(board)
    score += _king_safety(board)
    return score


def _mobility(board: chess.Board) -> int:
    """Cheap symmetric mobility: attacked non-own squares for sliders/knights.

    Uses raw attack bitboards (no legality filtering) so it costs a handful of
    popcounts rather than two full legal-move generations per leaf.
    """
    total = 0
    for color in (chess.WHITE, chess.BLACK):
        sign = 1 if color == chess.WHITE else -1
        own = board.occupied_co[color]
        count = 0
        for piece_type in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
            for sq in board.pieces(piece_type, color):
                count += chess.popcount(board.attacks_mask(sq) & ~own)
        total += sign * count
    return total * MOBILITY_WEIGHT


def _king_safety(board: chess.Board) -> int:
    """Reward an intact pawn shield; penalise a king loitering on the centre files."""
    score = 0
    for color in (chess.WHITE, chess.BLACK):
        king_sq = board.king(color)
        if king_sq is None:
            continue
        sign = 1 if color == chess.WHITE else -1
        file = chess.square_file(king_sq)
        rank = chess.square_rank(king_sq)

        shield_rank = rank + (1 if color == chess.WHITE else -1)
        if 0 <= shield_rank <= 7:
            for f in (file - 1, file, file + 1):
                if 0 <= f <= 7:
                    piece = board.piece_at(chess.square(f, shield_rank))
                    if piece is not None and piece.piece_type == chess.PAWN and piece.color == color:
                        score += sign * SHIELD_BONUS

        if file in (3, 4):
            score += sign * CENTER_KING_PENALTY
    return score


def _eval_pov(board: chess.Board) -> int:
    """Static evaluation from the side-to-move's perspective (for negamax)."""
    score = evaluate(board)
    return score if board.turn == chess.WHITE else -score


def _mvv_lva(board: chess.Board, move: chess.Move) -> int:
    """Most-Valuable-Victim / Least-Valuable-Attacker capture score."""
    victim_type = board.piece_type_at(move.to_square)
    if victim_type is None:  # en passant
        victim_type = chess.PAWN
    attacker_type = board.piece_type_at(move.from_square)
    attacker_value = PIECE_VALUES[attacker_type] if attacker_type else 0
    score = PIECE_VALUES[victim_type] * 10 - attacker_value
    if move.promotion:
        score += PIECE_VALUES[move.promotion]
    return score


def _order_moves(
    board: chess.Board,
    moves: list[chess.Move],
    tt_move: chess.Move | None,
    killers: list[chess.Move],
) -> list[chess.Move]:
    """Order moves: TT best move, then captures by MVV-LVA, then killers."""

    def key(move: chess.Move) -> int:
        if move == tt_move:
            return 1_000_000
        if board.is_capture(move):
            return 100_000 + _mvv_lva(board, move)
        if move in killers:
            return 90_000
        return 0

    return sorted(moves, key=key, reverse=True)


def _quiescence(board: chess.Board, ply: int, qdepth: int, alpha: int, beta: int, ctx: _SearchContext) -> int:
    """Search only forcing lines (captures, check evasions) to a quiet position.

    Prevents the horizon effect: a static eval at a leaf mid-capture would
    misjudge, e.g., a "free" pawn grab that hangs the queen on the reply.
    """
    ctx.tick()

    in_check = board.is_check()
    if in_check:
        moves = list(board.legal_moves)
        if not moves:
            return -MATE_SCORE + ply
        best = -INF
    else:
        best = _eval_pov(board)
        if best >= beta or qdepth >= MAX_QUIESCENCE_PLY:
            return best
        if best > alpha:
            alpha = best
        moves = [m for m in board.legal_moves if board.is_capture(m)]

    moves.sort(key=lambda m: _mvv_lva(board, m), reverse=True)
    for move in moves:
        board.push(move)
        try:
            score = -_quiescence(board, ply + 1, qdepth + 1, -beta, -alpha, ctx)
        finally:
            board.pop()
        if score > best:
            best = score
        if best > alpha:
            alpha = best
        if alpha >= beta:
            break
    return best


def _negamax(board: chess.Board, depth: int, ply: int, alpha: int, beta: int, ctx: _SearchContext) -> int:
    """Negamax alpha-beta with transposition-table probing and killer moves."""
    ctx.tick()

    if ply > 0 and (board.is_repetition(3) or board.is_fifty_moves()):
        return 0

    alpha_orig = alpha
    key = board._transposition_key()
    entry = ctx.tt.get(key)
    tt_move: chess.Move | None = None
    if entry is not None:
        tt_move = entry.move
        if entry.depth >= depth:
            if entry.flag == _EXACT:
                return entry.value
            if entry.flag == _LOWER:
                alpha = max(alpha, entry.value)
            elif entry.flag == _UPPER:
                beta = min(beta, entry.value)
            if alpha >= beta:
                return entry.value

    if depth <= 0:
        return _quiescence(board, ply, 0, alpha, beta, ctx)

    legal = list(board.legal_moves)
    if not legal:
        return -MATE_SCORE + ply if board.is_check() else 0

    killers = ctx.killers.get(ply, [])
    best_value = -INF
    best_move: chess.Move | None = None
    for move in _order_moves(board, legal, tt_move, killers):
        board.push(move)
        try:
            value = -_negamax(board, depth - 1, ply + 1, -beta, -alpha, ctx)
        finally:
            board.pop()
        if value > best_value:
            best_value = value
            best_move = move
        if best_value > alpha:
            alpha = best_value
        if alpha >= beta:
            if not board.is_capture(move):
                _record_killer(ctx, ply, move)
            break

    flag = _EXACT
    if best_value <= alpha_orig:
        flag = _UPPER
    elif best_value >= beta:
        flag = _LOWER
    ctx.tt[key] = _TTEntry(depth, best_value, flag, best_move)
    return best_value


def _record_killer(ctx: _SearchContext, ply: int, move: chess.Move) -> None:
    killers = ctx.killers.setdefault(ply, [])
    if move not in killers:
        killers.insert(0, move)
        del killers[2:]


def _search_root(board: chess.Board, depth: int, ctx: _SearchContext) -> tuple[int, chess.Move | None]:
    """Run one full-width search at ``depth`` and return (score, best move)."""
    alpha, beta = -INF, INF
    key = board._transposition_key()
    entry = ctx.tt.get(key)
    tt_move = entry.move if entry is not None else None

    best_value = -INF
    best_move: chess.Move | None = None
    for move in _order_moves(board, list(board.legal_moves), tt_move, ctx.killers.get(0, [])):
        board.push(move)
        try:
            value = -_negamax(board, depth - 1, 1, -beta, -alpha, ctx)
        finally:
            board.pop()
        if value > best_value:
            best_value = value
            best_move = move
        if best_value > alpha:
            alpha = best_value

    ctx.tt[key] = _TTEntry(depth, best_value, _EXACT, best_move)
    return best_value, best_move


def choose_move(
    board: chess.Board,
    max_depth: int = 4,
    time_limit: float | None = 1.0,
    node_budget: int | None = None,
) -> chess.Move | None:
    """Pick a move via iterative-deepening alpha-beta search.

    Deepens one ply at a time until ``max_depth`` is reached or a budget is hit,
    keeping the best move from the last *fully completed* depth (a partial depth
    interrupted by the budget is discarded, so the returned move is always sound
    to the depth it claims).

    Args:
        board: Position to move in. Not mutated (pushes are undone).
        max_depth: Maximum iterative-deepening depth in plies.
        time_limit: Wall-clock budget in seconds, or None for no time bound.
        node_budget: Optional hard cap on nodes searched, or None for no cap.

    Returns:
        The chosen move, or None only if there are no legal moves.
    """
    legal = list(board.legal_moves)
    if not legal:
        return None
    if len(legal) == 1:
        return legal[0]

    ctx = _SearchContext(time_limit, node_budget)
    best_move = legal[0]
    for depth in range(1, max_depth + 1):
        try:
            value, move = _search_root(board, depth, ctx)
        except _Timeout:
            break
        if move is not None:
            best_move = move
        # A found forced mate cannot be improved by searching deeper.
        if value >= MATE_SCORE - max_depth:
            break
        if ctx.deadline is not None and time.monotonic() >= ctx.deadline:
            break
    return best_move


def evaluate_move(
    board: chess.Board,
    move: chess.Move,
    max_depth: int = 4,
    time_limit: float | None = 1.0,
) -> int:
    """Searched evaluation (centipawns, mover's perspective) of playing ``move``.

    Unlike a bare :func:`evaluate` call on the resulting position, this
    searches ``max_depth`` plies past the move itself (the opponent's best
    reply and beyond), via the same iterative-deepening negamax used by
    :func:`choose_move`. A static eval of the position right after a move
    can't see a reply that wins material or forces mate next ply — it looks
    fine until the opponent actually plays it — so comparing static evals is
    blind to exactly the one-move-deep tactics a vetting check most needs to
    catch.

    Args:
        board: Position before ``move``. Not mutated (the push is undone).
        move: The candidate move to evaluate.
        max_depth: Iterative-deepening depth searched *after* ``move``.
        time_limit: Wall-clock budget in seconds, or None for no time bound.

    Returns:
        Centipawn score from the perspective of whoever is to move at
        ``board`` (i.e. the side playing ``move``), positive is better for
        them.
    """
    board.push(move)
    try:
        ctx = _SearchContext(time_limit, node_budget=None)
        value = 0
        for depth in range(1, max_depth + 1):
            try:
                value = _negamax(board, depth - 1, 0, -INF, INF, ctx)
            except _Timeout:
                break
            if ctx.deadline is not None and time.monotonic() >= ctx.deadline:
                break
        # _negamax returns the value from the perspective of whoever is to
        # move on `board` now, i.e. the opponent — negate back to the mover.
        return -value
    finally:
        board.pop()
