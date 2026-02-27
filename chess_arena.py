#!/usr/bin/env python3
"""
Chess Arena — AI vs AI terminal chess via OpenRouter
No humans, just vibes and blunders.

Usage:
  export OPENROUTER_API_KEY=your_key
  python chess_arena.py
  python chess_arena.py --white anthropic/claude-sonnet-4 --black openai/gpt-4o
  python chess_arena.py --white anthropic/claude-sonnet-4 --black openai/gpt-4o --delay 1.5
"""

import chess
import chess.pgn
import argparse
import json
import os
import re
import sys
import time
import datetime
import requests
from io import StringIO

# ── Config ────────────────────────────────────────────────────────────────────

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MAX_RETRIES = 3
DEFAULT_WHITE = "anthropic/claude-sonnet-4"
DEFAULT_BLACK = "openai/gpt-4o"

# ── Piece display ─────────────────────────────────────────────────────────────

# Use simple letters for reliable terminal alignment
PIECE_CHARS_WHITE = {"K": "K", "Q": "Q", "R": "R", "B": "B", "N": "N", "P": "P"}
PIECE_CHARS_BLACK = {"k": "k", "q": "q", "r": "r", "b": "b", "n": "n", "p": "p"}

# For captured display where alignment doesn't matter
CAPTURED_SYMBOLS = {
    "Q": "♛", "R": "♜", "B": "♝", "N": "♞", "P": "♟",
    "q": "♕", "r": "♖", "b": "♗", "n": "♘", "p": "♙",
}

PIECE_ORDER = {"Q": 0, "R": 1, "B": 2, "N": 3, "P": 4}

# ── Terminal colors ───────────────────────────────────────────────────────────

class C:
    RESET    = "\033[0m"
    BOLD     = "\033[1m"
    DIM      = "\033[2m"
    ITALIC   = "\033[3m"
    WHITE_SQ = "\033[48;5;180m"
    BLACK_SQ = "\033[48;5;94m"
    WHITE_PC = "\033[38;5;255m\033[1m"
    BLACK_PC = "\033[38;5;16m\033[1m"
    HEADER   = "\033[38;5;75m"
    MOVE_W   = "\033[38;5;255m"
    MOVE_B   = "\033[38;5;249m"
    ERROR    = "\033[38;5;203m"
    SUCCESS  = "\033[38;5;114m"
    INFO     = "\033[38;5;244m"
    LABEL    = "\033[38;5;137m"
    COMMENT  = "\033[38;5;103m"
    HIGHLIGHT = "\033[48;5;107m"


def clear_screen():
    print("\033[2J\033[H", end="")


def render_board(board: chess.Board, last_move: chess.Move = None) -> str:
    """Render the board with ASCII pieces for reliable alignment."""
    lines = []

    lines.append(f"   {C.DIM}+---+---+---+---+---+---+---+---+{C.RESET}")

    for rank in range(7, -1, -1):
        row = f" {C.LABEL}{rank + 1}{C.RESET} {C.DIM}|{C.RESET}"
        for file in range(8):
            sq = chess.square(file, rank)
            piece = board.piece_at(sq)
            is_light = (rank + file) % 2 == 1

            # Pick background
            if last_move and sq in (last_move.from_square, last_move.to_square):
                bg = C.HIGHLIGHT
            elif is_light:
                bg = C.WHITE_SQ
            else:
                bg = C.BLACK_SQ

            if piece:
                pc_color = C.WHITE_PC if piece.color == chess.WHITE else C.BLACK_PC
                sym = piece.symbol()
                # Use uppercase for white, lowercase for black
                display = sym.upper() if piece.color == chess.WHITE else sym.lower()
                row += f"{bg}{pc_color} {display} {C.RESET}{C.DIM}|{C.RESET}"
            else:
                row += f"{bg}   {C.RESET}{C.DIM}|{C.RESET}"

        lines.append(row)
        lines.append(f"   {C.DIM}+---+---+---+---+---+---+---+---+{C.RESET}")

    lines.append(f"   {C.LABEL}  a   b   c   d   e   f   g   h{C.RESET}")
    return "\n".join(lines)


def render_captured(captured_white: list, captured_black: list) -> str:
    """Render captured pieces for both sides."""
    def sort_pieces(pieces):
        return sorted(pieces, key=lambda p: PIECE_ORDER.get(p.upper(), 99))

    w_pieces = sort_pieces(captured_white)
    b_pieces = sort_pieces(captured_black)

    # "Captured by White" = black pieces that white took
    w_took = " ".join(CAPTURED_SYMBOLS.get(p.lower(), p) for p in b_pieces) if b_pieces else "—"
    # "Captured by Black" = white pieces that black took
    b_took = " ".join(CAPTURED_SYMBOLS.get(p, p) for p in w_pieces) if w_pieces else "—"

    lines = []
    lines.append(f"  {C.DIM}White took:{C.RESET} {w_took}")
    lines.append(f"  {C.DIM}Black took:{C.RESET} {b_took}")
    return "\n".join(lines)


def get_captured_pieces(board: chess.Board) -> tuple[list, list]:
    """Figure out which pieces have been captured by diffing from starting position."""
    start_white = {"P": 8, "N": 2, "B": 2, "R": 2, "Q": 1, "K": 1}
    start_black = {"p": 8, "n": 2, "b": 2, "r": 2, "q": 1, "k": 1}

    current_white = {}
    current_black = {}
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece:
            sym = piece.symbol()
            if piece.color == chess.WHITE:
                current_white[sym] = current_white.get(sym, 0) + 1
            else:
                current_black[sym] = current_black.get(sym, 0) + 1

    captured_white = []
    for p, count in start_white.items():
        current = current_white.get(p, 0)
        captured_white.extend([p] * (count - current))

    captured_black = []
    for p, count in start_black.items():
        current = current_black.get(p, 0)
        captured_black.extend([p] * (count - current))

    return captured_white, captured_black


# ── OpenRouter API ────────────────────────────────────────────────────────────

def get_ai_move(
    model: str,
    board: chess.Board,
    move_history: list[str],
    api_key: str,
    color: str,
) -> tuple[str | None, str]:
    """
    Ask a model for a chess move.
    Returns (move_san_or_none, full_commentary_text).
    """
    legal_moves = [board.san(m) for m in board.legal_moves]

    system_prompt = (
        "You are playing a game of chess. "
        "When you've decided on your move, include it in the format MOVE: <san> "
        "somewhere in your response. Example: MOVE: Nf3"
    )

    fen = board.fen()
    history_str = format_move_history(move_history) if move_history else "Game start."

    user_prompt = (
        f"You are {color}.\n"
        f"FEN: {fen}\n"
        f"Moves so far: {history_str}\n"
        f"Legal moves: {', '.join(legal_moves)}\n"
        f"Your move."
    )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://agentarcade.co",
        "X-Title": "Chess Arena",
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": 500,
        "temperature": 0.7,
    }

    try:
        resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        raw = data["choices"][0]["message"]["content"].strip()

        # Extract the move from MOVE: <move> pattern
        move_text = None
        move_match = re.search(r"MOVE:\s*([A-Za-z0-9+#=\-]+)", raw)
        if move_match:
            move_text = move_match.group(1).strip()
        else:
            # Fallback: try last line, last word
            last_line = raw.strip().split("\n")[-1].strip()
            last_line = last_line.strip("`.\"'*:")
            tokens = re.findall(r"[A-Za-z0-9+#=\-]+", last_line)
            if tokens:
                move_text = tokens[-1]

        # Extract commentary (everything before MOVE: line)
        commentary = raw
        if move_match:
            commentary = raw[:move_match.start()].strip()
        if not commentary:
            commentary = raw  # use full text if no MOVE: found

        return move_text, commentary

    except requests.exceptions.RequestException as e:
        print(f"    {C.ERROR}API error: {e}{C.RESET}")
        return None, ""
    except (KeyError, IndexError) as e:
        print(f"    {C.ERROR}Parse error: {e}{C.RESET}")
        return None, ""


def validate_and_push(board: chess.Board, san_move: str) -> chess.Move | None:
    """Validate a SAN move and push it. Returns the Move or None."""
    try:
        move = board.parse_san(san_move)
        if move in board.legal_moves:
            board.push(move)
            return move
    except (chess.InvalidMoveError, chess.IllegalMoveError, chess.AmbiguousMoveError, ValueError):
        pass
    return None


def format_move_history(moves: list[str]) -> str:
    """Format moves as numbered pairs."""
    lines = []
    for i in range(0, len(moves), 2):
        move_num = i // 2 + 1
        white_move = moves[i]
        black_move = moves[i + 1] if i + 1 < len(moves) else ""
        lines.append(f"{move_num}. {white_move} {black_move}".strip())
    return "  ".join(lines)


def wrap_text(text: str, width: int = 68, prefix: str = "    ") -> str:
    """Word-wrap text to a given width with a prefix on each line."""
    words = text.split()
    lines = []
    current = prefix
    for word in words:
        if len(current) + len(word) + 1 > width + len(prefix):
            lines.append(current)
            current = prefix + word
        else:
            current += (" " if current.strip() else "") + word
    if current.strip():
        lines.append(current)
    return "\n".join(lines)


# ── Game loop ─────────────────────────────────────────────────────────────────

def play_game(white_model: str, black_model: str, api_key: str, delay: float = 1.0):
    board = chess.Board()
    move_history: list[str] = []
    commentary_log: list[dict] = []
    last_move = None
    move_count = 0
    consecutive_errors = {chess.WHITE: 0, chess.BLACK: 0}
    last_commentary = ""

    models = {chess.WHITE: white_model, chess.BLACK: black_model}
    color_names = {chess.WHITE: "White", chess.BLACK: "Black"}
    color_codes = {chess.WHITE: C.MOVE_W, chess.BLACK: C.MOVE_B}

    short_name = lambda m: m.split("/")[-1] if "/" in m else m

    clear_screen()
    print(f"\n  {C.HEADER}{C.BOLD}+  CHESS ARENA  +{C.RESET}")
    print(f"  {C.DIM}{'─' * 40}{C.RESET}")
    print(f"  {C.MOVE_W}White:{C.RESET} {short_name(white_model)}")
    print(f"  {C.MOVE_B}Black:{C.RESET} {short_name(black_model)}")
    print(f"  {C.DIM}{'─' * 40}{C.RESET}\n")
    print(render_board(board))
    print(f"\n  {C.INFO}Starting in 2s...{C.RESET}")
    time.sleep(2)

    while not board.is_game_over():
        turn = board.turn
        model = models[turn]
        color = color_names[turn]

        success = False
        raw_commentary = ""
        clean_san = ""

        for attempt in range(1, MAX_RETRIES + 1):
            raw_move, raw_commentary = get_ai_move(model, board, move_history, api_key, color)

            if raw_move is None:
                print(f"    {C.ERROR}[{color}] No response (attempt {attempt}/{MAX_RETRIES}){C.RESET}")
                time.sleep(1)
                continue

            move = validate_and_push(board, raw_move)
            if move:
                board.pop()
                clean_san = board.san(move)
                board.push(move)
                move_history.append(clean_san)
                last_move = move
                success = True
                consecutive_errors[turn] = 0

                commentary_log.append({
                    "move_num": move_count + 1,
                    "color": color,
                    "model": model,
                    "move": clean_san,
                    "commentary": raw_commentary,
                })
                last_commentary = raw_commentary
                break
            else:
                legal_sample = [board.san(m) for m in list(board.legal_moves)[:8]]
                print(f"    {C.ERROR}[{color}] Illegal: \"{raw_move}\" "
                      f"(attempt {attempt}/{MAX_RETRIES}) "
                      f"Legal: {', '.join(legal_sample)}...{C.RESET}")
                time.sleep(0.5)

        if not success:
            consecutive_errors[turn] += 1
            if consecutive_errors[turn] >= 2:
                clear_screen()
                print(f"\n  {C.ERROR}{C.BOLD}{color} ({short_name(model)}) "
                      f"failed too many times. Game aborted.{C.RESET}\n")
                print(render_board(board, last_move))
                print(f"\n  {C.INFO}Moves: {format_move_history(move_history)}{C.RESET}\n")
                save_commentary_log(commentary_log, white_model, black_model)
                return
            print(f"    {C.ERROR}[{color}] Could not produce a legal move. Forfeiting.{C.RESET}")
            break

        move_count += 1

        # ── Redraw ────────────────────────────────────────────────────────
        clear_screen()
        print(f"\n  {C.HEADER}{C.BOLD}+  CHESS ARENA  +{C.RESET}")
        print(f"  {C.DIM}{'─' * 40}{C.RESET}")
        print(f"  {C.MOVE_W}White:{C.RESET} {short_name(white_model)}")
        print(f"  {C.MOVE_B}Black:{C.RESET} {short_name(black_model)}")
        print(f"  {C.DIM}{'─' * 40}{C.RESET}\n")

        cap_w, cap_b = get_captured_pieces(board)
        print(render_captured(cap_w, cap_b))
        print()

        print(render_board(board, last_move))

        print(f"\n  {C.INFO}Move {move_count}:{C.RESET} "
              f"{color_codes[turn]}{color}{C.RESET} "
              f"plays {C.BOLD}{clean_san}{C.RESET}")

        if board.is_check():
            print(f"  {C.ERROR}>>> Check!{C.RESET}")

        # Commentary bubble
        if last_commentary:
            display_comment = last_commentary.replace("\n", " ")
            if len(display_comment) > 300:
                display_comment = display_comment[:297] + "..."
            print(f"\n  {C.COMMENT}{C.ITALIC}[{short_name(model)} thinks]{C.RESET}")
            print(f"{C.COMMENT}{wrap_text(display_comment)}{C.RESET}")

        # Compact history
        hist = format_move_history(move_history)
        if len(hist) > 76:
            hist = "..." + hist[-(76 - 3):]
        print(f"\n  {C.DIM}{hist}{C.RESET}")

        time.sleep(delay)

    # ── Game over ─────────────────────────────────────────────────────────────
    print(f"\n  {C.DIM}{'=' * 40}{C.RESET}")

    result = board.result()
    if board.is_checkmate():
        winner = "Black" if board.turn == chess.WHITE else "White"
        winner_model = short_name(models[chess.BLACK if board.turn == chess.WHITE else chess.WHITE])
        print(f"  {C.SUCCESS}{C.BOLD}* CHECKMATE — {winner} ({winner_model}) wins!{C.RESET}")
    elif board.is_stalemate():
        print(f"  {C.INFO}{C.BOLD}Draw — Stalemate{C.RESET}")
    elif board.is_insufficient_material():
        print(f"  {C.INFO}{C.BOLD}Draw — Insufficient material{C.RESET}")
    elif board.is_fifty_moves():
        print(f"  {C.INFO}{C.BOLD}Draw — 50-move rule{C.RESET}")
    elif board.is_repetition():
        print(f"  {C.INFO}{C.BOLD}Draw — Threefold repetition{C.RESET}")
    else:
        print(f"  {C.INFO}Result: {result}{C.RESET}")

    print(f"  {C.DIM}Total moves: {move_count}{C.RESET}")
    print(f"  {C.DIM}{'=' * 40}{C.RESET}")

    save_pgn(board, move_history, commentary_log, white_model, black_model, result)
    save_commentary_log(commentary_log, white_model, black_model)


def save_pgn(board, move_history, commentary_log, white_model, black_model, result):
    """Save the game as a PGN file with AI commentary as annotations."""
    game = chess.pgn.Game()
    game.headers["Event"] = "Chess Arena"
    game.headers["Site"] = "AgentArcade / Terminal"
    game.headers["Date"] = datetime.datetime.now().strftime("%Y.%m.%d")
    game.headers["White"] = white_model
    game.headers["Black"] = black_model
    game.headers["Result"] = result

    comment_map = {}
    for entry in commentary_log:
        comment_map[entry["move_num"]] = entry["commentary"]

    node = game
    temp_board = chess.Board()
    for i, san in enumerate(move_history):
        move = temp_board.parse_san(san)
        node = node.add_variation(move)
        comment = comment_map.get(i + 1, "")
        if comment:
            if len(comment) > 500:
                comment = comment[:497] + "..."
            node.comment = comment
        temp_board.push(move)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    short_w = white_model.split("/")[-1] if "/" in white_model else white_model
    short_b = black_model.split("/")[-1] if "/" in black_model else black_model
    filename = f"chess_{short_w}_vs_{short_b}_{ts}.pgn"

    with open(filename, "w") as f:
        print(game, file=f)

    print(f"\n  {C.SUCCESS}PGN saved -> {filename}{C.RESET}")


def save_commentary_log(commentary_log, white_model, black_model):
    """Save the full untruncated commentary as a separate text file."""
    if not commentary_log:
        return

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    short_w = white_model.split("/")[-1] if "/" in white_model else white_model
    short_b = black_model.split("/")[-1] if "/" in black_model else black_model
    filename = f"chess_{short_w}_vs_{short_b}_{ts}_commentary.txt"

    with open(filename, "w") as f:
        f.write(f"Chess Arena — Full Commentary Log\n")
        f.write(f"White: {white_model}\n")
        f.write(f"Black: {black_model}\n")
        f.write(f"Date: {datetime.datetime.now().isoformat()}\n")
        f.write(f"{'=' * 60}\n\n")

        for entry in commentary_log:
            f.write(f"--- Move {entry['move_num']}: {entry['color']} plays {entry['move']} ---\n")
            f.write(f"Model: {entry['model']}\n\n")
            f.write(f"{entry['commentary']}\n\n")

    print(f"  {C.SUCCESS}Commentary saved -> {filename}{C.RESET}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Chess Arena — AI vs AI chess via OpenRouter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python chess_arena.py
  python chess_arena.py --white anthropic/claude-sonnet-4 --black google/gemini-2.5-flash
  python chess_arena.py --white openai/gpt-4o --black meta-llama/llama-4-maverick --delay 2
        """,
    )
    parser.add_argument("--white", default=DEFAULT_WHITE, help=f"White model (default: {DEFAULT_WHITE})")
    parser.add_argument("--black", default=DEFAULT_BLACK, help=f"Black model (default: {DEFAULT_BLACK})")
    parser.add_argument("--delay", type=float, default=1.0, help="Seconds between moves (default: 1.0)")

    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print(f"\n  {C.ERROR}Set OPENROUTER_API_KEY environment variable first.{C.RESET}")
        print(f"  {C.DIM}export OPENROUTER_API_KEY=your_key_here{C.RESET}\n")
        sys.exit(1)

    play_game(args.white, args.black, api_key, args.delay)


if __name__ == "__main__":
    main()
