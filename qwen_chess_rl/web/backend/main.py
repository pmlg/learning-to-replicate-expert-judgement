"""FastAPI backend for the Qwen Chess web application."""

import json
import logging
from pathlib import Path
from typing import Optional

import chess
import chess.svg
from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Qwen Chess API",
    description="Play chess against the Qwen fine-tuned model",
    version="0.1.0",
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

model = None
tokenizer = None
MODEL_LOADED = False


class GameState:
    """Manages a single chess game session."""

    def __init__(self):
        self.board = chess.Board()
        self.move_history: list[str] = []
        self.pgn = ""
        self.game_over = False
        self.result = ""

    def make_move(self, move_uci: str) -> dict:
        """Make a move on the board.

        Args:
            move_uci: Move in UCI notation (e.g., 'e2e4').

        Returns:
            Dictionary with move result.
        """
        try:
            move = chess.Move.from_uci(move_uci)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid UCI move: {move_uci}")

        if move not in self.board.legal_moves:
            raise HTTPException(status_code=400, detail=f"Illegal move: {move_uci}")

        self.board.push(move)
        self.move_history.append(move_uci)
        self._update_pgn()

        if self.board.is_game_over():
            self.game_over = True
            self.result = self._get_result()

        return {
            "move": move_uci,
            "fen": self.board.fen(),
            "is_game_over": self.board.is_game_over(),
            "result": self.result if self.game_over else None,
            "legal_moves": [m.uci() for m in self.board.legal_moves],
        }

    def get_ai_move(self) -> Optional[str]:
        """Get the model's recommended move.

        Returns:
            UCI move string, or None if model is not loaded.
        """
        global model, tokenizer

        from qwen_chess_rl.data.dataset import format_fen_prompt

        if not MODEL_LOADED:
            return None

        prompt = format_fen_prompt(self.board.fen())
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        import torch
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=8,
                temperature=0.1,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "assistant" in decoded:
            move_str = decoded.split("assistant")[-1].strip().split()[0]
        else:
            move_str = decoded.strip().split()[-1]

        try:
            move = chess.Move.from_uci(move_str)
            if move in self.board.legal_moves:
                return move.uci()
        except ValueError:
            pass

        # Fallback: return first legal move
        legal = list(self.board.legal_moves)
        return legal[0].uci() if legal else None

    def _update_pgn(self):
        """Update the PGN string from move history."""
        self.pgn = self.board.__str__()

    def _get_result(self) -> str:
        """Get the game result string."""
        if self.board.is_checkmate():
            return "1-0" if self.board.turn == chess.BLACK else "0-1"
        return "1/2-1/2"

    def reset(self):
        """Reset the game to initial position."""
        self.board = chess.Board()
        self.move_history = []
        self.pgn = ""
        self.game_over = False
        self.result = ""

    def set_fen(self, fen: str):
        """Set the board from a FEN string.

        Args:
            fen: FEN string.
        """
        try:
            self.board = chess.Board(fen)
            self.move_history = []
            self.pgn = fen
            self.game_over = self.board.is_game_over()
            self.result = self._get_result() if self.game_over else ""
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))


# Active game sessions
games: dict[str, GameState] = {}


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class MoveRequest(BaseModel):
    session_id: str
    move: str


class NewGameRequest(BaseModel):
    session_id: Optional[str] = None
    fen: Optional[str] = None
    color: str = "white"


class MoveResponse(BaseModel):
    session_id: str
    fen: str
    is_game_over: bool
    result: Optional[str] = None
    legal_moves: list[str]
    ai_move: Optional[str] = None
    ai_thinking: Optional[str] = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup_event():
    """Initialize model on server start (if checkpoint path is configured)."""
    import os
    ckpt_path = os.environ.get("QWEN_CHESS_CHECKPOINT", "")
    if ckpt_path and Path(ckpt_path).exists():
        load_model(ckpt_path)


def load_model(checkpoint_path: str):
    """Load the chess model into memory.

    Args:
        checkpoint_path: Path to the model checkpoint.
    """
    global model, tokenizer, MODEL_LOADED
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("Loading model from %s ...", checkpoint_path)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype="auto",
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    MODEL_LOADED = True
    logger.info("Model loaded successfully.")


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "model_loaded": MODEL_LOADED,
    }


@app.post("/new_game")
async def new_game(req: NewGameRequest):
    """Start a new game session.

    Args:
        req: New game request with optional FEN and color preferences.

    Returns:
        Initial game state.
    """
    import uuid
    session_id = req.session_id or str(uuid.uuid4())
    game = GameState()

    if req.fen:
        game.set_fen(req.fen)

    games[session_id] = game

    return {
        "session_id": session_id,
        "fen": game.board.fen(),
        "is_game_over": game.game_over,
        "turn": "white" if game.board.turn == chess.WHITE else "black",
        "legal_moves": [m.uci() for m in game.board.legal_moves],
    }


@app.post("/move", response_model=MoveResponse)
async def make_move(req: MoveRequest):
    """Make a move and get AI response.

    Args:
        req: Move request with session ID and UCI move.

    Returns:
        Updated game state with AI move.
    """
    if req.session_id not in games:
        raise HTTPException(status_code=404, detail="Game session not found")

    game = games[req.session_id]

    if game.game_over:
        raise HTTPException(status_code=400, detail="Game is already over")

    # Make player move
    result = game.make_move(req.move)

    if game.game_over:
        return MoveResponse(
            session_id=req.session_id,
            fen=game.board.fen(),
            is_game_over=True,
            result=game.result,
            legal_moves=[],
            ai_move=None,
        )

    # Get AI move
    ai_move = game.get_ai_move()
    if ai_move:
        game.make_move(ai_move)
        if game.game_over:
            result = {
                "fen": game.board.fen(),
                "is_game_over": True,
                "result": game.result,
                "legal_moves": [],
            }
        else:
            result = {
                "fen": game.board.fen(),
                "is_game_over": False,
                "result": None,
                "legal_moves": [m.uci() for m in game.board.legal_moves],
            }

    return MoveResponse(
        session_id=req.session_id,
        fen=result["fen"],
        is_game_over=result["is_game_over"],
        result=result.get("result"),
        legal_moves=result.get("legal_moves", []),
        ai_move=ai_move,
        ai_thinking=f"Evaluated position: {game.board.fen()}" if ai_move else None,
    )


@app.get("/game/{session_id}")
async def get_game(session_id: str):
    """Get current game state.

    Args:
        session_id: Game session ID.

    Returns:
        Current game state and history.
    """
    if session_id not in games:
        raise HTTPException(status_code=404, detail="Game session not found")

    game = games[session_id]
    return {
        "session_id": session_id,
        "fen": game.board.fen(),
        "pgn": game.pgn,
        "move_history": game.move_history,
        "is_game_over": game.game_over,
        "result": game.result,
        "turn": "white" if game.board.turn == chess.WHITE else "black",
        "legal_moves": [m.uci() for m in game.board.legal_moves],
        "model_loaded": MODEL_LOADED,
    }


@app.post("/undo/{session_id}")
async def undo_move(session_id: str):
    """Undo the last two moves (player + AI response).

    Args:
        session_id: Game session ID.

    Returns:
        Updated game state after undo.
    """
    if session_id not in games:
        raise HTTPException(status_code=404, detail="Game session not found")

    game = games[session_id]
    # Undo both AI and player moves
    for _ in range(2):
        if len(game.move_history) > 0:
            game.board.pop()
            game.move_history.pop()

    game.game_over = False
    game.result = ""

    return {
        "session_id": session_id,
        "fen": game.board.fen(),
        "move_history": game.move_history,
        "legal_moves": [m.uci() for m in game.board.legal_moves],
    }


@app.post("/import_fen")
async def import_fen(session_id: str, fen: str):
    """Import a FEN position into a game session.

    Args:
        session_id: Game session ID.
        fen: FEN string to import.

    Returns:
        Updated game state.
    """
    if session_id not in games:
        raise HTTPException(status_code=404, detail="Game session not found")

    game = games[session_id]
    game.set_fen(fen)

    return {
        "session_id": session_id,
        "fen": game.board.fen(),
        "is_game_over": game.game_over,
        "result": game.result,
        "legal_moves": [m.uci() for m in game.board.legal_moves],
    }


@app.get("/fen_export/{session_id}")
async def export_fen(session_id: str):
    """Export current position as FEN.

    Args:
        session_id: Game session ID.

    Returns:
        Current FEN string and PGN.
    """
    if session_id not in games:
        raise HTTPException(status_code=404, detail="Game session not found")

    game = games[session_id]
    return {
        "session_id": session_id,
        "fen": game.board.fen(),
        "pgn": game.pgn,
        "move_history": game.move_history,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
