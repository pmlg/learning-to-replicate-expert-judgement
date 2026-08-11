import React, { useState, useCallback, useEffect } from 'react';
import ChessBoard from './ChessBoard';
import GameControls from './GameControls';
import GameInfo from './GameInfo';
import api from './api';

/**
 * Main App component for the Qwen Chess web application.
 */
function App() {
  const [sessionId, setSessionId] = useState(null);
  const [fen, setFen] = useState('start');
  const [moveHistory, setMoveHistory] = useState([]);
  const [isGameOver, setIsGameOver] = useState(false);
  const [gameResult, setGameResult] = useState('');
  const [legalMoves, setLegalMoves] = useState([]);
  const [playerColor, setPlayerColor] = useState('white');
  const [isAiThinking, setIsAiThinking] = useState(false);
  const [modelLoaded, setModelLoaded] = useState(false);
  const [modelName, setModelName] = useState('');
  const [error, setError] = useState('');
  const [temperature, setTemperature] = useState(0.1);
  const [fenInput, setFenInput] = useState('');

  // Start a new game
  const startNewGame = useCallback(async (color = 'white', initialFen = null) => {
    try {
      const response = await api.newGame(initialFen, color);
      setSessionId(response.session_id);
      setFen(response.fen);
      setMoveHistory(response.move_history || (response.ai_move ? [response.ai_move] : []));
      setIsGameOver(Boolean(response.is_game_over));
      setGameResult(response.result || '');
      setLegalMoves(response.legal_moves || []);
      setPlayerColor(color);
      setError('');
      if (typeof response.model_loaded === 'boolean') {
        setModelLoaded(response.model_loaded);
      }
      if (response.model_name) {
        setModelName(response.model_name);
      }
      if (response.model_loaded === false) {
        setError(
          'Game started, but the model is not loaded. Restart the backend with QWEN_CHESS_CHECKPOINT=checkpoints/sft/final',
        );
      }
    } catch (err) {
      const detail = err.response?.data?.detail || err.message;
      setError('Failed to start new game: ' + detail);
      setModelLoaded(false);
    }
  }, []);

  // Make a move
  const makeMove = useCallback(async (move) => {
    if (!sessionId || isGameOver || isAiThinking) return;

    setIsAiThinking(true);
    setError('');

    try {
      const response = await api.makeMove(sessionId, move);
      setFen(response.fen);
      setIsGameOver(response.is_game_over);
      setGameResult(response.result || '');
      setLegalMoves(response.legal_moves || []);
      setMoveHistory(prev => [...prev, move]);

      if (response.ai_move) {
        setMoveHistory(prev => [...prev, response.ai_move]);
      }

      if (response.is_game_over) {
        setIsAiThinking(false);
      }
    } catch (err) {
      const detail = err.response?.data?.detail || err.message;
      setError('Move failed: ' + detail);
    } finally {
      setIsAiThinking(false);
    }
  }, [sessionId, isGameOver, isAiThinking]);

  // Import FEN
  const importFen = useCallback(async () => {
    if (!fenInput.trim()) return;
    try {
      const resp = await api.importFen(sessionId, fenInput.trim());
      setFen(resp.fen);
      setLegalMoves(resp.legal_moves || []);
      setIsGameOver(resp.is_game_over);
      setGameResult(resp.result || '');
      setFenInput('');
    } catch (err) {
      setError('Invalid FEN: ' + err.message);
    }
  }, [sessionId, fenInput]);

  // Undo last move
  const undoMove = useCallback(async () => {
    if (!sessionId || moveHistory.length < 2) return;
    try {
      const resp = await api.undoMove(sessionId);
      setFen(resp.fen);
      setMoveHistory(resp.move_history || []);
      setLegalMoves(resp.legal_moves || []);
      setIsGameOver(false);
      setGameResult('');
    } catch (err) {
      setError('Undo failed: ' + err.message);
    }
  }, [sessionId, moveHistory]);

  // Initialize on mount
  useEffect(() => {
    api.health()
      .then((h) => {
        setModelLoaded(Boolean(h.model_loaded));
        setModelName(h.model_name || '');
      })
      .catch(() => setModelLoaded(false));
    startNewGame('white');
  }, [startNewGame]);

  const subtitle = modelName
    ? `Play against ${modelName} fine-tuned for chess`
    : modelLoaded
      ? 'Play against your fine-tuned Qwen chess model'
      : 'Backend online — set QWEN_CHESS_CHECKPOINT to load a model';

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>♟ Qwen Chess</h1>
        <p className="subtitle">{subtitle}</p>
      </header>

      <div className="main-content">
        <div className="board-section">
          <ChessBoard
            fen={fen}
            onMove={makeMove}
            legalMoves={legalMoves}
            playerColor={playerColor}
            disabled={isGameOver || isAiThinking}
          />
        </div>

        <div className="side-panel">
          <GameControls
            onNewGame={startNewGame}
            onUndo={undoMove}
            onImportFen={importFen}
            fenInput={fenInput}
            onFenInputChange={setFenInput}
            temperature={temperature}
            onTemperatureChange={setTemperature}
            isAiThinking={isAiThinking}
            isGameOver={isGameOver}
            canUndo={moveHistory.length >= 2}
          />

          <GameInfo
            fen={fen}
            moveHistory={moveHistory}
            gameResult={gameResult}
            isGameOver={isGameOver}
            modelLoaded={modelLoaded}
            modelName={modelName}
          />

          {error && (
            <div className="error-message">
              {error}
              <button onClick={() => setError('')}>✕</button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
