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
  const [error, setError] = useState('');
  const [temperature, setTemperature] = useState(0.1);
  const [fenInput, setFenInput] = useState('');

  // Start a new game
  const startNewGame = useCallback(async (color = 'white', initialFen = null) => {
    try {
      const response = await api.newGame(initialFen, color);
      setSessionId(response.session_id);
      setFen(response.fen);
      setMoveHistory([]);
      setIsGameOver(false);
      setGameResult('');
      setLegalMoves(response.legal_moves || []);
      setPlayerColor(color);
      setError('');
      setModelLoaded(true);
    } catch (err) {
      setError('Failed to start new game: ' + err.message);
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
      setError('Invalid move: ' + err.message);
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
    startNewGame('white');
  }, [startNewGame]);

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>♟ Qwen Chess</h1>
        <p className="subtitle">Play against a Qwen-8B model fine-tuned for chess</p>
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
