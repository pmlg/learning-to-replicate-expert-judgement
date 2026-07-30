import React from 'react';

/**
 * Game control buttons and input forms.
 */
function GameControls({
  onNewGame,
  onUndo,
  onImportFen,
  fenInput,
  onFenInputChange,
  temperature,
  onTemperatureChange,
  isAiThinking,
  isGameOver,
  canUndo,
}) {
  return (
    <div className="game-controls">
      <h3>Controls</h3>

      <div className="control-group">
        <label>Play as:</label>
        <div className="button-group">
          <button onClick={() => onNewGame('white')} className="btn btn-primary">
            ♔ White
          </button>
          <button onClick={() => onNewGame('black')} className="btn btn-secondary">
            ♚ Black
          </button>
        </div>
      </div>

      <div className="control-group">
        <label>Model Temperature: {temperature.toFixed(1)}</label>
        <input
          type="range"
          min="0"
          max="1"
          step="0.1"
          value={temperature}
          onChange={(e) => onTemperatureChange(parseFloat(e.target.value))}
          className="slider"
        />
      </div>

      <div className="control-group">
        <button onClick={onUndo} disabled={!canUndo || isAiThinking} className="btn btn-secondary">
          ↩ Undo Move
        </button>
      </div>

      <div className="control-group">
        <label>Import FEN:</label>
        <div className="input-row">
          <input
            type="text"
            value={fenInput}
            onChange={(e) => onFenInputChange(e.target.value)}
            placeholder="Paste FEN string..."
            className="fen-input"
          />
          <button onClick={onImportFen} className="btn btn-small">
            Import
          </button>
        </div>
      </div>

      {isAiThinking && (
        <div className="thinking-indicator">
          <span className="spinner"></span> AI is thinking...
        </div>
      )}

      {isGameOver && (
        <div className="game-over-msg">
          <strong>Game Over!</strong>
          <button onClick={() => onNewGame('white')} className="btn btn-primary">
            Play Again
          </button>
        </div>
      )}
    </div>
  );
}

export default GameControls;
