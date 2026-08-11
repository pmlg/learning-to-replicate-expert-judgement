import React from 'react';

/**
 * Game information panel: FEN display, move history, result.
 */
function GameInfo({ fen, moveHistory, gameResult, isGameOver, modelLoaded, modelName }) {
  return (
    <div className="game-info">
      <h3>Game Info</h3>

      <div className="info-section">
        <label>FEN:</label>
        <code className="fen-display">{fen}</code>
      </div>

      <div className="info-section">
        <label>Status:</label>
        <span className={`status ${modelLoaded ? 'online' : 'offline'}`}>
          {modelLoaded
            ? `Model Ready${modelName ? ` (${modelName})` : ''}`
            : 'Model not loaded — restart backend with checkpoint'}
        </span>
      </div>

      {isGameOver && gameResult && (
        <div className="info-section result-section">
          <label>Result:</label>
          <span className="game-result">{gameResult}</span>
        </div>
      )}

      <div className="info-section">
        <label>Move History ({moveHistory.length}):</label>
        <div className="move-history">
          {moveHistory.length === 0 ? (
            <span className="empty-hint">No moves yet</span>
          ) : (
            moveHistory.map((move, i) => (
              <span key={i} className="move-tag">
                <span className="move-number">{Math.floor(i / 2) + 1}{i % 2 === 0 ? '.' : '...'}</span>
                {move}
              </span>
            ))
          )}
        </div>
      </div>
    </div>
  );
}

export default GameInfo;
