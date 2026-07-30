import React from 'react';
import { Chessboard } from 'react-chessboard';

/**
 * Interactive chess board component.
 */
function ChessBoard({ fen, onMove, legalMoves, playerColor, disabled }) {
  const onDrop = (sourceSquare, targetSquare) => {
    if (disabled) return false;

    const move = sourceSquare + targetSquare;
    // Check if it's a pawn promotion
    const piece = fen?.split(' ')[1]?.[0];
    const isPawn = piece === 'P' || piece === 'p';
    const fromRank = sourceSquare[1];
    const toRank = targetSquare[1];
    const isPromotion = isPawn && ((toRank === '8' && piece === 'P') || (toRank === '1' && piece === 'p'));

    if (isPromotion) {
      // Auto-promote to queen for simplicity
      onMove(move + 'q');
    } else {
      onMove(move);
    }
    return true;
  };

  return (
    <div className="chessboard-wrapper">
      <Chessboard
        id="qwen-chess-board"
        position={fen || 'start'}
        onPieceDrop={onDrop}
        boardOrientation={playerColor}
        boardWidth={480}
        customBoardStyle={{
          borderRadius: '8px',
          boxShadow: '0 4px 16px rgba(0, 0, 0, 0.15)',
        }}
        darkSquareStyle={{ backgroundColor: '#b58863' }}
        lightSquareStyle={{ backgroundColor: '#f0d9b5' }}
        arePiecesDraggable={!disabled}
      />
    </div>
  );
}

export default ChessBoard;
