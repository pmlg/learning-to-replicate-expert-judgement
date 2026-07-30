import axios from 'axios';

const API_BASE = import.meta.env.VITE_API_URL || '/api';

const client = axios.create({
  baseURL: API_BASE,
  timeout: 30000,
  headers: { 'Content-Type': 'application/json' },
});

/**
 * API wrapper for the Qwen Chess backend.
 */
const api = {
  /** Check server health */
  health: async () => {
    const resp = await client.get('/health');
    return resp.data;
  },

  /** Start a new game session */
  newGame: async (fen = null, color = 'white') => {
    const resp = await client.post('/new_game', {
      fen: fen,
      color: color,
    });
    return resp.data;
  },

  /** Make a move in the current game */
  makeMove: async (sessionId, move) => {
    const resp = await client.post('/move', {
      session_id: sessionId,
      move: move,
    });
    return resp.data;
  },

  /** Get current game state */
  getGame: async (sessionId) => {
    const resp = await client.get(`/game/${sessionId}`);
    return resp.data;
  },

  /** Undo last move pair */
  undoMove: async (sessionId) => {
    const resp = await client.post(`/undo/${sessionId}`);
    return resp.data;
  },

  /** Import a FEN position */
  importFen: async (sessionId, fen) => {
    const resp = await client.post(`/import_fen?session_id=${sessionId}&fen=${encodeURIComponent(fen)}`);
    return resp.data;
  },

  /** Export FEN */
  exportFen: async (sessionId) => {
    const resp = await client.get(`/fen_export/${sessionId}`);
    return resp.data;
  },
};

export default api;
