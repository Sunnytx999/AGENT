from __future__ import annotations

import unittest

import chess

from chess_agent.game import ChessGame


class ChessGameRuleTests(unittest.TestCase):
    def test_initial_position_has_20_legal_moves(self) -> None:
        game = ChessGame()

        self.assertEqual(len(game.legal_moves()), 20)

    def test_castling_moves_are_legal(self) -> None:
        game = ChessGame("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
        legal_ucis = {move["uci"] for move in game.legal_moves()}

        self.assertIn("e1g1", legal_ucis)
        self.assertIn("e1c1", legal_ucis)

    def test_en_passant_move_is_legal(self) -> None:
        game = ChessGame(
            "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3"
        )
        legal_ucis = {move["uci"] for move in game.legal_moves()}

        self.assertEqual(game.board.ep_square, chess.D6)
        self.assertIn("e5d6", legal_ucis)

    def test_promotion_moves_are_legal(self) -> None:
        game = ChessGame("8/P7/8/8/8/8/8/k6K w - - 0 1")
        legal_ucis = {move["uci"] for move in game.legal_moves()}

        self.assertIn("a8q", legal_ucis)
        self.assertIn("a8r", legal_ucis)
        self.assertIn("a8b", legal_ucis)
        self.assertIn("a8n", legal_ucis)

    def test_check_status(self) -> None:
        game = ChessGame("4k3/8/8/8/8/8/4r3/4K3 w - - 0 1")
        status = game.status()

        self.assertTrue(status["is_check"])
        self.assertFalse(status["is_checkmate"])
        self.assertFalse(status["is_game_over"])

    def test_checkmate_status(self) -> None:
        game = ChessGame("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1")
        status = game.status()

        self.assertTrue(status["is_check"])
        self.assertTrue(status["is_checkmate"])
        self.assertTrue(status["is_game_over"])
        self.assertEqual(status["result"], "1-0")

    def test_stalemate_status(self) -> None:
        game = ChessGame("k7/8/1Q6/8/8/8/8/7K b - - 0 1")
        status = game.status()

        self.assertFalse(status["is_check"])
        self.assertTrue(status["is_stalemate"])
        self.assertTrue(status["is_game_over"])
        self.assertEqual(status["result"], "1/2-1/2")

    def test_push_uci_rejects_illegal_move(self) -> None:
        game = ChessGame()

        with self.assertRaises(ValueError):
            game.push_uci("e2e5")
        self.assertEqual(game.board.move_stack, [])

    def test_push_uci_rejects_invalid_uci(self) -> None:
        game = ChessGame()

        with self.assertRaises(ValueError):
            game.push_uci("e9e9")
        self.assertEqual(game.board.move_stack, [])

    def test_push_uci_legal_move_updates_fen_and_move_stack(self) -> None:
        game = ChessGame()

        summary = game.push_uci("e2e4")

        self.assertEqual(summary["uci"], "e2e4")
        self.assertEqual(summary["san"], "e4")
        self.assertEqual(game.board.move_stack[-1].uci(), "e2e4")
        self.assertEqual(
            game.board.fen(),
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
        )


if __name__ == "__main__":
    unittest.main()
