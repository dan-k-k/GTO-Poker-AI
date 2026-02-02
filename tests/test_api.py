# tests/test_api.py

import unittest
from fastapi.testclient import TestClient
# Import the app from your NEW unified script
from unified_api import app 

class TestUnifiedPokerAPI(unittest.TestCase):
    
    def test_index_html(self):
        """Test that the root URL returns the HTML game page."""
        with TestClient(app) as client:
            response = client.get("/")
            self.assertEqual(response.status_code, 200)
            self.assertIn("text/html", response.headers["content-type"])
            # Check for a specific string we know exists in index.html
            self.assertIn("Poker vs NFSP Bot", response.text)

    def test_solver_recommendation(self):
        """Test the stateless solver endpoint (JSON in -> Recommendation out)."""
        with TestClient(app) as client:
            payload = {
                "pot": 3,
                "current_bets": [1, 2],
                "stacks": [199, 198],
                "initial_stacks": [200, 200], # Required field
                "hole_cards": [["Ah", "Kd"], ["Qs", "Js"]],
                "community_cards": [],
                "dealer_id": 0,
                "to_move": 0,
                "stage": 0,
                "past_actions": []
            }
            response = client.post("/get_optimal_action", json=payload)
            
            if response.status_code != 200:
                print("\nSOLVER ERROR:", response.json())
                
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertIn("action_type", data)
            self.assertIn(data["action_type"], ["fold", "call", "raise"])

    def test_game_session_flow(self):
        """Test the stateful game session flow (Cookies -> Game State)."""
        with TestClient(app) as client:
            # 1. Start a game (GET /game_state)
            # This should set a 'session_id' cookie
            response = client.get("/game_state")
            self.assertEqual(response.status_code, 200)
            
            # Check for cookie existence
            cookie = response.cookies.get("session_id")
            self.assertIsNotNone(cookie, "Session ID cookie was not set!")
            
            # Check response structure
            data = response.json()
            self.assertIn("table_image", data)
            self.assertIn("legal_actions", data)
            self.assertIn("pnl_history", data)

            # 2. Perform a Human Action (POST /action)
            # TestClient automatically persists cookies between requests in the same 'with' block,
            # so we don't strictly need to manually set it, but we can verify it works.
            
            # NOTE: We can only act if it is actually our turn (to_move == 0).
            # Since a new game always starts with player 0 (SB) or 1 (BB) depending on config,
            # we check the state first.
            if data['legal_actions']:
                action_payload = {"action": "call", "amount": 0}
                action_response = client.post("/action", json=action_payload)
                
                if action_response.status_code == 200:
                     self.assertIn("table_image", action_response.json())
                elif action_response.status_code == 400:
                     # 400 is acceptable if it wasn't our turn, but the API responded correctly
                     self.assertIn("error", action_response.json())

    def test_malformed_json_solver(self):
        """Test validation error for missing fields."""
        with TestClient(app) as client:
            payload = {"pot": 3} # Missing stacks, cards, etc.
            response = client.post("/get_optimal_action", json=payload)
            self.assertEqual(response.status_code, 422)

if __name__ == "__main__":
    unittest.main()

