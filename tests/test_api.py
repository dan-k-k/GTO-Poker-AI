# tests/test_api.py

import unittest
from fastapi.testclient import TestClient
from unified_api import app 

class TestUnifiedPokerAPI(unittest.TestCase):
    
    def test_index_html(self):
        """Test that the root URL returns the HTML game page."""
        with TestClient(app) as client:
            response = client.get("/")
            self.assertEqual(response.status_code, 200)
            self.assertIn("text/html", response.headers["content-type"])
            self.assertIn("Poker vs NFSP Bot", response.text)

    def test_solver_recommendation(self):
        """Test the stateless solver endpoint (JSON in and recommendation out)."""
        with TestClient(app) as client:
            payload = {
                "pot": 3,
                "current_bets": [1, 2],
                "stacks": [199, 198],
                "initial_stacks": [200, 200],
                "hole_cards": [["Ah", "Kd"], ["Qs", "Js"]],
                "community_cards": [],
                "dealer_id": 0,
                "to_move": 0,
                "stage": 0,
                "past_actions": []
            }
            response = client.post("/get_optimal_action", json=payload)
            
            if response.status_code != 200:
                print("\nSolver error:", response.json())
                
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertIn("action_type", data)
            self.assertIn(data["action_type"], ["fold", "call", "raise"])

    def test_game_session_flow(self):
        """Test the stateful game session flow (cookies, game state)."""
        with TestClient(app) as client:
            response = client.get("/game_state")
            self.assertEqual(response.status_code, 200)
            
            cookie = response.cookies.get("session_id")
            self.assertIsNotNone(cookie, "Session ID cookie was not set!")
            
            data = response.json()
            self.assertIn("table_image", data)
            self.assertIn("legal_actions", data)
            self.assertIn("pnl_history", data)
            
            if data['legal_actions']:
                action_payload = {"action": "call", "amount": 0}
                action_response = client.post("/action", json=action_payload)
                
                if action_response.status_code == 200:
                     self.assertIn("table_image", action_response.json())
                elif action_response.status_code == 400:
                     self.assertIn("error", action_response.json())

    def test_malformed_json_solver(self):
        """Test validation error for missing fields."""
        with TestClient(app) as client:
            payload = {"pot": 3} # Missing stacks, cards, etc.
            response = client.post("/get_optimal_action", json=payload)
            self.assertEqual(response.status_code, 422)

if __name__ == "__main__":
    unittest.main()

