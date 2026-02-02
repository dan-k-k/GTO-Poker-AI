# tests/test_api.py

import unittest
from fastapi.testclient import TestClient
from fastapi_app import app

class TestPokerAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Initialize the client once for the whole class
        cls.client = TestClient(app)

    def test_health_check(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "active")

    def test_successful_recommendation(self):
        # Using 'with' triggers the lifespan (startup) events!
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
            
            if response.status_code == 500:
                print("\nSERVER ERROR DETAIL:", response.json())
                
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertIn(data["action_type"], ["fold", "call", "raise"])

    def test_malformed_json(self):
        # Test missing required fields
        payload = {"pot": 3} 
        response = self.client.post("/get_optimal_action", json=payload)
        # FastAPI/Pydantic returns 422 for validation errors
        self.assertEqual(response.status_code, 422)

if __name__ == "__main__":
    unittest.main()