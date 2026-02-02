# api_schemas.py
from pydantic import BaseModel
from typing import List, Optional

class ActionLog(BaseModel):
    seat_id: int 
    stage: int 
    action_type: int 
    amount_added: int 

class GameStateInput(BaseModel):
    pot: int
    current_bets: List[int]
    stacks: List[int]
    initial_stacks: List[int] # <--- Required now
    hole_cards: List[List[str]] 
    community_cards: List[str]
    dealer_id: int
    to_move: int
    stage: int
    
    past_actions: List[ActionLog] = [] # <--- Required for history
    legal_actions: Optional[List[int]] = None

class ActionResponse(BaseModel):
    action_type: str
    amount: Optional[int] = None
    confidence: Optional[float] = None
    policy: str

