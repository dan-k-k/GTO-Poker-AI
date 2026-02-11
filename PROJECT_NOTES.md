# Poker AI Development Notes
  
## Current Focus
- Remote API/Docker so anyone can play against the trained agent
- Performance
  
## Long-term Goals
- Support for > 2 players per table (more expressive features needed, e.g. blocker ranks)
- Exploitative layer, player profiles.

## Current Focus
- Add a database to log previous hands and lookup?
- Add CD. Containerise? *.pt files are small, so is fine as is. 

## Done / Implemented
- Graceful resumption
- Hard fails - no pollution
- Atomic saving
- Reservoir sampling for the strategy buffer
- Discrete bet sizes (Fold, Call, 50%, Pot, All-in)
- Basic metrics (VPIP, Win Rate)

