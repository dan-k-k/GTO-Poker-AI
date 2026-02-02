# Poker AI Development Notes
  
## Current Focus
- Remote API so anyone can play against the trained agent
- Performance
- Better evaluation logic for saving best models. Currently assuming latest model is best
- Possible addition of features on board/hand texture for more context on when to bluff etc.
- Counter for the number of episodes ran since having to start from scratch
  
## Long-term Goals
- An exploitative layer that tracks opponent tendencies
- Support for > 2 players per table (more expressive features needed, e.g. blocker ranks)

## Current Focus
- Add a database to log previous hands and lookup.
- Add CI/CD.
- Containerise?
- Decide if other intricate features have room to exist.
- Discrete bet sizes.
- Plots for metrics such as VPIP
- Reservoir sampling for SL buffer.
- Test in a live environment.

