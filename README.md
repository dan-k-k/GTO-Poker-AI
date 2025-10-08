## GTO Poker Bot using Neural Fictitious Self-Play

This project is a Python-based implementation of a Texas Hold'em poker bot that approximates a Game Theory Optimal strategy heads-up (two-player).

The agent learns by playing against itself, using a combination of Reinforcement Learning (for finding the best response) and Supervised Learning (for building an average strategy).

The agent intelligently predicts its opponent's range by using their trained network to find the probability of holding random hands (Monte Carlo) for its most recent play given the action throughout the hand, which is then multiplied by the outcome (1 for win, 0 for loss, averaging its probability of winning the hand when the board is run-out). e.g. if the opponent may only bluff via triple barrel, our equity on the river increases for this specific scenario. As agents continue to play, these learned traits do not last in the extremes. In deployment, the agent can then use its own network to predict what it'd have done as its own opponent.


![Poker Live Demo](images/Pokerexamplelog2.gif)

###### 1. Clone the Repository
```bash
git clone https://github.com/dan-k-k/GTO-Poker-AI
cd GTO-Poker-AI
```

###### 2. Set Up a Virtual Environment
```bash
# Create the virtual environment
python -m venv venv

# Activate (macOS/Linux)
source venv/bin/activate

# Activate (Windows)
.\venv\Scripts\activate
```

###### 3. Install requirements
```bash
pip install -r requirements.txt
```

##### Train!
```bash
python -m app.train_nfsp
# You can monitor training_output/hand_history.log to see agents' plays

# If you make performance improvements, benchmark with:
python -m cProfile -o testtrain.pstats -m unittest tests.test_training
snakeviz testtrain.pstats
```

### Play against a trained Agent
```bash
python main.py
# Go to: http://127.0.0.1:5001
```