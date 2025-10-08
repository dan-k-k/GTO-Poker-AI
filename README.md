# Advanced GTO Poker AI via Reinforcement Learning

An implementation of a Game Theory Optimal (GTO) poker AI using Neural Fictitious Self-Play (NFSP). The agent learns a near-unexploitable strategy for two-player No-Limit Texas Hold'em. The user interface is built with Streamlit for easy interaction.

## How to Play (with Docker)

This project is containerized with Docker, which is the recommended way to run the application. It guarantees a consistent, working environment without needing to install Python, dependencies, or configure anything on your local machine.

**Note on the AI Agent:** The default setup runs a basic **`RandomBot`** opponent for immediate playability. The trained GTO models (`.pt` files) are **not included** in this repository to keep it lightweight. If you wish to play against your own trained agent, please see the instructions at the bottom.

### Prerequisites
- You must have **Docker Desktop** installed and running on your computer.

### Step 1: Get the Docker Image
You have two options: pull the pre-built image from Docker Hub (easiest) or build it yourself from the source code.

#### Option A: Pull from Docker Hub (Recommended)
This downloads the ready-to-run application.

```bash
docker pull kingdaniel9/poker-bot:latest
```

#### Option B: Build the Image Yourself

If you want to build the image from the code in this repository:

```bash
# 1. Clone the repository
git clone https://github.com/dan-k-k/Poker-GTO-Bot.git
cd Poker-GTO-Bot

# 2. Instructions for training your own poker bot are at the end of the README.

# 3. Build the Docker image
docker build -t kingdaniel9/poker-bot:latest .
```

#### Step 2: Run the Game

This single command starts the poker bot and makes it accessible.

```bash
docker run --rm -it -p 8501:8501 kingdaniel9/poker-bot:latest
```

#### Step 3: Open Your Browser

Open your web browser (Chrome, Safari, etc.) and navigate to the following URL:

http://localhost:8501

The poker game interface should appear, ready to play against the RandomBot.

### Key Features
- **NFSP Architecture**: A Best Response (BR) agent learns to exploit, while an Average Strategy (AS) agent learns to be unexploitable.

- **Advanced Feature Engineering**: Uses a comprehensive feature vector including raw game state, action sequences, and strategic features.

- **Equity-Based Rewards**: The BR agent learns from a dense, equity-based reward signal for faster learning.

- **Opponent and Self Modeling**: Leverages statistical models of both players to inform strategy.

<details>
<summary><b>Click here for Local Development and Training Instructions</b></summary>

#### Local Installation

This is for developers who want to modify the code or run the training scripts directly.

```bash
# 1. Clone the repository
git clone https://github.com/dan-k-k/Poker-GTO-Bot.git
cd Poker-GTO-Bot

# 2. Create and activate a Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install the required packages
pip install -r requirements.txt

# 4. Install the application in editable mode
pip install -e .
```

#### Training Workflow

Training follows a curriculum to ensure stable convergence.

#### 1. Bootstrap Agent & Generate Initial Data

Run the main training for ~100 episodes. This uses a heuristic to teach the agent basics and create the first dataset for the range predictor.

```bash
python -m app.trainingL1.train_L1
# Note: You can pause training at any time with Ctrl + C and resume with the same command.
```
#### 2. Train Initial Range Predictor

Use the data from Step 1 to train the first version of the RangeNetwork.

```bash
python -m app.range_predictor.train_range_predictor
```
#### 3. Iterative Refinement

Now, alternate between running the main agent training (which will automatically load and use the range model) and re-training the range predictor with the new, higher-quality data.

```bash
# Run for another 200-300 episodes to generate better data
python -m app.trainingL1.train_L1

# Re-train the range predictor with the new data
python -m app.range_predictor.train_range_predictor

# Repeat this cycle
```
</details>