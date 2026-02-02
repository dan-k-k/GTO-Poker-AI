# tests/test_training.py
# find . -type d -name "__pycache__" -exec rm -r {} +

# Begins two models from scratch, making it a decent benchmark, 
# but there will be some small variance still ~5%:
# python -m cProfile -o testtrain.pstats -m unittest tests.test_training
# snakeviz testtrain.pstats
import unittest
import os
import shutil
import yaml
import json

from app.train_nfsp import main as run_training

# --- Test Configuration ---
TEST_OUTPUT_DIR = os.path.join("tests", "test_output")
TEST_CONFIG_PATH = os.path.join("tests", "test_config.yaml")
TEST_CONFIG_DATA = {
    'training': {
        'num_episodes': 100,
        'save_interval': 50,
        'eval_interval': 50,
    },
    'agent': {
        'eta': 0.1,
        'learning_rate': 0.001,
        'gamma': 1,
        'batch_size': 4,
        'update_frequency': 1,
        'target_update_frequency': 100, 
    },
    'buffers': {
        'rl_buffer_capacity': 100,
        'sl_buffer_capacity': 100,
    },
    'simulations': {
        'random_equity_trials': 200,
        'intelligent_equity_trials': 200,
    },
    'logging': {
        'dump_features': False 
    }
}


class TestTrainingPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a backup of the real training_output if it exists
        if os.path.exists("training_output"):
            os.rename("training_output", "training_output_backup")
        # Ensure the test output directory is clean
        if os.path.exists(TEST_OUTPUT_DIR):
            shutil.rmtree(TEST_OUTPUT_DIR)

        # Create the test config file
        os.makedirs(os.path.dirname(TEST_CONFIG_PATH), exist_ok=True)
        with open(TEST_CONFIG_PATH, 'w') as f:
            yaml.dump(TEST_CONFIG_DATA, f)

    @classmethod
    def tearDownClass(cls):
        # Clean up test artifacts
        if os.path.exists("training_output"):
            shutil.rmtree("training_output")
        if os.path.exists(TEST_CONFIG_PATH):
            os.remove(TEST_CONFIG_PATH)
        # Restore the original training_output directory
        if os.path.exists("training_output_backup"):
            os.rename("training_output_backup", "training_output")

    def test_training_dry_run_produces_all_artifacts(self):
        try:
            # The trainer writes to a hardcoded "training_output" dir.
            # The test will check for artifacts there.
            run_training(config_path=TEST_CONFIG_PATH)
        except Exception as e:
            # Re-raise the exception with its full traceback for better debugging
            raise e.__class__(f"The training script crashed with an exception: {e}") from e

        output_dir = "training_output"
        self.assertTrue(os.path.isdir(output_dir), "Main output directory was not created.")

        log_path = os.path.join(output_dir, "hand_history.log")
        self.assertTrue(os.path.isfile(log_path), "Hand history log file was not created.")

        models_dir = os.path.join(output_dir, "models")
        self.assertTrue(os.path.isdir(models_dir), "Models directory was not created.")

        saved_files = os.listdir(models_dir)
        self.assertTrue(any(f.endswith('_best.pt') for f in saved_files),
                        "No '_best.pt' model files were saved after evaluation.")
        self.assertTrue(any(f.endswith('_latest.pt') for f in saved_files),
                        "No '_latest.pt' model files were saved after evaluation.")
        
        # === Buffers ===
        buffers_dir = os.path.join(output_dir, "buffers")
        self.assertTrue(os.path.isdir(buffers_dir), "Buffers directory was not created.")

        # Check that buffer files were created for each agent (assuming 2 agents)
        for i in range(2):
            rl_buffer_path = os.path.join(buffers_dir, f"agent{i}_rl_buffer.pkl")
            sl_buffer_path = os.path.join(buffers_dir, f"agent{i}_sl_buffer.pkl")
            self.assertTrue(os.path.isfile(rl_buffer_path), f"RL buffer for agent {i} was not created.")
            self.assertTrue(os.path.isfile(sl_buffer_path), f"SL buffer for agent {i} was not created.")

if __name__ == '__main__':
    unittest.main()

