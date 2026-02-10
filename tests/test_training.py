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
import tempfile
from app.train_nfsp import main as run_training

# --- Test Configuration ---
# We define the structure relative to the sandbox we will create
TEST_CONFIG_RELATIVE_PATH = os.path.join("tests", "test_config.yaml")
TEST_CONFIG_DATA = {
    'training': {
        'num_episodes': 10,  # Reduced for faster testing
        'save_interval': 5,
        'eval_interval': 5,
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
        'random_equity_trials': 10, # Reduced for faster testing
        'intelligent_equity_trials': 10,
    },
    'logging': {
        'verbose': True,
        'dump_features': False 
    }
}

class TestTrainingPipeline(unittest.TestCase):
    def setUp(self):
        # 1. Save the original location so we can return later
        self.original_cwd = os.getcwd()
        
        # 2. Create a temporary directory (The Sandbox)
        self.test_dir = tempfile.mkdtemp()
        
        # 3. Setup the config file inside the sandbox
        # We mimic the folder structure "tests/test_config.yaml" inside the temp dir
        sandbox_tests_dir = os.path.join(self.test_dir, "tests")
        os.makedirs(sandbox_tests_dir, exist_ok=True)
        
        self.sandbox_config_path = os.path.join(self.test_dir, TEST_CONFIG_RELATIVE_PATH)
        with open(self.sandbox_config_path, 'w') as f:
            yaml.dump(TEST_CONFIG_DATA, f)

        # 4. Switch context: Make the sandbox the current working directory
        # The script will now think the temp dir is the project root
        os.chdir(self.test_dir)

    def tearDown(self):
        # 1. Return to the original directory
        os.chdir(self.original_cwd)
        
        # 2. Delete the sandbox and all the garbage outputs created by the test
        shutil.rmtree(self.test_dir)

    def test_training_dry_run_produces_all_artifacts(self):
        try:
            # Run training using the config inside the sandbox
            # The script will write to "./training_output", which is now inside our temp dir
            run_training(config_path=TEST_CONFIG_RELATIVE_PATH)
        except Exception as e:
            raise e.__class__(f"The training script crashed: {e}") from e

        # All assertions check the SANDBOX directory, not your real project
        output_dir = "training_output"
        
        self.assertTrue(os.path.isdir(output_dir), "Main output directory was not created in sandbox.")

        log_path = os.path.join(output_dir, "hand_history.log")
        self.assertTrue(os.path.isfile(log_path), "Hand history log file was not created.")

        models_dir = os.path.join(output_dir, "models")
        self.assertTrue(os.path.isdir(models_dir), "Models directory was not created.")

        saved_files = os.listdir(models_dir)
        self.assertTrue(any(f.endswith('_best.pt') for f in saved_files),
                        f"No '_best.pt' found in {saved_files}")
        self.assertTrue(any(f.endswith('_latest.pt') for f in saved_files),
                        f"No '_latest.pt' found in {saved_files}")
        
        # === Buffers ===
        buffers_dir = os.path.join(output_dir, "buffers")
        self.assertTrue(os.path.isdir(buffers_dir), "Buffers directory was not created.")

        # Check that buffer files were created for each agent (assuming 2 agents)
        for i in range(2):
            rl_buffer_path = os.path.join(buffers_dir, f"agent{i}_rl_buffer.pkl")
            sl_buffer_path = os.path.join(buffers_dir, f"agent{i}_sl_buffer.pkl")
            self.assertTrue(os.path.isfile(rl_buffer_path), f"RL buffer for agent {i} missing.")
            self.assertTrue(os.path.isfile(sl_buffer_path), f"SL buffer for agent {i} missing.")

if __name__ == '__main__':
    unittest.main()

