"""farm: A Flower / PyTorch app."""

import torch
import numpy as np
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from farm.task import Net, load_test_data, test_final

import json
from pathlib import Path

# Create ServerApp
app = ServerApp()


def convert_to_python_types(obj):
    """Recursively convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, dict):
        return {key: convert_to_python_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_python_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

class FedAvgWithRound(FedAvg):
    """Custom FedAvg strategy that passes round information to clients."""
    
    def __init__(self, base_config_dict, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_config_dict = base_config_dict
        self.current_round = 0
    
    def configure_train(self, round, *args, **kwargs):
        """Override to inject current round into config."""
        self.current_round = round
        
        # FIXED: Call parent method first to get proper Message format
        parent_result = super().configure_train(round, *args, **kwargs)
        
        # Create config with current round
        config_dict = self.base_config_dict.copy()
        config_dict["current-round"] = round
        
        # Update the config in the parent result
        if parent_result:
            for msg in parent_result:
                msg.content["config"] = ConfigRecord(config_dict)
        
        return parent_result
    
    def configure_evaluate(self, round, *args, **kwargs):
        """Override to inject current round into eval config."""
        # FIXED: Call parent method first to get proper Message format
        parent_result = super().configure_evaluate(round, *args, **kwargs)
        
        # Create config with current round
        config_dict = self.base_config_dict.copy()
        config_dict["current-round"] = round
        
        # Update the config in the parent result
        if parent_result:
            for msg in parent_result:
                msg.content["config"] = ConfigRecord(config_dict)
        
        return parent_result


from typing import List, Optional
def set_seed(seed):
    """Set all random seeds for reproducibility"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Read run config
    num_rounds_for_sim_train: int = context.run_config["num-rounds-for-sim-train"]
    fraction_train: float = context.run_config["fraction-train"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["lr"]
    local_epochs: int = context.run_config["local-epochs"]

    # Get missing configs list (iterate through multiple configs)
    missing_configs_str: str = context.run_config.get("missing-configs", "100_text_100_audio_100_video")
    missing_configs = [config.strip() for config in missing_configs_str.split(',')]
    # Get num-supernodes from federation config
    num_supernodes: int = context.run_config.get("num-supernodes", 4)

    # Get alpha and beta from run config (with defaults)
    alpha: float = context.run_config.get("alpha", 1.0)
    beta: float = context.run_config.get("beta", 1.0)
    
    num_of_runs: int = context.run_config["num-of-runs"]
    
    print(f"\n{'='*80}")
    print(f"Starting Federated Learning with Farm")
    print(f"  Number of configs to run: {len(missing_configs)}")
    print(f"  Configs: {missing_configs}")
    print(f"  Num supernodes: {num_supernodes}")
    print(f"  Rounds per config: {num_rounds}")
    print(f"  Local epochs: {local_epochs}")
    print(f"  Fraction train: {fraction_train}")
    print(f"  Learning rate: {lr}")
    print(f"  Alpha (sim loss weight): {alpha}")
    print(f"  Beta (cls loss weight): {beta}")
    print(f"{'='*80}\n")

    my_seeds = [
    42, 123, 256, 512, 1024,
    2048, 3141, 5000, 7777, 9999,
    12345, 54321, 11111, 22222, 33333,
    44444, 55555, 66666, 77777, 88888,
    99999, 13579, 24680, 31415, 27182,
    16180, 86753, 10101, 20202, 30303
    ]    
    
    for config_idx, missing_config in enumerate(missing_configs, 1):
        # Iterate through each config
        total_test_fmic = 0.0
        ave_test_fmic = 0.0
        min_test_fmic = float('inf')
        for run_id in range(1, num_of_runs + 1):
            seed = my_seeds[run_id - 1]
            set_seed(seed)
        
                # FIXED: Create a plain dictionary (not ConfigRecord) for base_config_dict
            base_config_dict = {
                "num-rounds": num_rounds,
                "lr": lr, 
                "missing-config": missing_config,
                "alpha": alpha,
                "beta": beta,
                "num-rounds-for-sim-train": num_rounds_for_sim_train,
                "seed": seed,
                "run-id": run_id,
                "ave-test-f_mic": ave_test_fmic,
                "min-test-f_mic": min_test_fmic
            }

            # Load global model
            global_model = Net()
            arrays = ArrayRecord(global_model.state_dict())

            # Initialize FedAvg strategy with dictionary (not ConfigRecord)
            strategy = FedAvgWithRound(
                base_config_dict=base_config_dict,  # FIXED: Pass dict, not ConfigRecord
                fraction_train=fraction_train
            )

            # FIXED: Don't pass train_config and evaluate_config here
            # The custom strategy handles it in configure_train/configure_evaluate
            result = strategy.start(
                grid=grid,
                initial_arrays=arrays,
                num_rounds=num_rounds
            )

            # Generate filename
            folder_name = f"Farm-XCLP-n_{num_supernodes}-tr_{num_rounds}-sr_{num_rounds_for_sim_train}-e_{local_epochs}"
            filename_base = f"{missing_config}-{run_id}"
            model_filename = f"{filename_base}.pt"
            results_filename = f"{filename_base}.json"

            folder_path = Path(folder_name)
            folder_path.mkdir(exist_ok=True)  

            # Create a text file inside that folder
            results_file_path = folder_path / results_filename  
            model_file_path = folder_path / model_filename  
        
            # Save final model to disk
            state_dict = result.arrays.to_torch_state_dict()
            torch.save(state_dict, model_file_path)
            print(f"Model saved to: {model_file_path}")
        
            # Final evaluation on test set
        
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            final_model = Net()
            final_model.load_state_dict(state_dict)
            final_model.to(device)
        
            # Load test data with same missing config
            testloader = load_test_data(seed, missing_config=missing_config)
        
            # Evaluate on test set
            test_metrics = test_final(final_model, testloader, device)

            #just to debug and see results in terminal during the run.
            f1_mic = test_metrics['f1_micro']
            total_test_fmic += f1_mic
            if f1_mic < min_test_fmic:
                min_test_fmic = f1_mic
            ave_test_fmic = total_test_fmic / (run_id + 0.0)
        
            # Save test results - Convert numpy types to Python native types
            
            results_summary = {
                'method': 'farm2sep',
                'num_supernodes': num_supernodes,
                'config': missing_config,
                'num_rounds': num_rounds,
                'local_epochs': local_epochs,
                'run-id': run_id,
                'seed': seed,
                'test_metrics': convert_to_python_types(test_metrics)  # Convert here!
            }
        
            with open(results_file_path, 'w') as f:
                json.dump(results_summary, f, indent=2)
        
            print(f"\nTest results saved to: {results_filename}")
            print(f"\n{'='*80}")
            print(f"Config {config_idx}/{len(missing_configs)} Complete!")
            print(f"{'='*80}\n")
    
       