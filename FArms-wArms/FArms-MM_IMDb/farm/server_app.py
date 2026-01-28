"""Federated Cross-Modal Simulation: Server App"""

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from farm.task import Net, load_test_data, test_final

# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""
    
    # Read run config
    fraction_train: float = context.run_config["fraction-train"]
    num_rounds: int = context.run_config["num-server-rounds"]
    local_epochs: int = context.run_config["local-epochs"]
    lr: float = context.run_config["lr"]
    alpha: float = context.run_config["alpha"]
    beta: float = context.run_config["beta"]
    num_supernodes: int = context.run_config.get("num-supernodes", 5)
    
    # Parse missing configs from comma-separated string
    missing_configs_str: str = context.run_config.get("missing-configs", "100_image_100_text")
    missing_configs: list = [config.strip() for config in missing_configs_str.split(",")]
    
    # Iterate through all configs
    for missing_config in missing_configs:
        print(f"\n{'='*80}")
        print(f"Starting Federated Learning with Cross-Modal Simulation")
        print(f"{'='*80}")
        print(f"  Config: {missing_config}")
        print(f"  Rounds: {num_rounds}")
        print(f"  SuperNodes: {num_supernodes}")
        print(f"  Local epochs: {local_epochs}")
        print(f"  Fraction train: {fraction_train}")
        print(f"  Learning rate: {lr}")
        print(f"  Alpha (simulation loss): {alpha}")
        print(f"  Beta (classification loss): {beta}")
        print(f"{'='*80}\n")
        
        # Create config record that will be sent to both train AND evaluate
        config = ConfigRecord({
            "lr": lr,
            "alpha": alpha,
            "beta": beta,
            "missing-config": missing_config
        })
        
        # Get device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Load global model
        global_model = Net(device=device)
        arrays = ArrayRecord(global_model.state_dict())
        
        # Initialize FedAvg strategy
        strategy = FedAvg(
            fraction_train=fraction_train
        )
        
        # Pass config to both training AND evaluation
        result = strategy.start(
            grid=grid,
            initial_arrays=arrays,
            train_config=config,
            evaluate_config=config,
            num_rounds=num_rounds,
        )
        
        # Create filename with all parameters
        model_filename = f"final_model-n_{num_supernodes}-r_{num_rounds}-e_{local_epochs}-{missing_config}.pt"
        results_filename = f"results-n_{num_supernodes}-r_{num_rounds}-e_{local_epochs}-{missing_config}.json"
        
        # Save final model to disk
        print(f"\nSaving final model to disk...")
        state_dict = result.arrays.to_torch_state_dict()
        torch.save(state_dict, model_filename)
        print(f"Model saved to: {model_filename}")
        
        # Final evaluation on test set
        print("\n" + "="*80)
        print("Loading final model for test set evaluation...")
        print("="*80)
        
        final_model = Net(device=device)
        final_model.load_state_dict(state_dict)
        final_model.to(device)
        
        # Load test data with same missing config
        testloader = load_test_data(missing_config=missing_config)
        
        # Evaluate on test set
        test_metrics = test_final(final_model, testloader, device)
        
        # Save test results
        import json
        results_summary = {
            'method': 'cross-modal-simulation-federated',
            'config': missing_config,
            'num_supernodes': num_supernodes,
            'num_rounds': num_rounds,
            'local_epochs': local_epochs,
            'alpha': alpha,
            'beta': beta,
            'test_metrics': test_metrics
        }
        
        with open(results_filename, 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        print(f"\nTest results saved to: {results_filename}")
        print(f"\n{'='*80}")
        print(f"Federated Learning Complete for config: {missing_config}")
        print(f"{'='*80}\n")
    
    print(f"\n{'#'*80}")
    print(f"# ALL CONFIGURATIONS COMPLETED!")
    print(f"{'#'*80}\n")