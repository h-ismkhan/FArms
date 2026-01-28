"""farm: A Flower / PyTorch app."""

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from farm.task import Net, load_data
from farm.task import test as test_fn
from farm.task import train_task, train_sim, test_final, test
import copy

# Flower ClientApp
app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""

    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device used for testing: {device} ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    model.to(device)

    # Load the data with missing config from server
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    missing_config = msg.content["config"]["missing-config"]  # No default, comes from server
    num_r_for_sim_train: int = msg.content["config"]["num-rounds-for-sim-train"]    
    alpha: float = msg.content["config"]["alpha"]
    beta: float = msg.content["config"]["beta"]
    seed: int = msg.content["config"]["seed"]
    local_epoch: int =  context.run_config["local-epochs"]
    run_id: int = msg.content["config"]["run-id"]
    current_round = msg.content["config"]["current-round"]

    trainloader, _ = load_data(seed, partition_id, num_partitions, missing_config)

    ave_fmic: int =  msg.content["config"]["ave-test-f_mic"]
    min_fmic: int =  msg.content["config"]["min-test-f_mic"]    

    reg_loss = 0.0

    if current_round <= num_r_for_sim_train:
        for epoch in range(local_epoch):
            sim_loss = train_sim(
                model,
                trainloader,
                msg.content["config"]["lr"],
                device,
                f"{missing_config}, ave f-micro: {ave_fmic}, min f-micro: {min_fmic} ",
            )

    else:
        for epoch in range(local_epoch):
            reg_loss = train_task(
                model,
                trainloader,
                msg.content["config"]["lr"],
                device,
                f"{missing_config}, ave f-micro: {ave_fmic}, min f-micro: {min_fmic} ",
            )
            
            

    # Construct and return reply Message
    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "train_loss": reg_loss,
        "num-examples": len(trainloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""

    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data with missing config from server
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    missing_config = msg.content["config"]["missing-config"]  # No default, comes from server
         
    seed: int = msg.content["config"]["seed"]
    _, valloader = load_data(seed, partition_id, num_partitions, missing_config)

    # Call the evaluation function
    eval_loss, eval_mae = test_fn(
        model,
        valloader,
        device,
    )

    # Construct and return reply Message
    metrics = {
        "eval_loss": eval_loss,
        "eval_mae": eval_mae,
        "num-examples": len(valloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
