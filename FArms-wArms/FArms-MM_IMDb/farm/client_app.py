"""Federated Cross-Modal Simulation: Client App"""

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from farm.task import Net, load_data
from farm.task import test as test_fn
from farm.task import train as train_fn

# Flower ClientApp
app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""
    
    # Get device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Load the model and initialize it with the received weights
    model = Net(device=device)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    model.to(device)
    
    # Load the data with missing config from server
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    missing_config = msg.content["config"]["missing-config"]
    
    trainloader, _ = load_data(partition_id, num_partitions, missing_config)
    
    # Get training hyperparameters
    local_epochs = context.run_config["local-epochs"]
    lr = msg.content["config"]["lr"]
    alpha = msg.content["config"]["alpha"]
    beta = msg.content["config"]["beta"]
    
    # Call the training function
    train_loss = train_fn(
        model,
        trainloader,
        local_epochs,
        lr,
        alpha,
        beta,
        device,
    )
    
    # Construct and return reply Message
    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""
    
    # Get device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Load the model and initialize it with the received weights
    model = Net(device=device)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    model.to(device)
    
    # Load the data with missing config from server
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    missing_config = msg.content["config"]["missing-config"]
    
    _, valloader = load_data(partition_id, num_partitions, missing_config)
    
    # Call the evaluation function
    eval_f1_micro, eval_f1_macro = test_fn(
        model,
        valloader,
        device,
    )
    
    # Construct and return reply Message
    metrics = {
        "eval_f1_micro": eval_f1_micro,
        "eval_f1_macro": eval_f1_macro,
        "num-examples": len(valloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
