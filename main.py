import os

import constants
from data.StartingDataset import StartingDataset
from networks.StartingNetwork import StartingNetwork
from train_functions.starting_train import starting_train


def main():
    # Get command line arguments
    # hyperparameters = {"epochs": constants.EPOCHS, "batch_size": constants.BATCH_SIZE}

    # # : Add GPU support. This line of code might be helpful.
    # # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # print("Epochs:", constants.EPOCHS)
    # print("Batch size:", constants.BATCH_SIZE)

    # # Initalize dataset and model. Then train the model!
    # train_dataset = StartingDataset()
    # val_dataset = StartingDataset()
    # model = StartingNetwork()
    # starting_train(
    #     train_dataset=train_dataset,
    #     val_dataset=val_dataset,
    #     model=model,
    #     hyperparameters=hyperparameters,
    #     n_eval=constants.N_EVAL,
    # )

    train_loader = StartingDataset("./humpback-whale-identification/train.csv", "./humpback-whale-identification/train/")
    print(train_loader[3])
    pass

if __name__ == "__main__":
  main()
