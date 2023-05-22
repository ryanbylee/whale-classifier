import os
import pandas as pd
import numpy as np
import constants
from data.StartingDataset import StartingDataset
from networks.StartingNetwork import StartingNetwork
from train_functions.starting_train import starting_train
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import torchvision
import torchvision.transforms as transforms


def oneHotEncode_labels(labels):
    ogValues = np.array(labels)
    label_encoder = LabelEncoder()
    int_encoded_values = label_encoder.fit_transform(ogValues)
    int_encoded_values = int_encoded_values.reshape(len(int_encoded_values), 1)
    oneHot_encoder = OneHotEncoder(sparse=False)

    oneHotEncoded_values = oneHot_encoder.fit_transform(int_encoded_values)

    return oneHotEncoded_values, label_encoder

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

    #train_loader = StartingDataset("./humpback-whale-identification/train.csv", "./humpback-whale-identification/train/")
    #print(train_loader[3])

    train_df = pd.read_csv("./humpback-whale-identification/train.csv")
    print(train_df.head)
    print(train_df.Id.value_counts().head())
    
    data_transforms = transforms.Compose([
       transforms.Resize((100, 100)),
       transforms.ToTensor(),
       transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])


    oneHot_values, le = oneHotEncode_labels(train_df["Id"])

    train_dataset = StartingDataset(img_directory="./humpback-whale-identification/train/", dataMode="train", df = train_df, transform=data_transforms, labels=oneHot_values)

    print(train_dataset[3])



if __name__ == "__main__":
  main()
