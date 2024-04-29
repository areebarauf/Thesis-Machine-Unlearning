# Evaluating the perfomace of impairing and repairing of unlearning model in this file
import torch
from training.model import evaluate
from impair_repair_unlearn import ImpairRepair
from torch.utils.data import DataLoader


if __name__ == '__main__':
    num_classes = 10
    # Load the saved model from the .pt file
    model = torch.load("/media/homes/areeba/thesis_projects/FYEMU/training/ResNET18_CIFAR10_ALL_CLASSES.pt")
    batch_size = 256
    # retain validation set
    retain_valid = []
    for cls in range(num_classes):
        if cls not in classes_to_forget:
            for img, label in classwise_test[cls]:
                retain_valid.append((img, label))

    # forget validation set
    forget_valid = []
    for cls in range(num_classes):
        if cls in classes_to_forget:
            for img, label in classwise_test[cls]:
                forget_valid.append((img, label))

    forget_valid_dl = DataLoader(forget_valid, batch_size, num_workers=3, pin_memory=True)
    retain_valid_dl = DataLoader(retain_valid, batch_size*2, num_workers=3, pin_memory=True)


    # Initialize an instance of ImpairRepair with your model and other parameters
    impair_repair = ImpairRepair(model, classes_to_forget, classes_to_retain, batch_size)

    # Impair the model
    impair_repair.impair()

    # Evaluate the performance of the impaired model on forget class
    print("Performance of Impaired Model on Forget Class")
    history_forget = [evaluate(model, forget_valid_dl)]
    print("Accuracy: {}".format(history_forget[0]["Acc"] * 100))
    print("Loss: {}".format(history_forget[0]["Loss"]))

    # Evaluate the performance of the impaired model on retain class
    print("Performance of Impaired Model on Retain Class")
    history_retain = [evaluate(model, retain_valid_dl)]
    print("Accuracy: {}".format(history_retain[0]["Acc"] * 100))
    print("Loss: {}".format(history_retain[0]["Loss"]))

    # Repair the model
    impair_repair.repair()

    # Evaluate the performance of the repaired model on forget class
    print("Performance of Repaired Model on Forget Class")
    history_forget_repaired = [evaluate(model, forget_valid_dl)]
    print("Accuracy: {}".format(history_forget_repaired[0]["Acc"] * 100))
    print("Loss: {}".format(history_forget_repaired[0]["Loss"]))

    # Evaluate the performance of the repaired model on retain class
    print("Performance of Repaired Model on Retain Class")
    history_retain_repaired = [evaluate(model, retain_valid_dl)]
    print("Accuracy: {}".format(history_retain_repaired[0]["Acc"] * 100))
    print("Loss: {}".format(history_retain_repaired[0]["Loss"]))
