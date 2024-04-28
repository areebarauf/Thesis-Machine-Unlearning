import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import numpy as np
import logging 

# Defining the noise structure
class Noise(nn.Module):
    def __init__(self, *dim):
        super().__init__()
        self.noise = torch.nn.Parameter(torch.randn(*dim), requires_grad=True)

    def forward(self):
        return self.noise

class ImpairRepair:
    def __init__(self, model, classes_to_forget, classes_to_retain, batch_size=32):
        self.model = model
        self.classes_to_forget = classes_to_forget
        self.classes_to_retain = classes_to_retain
        self.batch_size = batch_size
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.02)
        self.noises = {}

    def generate_noise_for_forget_classes(self):
        try:
            for cls in self.classes_to_forget:
                logging.info("Optimizing loss for forget class {}".format(cls))
                self.noises[cls] = Noise(self.batch_size, 3, 32, 32).cuda()
                opt = torch.optim.Adam(self.noises[cls].parameters(), lr=0.1)

                num_epochs = 5
                num_steps = 8
                class_label = cls
                for epoch in range(num_epochs):
                    total_loss = []
                    for batch in range(num_steps):
                        inputs = self.noises[cls]()
                        labels = torch.zeros(self.batch_size).cuda() + class_label
                        outputs = self.model(inputs)
                        loss = -F.cross_entropy(outputs, labels.long()) + 0.1 * torch.mean(torch.sum(torch.square(inputs), [1, 2, 3]))
                        opt.zero_grad()
                        loss.backward()
                        opt.step()
                        total_loss.append(loss.cpu().detach().numpy())
                    logging.info("Loss: {}".format(np.mean(total_loss)))
        except Exception as e:
            logging.error(e)

    def generate_noise_for_retain_classes(self):
        try:
            for cls in self.classes_to_retain:
                logging.info("Optimizing loss for retain class {}".format(cls))
                self.noises[cls] = Noise(self.batch_size, 3, 32, 32).cuda()
                opt = torch.optim.Adam(self.noises[cls].parameters(), lr=0.1)

                num_epochs = 10
                num_steps = 8
                class_label = cls
                for epoch in range(num_epochs):
                    total_loss = []
                    for batch in range(num_steps):
                        inputs = self.noises[cls]()
                        labels = torch.zeros(self.batch_size).cuda() + class_label
                        outputs = self.model(inputs)
                        loss = -F.cross_entropy(outputs, labels.long()) + 0.1 * torch.mean(torch.sum(torch.square(inputs), [1, 2, 3]))
                        opt.zero_grad()
                        loss.backward()
                        opt.step()
                        total_loss.append(loss.cpu().detach().numpy())
                    logging.info("Loss: {}".format(np.mean(total_loss)))
        except Exception as e:
            logging.error(e)

    def impair(self):
        try:
            self.model.train(True)
            running_loss = 0.0
            running_acc = 0
            for cls in self.classes_to_forget + self.classes_to_retain:
                noisy_data = []
                num_batches = 20
                for _ in range(num_batches):
                    batch = self.noises[cls]().cpu().detach()
                    for i in range(batch[0].size(0)):
                        noisy_data.append((batch[i], torch.tensor(cls)))
                noisy_loader = torch.utils.data.DataLoader(noisy_data, batch_size=self.batch_size, shuffle=True)
                for i, data in enumerate(noisy_loader):
                    inputs, labels = data
                    inputs, labels = inputs.cuda(), labels.cuda()
                    self.optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = F.cross_entropy(outputs, labels)
                    loss.backward()
                    self.optimizer.step()
                    running_loss += loss.item() * inputs.size(0)
                    out = torch.argmax(outputs.detach(), dim=1)
                    assert out.shape == labels.shape
                    running_acc += (labels == out).sum().item()
            logging.info(f"Impairment loss: {running_loss / (len(self.classes_to_forget) + len(self.classes_to_retain) * 20 * self.batch_size)}, Impairment Acc: {running_acc * 100 / (len(self.classes_to_forget) + len(self.classes_to_retain) * 20 * self.batch_size)}%")
        except Exception as e:
            logging.error(e)

    def repair(self):
        try:
            self.model.train(True)
            running_loss = 0.0
            running_acc = 0
            for cls in self.classes_to_retain:
                noisy_repair_data = []
                num_batches = 20
                for _ in range(num_batches):
                    batch = self.noises[cls]().cpu().detach()
                    for i in range(batch[0].size(0)):
                        noisy_repair_data.append((batch[i], torch.tensor(cls)))
                noisy_repair_loader = torch.utils.data.DataLoader(noisy_repair_data, batch_size=self.batch_size, shuffle=True)
                for i, data in enumerate(noisy_repair_loader):
                    inputs, labels = data
                    inputs, labels = inputs.cuda(), labels.cuda()
                    self.optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = F.cross_entropy(outputs, labels)
                    loss.backward()
                    self.optimizer.step()
                    running_loss += loss.item() * inputs.size(0)
                    out = torch.argmax(outputs.detach(), dim=1)
                    assert out.shape == labels.shape
                    running_acc += (labels == out).sum().item()
            logging.info(f"Repair loss: {running_loss / (len(self.classes_to_retain) * 20 * self.batch_size)}, Repair Acc: {running_acc * 100 / (len(self.classes_to_retain) * 20 * self.batch_size)}%")
        except Exception as e:
            logging.error(e)
