
from abc import ABC, abstractmethod
import torch

class Loss(ABC):
    @abstractmethod
    def get_loss(self):
        pass

class L1Loss(Loss):
    def __init__(self, target, prediction, mean):
        self.mean = mean
        self.target = target
        self.prediction = prediction
    def get_loss(self):
        loss = (self.target - self.prediction).abs()
        if self.mean:
            loss = loss.mean()

        return loss



class L2Loss(Loss):
    def get_loss(target, prediction, mean=True):
        return torch.nn.functional.mse_loss(target, prediction).mean()

    

