import numpy as np
import pandas as pd

class WindowGenerator:
    def __init__(self, input_width=48, label_width=1):
        self.input_width = input_width   # Look back 48 hours
        self.label_width = label_width   # Predict next 1 hour

    def split_windows(self, data):
        """
        Converts a long series into (X, y) pairs.
        X: (Samples, Window_Size)
        y: (Samples, Label_Size)
        """
        X, y = [], []
        for i in range(len(data) - self.input_width - self.label_width + 1):
            X.append(data[i : i + self.input_width])
            y.append(data[i + self.input_width : i + self.input_width + self.label_width])
            
        return np.array(X), np.array(y)