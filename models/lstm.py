import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, src, pred_len):
        # x: (batch_size, seq_len, input_dim)
        # outputs = []
        # batch_size, _, input_dim = x.shape
        # out, (hidden, cell) = self.lstm(x)  # LSTM forward
        # last_output = out[:, -1, :]  # Last timestep's hidden state
        # prediction = self.fc(last_output)  # Initial prediction
        # outputs.append(prediction)
        
        # # Autoregressive multi-step forecasting
        # for _ in range(pred_len - 1):
        #     prediction = prediction.unsqueeze(1)  # Shape: (batch_size, 1, output_dim)
        #     out, (hidden, cell) = self.lstm(prediction, (hidden, cell))
        #     prediction = self.fc(out[:, -1, :])  # Predict next step
        #     outputs.append(prediction)
        
        # return torch.stack(outputs, dim=1)  # (batch_size, pred_len, output_dim)

        batch_size, seq_len, num_nodes, input_dim = src.shape
        predictions = []

        for node in range(num_nodes):
            node_src = src[:, :, node, :]  # (batch_size, seq_len, input_dim)
            out, (hidden, cell) = self.lstm(node_src)  # (batch_size, seq_len, hidden_dim)
            last_hidden = hidden[-1]  # (batch_size, hidden_dim)
            node_predictions = []

            # Autoregressive prediction
            for _ in range(pred_len):
                out_step = self.fc(last_hidden)  # (batch_size, output_dim)
                node_predictions.append(out_step)
                out_step = out_step.unsqueeze(1)  # (batch_size, 1, output_dim)
                _, (hidden, cell) = self.lstm(out_step, (hidden, cell))

            node_predictions = torch.stack(node_predictions, dim=1)  # (batch_size, pred_len, output_dim)
            predictions.append(node_predictions)

        predictions = torch.stack(predictions, dim=2)  # (batch_size, pred_len, num_nodes, output_dim)
        return predictions
