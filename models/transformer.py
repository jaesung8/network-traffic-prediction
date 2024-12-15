import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers, output_dim, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.transformer = nn.Transformer(
            d_model=hidden_dim, 
            nhead=num_heads, 
            num_encoder_layers=num_layers, 
            num_decoder_layers=num_layers, 
            dropout=dropout
        )
        self.output_projection = nn.Linear(hidden_dim, output_dim)
        self.hidden_dim = hidden_dim

    # def forward(self, src, tgt):
    #     # src: (seq_len, batch_size, input_dim)
    #     # tgt: (seq_len, batch_size, input_dim)
    #     src = self.input_projection(src)
    #     tgt = self.input_projection(tgt)
    #     output = self.transformer(src, tgt)
    #     output = self.output_projection(output)
    #     return output

    def forward(self, src, pred_len):
        """
        src: (batch_size, seq_len , input_dim) - Encoder input
        pred_len: int - Number of steps to predict
        """
        # src = self.input_projection(src)  # Project input to hidden_dim
        # # print(src.size())
        # src = src.permute(1, 0, 2)  # (seq_len, batch_size, hidden_dim)
        # memory = self.transformer.encoder(src)

        # # Initialize decoder input with the last step of src
        # tgt = src[-1:, :, :]  # (1, batch_size, hidden_dim)

        # predictions = []
        # for _ in range(pred_len):
        #     output = self.transformer.decoder(tgt, memory)  # (tgt_len, batch_size, hidden_dim)
        #     output_step = self.output_projection(output[-1])  # Last timestep output
        #     predictions.append(output_step)

        #     # Update tgt for next step
        #     output_step = output_step.unsqueeze(0)  # (1, batch_size, output_dim)
        #     # tgt = torch.cat([tgt, output_step], dim=0)
        #     tgt = output

        # predictions = torch.stack(predictions, dim=0)  # (pred_len, batch_size, output_dim)
        # # print(predictions.size())
        # return predictions.permute(1, 0, 2)  # (batch_size, pred_len, output_dim)


        batch_size, seq_len, num_nodes, input_dim = src.shape
        predictions = []
        for node in range(num_nodes):
            node_src = src[:, :, node, :]  # (batch_size, seq_len, input_dim)
            node_src = self.input_projection(node_src)  # (batch_size, seq_len, hidden_dim)
            node_src = node_src.permute(1, 0, 2)  # (seq_len, batch_size, hidden_dim)
            memory = self.transformer.encoder(node_src)

            # Initialize decoder input with the last step of node_src
            tgt = node_src[-1:, :, :]  # (1, batch_size, hidden_dim)
            node_predictions = []

            for _ in range(pred_len):
                output = self.transformer.decoder(tgt, memory)  # (tgt_len, batch_size, hidden_dim)
                output_step = self.output_projection(output[-1])  # Last timestep output
                # print(output.size(), output_step.size())
                # output_step = output_step.un # Expand to (1, batch_size, output_dim)
                node_predictions.append(output_step)
                # print(tgt.size(), output_step.size())
                # Update tgt for next step
                tgt = output

            node_predictions = torch.stack(node_predictions, dim=0)  # (pred_len, batch_size, output_dim)
            predictions.append(node_predictions)
        
        predictions = torch.stack(predictions, dim=2)  # (pred_len, batch_size, num_nodes, output_dim)
        return predictions.permute(1, 0, 2, 3)