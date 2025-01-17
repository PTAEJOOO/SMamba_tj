# import torch
# import torch.nn as nn
# from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# class VariableLengthRNN(nn.Module):
#     def __init__(self, input_dim, hidden_dim, num_layers=1):
#         super(VariableLengthRNN, self).__init__()
#         self.hidden_dim = hidden_dim
#         self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
    
#     def forward(self, x, lengths):
#         # Pack the padded sequence to handle variable-length sequences
#         packed_x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
#         packed_out, hidden = self.gru(packed_x)  # RNN output
#         # Unpack the sequence
#         out, _ = pad_packed_sequence(packed_out, batch_first=True)
#         return out, hidden

# # Example usage
# batch_size = 4
# num_channels = 3  # Number of features per time step
# hidden_dim = 64

# # Simulate variable-length sequences
# sequence_lengths1 = [7, 5, 6, 4]  # Different lengths for each sequence in the batch
# max_seq_len = max(sequence_lengths1)
# input_data_1 = torch.randn(batch_size, max_seq_len, num_channels)  # Shape: (batch, seq_len, channels)

# sequence_lengths2 = [7, 9, 6, 4]  # Different lengths for each sequence in the batch
# max_seq_len2 = max(sequence_lengths2)
# input_data_2 = torch.randn(batch_size, max_seq_len2, num_channels)  # Shape: (batch, seq_len, channels)

# # # Instantiate the model
# model = VariableLengthRNN(input_dim=num_channels, hidden_dim=hidden_dim)

# # # Forward pass 1
# output1, hidden1 = model(input_data_1, sequence_lengths1)

# print("Output shape:", output1.shape)  # Output of RNN for each timestep
# print("Hidden shape:", hidden1.shape)  # Final hidden state for each sequence

# # # Forward pass 2
# output2, hidden2 = model(input_data_2, sequence_lengths2)

# print("Output shape:", output2.shape)  # Output of RNN for each timestep
# print("Hidden shape:", hidden2.shape)  # Final hidden state for each sequence

# print(output2[:,-1,:])
# print(hidden2[:,-1,:])

import torch
import torch.nn as nn

# RNN 정의
rnn = nn.RNN(input_size=2, hidden_size=8, batch_first=True)
# 입력 데이터 (batch_size=32, seq_len=10, input_size=8)
x = torch.randn(3, 10, 2)
# RNN forward pass
output, hidden = rnn(x)
# output: (batch_size, seq_len, hidden_size)
# hidden: (1, batch_size, hidden_size)

# 방법 1: output에서 마지막 시간 스텝의 출력 가져오기
last_output = output[:, -1, :]  # (batch_size, hidden_size)
# 방법 2: hidden 값 사용하기 (hidden은 마지막 시간 스텝의 상태)
last_output_hidden = hidden.squeeze(0)  # (batch_size, hidden_size)
print(last_output)
print(last_output_hidden)


# lstm = nn.LSTM(input_size=8, hidden_size=16, batch_first=True)
# output, (hidden, cell) = lstm(x)
# # 마지막 출력
# last_output = output[:, -1, :]  # (batch_size, hidden_size)
# last_output_hidden = hidden.squeeze(0)  # (batch_size, hidden_size)