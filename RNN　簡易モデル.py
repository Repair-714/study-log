import torch
import torch.nn as nn

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        
        # 入力xと前の隠れ状態hを混ぜるための線形層
        # [h_t = tanh(W_ih * x_t + W_hh * h_{t-1} + b)] に相当
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        
        # 隠れ状態から最終的な出力を出す層
        self.h2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        # 入力と隠れ状態を結合（concatenate）
        combined = torch.cat((input, hidden), 1)
        
        # 次の隠れ状態を計算
        hidden = torch.tanh(self.i2h(combined))
        
        # 出力を計算
        output = self.h2o(hidden)
        output = self.softmax(output)
        
        return output, hidden

    def initHidden(self):
        # 最初の時刻t=0で使う隠れ状態（最初はゼロベクトル）
        return torch.zeros(1, self.hidden_size)


if __name__ == "__main__":
    # このファイルを直接実行したときの動作確認用
    input_size = 5
    hidden_size = 4
    output_size = 3

    model = SimpleRNN(input_size, hidden_size, output_size)
    hidden = model.initHidden()

    # ランダムな入力データを1ステップ分だけ渡す
    sample_input = torch.randn(1, input_size)
    output, next_hidden = model(sample_input, hidden)

    print("sample_input shape:", sample_input.shape)
    print("output shape:", output.shape)
    print("next_hidden shape:", next_hidden.shape)
    print("output:", output)
