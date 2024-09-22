import torch
import torch.nn as nn
import torch.nn.functional as F
import pefile
import argparse
import os

class MalConvPlus(nn.Module):
    def __init__(self, embed_dim, max_len, out_channels, window_size, dropout=0.5):
        super(MalConvPlus, self).__init__()
        self.tok_embed = nn.Embedding(257, embed_dim)
        self.pos_embed = nn.Embedding(max_len, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.conv = nn.Conv1d(
            in_channels=embed_dim,
            out_channels=out_channels * 2,
            kernel_size=window_size,
            stride=window_size,
        )
        self.fc = nn.Linear(out_channels, 1)

    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1)
        tok_embedding = self.tok_embed(x)
        pos = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1).to(x.device)
        pos_embedding = self.pos_embed(pos)
        embedding = self.dropout(tok_embedding + pos_embedding)
        conv_in = embedding.permute(0, 2, 1)
        conv_out = self.conv(conv_in)
        glu_out = F.glu(conv_out, dim=1)
        values, _ = glu_out.max(dim=-1)
        output = self.fc(values).squeeze(1)
        return output

def load_model(model_class, file_path, device='cpu'):
    model = model_class(embed_dim=8, max_len=4096, out_channels=128, window_size=32, dropout=0.5)  # Tạo lại mô hình với cấu trúc giống hệt
    model.load_state_dict(torch.load(file_path, map_location=device))  # Tải trạng thái mô hình đã lưu
    model.to(device)
    model.eval()  # Chuyển mô hình sang chế độ đánh giá
    return model

def extract_features_from_pe(file_path, max_len=4096):
    try:
        file = pefile.PE(file_path)
        header = list(file.header)
        if len(header) > max_len:
            header = header[:max_len]
        else:
            header += [0] * (max_len - len(header))
        header_tensor = torch.tensor(header, dtype=torch.long)
        header_tensor = header_tensor.unsqueeze(0)
        return header_tensor
    except pefile.PEFormatError:
        print(f"Skipping {file_path}, not a valid PE file.")
        return None

def predict_pe_file(model, file_path, max_len, device='cpu'):
    input_tensor = extract_features_from_pe(file_path, max_len)
    if input_tensor is None:
        return None, None
    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        output = model(input_tensor)
        prediction = torch.sigmoid(output).item()
    label = 'malware' if prediction > 0.5 else 'benign'
    return prediction, label

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    model = load_model(MalConvPlus, args.model_path, device=args.device)
    for file_name in os.listdir(args.input_dir):
        file_path = os.path.join(args.input_dir, file_name)
        prediction, label = predict_pe_file(model, file_path, args.max_len, device=args.device)
        if prediction is not None:
            print(f"File: {file_name}, Prediction: {prediction:.4f}, Label: {label}")
        else:
            print(f"Skipping {file_path}, not a valid PE file.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Directory of input PE files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the output results.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the saved model (.pt file).")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run the model on.")
    parser.add_argument("--max_len", type=int, default=4096, help="Maximum length of the input sequence.")
    args = parser.parse_args()
    
    main(args)

