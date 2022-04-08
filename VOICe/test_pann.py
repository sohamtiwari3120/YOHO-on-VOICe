import torch
from torchsummary import summary
from config import hparams
from models.pann_encoder import Cnn14, VOICePANN

hp = hparams()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Cnn14()
    summary(model.to(device), (hp.input_height, 64))
    print(VOICePANN())

if __name__ == '__main__':
    main()