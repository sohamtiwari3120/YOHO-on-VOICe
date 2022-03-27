import torch
from torchsummary import summary
from config import hparams
from models.pann_encoder import Cnn10

hp = hparams()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Cnn10()
    summary(model.to(device), (1, hp.input_height, hp.input_width))

if __name__ == '__main__':
    main()