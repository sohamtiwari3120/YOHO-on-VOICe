from sklearn.ensemble import VotingClassifier
import torch
from config import add_EAP_to_path
from torchsummary import summary
add_EAP_to_path()
from model.attention.CoAtNet import CoAtNet as CN
from models.VOICeCoAtNet import VOICeCoAtNet

def main():

    model = VOICeCoAtNet()
    print(model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summary(model.to(device), (1, 257, 40))

if __name__ == '__main__':
    main()