import torch
from loguru import logger
from torchsummary import summary
from models.YOHO import VOICeConvNeXt
from config import hparams
hp = hparams()

@logger.catch
def run():
    model = VOICeConvNeXt()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summary(model.to(device), (3, hp.input_height, hp.input_width))

if __name__=='__main__':
    run()