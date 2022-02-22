import argparse
from models.YOHO import YohoModel
from utils.data_utils import VOICeDataModule
from pytorch_lightning import Trainer
from utils.torch_utils import MonitorSedF1Callback


def main(args):
    env = args.env
    model = YohoModel()
    trainer = Trainer(callbacks=[MonitorSedF1Callback(env)])
    indoor_voice_dm = VOICeDataModule(env)
    trainer.fit(model, indoor_voice_dm)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='For making realtime predictons.')
    parser.add_argument('-e', '--env', type=str, default='indoor')

    args = parser.parse_args()
    main(args)
