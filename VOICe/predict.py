import argparse
from utils.torch_utils import predict_audio_path
from utils.pl_utils import LM
from loguru import logger
from config import env

@logger.catch
def main(args):
    env = args.env
    audio_path = args.audio_path
    logger.add(f'predict_{env}.log', rotation='500 KB')
    logger.info(f'Starting prediction using model for {env} audio.')
    if args.chkpt_path is not None:
        model = LM.load_from_checkpoint(args.chkpt_path)
        logger.info(f'Loaded model checkpoint: {args.chkpt_path}')
    else:
        model = LM()
        logger.info(f'Using a fresh model.')
    preds = predict_audio_path(model, audio_path)
    logger.info(f'Predictions:')
    logger.info(preds)
    logger.info(f'Finished prediction of {audio_path} using model for {env} audio.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='For making realtime predictons.')
    parser.add_argument('-e', '--env', type=str, default=env)
    parser.add_argument('-cp', '--chkpt_path', type=str, default=None)
    parser.add_argument('-ap', '--audio_path', type=str)

    args = parser.parse_args()
    main(args)
