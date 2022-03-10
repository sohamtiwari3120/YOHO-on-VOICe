from email.mime import audio
from loguru import logger
from config import snr
from utils.data_utils import process_audio_file
import argparse
logger.add('process_audiofile.log', rotation='500 KB')


@logger.catch
def main(args):
    audio_path = args.audio_path
    logger.info(f'Starting processing of for audiofile {audio_path}')
    process_audio_file(audio_path)
    logger.info(f'Finished processing of {audio_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='For processing individual audiofiles')
    parser.add_argument('-a', '--audio_path', type=str)

    args = parser.parse_args()
    main(args)