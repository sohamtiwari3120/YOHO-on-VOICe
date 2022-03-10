from email.mime import audio
from loguru import logger
from config import snr
from utils.data_utils import process_audio_file
from utils.torch_utils import convert_model_preds_to_soundevents, merge_sound_events
import argparse
logger.add('process_audiofile.log', rotation='500 KB')


@logger.catch
def main(args):
    audio_path = args.audio_path
    logger.info(f'Starting processing of for audiofile {audio_path}')
    audio_wins, window_ranges, all_anns, all_model_compatible_anns = process_audio_file(audio_path)
    for i, w in enumerate(window_ranges):
        print(audio_wins[i])
        print(w)
        print(all_anns[i])
        print(all_model_compatible_anns[i])
        se = convert_model_preds_to_soundevents([all_model_compatible_anns[i]], win_ranges=w)
        print(se)
        print(merge_sound_events(se))
    logger.info(f'Finished processing of {audio_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='For processing individual audiofiles')
    parser.add_argument('-a', '--audio_path', type=str)

    args = parser.parse_args()
    main(args)