import argparse
from loguru import logger
import os
from config import hparams
from utils.torch_utils import generate_save_predictions
from utils.evaluate_utils import compute_sed_f1_errorrate
from utils.data_utils import data_modes
from utils.pl_utils import LM

hp = hparams()


@logger.catch
def evaluate(args):
    source_env = args.source_env
    target_env = args.target_env
    data_mode = args.data_mode
    expt_folder = args.expt_folder
    if not os.path.exists(expt_folder):
        raise Exception(f'Folder not found: {expt_folder}')
    logger.info(
        f'Starting evalution on {data_mode} data for source - {source_env} and target - {target_env}.')
    logger.add(os.path.join(expt_folder,
                            f'{args.backend}_eval_src_{source_env}_target_{target_env}.log'), rotation='500 KB')
    logger.info(f'Loading best f1 model for {source_env} audio.')
    model_ckpt_folder_path = expt_folder
    chkpt_path = os.path.join(model_ckpt_folder_path,
                              f"model-{source_env}-best-f1.ckpt")
    if not os.path.exists:
        raise Exception(f'Model checkpoint not found: {chkpt_path}.')
    model = LM.load_from_checkpoint(chkpt_path)
    logger.info(f'Loaded model checkpoint: {chkpt_path}')
    reference_files, estimated_files = generate_save_predictions(
        model, data_mode, target_env)
    curr_f1, curr_error = compute_sed_f1_errorrate(
        reference_files, estimated_files)

    logger.success("F-measure: {:.3f}".format(curr_f1))
    logger.success("Error rate: {:.3f}".format(curr_error))

    # Or print all metrics as reports
    return (curr_f1, curr_error)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='For making realtime predictons.')
    parser.add_argument('-ef', '--expt_folder', type=str, required=True)
    parser.add_argument('-b', '--backend', type=str,
                        default=hp.backend, choices=hp.backends)
    parser.add_argument('-se', '--source_env', type=str, default=hp.source_env)
    parser.add_argument('-te', '--target_env', type=str, default=hp.target_env)
    parser.add_argument('-m', '--data_mode', type=str,
                        default=hp.data_mode, choices=data_modes)

    args = parser.parse_args()
    f1, error = evaluate(args)
