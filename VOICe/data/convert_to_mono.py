import os
from subprocess import Popen, PIPE
from config import snr
from data.data_paths import file_paths, data_mode, envs

def main():
    base_dir = os.path.dirname(__file__)
    os.makedirs(os.path.join(base_dir, f'{snr}-mono'), exist_ok=True)

    training_files = [ele for sublist in list(file_paths['training'].values()) for ele in sublist ]
    test_files = [ele for sublist in list(file_paths['test'].values()) for ele in sublist ]
    validation_files = [ele for sublist in list(file_paths['validation'].values()) for ele in sublist ]

    # conversion to mono
    for sound in training_files+test_files+validation_files:
      temp_file = sound.replace(f"{snr}", f"{snr}-mono")
      command = command = "sox " + sound + " " + temp_file + " channels 1"
      p = Popen(command, stdin=PIPE, stdout=PIPE, stderr=PIPE, shell=True)
      output, err = p.communicate()

    # adding subfolders in place
    for mode in data_mode:
        for e in envs:
            os.makedirs(os.path.join(base_dir, f'{snr}-mono', f"{mode}-data", f"{e}"), exist_ok=True)

if __name__ == '__main__':
    main()