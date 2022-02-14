from utils.data_utils import read_annotation
from utils.types import file_paths_type, event_type
from config import snr
envs = ['vehicle', 'outdoor', 'indoor']
data_mode = ['training', 'test', 'validation']

file_paths: file_paths_type = {}
events: event_type = {}

for mode in data_mode:
    file_paths[mode] = {}
    events[mode] = {}
    for e in envs:
        file_paths[mode][e] = [f"./{snr}/" + p[0]
                               for p in read_annotation(f"./{snr}/{e}_source_{mode}.txt")]
        file_paths[mode][f'mono_{e}'] = [
            p.replace(f"{snr}", f"{snr}-mono") for p in file_paths[mode][e]]
            
        events[mode][e] = []
        for audio in file_paths[mode][e]:
            path = audio.replace("wav", "txt")
            tmp = [[audio] + ann for ann in read_annotation(path)]
            events[mode][e] += tmp
