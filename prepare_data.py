import os
import numpy as np

def prepare_event_data(root):
    source_path = os.path.join(root, 'events_np')
    target_path = os.path.join(root, 'events')

    dir_list = [dir for dir in os.listdir(source_path) if os.path.isdir(os.path.join(source_path, dir))]
    for dir in dir_list:
        dir_path = os.path.join(source_path, dir)
        file_list = [f for f in os.listdir(dir_path) if f.endswith(".npz")]
        if not os.path.exists(os.path.join(target_path, dir)):
            os.mkdir(os.path.join(target_path, dir))
        for file in file_list:
            file_path = os.path.join(dir_path, file)
            data = np.load(file_path)
            t = data['t']  
            x = data['x']
            y = data['y']
            p = data['p']
            events = np.stack((t, x, y, p), axis=1)  # (seq_len, 4)
            np.save(os.path.join(target_path, dir, file[:-4]), events)
            

if __name__ == "__main__":
    root = "/home/haohq/datasets/elm/CIFAR10DVS"
    loaded_data = prepare_event_data(root)
    print('convert event data to frames finished')