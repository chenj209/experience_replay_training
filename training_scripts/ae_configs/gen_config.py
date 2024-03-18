import sys
import re
import json

def gen_config(config_name):
    pattern = "channel_(.*)_latent(\d+)_sample(\d+)_lr(.*)\.json"
    m = re.match(pattern, config_name)
    if m is None:
        print("Invalid config name:", config_name)
    else:
        print("Matched config name:", config_name)
        channel_size = m.group(1).split('_')
        channel_size = list(map(int, channel_size))
        latent = int(m.group(2))
        sample = int(m.group(3))
        lr = float(m.group(4))
        config = {
            "input_size": [309,32,56],
            "channel_sizes": channel_size,
            "latent_size": latent,
            "kernel_size": 3,
            "stride": [1]*len(channel_size),
            "padding": [1]*len(channel_size),
            "output_padding": [0]*len(channel_size),
            "sample_rate": sample,
            "lr": lr
        }
        if latent < 10:
            config["fc_sizes"] = [latent*32*56]*2
            config["num_em_layers"] = 0
        else:
            config["fc_sizes"] = []
            config["num_em_layers"] = 2
        with open(config_name, 'w') as f:
            f.write(json.dumps(config, indent=4))

if __name__ == "__main__":
    config_name = sys.argv[1]
    gen_config(config_name)