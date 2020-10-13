from gnn import *
from pathlib import Path
import json
from util import files_with_extension


def load_GNN1_drat(model_cfg_path, ckpt_path):
    with open(model_cfg_path, "r") as f:
        cfg = json.load(f)
    model = GNN1_drat(**cfg)

    if ckpt_path is not None:
        DEVICE = torch.device("cpu")
        if torch.cuda.is_available():
            print("GPU AVAILABLE")
        print("LOADING CKPT FROM", ckpt_path)
        ckpt = torch.load(ckpt_path, map_location = DEVICE)

        try:
            model.load_state_dict(ckpt["model_state_dict"], strict = False)
        except:
            model.load_state_dict(ckpt["models"][0], strict = False)
    else:
        print("WARNING: serializing randomly initialized network")

    model.eval()
    return model


def serialize_GNN1_drat(model, save_path):
    m = torch.jit.script(model)
    m.save(save_path)


def deploy_GNN1_drat(model_cfg_path, ckpt_path, save_path):
    serialize_GNN1_drat(load_GNN1_drat(model_cfg_path, ckpt_path), save_path)


def get_ckpt_from_index(ckpt_dir):
    index = files_with_extension(ckpt_dir, "index")[0]
    with open(index, "r") as f:
        cfg_dict = json.load(f)
    return cfg_dict["latest"]


def _parse_main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", dest = "cfg", action = "store")
    parser.add_argument("--ckpt", dest = "root_path", type = str, action = "store")
    # parser.add_argument("--dest", dest = "dest", type = str, action = "store")
    opts = parser.parse_args()
    return opts


def _main():
    opts = _parse_main()
    root_path = opts.root_path
    ckpt_path = get_ckpt_from_index(root_path)
    dest = '/'.join(ckpt_path.split('/')[:-1])
    rootname = str(Path(opts.cfg).stem)
    save_path_drat = os.path.join(dest, rootname + "_drat.pt")
    deploy_GNN1_drat(model_cfg_path = opts.cfg, ckpt_path = ckpt_path, save_path = save_path_drat)
    print(f"Saved model to {save_path_drat}")


if __name__ == "__main__":
    _main()
