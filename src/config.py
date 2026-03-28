import yaml
import argparse


class Config():
    """
    Converts a dict object into something similar to argparse.Namespace.
    Also provides a human readable repr similar to the yaml syntax.
    """
    def __init__(self, dict_like):
        input_type = type(dict_like)
        if input_type != dict:
            config_dict = vars(dict_like)
        else:
            config_dict = dict_like
        
        for k, v in config_dict.items():
            if isinstance(v, input_type):
                v = Config(v)
            setattr(self, k, v)
    
    @staticmethod
    def add_indent(string: str):
        lines = string.split('\n')
        indented_lines = ['  ' + line for line in lines]
        indented_string = '\n'.join(indented_lines)
        return indented_string

    def __repr__(self):
        item_reprs = []
        for k, v in self.__dict__.items():
            if isinstance(v, Config):
                item_repr = f"{k}:\n{self.add_indent(repr(v))}"
            else:
                item_repr = f"{k}: {v}"
            item_reprs.append(item_repr)
        return '\n'.join(item_reprs)


def get_default_config(config_path) -> dict:
    """Loads the YAML configuration file"""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def process_arguments(config_dict: dict, args_dict: dict, is_root: bool=True) -> dict:
    """Updates the config with CLI arguments"""
    if args_dict:
        for k, v in config_dict.items():
            if isinstance(v, dict):
                process_arguments(v, args_dict, False)
            elif k in args_dict:
                config_dict[k] = args_dict[k]
                args_dict.pop(k)
    if "config_file" in args_dict:
        args_dict.pop("config_file")
    if is_root and args_dict:
        config_dict["extra"] = {}
        for k, v in args_dict.items():
            config_dict["extra"][k] = v
    return config_dict


def args_to_dict(args: argparse.Namespace) -> dict:
    args_dict = {}
    for k, v in vars(args).items():
        args_dict[k] = v
    return args_dict


def get_config(args: argparse.Namespace):
    config_dict = get_default_config(args.config_file)
    config_dict = process_arguments(config_dict, args_to_dict(args))
    return Config(config_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', default="./config.yaml", type=str, help='Path to the default config file.')
    parser.add_argument('--res', type=int, default=224)
    parser.add_argument('--guidance_scale', type=float, default=2.0)
    
    args = parser.parse_args()
    config = get_config(args)
    print(config)
    print(config.training.lr)
