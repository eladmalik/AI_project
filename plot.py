import os.path

from utils.argparser import parse_plot_arguments
from utils.general_utils import get_plot_output_folder
import json


def main():
    plot_type, run_function, kwargs = parse_plot_arguments()
    save_folder = get_plot_output_folder(plot_type)
    if plot_type == "normal" or plot_type == "genetic":
        with open(os.path.join(save_folder, "plotted.json"), "w") as file:
            json.dump(kwargs["folder_path"], file)
    else:
        with open(os.path.join(save_folder, "plotted.json"), "w") as file:
            json.dump(kwargs["folders"], file)
    kwargs["save_folder"] = save_folder
    run_function(**kwargs)


if __name__ == '__main__':
    main()
