from utils.argparser import parse_run_arguments
from utils.lot_generator import example_easy, example_medium, example2

if __name__ == '__main__':
    func, kwargs = parse_run_arguments()

    if not kwargs["load_folder"]:
        assert False, "Missing model"
    
    kwargs["n_simulations"] = 100
    
    print("-== example_easy ==-")
    kwargs["lot_generator"] = example_easy
    func(**kwargs)

    print("-== example_medium ==-")
    kwargs["lot_generator"] = example_medium
    func(**kwargs)

    print("-== example2 ==-")
    kwargs["lot_generator"] = example2
    func(**kwargs)