from utils.argparser import parse_run_arguments

if __name__ == '__main__':
    func, kwargs = parse_run_arguments()
    func(**kwargs)
