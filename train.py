from utils.argparser import parse_train_arguments

if __name__ == '__main__':
    func, kwargs = parse_train_arguments()
    func(**kwargs)
