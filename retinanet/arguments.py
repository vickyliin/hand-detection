import argparse
from argparse import ArgumentDefaultsHelpFormatter as DefaultsHelpFormatter

class ArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args, 
                 formatter_class=DefaultsHelpFormatter, 
                 add_help=False,
                 **kwargs):
        super().__init__(*args, 
                         formatter_class=formatter_class, 
                         add_help=add_help,
                         **kwargs)
    def add_argument(self, *args, help=' ', **kwargs):
        super().add_argument(*args, help=help, **kwargs)

    def parse_args(self, args):
        if type(args) == str:
            args = args.split()
        return super().parse_args(args)

class ParserGroup:
    def __init__(self, *parents, name):
        self.parents = parents
        self.parser = ArgumentParser(name, add_help=True, parents=parents)
    def parse(self, args):
        args_all = self.parser.parse_args(args)
        self.print_args(args_all)
        return [args_all] + self.parse_all(self.parents, args)
    @staticmethod
    def print_args(args):
        for key, val in vars(args).items():
            print(key.ljust(20), str(val)[:100])
    @staticmethod
    def parse_all(parsers, args):
        return [ap.parse_known_args(args)[0] for ap in parsers]

def GeneratorArgparser():
    parser = ArgumentParser('generator')
    parser.add_argument('--image-min-side', '--img-min-side', type=int, default=460)
    parser.add_argument('--image-max-side', '--img-max-side', type=int, default=1024)
    parser.add_argument('--batch-size', type=int, default=4)
    return parser

def CSVGeneratorArgparser(data_train=True):
    parser = ArgumentParser('CSVGenerator', parents=[GeneratorArgparser()])
    if data_train:
        parser.add_argument('--data-train', nargs='+', default=['data/DeepQ-Synth-Hand-01/data/annotations.csv', 'data/DeepQ-Synth-Hand-02/data/annotations.csv'])
        parser.add_argument('--data-valid', nargs='+', default=['data/DeepQ-Vivepaper/data/annotations.csv'])
    else:
        parser.add_argument('--data-valid', nargs='*', default=['data/DeepQ-Vivepaper/data/annotations.csv'])
    parser.add_argument('--csv-class-file', '--class-map', default='data/class-map.csv')
    parser.add_argument('--group-method', choices=['none', 'random', 'ratio'], default='random')
    return parser


def ModelArgparser():
    parser = ArgumentParser('model')
    parser.add_argument('--backbone', type=int, choices=[50, 101, 152], default=50)
    parser.add_argument('--weights')
    return parser

def LRArgparser():
    parser = ArgumentParser('LRScheduler')
    parser.add_argument('--monitor', default='loss')
    parser.add_argument('--factor', type=float, default=0.1)
    parser.add_argument('--patience', type=int, default=2)
    parser.add_argument('--epsilon', type=float, default=0.0001)
    parser.add_argument('--cooldown', type=int, default=0)
    parser.add_argument('--min-lr', type=float, default=0)
    return parser

def TrainArgparser():
    parser = ArgumentParser('training')
    parser.add_argument('--name')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--train-steps', type=int, default=100)
    parser.add_argument('--valid-steps', type=int)
    return parser

def NMSArgparser(set_defaults=True):
    # set_defaults only when build model
    parser = ArgumentParser('nms')
    parser.add_argument('--score-thresh', type=float)
    parser.add_argument('--iou-thresh', type=float)
    parser.add_argument('--max-size-per-class', type=int)
    parser.add_argument('--max-total-size', type=int)
    if set_defaults:
        parser.set_defaults(score_thresh=0.4, 
                            iou_thresh=0.5,
                            max_size_per_class=100,
                            max_total_size=10000)
    return parser

def TFArgparser():
    parser = ArgumentParser('tensorflow')
    parser.add_argument('--visible-devices', '--device')
    parser.add_argument('--log-level', default='3')
    return parser

def InferenceArgparer():
    from argparse import FileType
    from functools import partial
    from glob import glob
    parser = ArgumentParser('inference')
    parser.add_argument('--weights', required=True)
    parser.add_argument('--inputs', '-i', type=partial(glob, recursive=True))
    parser.add_argument('--output', '-o', type=FileType('wb'))
    return parser
