import json
import os
from pprint import pprint
import time

class Logger():
    '''
    log structure:
    {
        results: {
            seed: {
                method: {
                    value1: value1,
                    value2: value2,
                },
                ...
            },
            ...
        },
        arguments: {
            arg1: arg1,
            arg2: arg2,
            ...
        }
    }
    '''
    
    def __init__(self, args, filename):
        os.makedirs(f'logs/{args.log_name}/{args.dataset}', exist_ok=True)
        if args.corrective_frac < 1.0:
            filename += f"_cf_{args.corrective_frac}"
        self.filename = f"logs/{args.log_name}/{args.dataset}/{filename}.json"
        
        # we are appending to the file, so get the old logs to append to
        try:
            with open(self.filename, 'r') as f:
                self.logs = json.load(f)
        except:
            self.logs = {
                'results': {},
                'arguments': {}
            }
            
    def log_arguments(self, args):
        '''
        log the arguments of the run
        
        args: argparse.Namespace, arguments of the run
        '''
        for key, value in vars(args).items():
            self.logs['arguments'][key] = value
        
        with open(self.filename, 'w') as f:
            json.dump(self.logs, f, indent=4)
        
    def log_result(self, seed, method, result):
        '''
        log the result of a method
        
        seed: int, seed of the run
        method: str, name of the method
        result: dict, values to log
        '''
        seed = str(seed)
        if seed not in self.logs['results']:
            print(f"seed {seed} not in logs")
            self.logs['results'][seed] = {}
            
        if method not in self.logs['results'][seed]:
            print(f"method {method} not in logs")
            self.logs['results'][seed][method] = {}
        
        print(f"logging {method} result for seed {seed} at location {self.filename}")
        for key, value in result.items():
            self.logs['results'][seed][method][key] = value
        
        os.makedirs(os.path.dirname(self.filename), exist_ok=True)

        with open(self.filename, 'w') as f:
            json.dump(self.logs, f, indent=4)