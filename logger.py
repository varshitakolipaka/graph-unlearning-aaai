import json

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
    
    def __init__(self, filename):
        self.filename = filename
        
        # we are appending to the file, so get the old logs to append to
        try:
            with open(filename, 'r') as f:
                self.logs = json.load(f)
        except:
            self.logs = {
                'results': {},
                'arguments': {}
            }
            
        print(self.logs)
            
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
        if seed not in self.logs['results']:
            self.logs['results'][seed] = {}
            
        if method not in self.logs['results'][seed]:
            self.logs['results'][seed][method] = {}
        
        for key, value in result.items():
            self.logs['results'][seed][method][key] = value
        
        with open(self.filename, 'w') as f:
            json.dump(self.logs, f, indent=4)