from warnings import warn
   
def jsonize_configs(config):
    '''
        Remove dictionary values that cannot be jsonized (i.e. objects)
        Constants:
            object_names (list of str) - the list of objects in the dictionary
        Inputs:
            config (dictionary) - the dictionary to be jsonized
        Returns:
            config (dict) - the dictionary with only valid values
    '''
    object_names = [
        'replay_buffer_class',
        'env',
        'model_class',
        'lr_schedule'
    ]

    for ob_name in object_names:
        if ob_name in list(config.keys()):
            config.pop(ob_name)

    return config

def flatten_dict(dict_in, new_dict=None):
    '''
        Flatten a dictionary (no nested structures):
        Inputs:
            dict_in (dict) - the dictionary to flatten
            new_dict (dict, default=None) - the dictionary to add keys to
                If no dict is passed, a new dictionary will be created.
                This input is necessary for recursion
        Returns:
            new_dict (dict) - the flattened dict
    '''
    if new_dict is None:
        new_dict = {}

    for key, value in dict_in.items():
        if isinstance(value, dict):
            new_dict = flatten_dict(value, new_dict)
        else:
            if key not in [new_dict.keys()]:
                new_dict[key] = value
            else:
                warn('Duplicate value for key with not be flattened: ' + str(key))


    return new_dict

def alphabetize_dict(dict_in):
    '''
        Alphabetize a dictionary by key
        Inputs:
            dict_in (dict) - the dictionary to alphabetize
        Returns:
            dict_out (dict) - the alphabetized dictionary
    '''
    new_keys = sorted(list(dict_in.keys()))
    dict_out = {}
    for key in new_keys:
        dict_out[key] = dict_in[key]
    return dict_out

def format_dict(config):
    '''
        Format a config dictionary to be json-ready, flat, and alphabetized by key
        Inputs:
            config (dict) - the dictionary to format
        Returns:
            config (dict) - the formated dictionary
    '''
    config = jsonize_configs(config)
    config = flatten_dict(config)
    config = alphabetize_dict(config)
    return config
        
def config_to_string(config):
    '''
        Create a string to represent a config dictionary
        Inputs:
            config (dict) - the config dictionary
        Returns:
            string (str) - the string representations
    '''
    config = format_dict(config)

    string = ''
    for key, value in config.items():
        new_key = str(key)[:2]
        string += '_'
        string += new_key
        string += '_'
        string += str(value)

    string = string[1:]

    return string

def generate_config(default_config, config):
    '''
        Generate a config file by replacing defaults with actual values
        Inputs:
            default_config (dict) - the default values
            config (dict) - a flattened dict that contains the values to replace
        Returns:
            default_config (dict) - the new config values
    '''
    for key, value in default_config.items():
        if isinstance(value, dict):
            default_config[key] = generate_config(value, config)
        else:
            if key in config.keys():
                default_config[key] = config[key]

    return default_config


if __name__ == '__main__':
    from config import DEFAULT_CONFIG

    config = DEFAULT_CONFIG
    string_config = config_to_string(config)
    print(string_config)
