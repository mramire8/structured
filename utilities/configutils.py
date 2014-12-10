
from ConfigParser import SafeConfigParser
import ast

def get_config(config_file):
    config = SafeConfigParser()
    config.read(config_file)
    return config


def get_section_names(config):    
    return  config.sections()

    for section_name in config.sections():
        print 'Section:', section_name
        print '  Options:', config.options(section_name)
        for name, value in config.items(section_name):
            print '  %s = %s' % (name, value)
        print

def has_section(config, section):
    return section in get_section_names(config)

def get_section_options(config, section):
    dict1 = {}
    for k,v in  config.items(section):
        try: 
            dict1[k] = ast.literal_eval(v)
        except ValueError:
            dict1[k] = v
    return dict1

def get_section_option(config, section, option):
    dict1 = {}
    for k,v in  config.items(section):
        dict1[k] = ast.literal_eval(v)
    return dict1


