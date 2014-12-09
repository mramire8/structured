from ConfigParser import SafeConfigParser
import sys

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
        dict1[k] = v
    return dict1

def config_section_map(section):
    dict1 = {}
    options = Config.options(section)
    for option in options:
        try:
            dict1[option] = Config.get(section, option)
            if dict1[option] == -1:
                DebugPrint("skip: %s" % option)
        except:
            print("exception on %s!" % option)
            dict1[option] = None
    return dict1
