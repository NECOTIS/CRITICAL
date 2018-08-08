'''
Created on 22.02.2012

@author: "Simon Brodeur"
'''

def enum(*sequential, **named):
    enums = dict(zip(sequential, range(len(sequential))), **named)
    return type('Enum', (), enums)

def getParameter(params, name, defaultValue=None):
    value = defaultValue
    if name in params:
        value = params[name]
    return value
