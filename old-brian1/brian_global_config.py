'''
Created on 02.01.2012

@author: "Simon Brodeur"
'''

from brian.globalprefs import *
set_global_preferences(useweave=True)
#set_global_preferences(usecodegen=True)
set_global_preferences(weavecompiler='gcc')
set_global_preferences(gcc_options=['-march=native'])
