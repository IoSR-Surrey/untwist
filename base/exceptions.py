"""
Common exceptions

"""

class ChannelLayoutException(Exception):    
    def __str__(self):
        return "Unsupported channel layout"
        
class ArgumentException(Exception):    
    pass
    