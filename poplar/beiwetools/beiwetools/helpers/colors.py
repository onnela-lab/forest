'''
Palettes & color maps for plots.
'''
from seaborn import color_palette

# View palettes with:
# >>> seaborn.palplot(palette)
# For ColorBrewer palette names, see:
# https://s3.amazonaws.com/codecademy-content/programs/dataviz-python/unit-5/seaborn-design-2/article2_image9.png

# Some saturated ColorBrewer palettes.
# Palettes repeat for n larger than default.
def set_1(n = 9): return(color_palette('Set1', n))
def dark_2(n = 8): return(color_palette('Dark2', n))

# Some desaturated ColorBrewer palettes.
# Palettes repeat for n larger than default.
def set_2(n = 8): return(color_palette('Set2', n))
def set_3(n = 12): return(color_palette('Set3', n))
def pastel_1(n = 9): return(color_palette('Pastel1', n))
def pastel_2(n = 8): return(color_palette('Pastel2', n))

# Some sequential ColorBrewer palettes.
# If reverse = 1, then palettes go from dark to light.
def reds (n, reverse = 0): return(color_palette('Reds' + '_r'*reverse, n))
def blues(n, reverse = 0): return(color_palette('Blues'+ '_r'*reverse, n))
def greys(n, reverse = 0): return(color_palette('Greys'+ '_r'*reverse, n))


def circular_palette(n):
    '''
    Circular palette with n colors.    
    '''
    return(color_palette("hls", n))


def paired_palette(n = 12):
    '''
    Color Brewer paired palette with n colors.
    The palette repeats if n > 12.
    '''    
    return(color_palette("Paired", n))


# To add:
#   24-hour color map to grayscale
#   24-hour color map to yellow/blue scale    
#   Maybe one coarse map (6-hour blocks) and one fine map (1-hour blocks)
#   With labels, e.g. 'Midnight - 6AM', '6AM - Noon'
