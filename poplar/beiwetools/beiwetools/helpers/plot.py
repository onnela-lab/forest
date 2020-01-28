'''Functions for generating visual summaries of Beiwe data.
'''
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D

from .time import to_timestamp, filename_time_format, UTC
from .colors import *

                    
def plot_timestamps(ax, timestamp_dictionary, labels = [],
                    zero_at = 0, palette = paired_palette(12),
                    hide_spines = ['top', 'bottom', 'left', 'right']):
    '''
    Plot a vertical line for each timestamp.
    
    Args:
        ax (subplots.AxesSubplot):  Where to plot the timestamps.
        timestamp_dictionary (OrderedDict):  Each value should be a list of timestamps.
        labels (list): Keys of timestamp_dictionary to plot, in order from top to bottom.
        zero_at (int):  Number to subtract from all timestamps before plotting. 
        palette (seaborn.palettes._ColorPalette):  Palette to use.
        hide_spines (list): Which axis spines to hide.

    Returns:
        t_min, t_max (int):  First and last timestamps in timestamp_dictionary.
    '''
    if len(labels) == 0:
        labels = list(timestamp_dictionary.keys())
    y_cutoffs = np.linspace(1, 0, len(labels)+1)
    t_min = None
    t_max = None
    for i in range(len(labels)):
        s = labels[i]
        try:
            timestamps = [t - zero_at for t in timestamp_dictionary[s]]
        except:
            timestamps = []
        if len(timestamps) > 0:
            ax.vlines(timestamps, ymin = y_cutoffs[i], ymax = y_cutoffs[i+1], color = palette[i]) 
            if t_min is None:
                t_min = min(timestamps)
                t_max = max(timestamps)
            else:
                t_min = min(t_min, min(timestamps))
                t_max = max(t_max, max(timestamps))    
        ax.set_ylim([0, 1])
        ax.set_yticks([])
        for loc in hide_spines:
            ax.spines[loc].set_color('none')
    return(t_min, t_max)


def make_legend(ax, labels, palette, loc = 'best', lw = 10):
    '''
    Make a basic legend.
    '''
    dummy_lines = []
    for i in range(len(labels)):
        dummy_lines.append(Line2D([0], [0], color = palette[i], lw = lw))
    ax.legend(dummy_lines, labels, loc = loc, framealpha = 1, frameon = False)


def make_fancy_legend(ax, labels_dict, palette, loc = 'best', lw = 10):
    '''
    Make a legend with labels organized under headers.

    Args:
        ax (subplots.AxesSubplot):  Where to place the legend.
        labels_dict (OrderedDict()):
            Keys are headers.
            Each value is a list of labels to appear under the corresponding header.
        palette (seaborn.palettes._ColorPalette):  Palette to use.
        loc (str): Legend location.
        lw (float): Linewidth.
        
    Returns:
        None
    '''
    dummy_lines = []
    labels = []
    count = 0
    for k in labels_dict:
        dummy_lines.append(Line2D([0], [0], alpha = 0, color = 'k', lw = lw))
        labels.append(k)
        for i in labels_dict[k]:
            dummy_lines.append(Line2D([0], [0], color=palette[count], lw = lw))
            labels.append(i)
            count += 1
        dummy_lines.append(Line2D([0], [0], alpha = 0, color = 'k', lw = lw))
        labels.append('')
    ax.legend(dummy_lines, labels, loc = loc, framealpha = 1, frameon = False)


   
    

def date_time_axis(t_start, t_end):
    '''
    Define axis ticks and labels for various followup periods.
    Labels are date or date/time strings.    
    
    Args:
        t_start, t_end (int):  Range in millisecond timestamps.
        
    Returns:
        
    '''
    # duration, 
    # tick condition, tick condition met
    # label condition, label condition met, time format
    date_axis_settings = [
            [2*year_ms, '%b', ['January', 'June'], '%b Y']
            ]
    #    x_min = x_min - x_min % day_ms
    #    x_max = x_max - x_max % day_ms
    #    x_range = range(x_min, x_max, day_ms)
    #    x_locs = []
    #    x_sublocs = []
    #    x_labs = []
    #    for t in x_range:
    #        d = to_readable(t, to_format = '%m/%d/%Y', to_tz = UTC)
    #        if d.split('/')[1] == '01':
    #            x_locs.append(t)
    #            if d.split('/')[0] in ['01', '07']:
    #                x_labs.append(to_readable(t, to_format = '%b %Y', to_tz = UTC))
    #                x_sublocs.append(t)
    #            else:
    #                x_labs.append('')

    pass


def elapsed_time_axis(t_start, t_end):
    '''
    Define axis ticks and labels for various followup periods.
    Return tick labels in units of days, weeks, etc. from start time.
    Especially for the case when t_start = 0, but works if not.
    Args:
        t_start, t_end (int):  Range in millisecond timestamps.
        
    Returns:
        
    '''
    # duration, tick unit, tick unit spacing, tick spacing, label spacing
    zero_axis_settings = [
            [  year_ms, 'Weeks', week_ms,   week_ms,  5*week_ms],
            [2*year_ms, 'Weeks', week_ms, 5*week_ms, 25*week_ms]
            ]

    # figure out closest time period
    t = t_end - t_start    
    durations = [s[0] for s in axis_settings]
    differences = [abs(t - i) for i in durations] 
    index = differences.index(min(differences))
    d, tu, tus, ts, ls = zero_axis_settings[0]
    # add buffers of +/- one tick space
    start = t_start - t_start % ts
    end = (t_end + ts) - (t_end + ts)% ts
    # get ticks and labels
    tick_locs = np.arange(start, end+ts, ts)
    tick_sublocs = np.arange(start, end+ls, ls)
    tick_labels = [''] * len(tick_locs)       
    label_locs = np.arange(0, len(tick_labels), int(ls / ts))
    for i in label_locs:
        tick_labels[i] = str(int( (tick_locs[i] - tick_locs[0]) / tus))
    return(start, end, tick_locs, tick_sublocs, tick_labels, tu)    