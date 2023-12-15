import time
from matplotlib import pyplot as plt, ticker
from matplotlib.axes import Axes
import numpy as np
import matplotlib.dates as mdates
from time_conversion_util import sec_to_timestamp


def sec_to_timestamp_format(x, pos):
    return time.strftime('%M:%S', time.gmtime(x))

def plot_timeline_data(title: str, target_plot_ax: Axes, data: np.ndarray, sample_rate: float, fill_down=False):

    data_sample_length = data.shape[0]

    timeline_data = np.linspace(
        0,  # start
        1,
        num=data_sample_length
    ) * data_sample_length / sample_rate

    # time_labels = list(map(lambda time_sec: sec_to_timestamp(time_sec), timeline_data))

    # plt.figure(1)

    # title of the plot
    # plt.title("Sound Wave")

    subplot_ax = target_plot_ax

    subplot_ax.set_title(title)

    # ax = plt.gca()

    # import matplotlib.dates as md
    # min_sec_formatter = md.DateFormatter('%M:%S')
    # plt.set_major_formatter(xfmt)
    # plt.gcf().autofmt_xdate()
    # plt.gca().xaxis.set_major_formatter(min_sec_formatter)

    subplot_ax.xaxis.set_major_formatter(ticker.FuncFormatter(sec_to_timestamp_format))

    # major tick every 60 seconds/units
    subplot_ax.xaxis.set_major_locator(ticker.MultipleLocator(30))
    # ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))

    # ax.tick_params(which='minor', length=10, color='r', )
    # ax.locator_params(axis='both', nbins=10)
    # ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    # label of x-axis
    subplot_ax.set_xlabel("Time (seconds)")
    #plt.xlabel("Time (seconds)")

    # plt.xticks(time, time_labels)
    subplot_ax.set_xticks()
    #plt.xticks()

    # plt.grid()
    #plt.grid(which='major', linestyle='-', linewidth='1', color='black')
    #plt.grid(which='minor', linestyle='-', linewidth='0.2', color='gray')

    subplot_ax.grid(which='major', linestyle='-', linewidth='1', color='black')
    subplot_ax.grid(which='minor', linestyle='-', linewidth='0.2', color='gray')
    # plt.grid(which='both', color='0.65', linestyle='-')

    subplot_ax.minorticks_on()
    #plt.minorticks_on()
    # Needs to be after minor ticks on to adjust number of minor ticks
    subplot_ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(6))
    

    # actual plotting
    #plt.plot(timeline_data, audio_channel_data)
    subplot_ax.plot(timeline_data, data)

    if (fill_down):
        #plt.fill_between(timeline_data, audio_channel_data, color='blue', alpha=0.3)
        subplot_ax.fill_between(timeline_data, data, color='blue', alpha=0.3)

def show_waveforms(audio_data, sample_rate, fill_down=False):

    data_sample_length = audio_data.shape[0]
    audio_channels_count = audio_data.shape[1]

    plt.rcParams["figure.figsize"] = (16, 12)
    figure, subplot_axs: list[Axes] = plt.subplots(audio_channels_count)

    figure.suptitle("Audio channel waves")

    for audio_channel_index in range(0, audio_channels_count):

        audio_channel_data = audio_data[:, audio_channel_index]
        subplot_ax: Axes = subplot_axs[audio_channel_index]
        plot_timeline_data(f'Channel {audio_channel_index}', subplot_ax, audio_channel_data, sample_rate, fill_down=fill_down)

    # shows the plot
    # in new window
    plt.show(block=True)