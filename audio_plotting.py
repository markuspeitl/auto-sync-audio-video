from collections.abc import Iterable
import time
from matplotlib import pyplot as plt, ticker
from matplotlib.axes import Axes
import numpy as np


def sec_to_timestamp_format(x, pos):
    return time.strftime('%M:%S', time.gmtime(x))


def plot_timeline_data(title: str, target_plot_ax: Axes, data: np.ndarray, sample_rate: float, marker_indices: list[int] = [], marked_ranges: list[tuple[int]] = [], fill_down=False):

    data_sample_length = data.shape[0]

    """timeline_data = np.linspace(
        0,  # start
        1,
        num=data_sample_length
    ) * data_sample_length / sample_rate"""

    timeline_data = np.linspace(
        0,
        1,
        num=data_sample_length
    ) * (data_sample_length - 1) / sample_rate

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
    subplot_ax.set_xlim(0, data_sample_length / sample_rate)
    # plt.xlabel("Time (seconds)")

    # plt.xticks(time, time_labels)
    # subplot_ax.set_xticks()
    # plt.xticks()

    # plt.grid()
    # plt.grid(which='major', linestyle='-', linewidth='1', color='black')
    # plt.grid(which='minor', linestyle='-', linewidth='0.2', color='gray')

    subplot_ax.grid(which='major', linestyle='-', linewidth='1', color='black')
    subplot_ax.grid(which='minor', linestyle='-', linewidth='0.2', color='gray')
    # plt.grid(which='both', color='0.65', linestyle='-')

    subplot_ax.minorticks_on()
    # plt.minorticks_on()
    # Needs to be after minor ticks on to adjust number of minor ticks
    subplot_ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(6))

    # actual plotting
    # plt.plot(timeline_data, audio_channel_data)
    # subplot_ax.plot(timeline_data, data, marker='^', markerfacecolor='red', markeredgecolor='red', markersize=15, markevery=list(marker_indices), color='blue')
    subplot_ax.plot(timeline_data, data, zorder=5)
    # subplot_ax.bar(timeline_data, data, zorder=1)
    # subplot_ax.stem(timeline_data, data, '-gD', marker='^', markerfacecolor='red', markeredgecolor='red', markersize=15, markevery=list(marker_indices), color='blue')

    if (marked_ranges and len(marked_ranges) > 0):
        for marked_range in marked_ranges:
            marked_range_scaled = (marked_range[0] * 1 / sample_rate, marked_range[1] * 1 / sample_rate)
            subplot_ax.axvspan(marked_range_scaled[0], marked_range_scaled[1], color='green', alpha=0.4)

    marker_range_scaled = (int(marker_indices[0] * 1 / sample_rate), int(marker_indices[1] * 1 / sample_rate))
    subplot_ax.scatter(marker_range_scaled[0], data[int(marker_indices[0])], marker='^', color="red", s=(15**2), zorder=10)
    subplot_ax.scatter(marker_range_scaled[1], data[int(marker_indices[1])], marker='^', color="red", s=(15**2), zorder=10)

    if (fill_down):
        # plt.fill_between(timeline_data, audio_channel_data, color='blue', alpha=0.3)
        subplot_ax.fill_between(timeline_data, data, color='blue', alpha=0.3, zorder=0)


def get_audio_channel(channel_index: int, audio_data: np.ndarray):

    if (len(audio_data.shape) <= 1):
        return audio_data

    return audio_data[:, channel_index]


def show_waveforms(audio_data: np.ndarray, sample_rate: float, marker_indices: list[int] = [], marked_ranges: list[tuple[int]] = [], fill_down=False, block=True):

    data_sample_length = audio_data.shape[0]

    audio_channels_count = 1
    if (len(audio_data.shape) > 1):
        audio_channels_count = audio_data.shape[1]

    plt.rcParams["figure.figsize"] = (16, 10)
    figure, subplot_axs = plt.subplots(audio_channels_count)

    figure.suptitle("Audio channel waves")

    for audio_channel_index in range(0, audio_channels_count):

        audio_channel_data = get_audio_channel(audio_channel_index, audio_data)

        if (isinstance(subplot_axs, Iterable)):
            subplot_ax: Axes = subplot_axs[audio_channel_index]
        else:
            subplot_ax = subplot_axs

        plot_timeline_data(f'Channel {audio_channel_index}', subplot_ax, audio_channel_data, sample_rate, marker_indices, marked_ranges, fill_down=fill_down)

    # shows the plot
    # in new window
    plt.show(block=block)

    print("Closed window")
