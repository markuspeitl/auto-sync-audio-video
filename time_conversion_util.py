
def msec_to_samples(msec, sample_rate):
    return int((msec / 1000) * sample_rate)


def sec_to_samples(sec, sample_rate):
    return int((sec) * sample_rate)


def samples_to_msec(sample_index, sample_rate):
    return int(samples_to_sec(sample_index, sample_rate) * 1000)


def samples_to_sec(sample_index, sample_rate):
    return sample_index / sample_rate


msec_precision_digits = 3


def msec_to_timestamp(sample_msec_time):
    return sec_to_timestamp(sample_msec_time / 1000)


def sec_to_timestamp(sample_sec_time):

    number_sign = ''
    if (sample_sec_time < 0):
        number_sign = '-'

    sample_sec_time = abs(sample_sec_time)

    sample_sec_time = round(sample_sec_time, 3)

    msec = round((sample_sec_time * pow(10, msec_precision_digits)) % pow(10, msec_precision_digits))
    sec = int(sample_sec_time) % 60
    min = int(sample_sec_time / 60) % 3600
    hours = int(sample_sec_time / 3600)

    return f"{number_sign}{hours:0>2}:{min:0>2}:{sec:0>2}." + str(msec).zfill(msec_precision_digits)


"""def sec_to_timestamp(sample_sec_time):

    timestamp_string_microseconds = time.strftime('%H:%M:%S.%f', time.gmtime(sample_sec_time))

    # Trim last 3 characters from string (discard microseconds)
    # timestamp_string_msec = timestamp_string_microseconds[:-3]

    return timestamp_string_microseconds
"""


def timestamp_to_sec(timestamp):

    time_and_msec = timestamp.split('.')
    time_parts = time_and_msec[0].split(':')

    index = len(time_parts) - 1

    seconds_sum: float = 0
    for i in range(0, time_parts[index]):
        factor = pow(60, index)
        reverse_index = len(time_parts) - 1 - i
        seconds_sum += time_parts[reverse_index] * factor

    return seconds_sum + time_and_msec / 1000


def sample_index_to_timestamp(sample_index, sample_rate):
    time_per_sample_frame_sec = 1 / sample_rate

    time_at_sample_sec = time_per_sample_frame_sec * sample_index

    return sec_to_timestamp(time_at_sample_sec)


def sample_range_to_timestamps(sample_range_tuple: tuple, sample_rate: float):
    start_timestamp = sample_index_to_timestamp(sample_range_tuple[0], sample_rate)
    end_timestamp = sample_index_to_timestamp(sample_range_tuple[1], sample_rate)

    return (start_timestamp, end_timestamp)


def print_sample_range_timestamps(sample_range_tuple: tuple, sample_rate: float):
    print(sample_range_tuple)

    sample_range_timestamps = sample_range_to_timestamps(sample_range_tuple, sample_rate)

    print(sample_range_timestamps[0])
    print(sample_range_timestamps[1])
