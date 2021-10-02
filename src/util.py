import logging
import numpy as np
import pandas as pd


def human_readable_size(data):
    size = data.memory_usage(index=True).sum()
    units = ['KB', 'MB', 'GB']
    for unit in units:
        if size <= 1024:
            return f"{round(size, 2)} {unit}"
        size /= 1024
    return f"{round(size, 2)} {units[-1]}"


def downsize(data):
    logging.info(f"Memory usage before downsizing {human_readable_size(data)}")
    translate = dict()
    for column in data.columns:
        if data[column].dtype == object \
                or data[column].dtype.name == 'category':
            data[column], translate[column] = pd.factorize(data[column])
        if data[column].dtype in [np.int8, np.int16, np.int32, np.int64]:
            data[column] = pd.to_numeric(data[column], downcast='unsigned')
        elif data[column].dtype in [np.float16, np.float32, np.float64]:
            data[column] = pd.to_numeric(data[column], downcast='float')
    logging.info(f"Memory usage after downsizing {human_readable_size(data)}")
    return data, translate
