import numpy as np
import polars as pl

def train_test_split(df, order_by='status_as_of', partition_by='event_id', test_ratio=0.2):
    partitions = df.group_by(partition_by).agg(pl.col(order_by).max()).sort(by=order_by, descending=True)
    number_of_partitions_in_test_set = int(np.ceil(test_ratio*partitions.shape[0]))
    test_partitions = partitions.head(number_of_partitions_in_test_set).get_column(partition_by).to_list()

    cond = pl.col(partition_by).is_in(test_partitions)
    test_data = df.filter(cond)
    train_data = df.filter(~cond)
    

    return train_data, test_data