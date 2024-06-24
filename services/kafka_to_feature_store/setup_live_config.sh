export KAFKA_TOPIC=ohlc
export KAFKA_CONSUMER_GROUP=ohlc_consumer_group_99
export FEATURE_GROUP_NAME=ohlc_feature_group
export FEATURE_GROUP_VERSION=1

# Number of elements to save at once to the feature store. For live data, should be saved to ASAP to avoid data loss.
export BUFFER_SIZE=1

export LIVE_OR_HISTORICAL=live