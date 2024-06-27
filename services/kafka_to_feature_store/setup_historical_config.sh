export KAFKA_TOPIC=ohlc_historical
export KAFKA_CONSUMER_GROUP=ohlc_historical_consumer_group_123
export FEATURE_GROUP_NAME=ohlc_feature_group
export FEATURE_GROUP_VERSION=1

# Number of elements to save at once to the feature store. For historical data, we can save more elements at once
export BUFFER_SIZE=150000

# this way we tell  our `kafka_to_feature_store` service to save features to the
# offline store, because we are basically generating historical data we will use for
# training our models
export LIVE_OR_HISTORICAL=historical

export SAVE_EVERY_N_SEC=30

export CREATE_NEW_COMSUMER_GROUP=true