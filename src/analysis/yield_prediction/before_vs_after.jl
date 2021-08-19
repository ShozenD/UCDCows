# Analyze the difference in curves before and after mastitis infection in cows.
# ------------------------------------------------------------------------------
#
# Steps:
#   1. For each cow, split the data into two data frames: one containing data
#      before mastitis, and the other one after the infection.
#      (Optional): Remove cows that have always been healthy, ie. never 
#                  had any cases of mastitis. Only do this is the amount of 
#                  data after the removal is still reasonable.
#   2. Build 2 mixed-effect models based on the 2 different sets of data from 
#      (1).
#   3. For each cow, make predictions on the entire time series using both 
#      models. Then, determine whether one curve is lower than the other.
#   4. Model inferences. Metrics to check:
#           a. MSE: MSE from after-mastitis model should be lower when
#              predicting after-mastitis yield. The same concept applies to
#              before-mastitis model.
#           b. MAE: While MSE is a good metric to use, if the errors have
#              magnitudes less than 1, it is not a good indicator.
#           c. Average differences between before vs after models. This will
#              be used to determine whether the average yield before mastitis
#              is actually higher than after mastitis.
# ------------------------------------------------------------------------------

# Import packages --------------------------------------------------------------
using DataFrames, DataFramesMeta

# Functions --------------------------------------------------------------------
"""
Splits data into healthy and unhealthy data based on `n_days` and 
`criterion`. Specifically, we consider `n_days` days before and after an 
occurrence of a `criterion` as unhealthy data.

# Arguments
- `df::DataFrame`: Cow data frame
- `n_days::Integer`: (Default: 7) Number of days before and after an
                        occurrence of a criterion to be considered unhealthy.
- `criterion::Symbol`: (Default: :sick) The criterion used to determine
                        whether a cow is unhealthy. There are 3 possible
                        criterions: `:sick` (when a cow has mastitis) and `:mdi` 
                        (when a cow has MDi ≥ 1.4).

# Returns
- `df_healthy::DataFrame`: Data frame of healthy data.
- `df_unhealthy::DataFrame`: Data frame of unhealthy data.
"""
function split_by_period(df::DataFrame,
                         n_days::Integer = 7,
                         criterion::Symbol = :sick)
    @assert criterion in [:sick, :mdi]
    @assert n_days > 0

    # Group data frame by animal number / id
    by_id = groupby(df, :id)
    # Extract healthy and unhealthy data for each cow
    df_healthy = DataFrame()
    df_unhealthy = DataFrame()
    for ((id,), group) in pairs(by_id)
        id_group = sort(DataFrame(group), :date)
        # ----------------------------------------------------------------------
        # Collect the unique dates of recorded yield for each cow. If within
        # this date range there exists any gaps, the cow is presumed to be
        # sick with mastitis.
        # ----------------------------------------------------------------------
        # Collect unique dates and find the start and end dates
        dts = unique(id_group.date)
        start_, end_ = minimum(id_group.date), maximum(id_group.date)
        # Get sick / unhealthy days
        sick_days, unhealthy_days = [], []
        for dt in daterange(start_, end_)
            # `dt` is sick day if yield data for the date `dt` is not recorded.
            # If `criterion` is `:mdi`, `dt` is also considered a sick day if
            # cow has MDi ≥ 1.4 on `dt`.
            if issickday(start_, end_, id_group, criterion)
                push!(sick_days, dt)
            elseif isunhealthyday(dt, sick_days, n_days)
                push!(unhealthy_days, dt)
            end
        end
    end
end

