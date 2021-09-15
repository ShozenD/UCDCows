# ========== Analysis with cows where first N days are recorded ============================

# ===== Import Packages =====
using DataFrames,
      DataFramesMeta,
      CSV,
      CategoricalArrays,
      Dates,
      Statistics,
      Plots

# ========== Functions ==========
# ----- Filter data frame -----
"""
    filter_cows(df, N)

Filter data frame to contain only cows where the first N days of milk yield are recorded,
ie. filter the cows where the 1st day in milk is on or after 2021/04/27.

# Arguments
- `df::DataFrame`: Raw data frame.
- `N::Integer`: (Default: 30) Criterion where first `N` days must of milk yield must be
                recorded.

# Returns
- `::DataFrame`: Filtered data frame.
- `::DataFrame`: Summary of cows where the first N days of milk yield are recorded.
"""
function filter_cows(df::DataFrame, N::Integer = 30)
    # Data frame containing cows where first N days are recorded
    cow_data = DataFrame()
    # Summary of the cows that were filtered
    summary = DataFrame()
    # Group data frame by cows
    by_id = groupby(df, :id)
    for ((id,), group) in pairs(by_id)
        # --- Check if first N days are recorded ---
        # Check 1st day in milk exists after the 1st recorded date in `df`, and `N`th day in
        # milk exists before the most recent recorded date in `df`. If they aren't, skip
        # this loop
        dinmilk_1 = minimum(group.date) - Day(minimum(group.dinmilk)) + Day(1)
        dinmilk_N = minimum(group.date) - Day(minimum(group.dinmilk)) + Day(N)
        # 1st day is before dataset first recorded date and Nth day is after dataset last
        # recorded date
        if dinmilk_1 < minimum(df.date) || dinmilk_N > maximum(df.date)
            continue
        # No data for first N days
        elseif all(group.date .> dinmilk_N)
            continue
        end
        # --- Add cow data into filtered data ---
        cow_data = vcat(cow_data, group)
        # --- Add summary of current cow ---
        temp = DataFrame(id = [id], 
                         first_day_in_milk = [dinmilk_1],
                         lactation_number = unique(group.lactnum))
        summary = vcat(summary, temp)
    end
    return cow_data, summary
end

# Checks if cow is healthy in the first N days
"""
    ishealthyfirstNdays(data[, N])

Checks if cow is healthy in the first N days.

# Arguments
- `data::DataFrame`: Data of a particular cow.
- `N::Integer`: (Default: 30) The number of days to be tracked after a cow gives birth to
  determine whether it has been healthy or not.
- `criterion::Symbol`: (Default: `:mdi`) Criterion to determine whether a cow is sick.
  Accepted inputs are `:mdi` and `:sick`.
- `threshold::AbstractFloat`: (Default: 1.4) The MDi threshold value to determine whether a
  cow is sick. Only applicable to `criterion = :mdi`.

# Returns
`::Bool`: whether a cow is healthy in the first N days.
"""
function ishealthyfirstNdays(data::DataFrame, 
                             N::Integer = 30; 
                             criterion::Symbol = :mdi,
                             threshold::AbstractFloat = 1.4)
    # Filter out the data to first N days of interest
    df = @subset(data, :dinmilk .<= N)
    # Determine if cow has been sick in the first N days
    if criterion == :mdi
        # --- Check if all MDi values are below threshold ---
        low_mdi = all(skipmissing(df.mdi .< threshold))
        # --- Check if there's a gap between the first recorded day and the N-th day ---
        # Get all unique dates within first N days
        dts = unique(df.date[df.dinmilk .≤ N])
        # Get number of days from the first recorded day to the N-th day
        n_days = maximum(dts) - minimum(dts) + Day(1)
        # --- Return true if there is no gap and all MDi are below threshold ---
        return n_days.value == length(dts) && low_mdi
    elseif criterion == :sick
        # --- Check if there's a gap between the first recorded day and the N-th day ---
        # Get all unique dates within df
        dts = unique(df.date[df.dinmilk .≤ N])
        # Get number of days from the first recorded day to the N-th day
        n_days = maximum(dts) - minimum(dts) + Day(1)
        # --- Return true if there is no gap ---
        return n_days.value == length(dts)
    else
        throw(ArgumentError("Unknown criterion $criterion."))
    end
end

# Splits data by health status, ie. sick vs healthy
"""

"""
function splitbyhealth(data::DataFrame, args...; kwargs...)
    # Declare healthy and sick data frames
    df_healthy = DataFrame()
    df_sick = DataFrame()
    # Group by cow ID
    bycows = groupby(data, :id)
    # Add cow data into healthy or sick data frame depending on its health status
    for (_, group) in pairs(bycows)
        # Convert to data frame
        df = DataFrame(group)
        # Cow is healthy
        if ishealthyfirstNdays(df, args...; kwargs...)
            df_healthy = vcat(df_healthy, df)
        # Cow is unhealthy
        else
            df_sick = vcat(df_sick, df)
        end
    end
    return df_healthy, df_sick
end

# ========== Workflow ==========
# ----- Import Data -----
# Move to directory of current file
cd(@__DIR__)
# Read file
path = "../../../data/analytical/cows-analytic.csv"
df = CSV.read(path, DataFrame)
df[!, :id] = categorical(df.id)
df[!, :lactnum] = categorical(df.lactnum)

# ----- Data Processing -----
# Keep data of cows with records of their first 30 days in milk, discard the rest of the
# data.
filtered, summary = filter_cows(df)
# Split data into healthy and sick
mdi_threshold = 1.4
df_healthy, df_sick = splitbyhealth(filtered, 30, criterion=:mdi, threshold=mdi_threshold)

# Remove certain cows that have missing teats
# Healthy data: remove 9064 (LR) and 49236 (RF)
df_healthy = df_healthy[df_healthy.id .∉ Ref([9064, 49236]), :]
# Sick data: remove 7984 (RF), 42130 (LR), and 48695 (LR)
df_sick = df_sick[df_sick.id .∉ Ref([7984, 42130, 48695]), :]

# Aggregate the yield and mdi
df_healthy = groupby(df_healthy, [:id, :dinmilk]) |>
             grps -> combine(grps,
                             :lactnum,
                             :yield => sum => :yield,
                             :mdi => (x -> mdi=maximum(skipmissing(x), init=-1)) => :mdi,
                             :date) |>
             comb -> unique(comb, [:id, :dinmilk])
df_sick = groupby(df_sick, [:id, :dinmilk]) |>
          grps -> combine(grps,
                          :lactnum,
                          :yield => sum => :yield,
                          :mdi => (x -> mdi=maximum(skipmissing(x), init=-1)) => :mdi,
                          :date) |>
          comb -> unique(comb, [:id, :dinmilk])

# Add new labels: Max MDi level and MDi level
# MDi level
df_healthy[!, :mdi_level] = map(x -> x≥mdi_threshold ? "high" : "normal", df_healthy.mdi)
df_sick[!, :mdi_level] = map(x -> x≥mdi_threshold ? "high" : "normal", df_sick.mdi)
# Max MDi level: Maximum MDi level a particular cow has/will experience based on the data


# Healthy plot
p = plot()
by_mdi_level = groupby(df_healthy, :mdi_level)
for ((level,), group) in pairs(by_mdi_level)
    scatter!(p, group.dinmilk, group.yield, label=level)
end
# Sick plot
p = plot()
by_mdi_level = groupby(df_sick, :mdi_level)
for ((level,), group) in pairs(by_mdi_level)
    scatter!(p, group.dinmilk, group.yield, label=level)
end