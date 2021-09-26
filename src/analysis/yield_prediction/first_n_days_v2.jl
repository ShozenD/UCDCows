# ========== Analysis with cows where first N days are recorded ============================
# Diff from v1:
#   - Uses dummy variables and have everything fitted into one linear regression equation
#     instead of having multiple equations
#   

# ===== Import Packages =====
using DataFrames,
      DataFramesMeta,
      CSV,
      CategoricalArrays,
      Dates,
      Statistics,
      Plots,
      GLM

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
    splitbyhealth(data, args...; kwargs...)

Splits data to 2 data frames based on health status (healthy vs sick).

# Arguments
- `data::DataFrame`: Filtered data frame. Results from `filter_cows`.
- `args...`: Additional arguments for `ishealthyfirstNdays`.
- `kwargs`: Additional keyword arguments for `ishealthyfirstNdays`.

# Returns
- `::DataFrame`: Data frame containing cows that are healthy in the first N days.
- `::DataFrame`: Data frame containing cows that either have gaps (mastitis occurence) or
  high MDi in the first N days.
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

# Compute the day in milk of specific date
"""
    compute_dinmilk(dt, ref)

Compute the day in milk of a given date based on the corresponding first date in milk.

# Arguments
- `dt::Date`: Date to compute day in milk.
- `ref::Date`: Date of first day in milk.

# Returns
`::Int64`: The day in milk of `dt.`
"""
compute_dinmilk(dt::Date, ref::Date) = (dt - ref + Day(1)).value

"""
    compute_first_dinmilk(data)
    compute_first_dinmilk(dt, dinmilk)

Compute the actual first day in milk of a cow.

# Arguments
- `data::DataFrame`: Data frame containing `:id`, `:dinmilk`, `:date` columns.
- `dt::Date`: A proposed date.
- `dinmilk::Integer`: The corresponding day in milk to the proposed `dt`.

# Returns
`::Date`: First date in milk.
"""
function compute_first_dinmilk(data::AbstractDataFrame)
    # Ensure there is only 1 cow in data frame
    @assert length(unique(data.id)) == 1

    return compute_first_dinmilk(data.date[1], data.dinmilk[1])
end

compute_first_dinmilk(dt::Date, dinmilk::Integer) = dt - Day(dinmilk) + Day(1)

# Fill missing dates (gaps representing mastitis occurence) with yield = 0
"""
    fillmissingdates(data)

For each cow, fill the gaps (missing records in the middle of milk production) with default
yield value of 0.

!!! Note: This function cannot be used before running `categorize_data`.

# Arguments
- `data::DataFrame`: Data frame containing data of multiple cows.

# Returns
- `::DataFrame`: Data frame with filled gaps.
"""
function fillmissingdates(data::DataFrame)
    results = DataFrame()
    by_cows = groupby(data, :id)
    for ((id,), group) in pairs(by_cows)
        # Find missing dates and their corresponding days in milk
        missing_dates = findmissingdates(group.date)
        dinmilk_1 = compute_first_dinmilk(group)    # First day in milk
        missing_dinmilk = [compute_dinmilk(dt, dinmilk_1) for dt in missing_dates]
        # Get the lactation number and group of current cow
        lactnum = unique(group.lactnum)[1]
        grp = unique(group.group)[1]
        # Create temporary data frame for missing dates
        temp = DataFrame(id = id, date = missing_dates, yield = 0, 
                         dinmilk = missing_dinmilk, lactnum = lactnum,
                         group = grp, status = "sick")
        # Join temporary data with data frame of current cow
        temp = outerjoin(group, temp, 
                         on = [:id, :yield, :date, :dinmilk, :lactnum, :group, :status])
        results = vcat(results, temp)
    end
    return results
end

# Find the set of missing dates from a list of dates
"""
    findmissingdates(list)

Finds the list of missing dates from a list of dates.

# Arguments
- `list::AbstractVector{<:Dates}`: List of dates.
"""
function findmissingdates(list::AbstractVector{T}) where T<:Date
    result = Vector{T}()
    rng = minimum(list):Day(1):maximum(list)
    for dt in rng
        dt ∉ list ? push!(result, dt) : nothing
    end
    return result
end

# Check if the record of a cow contains gaps
"""
    containsgaps(data[, bycows])

Checks if the records of a cow contains any gaps.

# Arguments
- `data::DataFrame`: Data frame containing cow records.
- `return_summary::Bool`: (Default: true) Whether to return the results of each cow or all
  cows in general. 

# Returns
- `::Bool / ::DataFrame`: If `return_summary=false`, returns a data frame describing whether
  each cow has a gap or not. If `return_summary=true`, returns a boolean describing if
  there's a gap overall.
"""
function containsgaps(data::DataFrame, return_summary::Bool = true)
    by_cows = groupby(data, :id)
    result = DataFrame()
    # Check if there's gaps in each cow
    for ((id,), group) in pairs(by_cows)
        # Temporary data frame storing the row result of current cow
        temp = DataFrame("id" => id, "has_gaps" => !iscompleteperiod(group))
        # Append result
        result = vcat(result, temp)
    end
    return (return_summary ? any(result.has_gaps) : result)
end

# Remove cows with gaps from healthy group
"""
    removecowswithgaps(data)

Remove cows with gaps in their milk production as it is assumed that these gaps suggest the
occurence of mastitis.

# Arguments
- `data::DataFrame`: Data frame, generally data frame of healthy cows are used.

# Returns
`::DataFrame`: Subset of input data frame without cows with gaps in milk production.
"""
function removecowswithgaps(data::DataFrame)
    # Declare variable to track cows with gaps
    cows_with_gaps = Vector{eltype(data.id)}()
    # For each cow, check if cow contains gap, if true, push cow into cows_with_gaps
    by_cows = groupby(data, :id)
    for ((id,), group) in pairs(by_cows)
        iscompleteperiod(group) ? nothing : push!(cows_with_gaps, id)
    end
    # Return subset of data frame without cows with gaps
    return @subset(data, :id .∉ Ref(cows_with_gaps))
end

"""
    iscompleteperiod(data)

Check if all dates in the specified range is available.

# Arguments
- `data::AbstractDataFrame`: Data frame containing cow data.

# Returns
- `::Bool`:  Whether all dates in the specified range is available.
"""
function iscompleteperiod(data::AbstractDataFrame)
    # Collect all available dates in data frame
    dates = unique(data.date)
    # Compute the length of date range
    ndays = maximum(dates) - minimum(dates) + Day(1)
    # Check if length of date range is equal to dates list
    return ndays.value == length(dates)
end

# Aggregate data frame
"""
    aggregate_data(data)

# Arguments
- `data::DataFrame`: Data frame to aggregate.

# Returns
`::DataFrame`: Aggregated data frame.
"""
function aggregate_data(data::DataFrame)
    # 1. Group data frame by ID and day in milk
    # 2. Aggregate data frame:
    #   - Yield => Sum of yield by day
    #   - LogYield => Log of sum of yield by day
    #   - MDi => Maximum MDi value each day
    results = groupby(data, [:id, :dinmilk]) |>
              grps -> combine(grps,
                              :lactnum,
                              :yield => sum => :yield,
                              :yield => (x -> logyield=log(sum(x)+1)) => :logyield,
                              :mdi => (x -> mdi=maximum(skipmissing(x), init=-1)) => :mdi,
                              :date) |>
              comb -> unique(comb, [:id, :dinmilk])
    return results
end

# Categorize the cows and their healthy status
"""
    categorize_data(healthy_data, sick_data[; before, after, criterion, mdi_threshold])
    categorize_data(data[; before, after, criterion, mdi_threshold])

Categorize data into: 
    - `group`: Healthy vs sick groups (Not available if `data` is input instead of
    `healthy_data` and `sick_data`)
    - `status`: Health status, healthy vs unhealthy

# Arguments
- `healthy_data::DataFrame`: Data frame containing healthy cows.
- `sick_data::DataFrame`: Data frame containing sick cows.
- `data::DataFrame`: Data frame containing either of healthy or sick cows (or both).

# Keyword Arguments
- `before::Integer`: (Default: 2) Number of days before an occurence of an event to consider
  a cow to be unhealthy.
- `after::Integer`: (Default: 5) Number of days after an occurence of an event to consider a
  cow to be unhealthy.
- `criterion::Symbol`: (Default: `:mdi`) Criterion to consider a cow to be sick. Applicable
  values are `:mdi` and `:sick`.
- `mdi_threshold::AbstractFloat`: (Default: 1.4) MDi threshold to consider a cow to be sick.
  Ignored when `criterion = :sick`.

# Returns
`::DataFrame`: A single data frame (result of the concatenation of `healthy_data` and
`sick_data`) with additional `group` and `status` columns.
"""
function categorize_data(healthy_data::DataFrame, sick_data::DataFrame;
                         before::Integer = 2,
                         after::Integer = 5,
                         criterion::Symbol = :mdi,
                         mdi_threshold::AbstractFloat = 1.4)
    # Add health status categories to healthy and sick data
    h_data = categorize_data(healthy_data, before=before, after=after,
                             criterion=criterion, mdi_threshold=mdi_threshold)
    s_data = categorize_data(sick_data, before=before, after=after,
                             criterion=criterion, mdi_threshold=mdi_threshold)
    # Add group categories to healthy and sick data
    h_data[!, :group] .= "healthy"
    s_data[!, :group] .= "sick"
    # Concatenate the 2 data frames
    data = vcat(h_data, s_data)
    return data
end

function categorize_data(data::DataFrame;
                         before::Integer = 2,
                         after::Integer = 5,
                         criterion::Symbol = :mdi,
                         mdi_threshold::AbstractFloat = 1.4)
    # Make copy of data frame
    data_copy = copy(data)
    # Add health status column
    data_copy[!, :status] .= "healthy"
    # Find the unhealthy (sick or a few days before/after sick) days for each cow
    by_cows = groupby(data, :id)
    for ((id,), group) in pairs(by_cows)
        rng = minimum(group.date):Day(1):maximum(group.date)
        sick_days, unhealthy_days = Date[], Date[]
        # Look for cow's sick days (occurence of event)
        for dt in rng
            if issickday(dt, group, criterion=criterion, mdi_threshold=mdi_threshold)
                push!(sick_days, dt)
            end
        end
        # Look for cow's unhealthy days (before or after occurence of event)
        for dt in rng
            if isunhealthyday(dt, sick_days, before=before, after=after)
                push!(unhealthy_days, dt)
            end
        end
        println(unhealthy_days)
        # Change the health status based on ID and unhealthy days
        cond = (data_copy.id .== id) .& (data_copy.date ∈ Ref(sick_days))
        data_copy[cond, :status] .= "sick"
        cond = (data_copy.id .== id) .& (data_copy.date ∈ Ref(unhealthy_days))
        data_copy[cond, :status] .= "unhealthy"
    end
    return data_copy
end

# Check if cow is sick on `dt`
"""
    issickday(dt, df[; criterion, mdi_threshold])

Check if `dt` is a sick day for a particular cow.

# Arguments
- `dt::Date`: Date of interest.
- `df::AbstractDataFrame`: Data for corresponding cow.

# Keyword Arguments
- `criterion::Symbol`: (Default: `:mdi`) Criterion to consider a cow as sick.
- `mdi_threshold::AbstractFloat`: (Default: 1.4) MDi threshold. Ignored if `criterion =
  :sick`.

# Returns
`::Bool`: Whether a cow is sick or not.
"""
function issickday(dt::Date, df::AbstractDataFrame; 
                   criterion::Symbol = :mdi,
                   mdi_threshold::AbstractFloat = 1.4)
    # ----- Check gap (mastitis case) -----
    # If `dt` is part of a gap, it is a sick day regardless of criterion
    if dt ∉ df.date
        return true
    end

    # ----- Check MDi -----
    max_mdi = all(ismissing.(df[df.date .== dt, :mdi])) ? 
              -1 : maximum(skipmissing(df[df.date .== dt, :mdi]))
    # If criterion is :sick, nothing to check
    if criterion == :sick
        return false
    # If criterion is :mdi, check if maximum MDi of `dt` is higher than threshold
    elseif max_mdi ≥ mdi_threshold
        return true
    # Criterion is :mdi, but maximum MDi of `dt` is less than threshold
    else
        return false
    end
end

"""
    isunhealthyday(dt, dts[; before, after])

Checks if cow is fully healthy.

# Arguments
- `dt::T where T<:Date`: Date of interest.
- `dts::Vector{T} where T<:Date`: List of sick dates.

# Keyword Arguments
- `before::Integer`: (Default: 2) Number of days before a sick event where a cow should
  still be considered unhealthy.
- `after::Integer`: (Default: 5) Number of days after a sick event where a cow should still
  be considered unhealthy.

# Returns
`::Bool`: Whether cow is unhealthy or not.
"""
function isunhealthyday(dt::T, dts::Vector{T};
                        before::Integer = 2, 
                        after::Integer = 5) where T<:Date
    for day in dts
        if (dt ≥ (day - Day(before))) & (dt ≤ (day + Day(after)))
            return true
        end
    end
    return false
end

# Print model statistics
function print_modelstatistics(model::StatsModels.TableRegressionModel,
                               train_data::DataFrame,
                               test_data::DataFrame)
    # model statistics
    println("Model Statistics:")
    display(model)

    # train data
    degree_of_freedom = dof(model)
    n_observation = nobs(model)
    r² = r2(model)
    pred = predict(model, train_data)
    error = train_data.logyield - pred
    train_sse = sum(error .^ 2)
    train_mse = train_sse / dof_residual(model)
    println("\nTrain data:")
    println("Degree of freedom: $degree_of_freedom")
    println("Number of observations: $n_observation")
    println("R²: $r²")
    println("Mean squared error: $(round(train_mse, digits=4))")

    # test data
    n_observation = nrow(test_data)
    pred = predict(model, test_data)
    error = test_data.logyield - pred
    test_sse = sum(error .^ 2)
    test_mse = test_sse / (n_observation - degree_of_freedom)
    println("\nTest data statistics:")
    println("Number of observations: $n_observation")
    println("Mean Squared Error: $(round(test_mse, digits=4))")
end

# Scatter plot and curve for healthy vs sick cows
function plot_healthyvssick(df_healthy::DataFrame,
                            df_sick::DataFrame,
                            model_healthy::StatsModels.TableRegressionModel,
                            model_sick::StatsModels.TableRegressionModel,
                            lactnum::Integer,
                            max_dinmilk::Integer;
                            kwargs...)
    # ----- Get prediction line -----
    # Make predictor dataset
    pred_data = DataFrame("lactnum" => lactnum, "dinmilk" => 1:max_dinmilk)
    # Compute fitted line, then convert predicted logyield to yield
    pred_healthy = predict(model_healthy, pred_data) |> x -> exp.(x) .- 1
    pred_sick = predict(model_sick, pred_data) |> x -> exp.(x) .- 1
    # ----- Build healthy and sick scatter plots -----
    # Create empty plot with titles
    healthy_plot = plot(title="Lactation $lactnum Healthy Cows"; kwargs...)
    sick_plot = plot(title="Lactation $lactnum Sick Cows"; kwargs...)
    # Color scatter plots by MDi level
    for (p, df) in [(healthy_plot, df_healthy), (sick_plot, df_sick)]
        by_mdi_level = groupby(df, :mdi_level)
        for ((level,), group) in pairs(by_mdi_level)
            # Color blue if level=normal, orange if level=high
            mc= level == "normal" ? 1 : 2
            scatter!(p, group.dinmilk, group.yield, mc=mc, ma=0.5, label=level)
        end
    end
    # Add fitted line
    plot!(healthy_plot, 1:max_dinmilk, pred_healthy, label="", lc=:black, lw=3)
    plot!(sick_plot, 1:max_dinmilk, pred_sick, label="", lc=:black, lw=3)
    # ----- Get healthy and sick plot side by side -----
    p = plot(healthy_plot, sick_plot, layout=(1,2))
    return p
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
# -- Remove certain cows that have missing teats --
# Healthy data: remove 9064 (LR) and 49236 (RF)
df_healthy = @subset(df_healthy, :id .∉ Ref([9064, 49236]))
# Sick data: remove 7984 (RF), 42130 (LR), and 48695 (LR)
df_sick = @subset(df_sick, :id .∉ Ref([7984, 42130, 48695]))
# -- Remove "healthy" cows with MDi above threshold at any point of milking -- 
# Only keep fully healthy cows
cows = @subset(df_healthy, :mdi .≥ mdi_threshold) |> x -> unique(x.id)
df_healthy = @subset(df_healthy, :id .∉ Ref(cows))
# Remove "healthy" cows with gaps
df_healthy = removecowswithgaps(df_healthy)
# --- Categorize data into groups and health status ---
df = categorize_data(df_healthy, df_sick)
# Fill up missing dates with yield = 0
df = fillmissingdates(df)
