# ----- Filter data frame -----
"""
    filter_cows(df, N)

Filter data frame to contain only cows where the first N days of milk yield are recorded,
ie. filter the cows where the 1st day in milk is on or after 2021/04/27. Additionally, cows
with lactation number larger than 3 is removed from the dataset.

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
    by_id = groupby(df, [:id, :lactnum])
    for ((id, lactnum), group) in pairs(by_id)
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
        # Remove data is lactation number is not 1,2,3
        if lactnum ∉ [1,2,3]
            continue
        end
        # --- Add cow data into filtered data ---
        cow_data = vcat(cow_data, group)
        # --- Add summary of current cow ---
        temp = DataFrame(id = [id], 
                         first_day_in_milk = [dinmilk_1],
                         lactation_number = [lactnum])
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
        low_mdi = (all ∘ skipmissing)(df.mdi .< threshold)
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
    fillmissingdinmilk(data)

For each cow, fill the gaps (missing records in the middle of milk production) with default
yield value of 0.

!!! Note: This function cannot be used before running `categorize_data`.

# Arguments
- `data::DataFrame`: Data frame containing data of multiple cows.

# Returns
- `::DataFrame`: Data frame with filled gaps.
"""
function fillmissingdinmilk(data::DataFrame)
    results = DataFrame()
    by_cows = groupby(data, :id)
    for ((id,), group) in pairs(by_cows)
        # Find missing days in milk
        missing_dinmilk = findmissingdinmilk(group.dinmilk)
        # Get the lactation number and group of current cow
        lactnum = unique(group.lactnum)[1]
        grp = unique(group.group)[1]
        # Create temporary data frame for missing dates
        temp = DataFrame(id = id, yield = 0, dinmilk = missing_dinmilk, lactnum = lactnum,
                         group = grp, status = "sick")
        # Join temporary data with data frame of current cow
        temp = outerjoin(group, temp, 
                         on = [:id, :yield, :dinmilk, :lactnum, :group, :status])
        results = vcat(results, temp)
    end
    # Make lactnum as categorical variable again
    results[!, :lactnum] = categorical(results.lactnum)
    return results
end

# Find the set of missing dates from a list of dates
"""
    findmissingdinmilk(list)

Finds the list of missing days in milk from a list of days in milk.

# Arguments
- `list::AbstractVector{<:Integer}`: List of days in milk.
"""
function findmissingdinmilk(list::AbstractVector{T}) where T<:Integer
    result = Vector{T}()
    rng = minimum(list):maximum(list)
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
    cows_with_gaps = Vector{Tuple{eltype(data.id), eltype(data.lactnum)}}()
    # For each cow, check if cow contains gap, if true, push cow into cows_with_gaps
    by_cows = groupby(data, [:id, :lactnum])
    for ((id, lactnum), group) in pairs(by_cows)
        iscompleteperiod(group) ? nothing : push!(cows_with_gaps, (id, lactnum))
    end
    # Return subset of data frame without cows with gaps
    return @subset(data, zip(:id, :lactnum) .∉ Ref(cows_with_gaps))
end

"""
    iscompleteperiod(data)

Check if all days in milk are available without any gaps in between.

# Arguments
- `data::AbstractDataFrame`: Data frame containing cow data.

# Returns
- `::Bool`:  Whether all dates in the specified range is available.
"""
function iscompleteperiod(data::AbstractDataFrame)
    # Collect all available days in milk in data frame
    days = unique(data.dinmilk)
    # Compute the expected length of days in milk if there are no gaps
    ndays = maximum(days) - minimum(days) + 1
    # Check if length of date range is equal to dates list
    return ndays == length(days)
end

# Aggregate data frame
"""
    aggregate_data(data)

1. Group data frame by ID and day in milk
2. Aggregate data frame:
  - Yield => Sum of yield by day
  - LogYield => Log of sum of yield by day
  - MDi => Maximum MDi value each day

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
                              :condtot => mean => :condtot,
                              :GPTAM,
                              :normalized_GPTAM,
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
        rng = minimum(group.dinmilk):maximum(group.dinmilk)
        sick_days, unhealthy_days = Int[], Int[]
        # Look for cow's sick days (occurence of event)
        for dinmilk in rng
            if issickday(dinmilk, group, criterion=criterion, mdi_threshold=mdi_threshold)
                push!(sick_days, dinmilk)
            end
        end
        # Look for cow's unhealthy days (before or after occurence of event)
        for dinmilk in rng
            if isunhealthyday(dinmilk, sick_days, before=before, after=after)
                push!(unhealthy_days, dinmilk)
            end
        end
        # Change the health status based on ID and unhealthy days
        cond = (data_copy.id .== id) .& (data_copy.dinmilk .∈ Ref(sick_days))
        data_copy[cond, :status] .= "sick"
        cond = (data_copy.id .== id) .& (data_copy.dinmilk .∈ Ref(unhealthy_days))
        data_copy[cond, :status] .= "unhealthy"
    end
    return data_copy
end

# Check if cow is sick on `dt`
"""
    issickday(dy, df[; criterion, mdi_threshold])

Check if `dy` is a sick day for a particular cow.

# Arguments
- `dy::Integer`: Day in milk of interest.
- `df::AbstractDataFrame`: Data for corresponding cow.

# Keyword Arguments
- `criterion::Symbol`: (Default: `:mdi`) Criterion to consider a cow as sick.
- `mdi_threshold::AbstractFloat`: (Default: 1.4) MDi threshold. Ignored if `criterion =
  :sick`.

# Returns
`::Bool`: Whether a cow is sick or not.
"""
function issickday(dy::Integer, df::AbstractDataFrame; 
                   criterion::Symbol = :mdi,
                   mdi_threshold::AbstractFloat = 1.4)
    # ----- Check gap (mastitis case) -----
    # If `dy` is part of a gap, it is a sick day regardless of criterion
    if dy ∉ df.dinmilk
        return true
    end

    # ----- Check MDi -----
    max_mdi = all(ismissing.(df[df.dinmilk .== dy, :mdi])) ? 
              -1 : maximum(skipmissing(df[df.dinmilk .== dy, :mdi]))
    # If criterion is :sick, nothing to check
    if criterion == :sick
        return false
    # If criterion is :mdi, check if maximum MDi of `dy` is higher than threshold
    elseif max_mdi ≥ mdi_threshold
        return true
    # Criterion is :mdi, but maximum MDi of `dy` is less than threshold
    else
        return false
    end
end

"""
    isunhealthyday(dy, dys[; before, after])

Checks if cow is fully healthy.

# Arguments
- `dy::T where T<:Integer`: Day in milk of interest.
- `dys::Vector{T} where T<:Integer`: List of sick days.

# Keyword Arguments
- `before::Integer`: (Default: 2) Number of days before a sick event where a cow should
  still be considered unhealthy.
- `after::Integer`: (Default: 5) Number of days after a sick event where a cow should still
  be considered unhealthy.

# Returns
`::Bool`: Whether cow is unhealthy or not.
"""
function isunhealthyday(dy::T, dys::Vector{T};
                        before::Integer = 2, 
                        after::Integer = 5) where T<:Integer
    for day in dys
        if (dy ≥ (day - before)) & (dy ≤ (day + after)) & (dy ≠ day)
            return true
        end
    end
    return false
end