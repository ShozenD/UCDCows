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
      GLM,
      LaTeXStrings,
      Random,
      StatsBase,
      AverageShiftedHistograms,
      Distributions

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

# Print model statistics
function print_modelstatistics(model::StatsModels.TableRegressionModel, data::DataFrame)
    # model statistics
    println("Model Statistics:")
    display(model)

    # train data
    degree_of_freedom = dof(model)
    n_observation = nobs(model)
    r² = r2(model)
    pred = predict(model, data)
    error = data.logyield - pred
    sse = sum(skipmissing(error) .^ 2)
    mse = sse / dof_residual(model)
    println("\nMetrics:")
    println("Degree of freedom: $degree_of_freedom")
    println("Number of observations: $n_observation")
    println("R²: $r²")
    println("Mean squared error: $(round(mse, digits=4))")
end

function gridsearch(df_healthy::DataFrame,
                    df_sick::DataFrame,
                    genomic_info::DataFrame,
                    list::Union{Vector{T}, UnitRange{T}},
                    criterion::Symbol = :mdi,
                    mdi_threshold::AbstractFloat = 1.4;
                    fm::FormulaTerm = @formula(logyield ~ 1 + log(dinmilk) + dinmilk + lactnum + status + group + normalized_GPTAM),
                    split_by::Symbol = :date,
                    split_date::Union{Date, Nothing} = nothing, 
                    train_size::Union{Real, Nothing} = nothing, 
                    test_size::Union{Real, Nothing} = nothing, 
                    random_state::Union{Integer, Nothing} = nothing) where T<:Integer
    # Grid search through list
    len = length(list)
    mse_list = Dict("train" => Vector{Float64}(undef, len),
                    "test" => Vector{Float64}(undef, len))
    coef_list = Dict("status (unhealthy)" => Vector{Float64}(undef, len),
                     "group (sick)" => Vector{Float64}(undef, len))
    for (i, n) in enumerate(list)
        mse_train, mse_test, coefs, _ = categorize_and_fit(
            df_healthy, df_sick, genomic_info, n, criterion, mdi_threshold, 
            fm=fm, split_by=split_by, split_date=split_date, train_size=train_size, 
            test_size=test_size, random_state=random_state
        )
        mse_list["train"][i] = mse_train
        mse_list["test"][i] = mse_test
        coef_list["status (unhealthy)"][i] = coefs["status (unhealthy)"]
        coef_list["group (sick)"][i] = coefs["group (sick)"]
    end
    # Return the best parameter along with its results
    idx = argmin(mse_list["test"])
    best_n = list[idx]
    best_mse = (train = mse_list["train"][idx], test = mse_list["test"][idx])
    best_coef = Dict("status (unhealthy)" => coef_list["status (unhealthy)"][idx],
                     "group (sick)" => coef_list["group (sick)"][idx])
    return (n = best_n, mse = best_mse, coef = best_coef), (mse = mse_list, coef = coef_list)
end

function categorize_and_fit(df_healthy::DataFrame, 
                            df_sick::DataFrame,
                            genomic_info::DataFrame,
                            n::Integer,
                            criterion::Symbol,
                            mdi_threshold::AbstractFloat;
                            fm::FormulaTerm = @formula(logyield ~ 1 + log(dinmilk) + dinmilk + lactnum + status + group + normalized_GPTAM),
                            split_by::Symbol = :date,
                            split_date::Union{Date, Nothing} = nothing, 
                            train_size::Union{Real, Nothing} = nothing, 
                            test_size::Union{Real, Nothing} = nothing, 
                            random_state::Union{Integer, Nothing} = nothing)
    # --- Categorize data into groups and health status ---
    df = categorize_data(df_healthy, df_sick, before=2, after=n, criterion=criterion, mdi_threshold=mdi_threshold)
    # Fill missing logyield values with 0
    df[ismissing.(df.logyield), :logyield] .= 0.0
    # Join genomic info with data
    df = leftjoin(df, genomic_info, on=:id)
    df[!,:GPTAM] = convert.(Union{Float64, Missing},df[!,:GPTAM])
    # --- Discard data ---
    # Discard cow data of lactation number ≥ 4 as we are unable to estimate the GPTAM values for
    # cows of those lactation numbers
    df = @subset(df, :lactnum .∈ Ref([1,2,3]))
    # --- Fill missing GPTAM values ---
    with_GPTAM = df[.!ismissing.(df.GPTAM),:] |> 
                z -> groupby(z, :id) |> 
                z -> combine(z, :id, :lactnum, :GPTAM, :normalized_GPTAM, :yield => mean => :yield) |> 
                unique
    null_GPTAM = df[ismissing.(df.GPTAM),:] |> 
                z -> groupby(z, :id) |> 
                z -> combine(z, :id, :lactnum, :GPTAM, :normalized_GPTAM, :yield => mean => :yield)
    # Model fit for model to predict GPTAM
    fm₀ = @formula(GPTAM ~ yield + lactnum)
    model = fit(LinearModel, fm₀, with_GPTAM)
    # Use fitted model to fill missing GPTAM values
    idx = ismissing.(df.GPTAM)
    df[idx, :GPTAM] = predict(model, null_GPTAM)
    df[idx, :normalized_GPTAM] = (df[idx, :GPTAM] .- mean_GPTAM) / std_GPTAM

    # ----- Model Fitting -----
    df[df.status .== "sick", :status] .= "unhealthy"
    # Split model to train-test sets
    df_train, df_test = train_test_split(df, split_by, split_date=split_date, train_size=train_size,
                                         test_size=test_size, random_state=random_state)
    model = fit(LinearModel, fm, df_train)

    # ----- Error Computation and Coefficient Extraction -----
    coefs = Dict("status (unhealthy)" => coef(model)[6], "group (sick)" => coef(model)[7])
    mse_train = sum(residuals(model).^2) / dof_residual(model)
    err_test = predict(model, df_test) - df_test.logyield
    mse_test = sum(err_test.^2) / (nrow(df_test) - dof(model) + 1)
    return (mse_train, mse_test, coefs, model)
end

function train_test_split(df::DataFrame, by::Symbol; 
                          split_date::Union{Date, Nothing} = nothing, 
                          train_size::Union{Real, Nothing} = nothing, 
                          test_size::Union{Real, Nothing} = nothing, 
                          random_state::Union{Integer, Nothing} = nothing)
    @assert by ∈ [:date, :random]

    if by == :date
        @assert !isnothing(split_date)
        df_train = @subset(df, :date .< split_date)
        df_test = @subset(df, :date .≥ split_date)
    else
        @assert !isnothing(train_size)
        @assert !isnothing(test_size)
        @assert train_size > 0
        @assert test_size > 0
        n = nrow(df)
        train_size = floor(typeof(n), train_size * n / (train_size+test_size))
        isnothing(random_state) || Random.seed!(random_state)
        nₚ = randperm(n)
        rng_tr = nₚ[1:(train_size)]
        rng_te = nₚ[(train_size+1):end]
        df_train = df[rng_tr,:]
        df_test = df[rng_te,:]
    end
    return df_train, df_test
end

## ========== Workflow ==========
# ----- Import Data -----
# Move to directory of current file
cd(@__DIR__)
# Read milk production data file
path = "../../../data/analytical/cows-analytic.csv"
df = CSV.read(path, DataFrame)
df[!, :id] = categorical(df.id)
df[!, :lactnum] = categorical(df.lactnum)
# Read genomic info data file (55 out of 695 cows have GPTAM values after data cleaning)
path = "../../../data/misc/genomic_info.csv"
genomic_info = CSV.read(path, DataFrame) |> x -> x[!, [:id, :GPTAM]]
mean_GPTAM = mean(genomic_info.GPTAM)
std_GPTAM = std(genomic_info.GPTAM)
genomic_info[!, :normalized_GPTAM] = (genomic_info.GPTAM .- mean_GPTAM) / std_GPTAM

## ----- Data Processing -----
# Keep data of cows with records of their first 30 days in milk, discard the rest of the
# data.
filtered, summary = filter_cows(df)
# -- Remove certain cows that have missing teats --
# Cows: 9064 (LR), 49236 (RF), 7984 (RF), 42130 (LR), and 48695 (LR)
filtered = @subset(filtered, :id .∉ Ref([9064, 49236, 7984, 42130, 48695]))
# Split data into healthy and sick
mdi_threshold₁ = 1.4
mdi_threshold₂ = 1.8
df_healthy₁, df_sick₁ = splitbyhealth(filtered, 30, criterion=:mdi, threshold=mdi_threshold₁)
df_healthy₂, df_sick₂ = splitbyhealth(filtered, 30, criterion=:mdi, threshold=mdi_threshold₂)
# -- Remove "healthy" cows with MDi above threshold at any point of milking -- 
# Only keep fully healthy cows
cows = @subset(df_healthy₁, :mdi .≥ mdi_threshold₁) |> x -> unique(x.id)
df_healthy₁ = @subset(df_healthy₁, :id .∉ Ref(cows))
cows = @subset(df_healthy₂, :mdi .≥ mdi_threshold₂) |> x -> unique(x.id)
df_healthy₂ = @subset(df_healthy₂, :id .∉ Ref(cows))
# Remove "healthy" cows with gaps
df_healthy₁ = removecowswithgaps(df_healthy₁)
df_healthy₂ = removecowswithgaps(df_healthy₂)
# --- Aggregate data ---
df_healthy₁ = aggregate_data(df_healthy₁)       # Data where all cows have MDi<1.4 at all times
df_healthy₂ = aggregate_data(df_healthy₂)       # Data where all cows have MDi<1.8 at all times
df_sick₁ = aggregate_data(df_sick₁)             # Data where cows have MDi≥1.4 at certain times
df_sick₂ = aggregate_data(df_sick₂)             # Data where cows have MDi≥1.8 at certain times

## ----- Grid search for best model fit -----
dinmilk_range = 0:100
# Grid search for mdi_threshold = 1.4
best₁, list₁ = gridsearch(df_healthy₁, df_sick₁, genomic_info, dinmilk_range, :mdi, mdi_threshold₁,
                          split_by=:random, train_size=0.9, test_size=0.1, random_state=1234)
p1 = plot(dinmilk_range, list₁.mse["train"], xlabel = "k days after event", ylabel = "MSE", 
          label = "(train) vs mid-high MDi cows", lc = 1, ls = :dash, title = "Model MSEs by n days", 
          titlefontsize = 10, xguidefontsize = 8, yguidefontsize = 8,
          legend=:topright)
plot!(p1, dinmilk_range, list₁.mse["test"], label = "(test) vs mid-high MDi cows", lc = 1, ls = :solid)
p2 = plot(dinmilk_range, abs.(list₁.coef["status (unhealthy)"]), label = "β₄ when vs mid-high MDi cows",
          xlabel = "k days after event", ylabel = "|coef.|", 
          title = "Coef. magnitude for status=unhealthy (β₄)", 
          titlefontsize = 7, xguidefontsize = 8, yguidefontsize = 8, legend=:topright)
p3 = plot(dinmilk_range, abs.(list₁.coef["group (sick)"]), label = "β₅ when vs mid-high MDi cows",
          xlabel = "k days after event", ylabel = "|coef.|",
          title = "Coef. magnitude for group=sick (β₅)",
          titlefontsize = 7, xguidefontsize = 8, yguidefontsize = 8, legend=:topleft)
# Grid search for mdi_threshold = 1.8
best₂, list₂ = gridsearch(df_healthy₁, df_sick₂, genomic_info, dinmilk_range, :mdi, mdi_threshold₂,
                          split_by=:random, train_size=0.9, test_size=0.1, random_state=1234)
plot!(p1, dinmilk_range, list₂.mse["train"], label = "(train) vs high MDi cows", lc = 2, ls = :dash)
plot!(p1, dinmilk_range, list₂.mse["test"], label = "(test) vs high MDi cows", lc = 2, ls = :solid)
plot!(p2, dinmilk_range, abs.(list₂.coef["status (unhealthy)"]), label = "β₄ when vs high MDi cows")
plot!(p3, dinmilk_range, abs.(list₂.coef["group (sick)"]), label = "β₅ when vs high MDi cows")
# Overall plot
p4 = plot(p2, p3, layout=(2,1))
p5 = plot(p1, p4, layout=(1,2))
savefig(p5, "gridsearch.png")

## ----- Search for `k` days for best model fit -----
dinmilk_range = 0:7
n_trials = 5000
random_states = sample(1:10000, n_trials, replace=false)
n_days₁ = Vector{Int64}(undef, n_trials); n_days₂ = Vector{Int64}(undef, n_trials)
mses₁ = Vector{Float64}(undef, n_trials); mses₂ = Vector{Float64}(undef, n_trials)
coefs₁ = Dict{String, Vector{Float64}}(); coefs₂ = Dict{String, Vector{Float64}}()
coefs₁["status (unhealthy)"] = Float64[]; coefs₂["status (unhealthy)"] = Float64[]
coefs₁["group (sick)"] = Float64[]; coefs₂["group (sick)"] = Float64[]
for (i, random_state) in enumerate(random_states)
    best₁, list₁ = gridsearch(df_healthy₁, df_sick₁, genomic_info, dinmilk_range, :mdi, mdi_threshold₁,
                              split_by=:random, train_size=0.9, test_size=0.1, random_state=random_state)
    best₂, list₂ = gridsearch(df_healthy₁, df_sick₂, genomic_info, dinmilk_range, :mdi, mdi_threshold₂,
                              split_by=:random, train_size=0.9, test_size=0.1, random_state=random_state)
    n_days₁[i] = best₁.n; n_days₂[i] = best₂.n
    mses₁[i] = best₁.mse.test; mses₂[i] = best₂.mse.test
    push!(coefs₁["status (unhealthy)"], best₁.coef["status (unhealthy)"])
    push!(coefs₂["status (unhealthy)"], best₂.coef["status (unhealthy)"])
    push!(coefs₁["group (sick)"], best₁.coef["group (sick)"])
    push!(coefs₂["group (sick)"], best₂.coef["group (sick)"])
end

d₁ = unique(n_days₁) |> sort!; c₁ = [count(==(i), n_days₁) for i in d₁]
d₂ = unique(n_days₂) |> sort!; c₂ = [count(==(i), n_days₂) for i in d₂]
p1 = plot(title="β₄ when vs mid-high MDi cows", titlefontsize=10, legend=:outerright) 
p2 = plot(title="β₄ when vs high MDi cows", titlefontsize=10, legend=:outerright)
p3 = plot(title="β₅ when vs mid-high MDi cows", titlefontsize=10, legend=:outerright) 
p4 = plot(title="β₅ when vs high MDi cows", titlefontsize=10, legend=:outerright)
p5 = plot(title="MSE when vs mid-high MDi cows", titlefontsize=10, legend=:outerright) 
p6 = plot(title="MSE when vs high MDi cows", titlefontsize=10, legend=:outerright)
for i in unique([d₁; d₂])
    lw = i < 2 ? 2 : 1
    la = i < 2 ? 1 : 0.5
    if i in d₁ && count(==(i), n_days₁) > 1
        plot!(p1, ash(coefs₁["status (unhealthy)"][n_days₁ .== i], m=50), hist=false, label="k=$i", xtickfontsize=6, lw=lw, la=la)
        plot!(p3, ash(coefs₁["group (sick)"][n_days₁ .== i], m=50), hist=false, label="k=$i", xtickfontsize=6, lw=lw, la=la)
        plot!(p5, ash(mses₁[n_days₁ .== i], m=50), hist=false, label="k=$i", xtickfontsize=6, lw=lw, la=la)
    end
    if i in d₂ && count(==(i), n_days₂) > 1
        plot!(p2, ash(coefs₂["status (unhealthy)"][n_days₂ .== i], m=50), hist=false, label="k=$i", xtickfontsize=6, lw=lw, la=la)
        plot!(p4, ash(coefs₂["group (sick)"][n_days₂ .== i], m=50), hist=false, label="k=$i", xtickfontsize=6, lw=lw, la=la)
        plot!(p6, ash(mses₂[n_days₂ .== i], m=50), hist=false, label="k=$i", xtickfontsize=6, lw=lw, la=la)
    end
end
p7 = bar(d₁, c₁, orientation=:h, legend=false, title="Best k", yticks=d₁)
p8 = bar(d₂, c₂, orientation=:h, legend=false, title="Best k", yticks=d₂)
px = plot(p1, p3, p5, p7, layout=@layout [[a;b;c] d])
py = plot(p2, p4, p6, p8, layout=@layout [[a;b;c] d])
savefig(px, "mdi1_estimate.png")
savefig(py, "mdi2_estimate.png")

## ----- Find confidence intervals for β₄ and β₅ -----
struct ConfidenceIntervals{T<:AbstractFloat}
    μ::T
    σ::T
    percent90::NamedTuple{(:lower, :upper), Tuple{T,T}}
    percent95::NamedTuple{(:lower, :upper), Tuple{T,T}}
    percent99::NamedTuple{(:lower, :upper), Tuple{T,T}}
end

function ConfidenceIntervals(μ::T, σ::T, dist::Type{S} = Normal) where 
                            {T<:AbstractFloat, S<:Distribution}
    D = dist(μ, σ)
    μₜ = mean(D)            # Transformed mean
    σₜ = std(D)             # Transformed standard deviation
    percent90 = (lower = quantile(D, 0.05), upper = quantile(D, 0.95))
    percent95 = (lower = quantile(D, 0.025), upper = quantile(D, 0.975))
    percent99 = (lower = quantile(D, 0.005), upper = quantile(D, 0.995))
    return ConfidenceIntervals(μₜ, σₜ, percent90, percent95, percent99)
end

μ₄₁ = mean(coefs₁["status (unhealthy)"][n_days₁ .== 1]); μ₅₁ = mean(coefs₁["group (sick)"][n_days₁ .== 1])
σ₄₁ = std(coefs₁["status (unhealthy)"][n_days₁ .== 1]); σ₅₁ = std(coefs₁["group (sick)"][n_days₁ .== 1])
μ₄₂ = mean(coefs₂["status (unhealthy)"][n_days₂ .== 1]); μ₅₂ = mean(coefs₂["group (sick)"][n_days₂ .== 1])
σ₄₂ = std(coefs₂["status (unhealthy)"][n_days₂ .== 1]); σ₅₂ = std(coefs₂["group (sick)"][n_days₂ .== 1])
CI₄₁ = ConfidenceIntervals(μ₄₁, σ₄₁)
CI₅₁ = ConfidenceIntervals(μ₅₁, σ₅₁)
CI₄₂ = ConfidenceIntervals(μ₄₂, σ₄₂)
CI₅₂ = ConfidenceIntervals(μ₅₂, σ₅₂)
# === How β₄ and β₅ predicts difference in milk yield between healthy vs sick cows ===
# --- β₄ and β₅ when MDi threshold = 1.4 ---
CI1₁ = ConfidenceIntervals(μ₅₁, σ₅₁, LogNormal)                     # healthy cows in sick vs healthy group   [99%: (0.964, 0.969)]
CI2₁ = ConfidenceIntervals(μ₄₁, σ₄₁, LogNormal)                     # healthy vs unhealthy in sick group      [99%: (0.880, 0.887)]
CI3₁ = ConfidenceIntervals(μ₄₁ + μ₅₁, √(σ₄₁^2 + σ₅₁^2), LogNormal)  # healthy in healthy vs unhealthy in sick [99%: (0.850, 0.858)]
# --- β₄ and β₅ when MDi threshold = 1.8 ---
CI1₂ = ConfidenceIntervals(μ₅₂, σ₅₂, LogNormal)                     # healthy cows in sick vs healthy group   [99%: (0.971, 0.977)]
CI2₂ = ConfidenceIntervals(μ₄₂, σ₄₂, LogNormal)                     # healthy vs unhealthy in sick group      [99%: (0.877, 0.888)]
CI3₂ = ConfidenceIntervals(μ₄₂ + μ₅₂, √(σ₄₂^2 + σ₅₂^2), LogNormal)  # healthy in healthy vs unhealthy in sick [99%: (0.853, 0.866)]

## ----- Model fit without accounting cow status -----
fm = @formula(logyield ~ 1 + log(dinmilk) + dinmilk + lactnum + status + group + normalized_GPTAM)
mse_train₁, mse_test₁, coefs₁, model₁ = categorize_and_fit(df_healthy₁, df_sick₁, genomic_info, 1, :mdi, mdi_threshold₁, fm=fm, split_by=:random, train_size=9, test_size=1, random_state=1234)
mse_train₂, mse_test₂, coefs₂, model₂ = categorize_and_fit(df_healthy₁, df_sick₂, genomic_info, 1, :mdi, mdi_threshold₂, fm=fm, split_by=:random, train_size=9, test_size=1, random_state=1234)
# Model predictions for average cows
dfₕ = DataFrame(dinmilk = repeat(1:200,3), lactnum = repeat([1,2,3],inner=200), group="healthy", status="healthy", normalized_GPTAM=0)
dfₛ = DataFrame(dinmilk = repeat(1:200,3), lactnum = repeat([1,2,3],inner=200), group="sick", status="unhealthy", normalized_GPTAM=0)
dfₕ[!,:yield₁] = predict(model₁, dfₕ) |> x -> exp.(x)
dfₛ[!,:yield₁] = predict(model₁, dfₛ) |> x -> exp.(x)
dfₕ[!,:yield₂] = predict(model₂, dfₕ) |> x -> exp.(x)
dfₛ[!,:yield₂] = predict(model₂, dfₛ) |> x -> exp.(x)
# Build plots
p1 = plot(1:200, dfₕ.yield₁[dfₕ.lactnum .== 1], label="Lactation 1, Healthy", lw=2, lc=:blue, ls=:solid, title="MDi=$mdi_threshold₁", legend=:bottomright, ylims=(0,150), xlabel="Days in milk", ylabel="Yield")
plot!(p1, 1:200, dfₕ.yield₁[dfₕ.lactnum .== 2], label="Lactation 2, Healthy", lw=2, lc=:blue, ls=:dash)
plot!(p1, 1:200, dfₕ.yield₁[dfₕ.lactnum .== 3], label="Lactation 3, Healthy", lw=2, lc=:blue, ls=:dot)
plot!(p1, 1:200, dfₛ.yield₁[dfₛ.lactnum .== 1], label="Lactation 1, Sick", lw=2, lc=:red, ls=:solid)
plot!(p1, 1:200, dfₛ.yield₁[dfₛ.lactnum .== 2], label="Lactation 2, Sick", lw=2, lc=:red, ls=:dash)
plot!(p1, 1:200, dfₛ.yield₁[dfₛ.lactnum .== 3], label="Lactation 3, Sick", lw=2, lc=:red, ls=:dot)

p2 = plot(1:200, dfₕ.yield₂[dfₕ.lactnum .== 1], label="Lactation 1, Healthy", lw=2, lc=:blue, ls=:solid, title="MDi=$mdi_threshold₂", legend=:bottomright, ylims=(0,150), xlabel="Days in milk", ylabel="Yield")
plot!(p2, 1:200, dfₕ.yield₂[dfₕ.lactnum .== 2], label="Lactation 2, Healthy", lw=2, lc=:blue, ls=:dash)
plot!(p2, 1:200, dfₕ.yield₂[dfₕ.lactnum .== 3], label="Lactation 3, Healthy", lw=2, lc=:blue, ls=:dot)
plot!(p2, 1:200, dfₛ.yield₂[dfₛ.lactnum .== 1], label="Lactation 1, Sick", lw=2, lc=:red, ls=:solid)
plot!(p2, 1:200, dfₛ.yield₂[dfₛ.lactnum .== 2], label="Lactation 2, Sick", lw=2, lc=:red, ls=:dash)
plot!(p2, 1:200, dfₛ.yield₂[dfₛ.lactnum .== 3], label="Lactation 3, Sick", lw=2, lc=:red, ls=:dot)
p3 = plot(p1,p2, layout=(1,2))
savefig(p3, "yield_curve.png")

## ----- Average difference in yield between sick vs non-sick -----
list₁.coef["overall"] = list₁.coef["status (unhealthy)"] + list₁.coef["group (sick)"]
list₂.coef["overall"] = list₂.coef["status (unhealthy)"] + list₂.coef["group (sick)"]
p1 = plot(dinmilk_range, exp.(list₁.coef["group (sick)"]), 
          label=L"\frac{healthy_{sick}}{healthy_{healthy}}", 
          title="Yield comparison when MDi threshold=$mdi_threshold₁",
          ylabel="sick/healthy group ratio", xlabel="k days after event",
          fillcolor=:blue, fillrange=0.7, fillalpha=0.2, ylims=(0.7,1.05), legend=:outerright)
plot!(p1, dinmilk_range, exp.(list₁.coef["overall"]), 
      label=L"\frac{unhealthy_{sick}}{healthy_{healthy}}",
      fillcolor=:red, fillrange=0.7, fillalpha=0.2)
p2 = plot(dinmilk_range, exp.(list₂.coef["group (sick)"]), 
          label=L"\frac{healthy_{sick}}{healthy_{healthy}}", 
          title="Yield comparison when MDi threshold=$mdi_threshold₂",
          ylabel="sick/healthy group ratio", xlabel="n days after event",
          fillcolor=:blue, fillrange=0.7, fillalpha=0.2, ylims=(0.7,1.05), legend=:outerright)
plot!(p2, dinmilk_range, exp.(list₂.coef["overall"]), 
      label=L"\frac{unhealthy_{sick}}{healthy_{healthy}}",
      fillcolor=:red, fillrange=0.7, fillalpha=0.2)
p3 = plot(p1, p2, layout = (2,1))
savefig(p3, "yield_compare.png")