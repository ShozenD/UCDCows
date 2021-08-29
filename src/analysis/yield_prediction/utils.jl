using MixedModels,
      GLM,
      Statistics,
      DataFrames,
      DataFramesMeta,
      CategoricalArrays,
      Plots,
      Statistics,
      Dates

# ========== Structs ==========
"""
    CowData(id, healthy_data, sick_data)

Struct containing the healthy and sick aggregated data of a cow.

# Arguments
- `id::CategoricalValue{Int64, UInt32}`: Cow ID.
- `healthy_data::DataFrame`: Healthy data of the cow.
- `healthy_count::Integer`: Number of healthy data.
- `sick_data::DataFrame`: Unhealthy data of the cow.
- `sick_count::Integer`: Number of unhealthy_data.
- `total_count::Integer`: Total number of data for the cow.
- `ratio::AbstractFloat`: Ratio between `sick_count` and `healthy_count`.
                          Computed using `sick_count`/`healthy_count`
"""
struct CowData
    id::CategoricalValue{Int64, UInt32}
    healthy_data::DataFrame
    healthy_count::Integer
    sick_data::DataFrame
    sick_count::Integer
    total_count::Integer
    ratio::AbstractFloat
    # Inner constructor
    function CowData(id, healthy_data, sick_data)
        healthy_count = nrow(healthy_data)
        sick_count = nrow(sick_data)
        total_count = healthy_count + sick_count
        ratio = sick_count / healthy_count
        return new(id, healthy_data, healthy_count, sick_data, sick_count, 
                   total_count, ratio)
    end
end

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
        if dinmilk_1 < minimum(df.date) || dinmilk_N > maximum(df.date)
            continue
        end
        # --- Check number of days missed in first N days ---
        # Collect all days in milk within the first N days
        dinmilk = unique(group.date) |> x -> x[x .<= dinmilk_N]
        # Compute number of days with no records in the first N days (number of days out
        # with mastitis in the first N days)
        sick_days = N - length(dinmilk)
        if sick_days > 20
            println(unique(group.date))
        end
        # --- Add cow data into filtered data ---
        cow_data = vcat(cow_data, group)
        # --- Add summary of current cow ---
        temp = DataFrame(id = [id], 
                         healthy = [sick_days == 0 ? true : false],
                         sick_days = [sick_days],
                         first_day_in_milk = [dinmilk_1],
                         lactation_number = unique(group.lactnum))
        summary = vcat(summary, temp)
    end
    return cow_data, summary
end

# ----- Split data frame to healthy and unhealthy data -----
"""
    splitbyperiod(df, n_days, criterion)

Splits data into healthy and unhealthy data based on `n_days` and 
`criterion`. Specifically, we consider `n_days` days before and after an 
occurrence of a `criterion` as unhealthy data.

# Arguments
- `df::DataFrame`: Cow data frame
- `n_days::Integer`: (Default: 7) Number of days before and after an
                        occurrence of a criterion to be considered unhealthy.
- `criterion::Symbol`: (Default: :sick) The criterion used to determine
                        whether a cow is unhealthy. There are 2 possible
                        criterions: `:sick` (when a cow has mastitis) and `:mdi` 
                        (when a cow has MDi ≥ 1.4).

# Returns
- `df_healthy::DataFrame`: Data frame of healthy data.
- `df_unhealthy::DataFrame`: Data frame of unhealthy data.
- `by_cows::Vector{CowData}`: List of data for each cow.
"""
function splitbyperiod(df::DataFrame,
                       n_days::Integer = 7,
                       criterion::Symbol = :sick)
    @assert criterion in [:sick, :mdi]
    @assert n_days > 0

    # Group data frame by animal number / id
    by_id = groupby(df, :id)
    # Extract healthy and unhealthy data for each cow
    df_healthy = DataFrame()
    df_sick = DataFrame()
    by_cows = Vector{CowData}()
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
        sick_days, unhealthy_days = Vector{Date}(), Vector{Date}()
        for dt in start_:Day(1):end_
            # `dt` is sick day if yield data for the date `dt` is not recorded.
            # If `criterion` is `:mdi`, `dt` is also considered a sick day if
            # cow has MDi ≥ 1.4 on `dt`.
            if issickday(dt, dts, id_group, criterion)
                push!(sick_days, dt)
            # `dt` is unhealthy day if it is `n_days` days before or after a 
            # sick day.
            elseif isunhealthyday(dt, sick_days, n_days)
                push!(unhealthy_days, dt)
            end
        end
        # ----------------------------------------------------------------------
        # Extract healthy and unhealthy days for a particular cow
        healthy_group = @linq id_group |> where(:date .∉ Ref(unhealthy_days))
        sick_group = @linq id_group |> where(:date .∈ Ref(unhealthy_days))
        # Group yield values by day --------------------------------------------
        healthy_group = groupby(healthy_group, :dinmilk) |>
                        grps -> combine(grps,
                                        :id,
                                        :lactnum,
                                        :yield => sum => :yield,
                                        :date) |>
                        comb -> unique(comb, :dinmilk)
        sick_group = groupby(sick_group, :dinmilk) |>
                     grps -> combine(grps,
                                     :id,
                                     :lactnum,
                                     :yield => sum => :yield,
                                     :date) |>
                     comb -> unique(comb, :dinmilk)
        # Add :logyield field into healthy_group and sick_group
        healthy_group[!, :logyield] = log.(healthy_group.yield .+ 1)
        sick_group[!, :logyield] = log.(sick_group.yield .+ 1)
        cow_data = CowData(id, healthy_group, sick_group)
        # Append data frames and list of cow data ------------------------------
        df_healthy = vcat(df_healthy, healthy_group)
        df_sick = vcat(df_sick, sick_group)
        push!(by_cows, cow_data)
    end
    return df_healthy, df_sick, by_cows
end

"""
    issickday(dt, dts, df, criterion)

Returns `true` if `dt` is a sick day.

# Arguments
- `dt::Date`: Date to be checked.
- `dts::Vector{Dates}`: List of dates that animal is not sick with mastitis.
- `df::DataFrame`: DataFrame containing date and MDi records for 1 animal.
- `criterion::Symbol`: Criterion to determine if animal is sick. There are 2 
                       possible criterions: `:sick` (when a cow has mastitis) 
                       and `:mdi` (when a cow has MDi ≥ 1.4).

# Returns
- `::Bool`: Whether animal is sick on `dt`.
"""
function issickday(dt::Date, 
                   dts::Vector{Date},
                   df::DataFrame,
                   criterion::Symbol)
    @assert criterion in [:sick, :mdi]
    # If cow has mastitis, it is sick regardless
    if dt ∉ dts
        return true
    # If `criterion` is `:mdi`, check `df` to see if cow has MDi levels ≥ 1.4
    # on `dt`
    elseif criterion == :mdi
        df_dt = @subset(df, :date .== dt)
        # Returns true (sick day) if MDi ≥ 1.4 at any time of the day.
        return any(skipmissing(df_dt.mdi) .>= 1.4)
    end
    # Otherwise, cow is not sick.
    return false
end

"""
    isunhealthyday(dt, dts, n_days)

Returns `true` if `dt` is considered an unhealthy day. Unhealthy days are 
defined as days where an animal is `n_days` before or after any sick days.

# Arguments
- `dt::Date`: Date to be checked.
- `dts::Vector{Dates}`: List of sick days.
- `n_days::Integer`: Number of days used to determine if animal is sick or not.

# Returns
- `::Bool`: Whether animal is unhealthy on `dt`.
"""
function isunhealthyday(dt::Date,
                        dts::Vector{Date},
                        n_days::Integer)
    # `dt` is considered an unhealthy day if it is `n_days` days before or
    # after any dates within `dts`
    for day in dts
        if (dt >= daysminusN(day, n_days)) & (dt <= daysplusN(day, n_days))
            return true
        end
    end
    # Otherwise, `dt` is not an unhealthy day.
    return false
end

"""
    daysminusN(dt, n)

Compute `n` days before `dt`.

# Arguments
- `dt::Date`: Date.
- `n::Integer`: Number of days.

# Returns
- `::Date`: `n` days before `dt`.
"""
daysminusN(dt::Date, n::Integer) = dt - Day(n)

"""
    daysplusN(dt, n)

Compute `n` days after `dt`.

# Arguments
- `dt::Date`: Date.
- `n::Integer`: Number of days.

# Returns
- `::Date`: `n` days after `dt`.
"""
daysplusN(dt::Date, n::Integer) = dt + Day(n)

# ----- Compare curves of healthy vs unhealthy cows -----
"""
    makedataset(df)

Builds dataset that contains the rows from day 1 to 300 of days in milk 
`dinmilk`. The purpose of this function is to produce a data frame that can be
entered into the `predict()` function from MixedModel.jl and predict the entire
sequence from day 1 to 300.

# Arguments
- `df::DataFrame`: Data frame of records for a particular cow.

# Returns
- `::DataFrame`: Data frame of 300 rows, where each row correspond to 1 day in
                 milk.
"""
function makedataset(df::DataFrame)
    # Get min and max date
    min_date, max_date = minimum(df.date), maximum(df.date)
    min_day, max_day = minimum(df.dinmilk), maximum(df.dinmilk)
    # Get start (day 1) and end (day 300) dates
    start_date = min_date - Day(min_day) + Day(1)
    end_date = max_date - Day(max_day) + Day(300)
    # Build the data frame consisting day 1 to day 300
    df_seq = DataFrame(dinmilk = Vector{Int64}(),
                       date = Vector{Date}())
    for (dinmilk, date) in enumerate(start_date:Day(1):end_date)
        # Add row for particular day and date
        push!(df_seq, [dinmilk, date])
    end
    # Add :id and :lactnum columns
    df_seq[!, :id] = repeat([df.id[1]], 300)
    df_seq[!, :lactnum] = repeat([df.lactnum[1]], 300)
    return df_seq
end

function make_prediction(healthy_model,
                         sick_model,
                         df::DataFrame)
    pred_healthy = predict(healthy_model, df)
    pred_sick = predict(sick_model, df)
    # Return exponential of predicted values since predicted values are 
    # log(yield)
    return exp.(pred_healthy), exp.(pred_sick)
end

"""
    compare_healthy_vs_sick(n_days, criterion, df)

Fit and compare the curve of healthys and sick cows. Here, a cow is defined to
be sick if it is `n_days` before or after an occurrence of `criterion`, where
`criterion` is either high MDi or contracting mastitis itself.

# Arguments
- `n_days::Integer`: Number of days before and after `criterion` where a cow
                     is still considered to be sick.
- `criterion::Symbol`: The criterion to determine if a cow is sick. Accepted
                       values are `:mdi` (MDi ≥ 1.4) and `:sick` (mastitis).
- `df::DataFrame`: Data frame containing the `:dinmilk`, `:lactnum`,
                   `:logyield` of each cow.

# Returns
`::Array`: A matrix of the expected difference in milk production in each
           lactation number. Column `j` corresponds to the lactation number `j`
           while row `i` corresponds to the `i`-th day in milk.
"""
function compare_healthy_vs_sick(n_days::Integer,
                                 criterion::Symbol,
                                 df::DataFrame)
    # ----- Split data into healthy and sick data frames -----
    healthy_data, sick_data, _ = splitbyperiod(df, n_days, criterion)
    # ----- Build model -----
    fm = @formula(logyield ~ 1 + log(dinmilk) + dinmilk + lactnum)
    model_healthy = fit(LinearModel, fm, healthy_data)
    model_sick = fit(LinearModel, fm, sick_data)
    # ----- Get unique lactation numbers for healthy and sick data -----
    healthy_lactnum = Set(healthy_data.lactnum)
    sick_lactnum = Set(sick_data.lactnum)
    unique_lactnum = intersect(healthy_lactnum, sick_lactnum)
    # ----- Predict curve for each lactation number -----
    diff = zeros(300, 7)
    for lactnum in unique_lactnum
        # Build data frame
        df_seq = DataFrame(dinmilk = 1:300, lactnum = lactnum)
        # Predict curve
        pred_healthy, pred_sick = make_prediction(model_healthy, model_sick, df_seq)
        # Compute expected difference (healthy - sick) for each day
        diff[:, convert(Int64, lactnum)] = pred_healthy - pred_sick
    end
    return diff
end

"""

"""
function gridsearch_expected_difference(df::DataFrame,
                                        days_range::Vector{<:Integer})
    # Define dictionary to track mean expected differences
    expected_diff = Dict{String, Dict}()
    # Loop through criteria and days_range and get expected differences for
    # each
    for criterion in [:mdi, :sick]
        # Define dictionary to track mean expected differences of current
        # criterion
        expected_diff[String(criterion)] = Dict{Int64, Vector{Float64}}()
        for n_days in days_range
            # Compute the expected differences in healthy vs sick model
            diff = compare_healthy_vs_sick(n_days, criterion, df)
            expected_diff[String(criterion)][n_days] = reshape(mean(diff, dims=1),:)
        end
    end
    return expected_diff
end

# ----- Plots -----
"""

"""
function plot_differences(results::Dict, days_range::Vector{Int64})
    # Make empty plot
    p = plot(xlabel="Number of days before and after 'sick' where considered unhealthy",
             ylabel="Healthy curve - Unhealthy curve",
             xlim=(minimum(days_range)-1, maximum(days_range)+1),
             ylim=(0,60),
             xticks=minimum(days_range):1:maximum(days_range),
             yticks=0:10:60)
    for criterion in ["mdi", "sick"]
        # Scatter points are red for MDi criterion and green for mastitis criterion
        color = criterion == "mdi" ? :red : :green
        # Vectors y and x are used to collect the expected differences across each n_days
        # of the same criterion
        y = Vector{Float64}()
        x = Vector{Int64}()
        for n_days in days_range
            # To avoid plotting zeros (cases where there were no sick data and therefore
            # no comparison), remove them from list
            filtered_diff = filter(x -> x≠0, results[criterion][n_days])
            append!(y, filtered_diff)
            append!(x, repeat([n_days], length(filtered_diff)))
        end
        # Draw plot
        scatter!(p, x, y, lc=color, labels=criterion)
    end
    return p
end

"""
    plot_yield_byhealth(healthy_data, sick_data[, lactnum=1; include_curve, formula])

Plots the yield by days in milk for each cow based on its health status and lactation 
number.

# Arguments
- `healthy_data::DataFrame`: Healthy data.
- `sick_data::DataFrame`: Unhealthy data.
- `lactnum::Integer`: (Default: `1`) Lactation number.
- `include_curve::Bool`: (Default: `true`) Whether or not to include prediction curves in 
                         the plots.
- `formula::FormulaTerm`: (Default: `@formula(logyield ~ 1 + log(dinmilk) + dinmilk + 
                          lactnum`)) Formula used to fit the linear model to plot the curve.

# Returns:
- `::Plots.Plot{Plots.GRBackend}`: A plot containing the yield for healthy and sick cows.
"""
function plot_yield_byhealth(healthy_data::DataFrame,
                             sick_data::DataFrame,
                             lactnum::Integer = 1;
                             include_curve::Bool = true,
                             formula::FormulaTerm = @formula(logyield ~ 1 + log(dinmilk) + dinmilk + lactnum))
    # Set up plot for yield of healthy and sick cows
    healthy_plot = plot(xlabel="Days In Milk", ylabel="Yield", title="Yield for Healthy Cows",
                        xlim=(1,400), ylim=(0,250), xticks=0:100:400, yticks=0:50:250)
    sick_plot = plot(xlabel="Days In Milk", ylabel="Yield", title="Yield for Sick Cows",
                     xlim=(1,400), ylim=(0,250), xticks=0:100:400, yticks=0:50:250)
    # Filter data frame into data with lactation number `lactnum`
    hdata = @subset(healthy_data, :lactnum .== lactnum)
    sdata = @subset(sick_data, :lactnum .== lactnum)
    # Group the data frames by cows
    hdata_byid = groupby(hdata, :id)
    sdata_byid = groupby(sdata, :id)
    # Plot points for each cow in different color
    for (_, group) in pairs(hdata_byid)
        group = DataFrame(group)
        scatter!(healthy_plot, group.dinmilk, group.yield, legend=false)
    end
    for (_, group) in pairs(sdata_byid)
        group = DataFrame(group)
        scatter!(sick_plot, group.dinmilk, group.yield, legend=false)
    end
    # Add model curve if include_curve = true
    if include_curve
        # Fit models
        model_healthy = fit(LinearModel, formula, healthy_data)
        model_sick = fit(LinearModel, formula, sick_data)
        # Make data frame for predictions
        df_seq = DataFrame(dinmilk = 1:400,
                           lactnum = categorical(repeat([lactnum], 400)))
        # Make predictions
        pred_healthy, pred_sick = make_prediction(model_healthy, model_sick, df_seq)
        # Plot predictions
        plot!(healthy_plot, 1:400, pred_healthy, lc=:black, lw=3)
        plot!(sick_plot, 1:400, pred_sick, lc=:black, lw=3)
    end
    # Combine both plots side by side
    return plot(healthy_plot, sick_plot, layout=(1,2))
end

# ----- Model Analysis -----
function print_modelstatistics(model::LinearMixedModel, 
                               train_data::DataFrame,
                               test_data::DataFrame)
    
    # model statistics
    println("Model Statistics:")
    display(model)

    # train data
    degree_of_freedom = dof(model)
    n_observation = nobs(model)
    pred = predict(model, train_data)
    error = train_data.logyield - pred
    train_sse = sum(error .^ 2)
    train_mse = train_sse / dof_residual(model)
    train_mpwrss = pwrss(model) / dof_residual(model)
    println("\nTrain data:")
    println("Degree of freedom: $degree_of_freedom")
    println("Number of observations: $n_observation")
    println("Mean squared error: $(round(train_mse, digits=4))")
    println("Mean squared penalized, weighted residual: $(round(train_mpwrss, digits=4))")

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