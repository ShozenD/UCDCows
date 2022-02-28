function gridsearch(df_healthy::DataFrame,
                    df_sick::DataFrame,
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
    mae_list = Dict("train" => Vector{Float64}(undef, len),
                    "test" => Vector{Float64}(undef, len))
    mpe_list = Dict("train" => Vector{Float64}(undef, len),
                    "test" => Vector{Float64}(undef, len))
    coef_list = Dict("status (unhealthy)" => Vector{Float64}(undef, len),
                     "group (sick)" => Vector{Float64}(undef, len))
    for (i, n) in enumerate(list)
        mae_train, mae_test, mpe_train, mpe_test, coefs, _ = categorize_and_fit(
            df_healthy, df_sick, n, criterion, mdi_threshold, 
            fm=fm, split_by=split_by, split_date=split_date, train_size=train_size, 
            test_size=test_size, random_state=random_state
        )
        mae_list["train"][i] = mae_train
        mpe_list["train"][i] = mpe_train
        mae_list["test"][i] = mae_test
        mpe_list["test"][i] = mpe_test
        coef_list["status (unhealthy)"][i] = coefs["status (unhealthy)"]
        coef_list["group (sick)"][i] = coefs["group (sick)"]
    end
    # Return the best parameter along with its results
    idx = argmin(mpe_list["test"])
    best_n = list[idx]
    best_mae = (train = mae_list["train"][idx], test = mae_list["test"][idx])
    best_mpe = (train = mpe_list["train"][idx], test = mpe_list["test"][idx])
    best_coef = Dict("status (unhealthy)" => coef_list["status (unhealthy)"][idx],
                     "group (sick)" => coef_list["group (sick)"][idx])
    return (n = best_n, mae = best_mae, mpe = best_mpe, coef = best_coef), 
           (mae = mae_list, mpe = mpe_list, coef = coef_list)
end

function categorize_and_fit(df_healthy::DataFrame, 
                            df_sick::DataFrame,
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

    # ----- Model Fitting -----
    df[df.status .== "sick", :status] .= "unhealthy"
    # Split model to train-test sets
    df_train, df_test = train_test_split(df, split_by, split_date=split_date, train_size=train_size,
                                         test_size=test_size, random_state=random_state)
    model = fit(LinearModel, fm, df_train)

    # ----- Error Computation and Coefficient Extraction -----
    coefs = Dict("status (unhealthy)" => coef(model)[6], "group (sick)" => coef(model)[7])
    logyield_pred = predict(model, df_train) |> x -> convert.(Float64, x)
    mae_train = mean(abs.(exp.(logyield_pred) - df_train.yield))
    mpe_train = compute_mpe(logyield_pred, df_train.logyield, transform = :log)
    logyield_pred = predict(model, df_test) |> x -> convert.(Float64, x)
    mae_test = mean(abs.(exp.(logyield_pred) - df_test.yield))
    mpe_test = compute_mpe(logyield_pred, df_test.logyield, transform = :log)
    return (mae_train, mae_test, mpe_train, mpe_test, coefs, model)
end

"""
    train_test_split(df, by[; split_date, train_size, test_size, random_state])

# Arguments
- `df::DataFrame`
- `by::Symbol`

# Keyword Arguments
- `split_date::Union{Date, Nothing}`
- `train_size::Union{Real, Nothing}`
- `test_size::Union{Real, Nothing}`
- `random_state::Union{Integer, Nothing}`
"""
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

@doc raw"""
    compute_mpe(ŷ, y[; transform])

Computes Mean Proportion Error (MPE), defined as:

``MPE =  \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y_i}| / y_i``

# Arguments
- `ŷ::AbstractVector{T} where T<:Number`: Predicted values.
- `y::AbstractVector{T} where T<:Number`: True values.

# Keyword Arguments
- `transform::Symbol`: (Default: `:none`) Transform type of ŷ and y. Accepted values are
  `:none` and `:log`.

# Returns
- `mpe::T`: Mean Proportion Error.
"""
function compute_mpe(ŷ::AbstractVector, y::AbstractVector; transform::Symbol = :none) where T<:Number
    @assert transform ∈ [:none, :log]
    @assert length(ŷ) == length(y)
    if transform == :log
        ŷ = exp.(ŷ)
        y = exp.(y)
    end
    return mean(abs.(y - ŷ) ./ y)
end