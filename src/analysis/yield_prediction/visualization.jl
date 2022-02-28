function true_vs_fitted(data::DataFrame, model::StatsModels.TableRegressionModel; 
                        split_date::Union{Date, Nothing} = nothing,
                        mdi_threshold::AbstractFloat = 1.4)
    @assert (length ∘ unique)(data.id) == 1
    df = copy(data)
    # Add "status" amd "group" columns if necessary
    if "status" ∉ names(data)
        df[!, :status] .= "healthy"
        df[df.mdi .≥ mdi_threshold, :status] .= "unhealthy"
    end
    if "group" ∉ names(data)
        df[!, :group] .= "unhealthy" ∈ df.status ? "sick" : "healthy"
    end
    # Obtain true and fitted values
    ŷ = predict(model, df) |> x -> exp.(x)
    y = df.yield
    # Build plot
    cow_id = unique(df.id)[1]
    plt = plot(df.dinmilk, y, shape = :circle, label = "True yield", title = "Cow $cow_id")
    plot!(plt, df.dinmilk, ŷ, shape = :circle, label = "Predicted yield", legend = :bottomright)
    if !isnothing(split_date)
        split_dinmilk = Dates.value(split_date - minimum(df.date) + Day(1))
        vline!(plt, [split_dinmilk], label = "", color = :red)
    end
    return plt
end

function plot_error(rng::UnitRange, data, metric::Symbol = :mpe, linecolor::Symbol = :blue)
    metric_str = (uppercase ∘ String)(metric)
    plt = plot(xlabel = "k days after event", ylabel = metric_str, 
               title = "Model $metric_str by k days", titlefontsize = 10, 
               xguidefontsize = 8, yguidefontsize = 8, legend=:topright)
    return plot_error!(plt, rng, data, metric)
end
function plot_error!(plt, rng::UnitRange, data, metric::Symbol, linecolor::Symbol)
    plot!(plt, rng, data[metric]["train"], label = "(train) vs mid-high MDi cows", lc = linecolor, ls = :dash)
    plot!(plt, rng, data[metric]["test"], label = "(test) vs mid-high MDi cows", lc = linecolor, ls = :solid)
    return plt
end