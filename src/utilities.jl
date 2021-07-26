using Statistics, Impute

"""
  minmax_cond_teat(D, i[, teats=[:lf,:lr,:rf,:rr], min=true])

Finds the teat with the minimum/maximum conductivity for a given time point.

*See also:* `minmax_teat`, `minmax_yield_teat`
"""
function minmax_cond_teat end

"""
  minmax_yield_teat(D, i[, teats=[:lf,:lr,:rf,:rr], min=true])

Finds the teat with the minimum/maximum yield for a given time point.

*See also:* `minmax_teat`, `minmax_cond_teat`
"""
function minmax_yield_teat end

function rolling_average(x::AbstractVector, n::Int)
  @assert 1<=n<=length(x)
  l = length(x)
  y = [mean(x[(i-n+1):i]) for i in n:l]
  return [x[1:n-1];y]
end

function zero_to_missing(x)
  if !ismissing(x) && x == 0
    return missing
  else
    return x
  end
end

function linear_interpolation(x::AbstractVector)
  y = zero_to_missing.(x)
  y = Impute.interp(y) |> Impute.locf()
  return convert(Vector{Float64}, y)
end

function minmax_cond_teat(D::DataFrame; min::Bool=true)
  C = hcat(D.condlf, D.condlr, D.condrf, D.condrr)
  T = [minmax_teat(C[i,:]; min=min) for i in 1:size(C,1)]
  return T
end

function minmax_yield_teat(D::DataFrame; min::Bool=true)
  C = hcat(D.yieldlf, D.yieldlr, D.yieldrf, D.yieldrr)
  T = [minmax_teat(C[i,:]; min=min) for i in 1:size(C,1)]
  return T
end                      

function minmax_teat(x::T; min::Bool= true) where T <: AbstractVector{Union{Missing, Float64}}
  teats=[:lf,:lr,:rf,:rr]
  x = skipmissing(x)
  min ? teats[argmin(x)] : teats[argmax(x)]
end

function minmax_cond(D::DataFrame; min::Bool=true)
  y = Vector{Union{Missing, Float64}}(undef, nrow(D))
  for i in 1:nrow(D)
    x = skipmissing([D.condlf[i], D.condlr[i], D.condrf[i], D.condrr[i]])
    if min
      y[i] = minimum(x)
    else
      y[i] = maximum(x)
    end
  end
  return y
end

function minmax_yield(D::DataFrame; min::Bool=true)
  y = Vector{Union{Missing, Float64}}(undef, nrow(D))
  for i in 1:nrow(D)
    x = skipmissing([D.yieldlf[i], D.yieldlr[i], D.yieldrf[i], D.yieldrr[i]])
    if min
      y[i] = minimum(x)
    else
      y[i] = maximum(x)
    end
  end
  return y
end