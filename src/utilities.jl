using Statistics, Impute

"""
  minmax_teat(D, i[, teats=[:rr,:rf,:lr,:lf], min=true])
  minmax_teat(a, b, c, d[, teats=[:rr,:rf,:lr,:lf], min=true])

Finds the teat with the minimum/maximum conductivity for a given time point.
"""
function minmax_teat end

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

function minmax_teat(D::DataFrame, i::Int, teats=[:rr,:rf,:lr,:lf]; min::Bool=true)
  return minmax_teat(D.condrr[i], D.condrf[i], D.condlr[i], D.condlf[i], teats, min=min)
end

function minmax_teat(a::T,b::T,c::T,d::T, 
  teats=[:rr,:rf,:lr,:lf];
  min::Bool= true) where T <: Number

  x = [a,b,c,d]
  min ? teats[argmin(x)] : teats[argmax(x)]
end