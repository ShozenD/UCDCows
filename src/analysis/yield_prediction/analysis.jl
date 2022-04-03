struct ConfidenceInterval{T₁<:Integer, T₂<:AbstractFloat}
    percent::T₁
    lower::T₂
    mean::T₂
    upper::T₂

    function ConfidenceInterval{T₁, T₂}(percent, lower, mean, upper) where {T₁<:Integer, T₂<:AbstractFloat}
        if 1 ≤ percent ≤ 100 
            return new(percent, lower, mean, upper)
        else
            error("Percent range needs to be between 1 and 100.")
        end
    end
end

struct SummaryStatistics{T₁<:Integer, T₂<:AbstractFloat}
    β₁::ConfidenceInterval{T₁,T₂}
    β₀::ConfidenceInterval{T₁,T₂}
    β₂::ConfidenceInterval{T₁,T₂}
    β₃::Union{Missing, ConfidenceInterval{T₁,T₂}}
    MAETrain::ConfidenceInterval{T₁,T₂}
    MAETest::ConfidenceInterval{T₁,T₂}
    MPETrain::ConfidenceInterval{T₁,T₂}
    MPETest::ConfidenceInterval{T₁,T₂}
end

function createconfidenceinterval(list::Vector{T}, α::T) where T<:AbstractFloat
    @assert 0 < α < 1

    _mean_ = mean(list)
    _std_ = std(list)

    percentile = Int8((1 - α) * 100)
    multiplier = quantile(Normal(), 1-α/2)
    _lower_, _upper_ = _mean_ .+ [-1, 1] * multiplier * _std_
    return ConfidenceInterval{Int8, T}(percentile, _lower_, _mean_, _upper_)
end

function createsummarystatistics(df::DataFrame, α::T) where T<:AbstractFloat
    @assert 0 < α < 1
    @assert "group" ∈ names(df)
    @assert (length ∘ unique)(df.group) == 1
    @assert df.group[1] ∈ ["sick", "healthy"]

    _group_ = df.group[1]
    _β₀_ = createconfidenceinterval(df.β₀, α)
    _β₁_ = createconfidenceinterval(df.β₁, α)
    _β₂_ = createconfidenceinterval(df.β₂, α)
    _β₃_ = _group_ == "sick" ? createconfidenceinterval(convert.(T, df.β₃), α) : missing
    _maetrain_ = createconfidenceinterval(df.MAETrain, α)
    _maetest_ = createconfidenceinterval(df.MAETest, α)
    _mpetrain_ = createconfidenceinterval(df.MPETrain, α)
    _mpetest_ = createconfidenceinterval(df.MPETest, α)
    return SummaryStatistics(_β₀_,_β₁_,_β₂_,_β₃_,_maetrain_,_maetest_,_mpetrain_,_mpetest_)
end