function summaryresults(results::DataFrame; 
                        α::AbstractFloat = 0.01, 
                        MPEThreshold::AbstractFloat = 0.3, 
                        verbose::Integer = 1,
                        tail::Symbol = :both)
    @assert verbose ∈ [0,1,2]
    @assert tail ∈ [:both, :single]
    @info "Total eligible samples $(sum(results.MPETest .< MPEThreshold)) (out of $(nrow(results)))"
    @info "Total eligible healthy samples $(sum((results.MPETest .< 0.3) .& (results.group .== "healthy"))) (out of $(sum((results.group .== "healthy"))))"
    @info "Total eligible sick samples $(sum((results.MPETest .< 0.3) .& (results.group .== "sick"))) (out of $(sum(results.group .== "sick")))"
    lactationNumbers = (sort ∘ unique)(results.lactationNumber)
    histogramList = []
    for lactationNumber in lactationNumbers
        verbose == 0 || @info "\tAnalysis on lactation $lactationNumber"
        
        resultsHealthy = @subset(results, :group .== "healthy", :lactationNumber .== lactationNumber, :MPETest .< MPEThreshold)
        resultsSick = @subset(results, :group .== "sick", :lactationNumber .== lactationNumber, :MPETest .< MPEThreshold)
        
        verbose == 0 || @info "\t\tMPE Threshold: $MPEThreshold"
        verbose == 0 || @info "\t\tNumber of healthy data: $(nrow(resultsHealthy)); Number of sick data: $(nrow(resultsSick))"
    
        histβ₀ = histogram(resultsHealthy.β₀, label="", title="β₀(L$lactationNumber)", alpha=0.5, color="red")
        histogram!(histβ₀, resultsSick.β₀, label="", alpha=0.5, color="blue")
        histβ₁ = histogram(resultsHealthy.β₁, label="", title="β₁(L$lactationNumber)", alpha=0.5, color="red")
        histogram!(histβ₁, resultsSick.β₁, label="", alpha=0.5, color="blue")
        histβ₂ = histogram(resultsHealthy.β₂, label="", title="β₂(L$lactationNumber)", alpha=0.5, color="red")
        histogram!(histβ₂, resultsSick.β₂, label="", alpha=0.5, color="blue")
        histβ₃ = histogram(resultsSick.β₃, label="", title="β₃(L$lactationNumber)", alpha=0.5, color="blue")
        append!(histogramList, [histβ₀, histβ₁, histβ₂, histβ₃])
    
        testβ₀ = UnequalVarianceTTest(resultsHealthy.β₀, resultsSick.β₀)
        testβ₁ = UnequalVarianceTTest(resultsHealthy.β₁, resultsSick.β₁)
        testβ₂ = UnequalVarianceTTest(resultsHealthy.β₂, resultsSick.β₂)
        testβ₃ = (OneSampleTTest ∘ convert)(Vector{Float64}, resultsSick.β₃)

        pvalue_β₀ = tail == :both ? pvalue(testβ₀, tail=:both) : pvalue(testβ₀, tail=:right)
        pvalue_β₁ = tail == :both ? pvalue(testβ₁, tail=:both) : pvalue(testβ₁, tail=:left)
        pvalue_β₂ = tail == :both ? pvalue(testβ₂, tail=:both) : pvalue(testβ₂, tail=:right)
        pvalue_β₃ = tail == :both ? pvalue(testβ₃, tail=:both) : pvalue(testβ₃, tail=:left)
        
        # @info "Mean: $(mean(resultsHealthy.β₀))  $(mean(resultsSick.β₀))\t SD: $(std(resultsHealthy.β₀))  $(std(resultsSick.β₀))， pvalue: $pvalue_β₀"
        # @info "Mean: $(mean(resultsHealthy.β₁))  $(mean(resultsSick.β₁))\t SD: $(std(resultsHealthy.β₁))  $(std(resultsSick.β₁))， pvalue: $pvalue_β₁"
        # @info "Mean: $(mean(resultsHealthy.β₂))  $(mean(resultsSick.β₂))\t SD: $(std(resultsHealthy.β₂))  $(std(resultsSick.β₂))， pvalue: $pvalue_β₂"
        # @info "Mean: $(mean(resultsSick.β₃))\t SD: $(std(resultsSick.β₃))， pvalue: $pvalue_β₃"
        # @info UnequalVarianceTTest(resultsHealthy.β₀, resultsSick.β₀)
        # @info UnequalVarianceTTest(resultsHealthy.β₁, resultsSick.β₁)
        # @info UnequalVarianceTTest(resultsHealthy.β₂, resultsSick.β₂)
        # @info (OneSampleTTest ∘ convert)(Vector{Float64}, resultsSick.β₃)

        if pvalue_β₀ < α
            verbose == 0 || @info "\t\tp-value for β₀ < α ($pvalue_β₀)"
            verbose ∈ [0,1] || @info testβ₀
        end
        if pvalue_β₁ < α
            verbose == 0 || @info "\t\tp-value for β₁ < α ($pvalue_β₁)"
            verbose ∈ [0,1] || @info testβ₁
        end
        if pvalue_β₂ < α
            verbose == 0 || @info "\t\tp-value for β₂ < α ($pvalue_β₂)"
            verbose ∈ [0,1] || @info testβ₂
        end
        if pvalue_β₃ < α
            verbose == 0 || @info "\t\tp-value for β₃ < α ($pvalue_β₃)"
            verbose ∈ [0,1] || @info testβ₃
        end
        if (pvalue_β₀ ≥ α) & (pvalue_β₁ ≥ α) & (pvalue_β₂ ≥ α) & (pvalue_β₃ ≥ α)
            verbose == 0 || @info "\t\tNo significant differences between healthy and sick samples."
        end
    end
    
    histogramCollection = plot(histogramList..., layout = (3,4))
    return histogramCollection
end