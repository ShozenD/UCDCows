using DataFrames, DataFramesMeta

function remove_unhealthydata(df::DataFrame, Ndays::Integer, verbose::Bool=false)
    # group data frame by animal number / id -----------------------------------
    by_id = groupby(df, :id)
    # remove data where animal is deemed "unhealthy" ---------------------------
    # An animal is considered sick (possibly with mastitis) when there is a gap
    # in the data where the animal's milk yield is not recorded.
    # An animal is considered unhealthy / not fully healthy one week before and 
    # after the animal is sick.
    # --------------------------------------------------------------------------
    healthy = DataFrame()   # healthy data
    for ((id,), group) in pairs(by_id)
        id_group = sort(DataFrame(group), :date)
        dts = unique(id_group.date)
        start_, end_ = minimum(id_group.date), maximum(id_group.date)
        # get sick and unhealthy days ------------------------------------------
        sick_days, unhealthy_days = [], []
        for dt in daterange(start_, end_)
            if issickday(dt, dts)
                push!(sick_days, dt)
            elseif isunhealthyday(dt, sick_days, Ndays)
                push!(unhealthy_days, dt)
            end
        end
        # remove unhealthy days ------------------------------------------------
        healthy_group = @linq id_group |> where(:date .∉ Ref(unhealthy_days))
        # group yield values by day --------------------------------------------
        healthy_group = groupby(healthy_group, :dinmilk) |>
                        grps -> combine(grps, 
                                        :id,
                                        :lactnum, 
                                        :yield => sum => :yield, 
                                        :date) |>
                        comb -> unique(comb, :dinmilk)
        healthy = vcat(healthy, healthy_group)
        if length(sick_days) > 0 && verbose
            println("Animal $id:")
            println("\tSick days: $(length(sick_days))")
            println("\tUnhealthy days: $(length(unhealthy_days))")
            println("\tHealthy days: $(nrow(healthy_group))")
            println(repeat("-", 30))
        end
    end
    return healthy
end

function daterange(start_date, end_date)
    numdays = (end_date-start_date).value
    dts = Vector{Date}(undef, numdays+1)
    for i in 0:numdays
        dts[i+1] = start_date + Day(i)
    end
    return dts
end

issickday(dt, dts) = dt ∉ dts

function isunhealthyday(dt, dts, Ndays)
    for day in dts
        if (dt >= daysminusN(day, Ndays)) & (dt <= daysplusN(day, Ndays))
            return true
        end
    end
    return false
end

daysminusN(dt, N) = dt - Day(N)
daysplusN(dt, N) = dt + Day(N)