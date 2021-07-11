colnames = Dict(
  "Animal Number" => "id",
  "Group Number" => "groupid",
  "Days In Milk" => "dinmilk",
  "Lactation Number" => "lactnum",
  "Date" => "date",
  "Begin Time" => "tbegin",
  "End Time" => "tend",
  "Milk duration (mm:ss)" => "milkdur",
  "Yield" => "yield",
  "Yield LF" => "yieldlf",
  "Yield LR" => "yieldlr",
  "Yield RF" => "yieldrf",
  "Yield RR" => "yieldrr",
  "Interval" => "interval",
  "Last Milking Interval" => "lastmilkint",
  "Conductivity RR" => "condrr",
  "Conductivity RF" => "condrf",
  "Conductivity LR" => "condlr",
  "Conductivity LF" => "condlf",
  "Total Conductivity" => "condtot",
  "Mean Flow LR" => "flowlr",
  "Mean Flow LF" => "flowlf",
  "Mean Flow RR" => "flowrr",
  "Mean Flow RF" => "flowrf",
  "Peak Flow LF" => "peaklf", 
  "Peak Flow LR" => "peaklr", 
  "Peak Flow RF" => "peakrf", 
  "Peak Flow RR" => "peakrr", 
  "MDi" => "mdi", 
  "Kickoff LF" => "kicklf", 
  "Kickoff LR" => "kicklr", 
  "Kickoff RF" => "kickrf", 
  "Kickoff RR" => "kickrr", 
  "Blood RF" => "bloodrf", 
  "Blood RR" => "bloodrr", 
  "Blood LF" => "bloodlf", 
  "Blood LR" => "bloodlr", 
  "Total Blood" => "bloodtot", 
  "Teats Not Found" => "teats_not_found", 
  "Smart Pulsation Ratio" => "spr", 
  "Incomplete" => "incomplete"
)

parse_date(x::AbstractArray) = Date.(x, "m/d/y")

function parse_tbegin(x::AbstractArray)
  x = DateTime.(x, "m/d/y H:MM p")
  return Dates.Time.(x)
end

parse_tend(x::AbstractArray) = Time.(x, "H:MM p")

function parse_interval(x)
  if typeof(x) == Time
    h = Dates.value(Dates.Hour(x))
    m = Dates.value(Dates.Minute(x))
    s = Dates.value(Dates.Second(x))
    return 60*h + m + s/60
  end

  if typeof(x) == String
    if !isnothing(match(r"[0-9]{2}:[0-9]{2}:[0-9]{2}", x))
      mt = match(r"[0-9]{2}:[0-9]{2}:[0-9]{2}", x)
      t = Time(mt.match, "H:M:S")
      h = Dates.value(Dates.Hour(t))
      m = Dates.value(Dates.Minute(t))
      s = Dates.value(Dates.Second(t))

      return 60*h + m + s/60
    end

    if !isnothing(match(r"([0-9]{2}) d ([0-9]{2}:[0-9]{2}:[0-9]{2})", x))
      mt = match(r"([0-9]{2}) d ([0-2]{2}:[0-9]{2}:[0-9]{2})", x)
      d = parse(Int, mt.captures[1])
      t = Time(mt.captures[2], "H:M:S")
      h = Dates.value(Dates.Hour(t))
      m = Dates.value(Dates.Minute(t))
      s = Dates.value(Dates.Second(t))

      return 24*60*d + 60*h + m + s/60
    end   
  end

  if ismissing(x) | isnothing(x)
    return missing
  end
end

function parse_all!(d::DataFrame)
  #--------------- Rename columns -----------------
  rename!(d, colnames)

  for col = propertynames(d)

    #--------------- Time -----------------
    if col == :date d.date = parse_date(d.date) end
    if col == :tbegin d.tbegin = parse_tbegin(d.tbegin) end
    if col == :tend 
      d.tend = parse_tend(d.tend)
      d.tend = d.tbegin + Dates.Minute.(d.tend) + Dates.Second.(d.tend)
    end
    if col == :milkdur
      d.mdurS = Second.(d.tend - d.tbegin)
      d.mdurM = Dates.value.(round.(d.mdurS, Dates.Minute))
      d.mdurS = Dates.value.(d.mdurS)
    end
    if col == :interval d.interval = parse_interval.(d.interval) end
    if col == :lastmilkint d.lastmilkint = parse_interval.(d.lastmilkint) end
    
    #--------------- Yield -----------------
    if col == :yieldrr d.ypmrr = d.yieldrr ./ d.mdurS * 60 end
    if col == :yieldrf d.ypmrf = d.yieldrf ./ d.mdurS * 60 end
    if col == :yieldlr d.ypmlr = d.yieldlr ./ d.mdurS * 60 end 
    if col == :yieldlf d.ypmlf = d.yieldlf ./ d.mdurS * 60 end
  end
end