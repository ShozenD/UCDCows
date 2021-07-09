using PrettyTables

function savehtml(filename, data)
    open("$filename.html", "w") do f
        pretty_table(f, data, backend = :html)
    end
end

using CSV, HTTP, DataFrames

url1 = "https://raw.githubusercontent.com/julia4ta/tutorials/master/Series%2002/Files/mydf.csv"

mydf_url = DataFrame(CSV.File(HTTP.get(url1).body))

savehtml("mydf_url", mydf_url)
CSV.write("mydf_url.csv", mydf_url)

# donwload 2019 gdp data from GitHub

url2 = "https://raw.githubusercontent.com/julia4ta/tutorials/master/Series%2002/Files/gdp_by_country_un.csv"

df_gdp = DataFrame(CSV.File(HTTP.get(url2).body))

savehtml("df_gdp", df_gdp)
CSV.write("df_gdp.csv", df_gdp)

################################################################################
# DataFrames
################################################################################

# outer join (include all countries)

df_outer = outerjoin(mydf_url, df_gdp, on = :country)
savehtml("df_outer", df_outer)

# inner join (keep only common countries)

df_inner = innerjoin(mydf_url, df_gdp, on = :country)
savehtml("df_inner", df_inner)

# convert gdp from millions to billions

df_inner.gdp = round.(df_inner.gdp ./ 1000, digits = 3)
savehtml("df_inner", df_inner)

# add column showing gdp per capita in thousands

df_inner.gdppc = round.(df_inner.gdp ./ df_inner.my19, digits = 3)
savehtml("df_inner", df_inner)

df_sa_base = filter(:subregion => x -> x == "Southern Asia", df_inner)
savehtml("df_sa_base", df_sa_base)

using DataFramesMeta

df_sa_dfm = @where(df_inner, :subregion .== "Southern Asia")
savehtml("df_sa_dfm", df_sa_dfm)

df_multi_filters = @where(df_inner,
    .|(:subregion .== "Southern Asia", :subregion .== "Northern America"),
    .&(:my19 .< 1, :gdppc .> 10)
)
savehtml("df_multi_filters", df_multi_filters)

df_by_r = @by(df_inner, :region,
    my18_sum = sum(:my18),
    my19_sum = sum(:my19),
    mydelta_sum = sum(:mydelta),
    mypct_c = round.(sum(:mydelta) ./ sum(:my18) .* 100, digits = 2),
    my100_c = round.(sum(:my19) ./ sum(df_inner.my19) .* 100, digits = 2),
    gdp_sum = sum(:gdp),
    gdppc_c = round.(sum(:gdp) ./ sum(:my19), digits = 3)
)
savehtml("df_by_r", df_by_r)

df_by_s = @by(df_inner, :subregion,
    region_unique = unique(:region),
    my18_sum = sum(:my18),
    my19_sum = sum(:my19),
    mydelta_sum = sum(:mydelta),
    mypct_c = round.(sum(:mydelta) ./ sum(:my18) .* 100, digits = 2),
    my100_c = round.(sum(:my19) ./ sum(df_inner.my19) .* 100, digits = 2),
    gdp_sum = sum(:gdp),
    gdppc_c = round.(sum(:gdp) ./ sum(:my19), digits = 3)
)
savehtml("df_by_s", df_by_s)

using Gadfly

p = plot(df_by_r,
    x = :my19_sum,
    y = :region,
    color = :region,
    Geom.bar(orientation = :horizontal),
    Guide.xlabel("Population (mils)"),
    Guide.ylabel("Region"),
    Guide.title("2019 Population by Region"),
    Guide.colorkey(title = "Region"),
    Scale.x_continuous(format = :plain),
    Theme(
        background_color = "white",
        bar_spacing = 1mm
    )
)

df_sub100 = @where(df_inner, :my19 .< 100)

p = plot(df_sub100,
    x = :my19,
    y = :mypct,
    color = :region,
    Geom.point,
    Guide.xlabel("Population (mils)"),
    Guide.ylabel("Growth Rate (%)"),
    Guide.title("2019 Population vs Growth Rate (sub 100 mil)"),
    Guide.colorkey(title = "Region"),
    Theme(background_color = "white")
)

# Beeswarm plot by region

p = plot(df_sub100,
    x = :region,
    y = :my19,
    color = :region,
    Geom.beeswarm,
    Guide.xlabel("Population (mils)"),
    Guide.ylabel("Growth Rate (%)"),
    Guide.title("2019 Population vs Growth Rate (sub 100 mil)"),
    Guide.colorkey(title = "Region"),
    Scale.y_continuous(format = :plain),
    Theme(background_color = "white")
)

p = plot(df_sub100,
    x = :region,
    y = :my19,
    color = :region,
    Geom.boxplot,
    Guide.xlabel("Population (mils)"),
    Guide.ylabel("Growth Rate (%)"),
    Guide.title("2019 Population vs Growth Rate (sub 100 mil)"),
    Guide.colorkey(title = "Region"),
    Scale.y_continuous(format = :plain),
    Theme(background_color = "white")
)

# Density plot by region
df_sub100_o = @where(df_sub100, :region .!= "Oceania")

p = plot(df_sub100_o,
    x = :my19,
    color = :region,
    Geom.density,
    Guide.xlabel("Population (mils)"),
    Guide.ylabel("Density"),
    Guide.title("2019 Population Density by Region (sub 100 mil ex Oceania"),
    Guide.colorkey(title = "Region"),
    Scale.x_continuous(format = :plain),
    Theme(background_color = "white")
)

p = plot(df_sub100,
    x = :region,
    y = :gdppc,
    color = :region,
    Geom.violin,
    Guide.xlabel("Region"),
    Guide.ylabel(
        "GDP per Capita (dollars in 000s)",
        orientation = :vertical
    ),
    Guide.title("2019 GPD per Capita by Region"),
    Guide.colorkey(title = "Region"),
    Scale.y_continuous(format = :plain),
    Coord.cartesian(ymin = -25, ymax = 75),
    Theme(background_color = "white")
)
