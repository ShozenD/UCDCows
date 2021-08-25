### A Pluto.jl notebook ###
# v0.15.1

using Markdown
using InteractiveUtils

# ╔═╡ 1800f9fa-5d20-4c9b-804e-091066f3a539
begin
	using Pkg
	Pkg.activate("../")
end

# ╔═╡ 936d0604-0b6d-4b98-a16c-8e434616b461
using 
	CSV, 
	Dates,
	DataFrames, 
	DataFramesMeta,
	Statistics,
	Impute,
	GLM, 
	Gadfly,
	PlutoUI

# ╔═╡ 04a46145-479c-4409-973e-c3adeca6dec1
begin
	include("../src/utilities.jl")
	include("../src/visualization.jl")
end

# ╔═╡ 5f895120-fa95-11eb-16d7-81f1927ff9ab
md"# Yield Prior Warning"

# ╔═╡ 4c061106-244d-4b06-ade4-802da1af5394
md"**Research Question**: Does milk yield decrease prior to contracting mastits?

* When does milk yield start to decrease?
* How much does it decrease compared to normal?
"

# ╔═╡ 81b6dc3f-9021-4922-bee4-3d2aced5cd78
df = CSV.read("../data/analytical/cows-analytic.csv", DataFrame);

# ╔═╡ e08f5110-7b09-4230-861c-82542c1180a8
# Identifying cows with abnormal (>=1.4) MDi values
begin
	d = @where(df, :mdi .>= 1.4)
	id_abnrm = unique(d.id)
	mask = [x ∈ id_abnrm for x in df.id]
	df_abnrm = df[mask,:]
	cowid = unique(df_abnrm.id)
end

# ╔═╡ d4c61cf4-4112-4ae1-b557-2a0835381507
begin
	ncows = length(cowid)
	md"There are **$ncows cows** with MDi values over 1.4"
end

# ╔═╡ a2ad7978-d54a-44fd-b656-14080bfbe3ad
begin
	function priorwarn_yield(D::DataFrame, id::Int, N::Int=20)	
		# Filter for a specific cow
		Dx = D[D.id .== id,:]

		# Fill missing values
		Dx.mdi = linterp(Dx.mdi)

		# Find when the first alert occurred
		h, t = find_headtail(Dx.mdi .> 1.3)

		# Obtain the yield info prior to the alert
		if length(h) > 0 && h[1] > 1
			dwarn = Dx.date[h[1]] + Dx.tbegin[h[1]]

			Dy = select(
				Dx, 
				[:id,:date,:tbegin,:yieldlf,:yieldlr,:yieldrf,:yieldrr]
			)[max(1,h[1]-N):h[1]-1,:]


			Dy.twarn = Dates.Minute.(dwarn .- (Dy.date + Dy.tbegin))
			Dy.twarn = Dates.value.(Dy.twarn) / 60
			return Dy
		end
	end
	
	function priorwarn_cond(D::DataFrame, id::Int, N::Int=20)
		Dx = D[D.id .== id,:]
		Dx.mdi = linterp(Dx.mdi)
		h, t = find_headtail(Dx.mdi .> 1.3)

		if length(h) > 0 && h[1] > 1
			dwarn = Dx.date[h[1]] + Dx.tbegin[h[1]]

			Dy = select(
				Dx, 
				[:id,:date,:tbegin,:condlf,:condlr,:condrf,:condrr]
			)[max(1,h[1]-N):h[1]-1,:]


			Dy.twarn = Dates.Minute.(dwarn .- (Dy.date + Dy.tbegin))
			Dy.twarn = Dates.value.(Dy.twarn) / 60
			return Dy
		end
	end
end

# ╔═╡ 0cc44c93-c706-4ab2-95fe-0b1ae7e76724
begin
	Dy₁ = priorwarn_yield(df_abnrm, cowid[1])
	Dc₁ = priorwarn_cond(df_abnrm, cowid[1])
	
	for id in cowid[2:end]
		try
			Tmpy = priorwarn_yield(df_abnrm, id)
			Tmpc = priorwarn_cond(df_abnrm, id)
			if !isnothing(Tmpy)
				append!(Dy₁,Tmpy)
			end

			if !isnothing(Tmpc)
				append!(Dc₁,Tmpc)
			end
		catch
			# Return Nothing
		end
	end
end

# ╔═╡ cf69fb5e-6998-4b89-863d-efbadff939d0
begin
	Dy₂ = @transform(Dy₁, 
		EX1 = (:yieldlf.^2 + :yieldlr.^2 + :yieldrf.^2 + :yieldrr.^2)./4,
		EX2 = ((:yieldlf + :yieldlr + :yieldrf + :yieldrr)./4).^2,
		min = minimum([:yieldlf, :yieldlr, :yieldrf, :yieldrr]),
		max = maximum([:yieldlf, :yieldlr, :yieldrf, :yieldrr])
	)
	
	Dc₂ = @transform(Dc₁, 
		EX1 = (:condlf.^2 + :condlr.^2 + :condrf.^2 + :condrr.^2)./4,
		EX2 = ((:condlf + :condlr + :condrf + :condrr)./4).^2,
		min = minimum([:condlf, :condlr, :condrf, :condrr]),
		max = maximum([:condlf, :condlr, :condrf, :condrr])
	)
	
	Dy₂.twarn = Dy₂.twarn * -1
	Dy₂.var = Dy₂.EX1 - Dy₂.EX2
	Dy₂.sd = sqrt.(Dy₂.var)
	Dy₂.range = abs.(Dy₂.max - Dy₂.min)
	
	Dc₂.twarn = Dc₂.twarn * -1
	Dc₂.var = Dc₂.EX1 - Dc₂.EX2
	Dc₂.sd = sqrt.(Dc₂.var)
	Dc₂.range = abs.(Dc₂.max - Dc₂.min)
end

# ╔═╡ 9ad177a9-afed-47d6-beef-f0dc6fde1dd5
begin
	Dy₃ = @where(Dy₂, :var .!= ismissing.(Dy₂.var))
	Dc₃ = @where(Dc₂, :var .!= ismissing.(Dc₂.var))
end;

# ╔═╡ ee273b6a-17d4-4c40-b1ec-af1839878068
begin
	Gadfly.set_default_plot_size(18cm, 10cm)
	plot(
		@where(Dy₃, :twarn .>= -300),
		x = :twarn, y = :sd,
		layer(
			Geom.smooth(method=:loess),
			color = [colorant"#de425b"],
			style(line_width = 0.7mm)
		),
		layer(
			Geom.point,
			color = [colorant"black"]
		),
		Guide.xticks(ticks=[0:-24:-288;]),
		Guide.ylabel("Yield [SD]"),
		Guide.xlabel("Time to Warning [H]"),
		Theme(point_size = 0.7mm, alphas=[0.1], highlight_width=0mm)
	)
end

# ╔═╡ 7531313c-b368-43bc-aec9-ee0562eb6d92
begin
	Gadfly.set_default_plot_size(18cm, 10cm)
	plot(
		@where(Dc₃, :twarn .>= -300),
		x = :twarn, y = :sd,
		layer(
			Geom.smooth(method=:loess),
			color = [colorant"#de425b"],
			style(line_width = 0.7mm)
		),
		layer(
			Geom.point,
			color = [colorant"black"]
		),
		Guide.xticks(ticks=[0:-24:-288;]),
		Guide.ylabel("Conductivity [SD]"),
		Guide.xlabel("Time to Warning [H]"),
		Theme(point_size = 0.7mm, alphas=[0.1], highlight_width=0mm)
	)
end

# ╔═╡ 45b96901-8f38-4b92-96fd-2971e5aff535
begin
	# Filter for a specific cow
	cow = df_abnrm[df_abnrm.id .== cowid[1],:]
	
	# Fill missing values
	cow.mdi = linterp(cow.mdi)
	
	# Find when the first alert occurred
	h, t = find_headtail(cow.mdi .> 1.3)
	
	cow[h[1],[:condlf, :condrf, :condlr, :condrr]]
end

# ╔═╡ 41f436f7-c0fb-4c1e-8dd9-4ba82401c676
function warn_cond(D::DataFrame, id::Int)	
		# Filter for a specific cow
		Dx = D[D.id .== id,:]

		# Fill missing values
		Dx.mdi = linterp(Dx.mdi)

		# Find when the first alert occurred
		h, t = find_headtail(Dx.mdi .> 1.3)

		# Obtain the yield info prior to the alert
		if length(h) > 0 && h[1] > 1
			Dy = select(
				Dx, 
				[:id,:condlf,:condlr,:condrf,:condrr]
			)[[1],:]
		
			return Dy
		end
	end

# ╔═╡ c1bce6eb-1a1c-4c88-8f23-791098c07492
begin
	Dwc₁ = warn_cond(df_abnrm, cowid[1])
	
	for id in cowid[2:end]
		try
			Tmpwc = warn_cond(df_abnrm, id)
			if !isnothing(Tmpwc)
				append!(Dwc₁,Tmpwc)
			end
		catch
			# Return Nothing
		end
	end
end

# ╔═╡ c14a5c10-1308-48ef-baa4-6e374f6f1f37
begin
	Dwc₂ = @transform(Dwc₁, 
		EX1 = (:condlf.^2 + :condlr.^2 + :condrf.^2 + :condrr.^2)./4,
		EX2 = ((:condlf + :condlr + :condrf + :condrr)./4).^2,
		min = minimum([:condlf, :condlr, :condrf, :condrr]),
		max = maximum([:condlf, :condlr, :condrf, :condrr])
	)
	Dwc₂.var = Dwc₂.EX1 - Dwc₂.EX2
	Dwc₂.sd = sqrt.(Dwc₂.var)
	Dwc₂.range = abs.(Dwc₂.max - Dwc₂.min)
end

# ╔═╡ 38b5cf1b-2166-42f5-8054-f2ef7d246641
begin
	Dwc₃ = @where(Dwc₂, :var .!= ismissing.(Dwc₂.var))
	q5 = quantile(Dwc₃.sd, [0.25,0.50,0.75,0.95])

	plot(
		Dwc₂,
		layer(xintercept=q5, Geom.vline(style=[:dash],color="gray10")),
		layer(x=:sd, color=[colorant"#de425b"], Geom.histogram),
		Guide.xticks(ticks=[0:0.25:3;]),
		Guide.xlabel("Conductivity [SD]"),
		Guide.ylabel("Frequency"),
		Theme()
	)
end

# ╔═╡ 4752bde8-d147-453f-9840-53fb5798cd42
begin
	Dwc_subset = @where(Dwc₂, Dwc₂.sd .>= q5[4])
	mask₂ = [x ∈ Dwc_subset.id for x in Dc₃.id]
end

# ╔═╡ d68a5855-5da9-4d21-877b-939b667612e8
begin
	Gadfly.set_default_plot_size(18cm, 10cm)
	plot(
		@where(filter(row -> row.id ∈ Dwc_subset.id, Dy₃), :twarn .>= -300),
		x = :twarn, y = :sd,
		layer(
			Geom.smooth(method=:loess),
			color = [colorant"#de425b"],
			style(line_width = 0.7mm)
		),
		layer(
			Geom.point,
			color = [colorant"black"]
		),
		Guide.xticks(ticks=[0:-24:-288;]),
		Guide.ylabel("Yield [SD]"),
		Guide.xlabel("Time to Warning [H]"),
		Theme(point_size = 0.7mm, alphas=[0.1], highlight_width=0mm)
	)
end

# ╔═╡ 8dfaedda-a962-4ebe-aa88-8223a88a458f
begin
	Gadfly.set_default_plot_size(18cm, 10cm)
	plot(
		@where(filter(row -> row.id ∈ Dwc_subset.id, Dc₃), :twarn .>= -300),
		x = :twarn, y = :sd,
		layer(
			Geom.smooth(method=:loess),
			color = [colorant"#de425b"],
			style(line_width = 0.7mm)
		),
		layer(
			Geom.point,
			color = [colorant"black"]
		),
		Guide.xticks(ticks=[0:-24:-288;]),
		Guide.ylabel("Conductivity [SD]"),
		Guide.xlabel("Time to Warning [H]"),
		Theme(point_size = 0.7mm, alphas=[0.1], highlight_width=0mm)
	)
end

# ╔═╡ Cell order:
# ╟─5f895120-fa95-11eb-16d7-81f1927ff9ab
# ╟─4c061106-244d-4b06-ade4-802da1af5394
# ╠═1800f9fa-5d20-4c9b-804e-091066f3a539
# ╠═936d0604-0b6d-4b98-a16c-8e434616b461
# ╠═04a46145-479c-4409-973e-c3adeca6dec1
# ╠═81b6dc3f-9021-4922-bee4-3d2aced5cd78
# ╠═e08f5110-7b09-4230-861c-82542c1180a8
# ╟─d4c61cf4-4112-4ae1-b557-2a0835381507
# ╠═a2ad7978-d54a-44fd-b656-14080bfbe3ad
# ╠═0cc44c93-c706-4ab2-95fe-0b1ae7e76724
# ╠═cf69fb5e-6998-4b89-863d-efbadff939d0
# ╠═9ad177a9-afed-47d6-beef-f0dc6fde1dd5
# ╟─ee273b6a-17d4-4c40-b1ec-af1839878068
# ╟─7531313c-b368-43bc-aec9-ee0562eb6d92
# ╠═45b96901-8f38-4b92-96fd-2971e5aff535
# ╠═41f436f7-c0fb-4c1e-8dd9-4ba82401c676
# ╠═c1bce6eb-1a1c-4c88-8f23-791098c07492
# ╠═c14a5c10-1308-48ef-baa4-6e374f6f1f37
# ╟─38b5cf1b-2166-42f5-8054-f2ef7d246641
# ╠═4752bde8-d147-453f-9840-53fb5798cd42
# ╟─d68a5855-5da9-4d21-877b-939b667612e8
# ╟─8dfaedda-a962-4ebe-aa88-8223a88a458f
