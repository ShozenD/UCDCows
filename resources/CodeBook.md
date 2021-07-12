## Dairy Cow Data Code Book
A dictionary of variable names and their definition

### Cow Informaton
`id` (*Int*): Unique cow ID number

`groupid` (*Int*): Unique group ID

`lactnum` (*Int*): Lactation number

### Time
`dinmilk` (*Int*): Days in milk
![dist-dinmilk](figures/dist-dinmilk.png)

`date` (*Date*): Date of the milking

`tbegin` (*Time*): Time when the milking began

`tend` (*Time*): Time when the milking ended

`milkdur` (*Time*): Milking duration

`interval` (*Float*): Time interval between the end time of the last milking and the begin time of the current milking
![dist-interval](figures/dist-interval.png)

`lastmilkint` (*Float*): Time interval between the begin time of the last milking and the begin time of the current milking
![dist-lastmilkint](figures/dist-lastmilkint.png)

### Conductivity
`condlf`, `condlr`, `condrf`, `condrr` (*Float*): Raw conductivity for each teat
|Left Front|Left Rear|Right Front|Right Reat|
|:---:|:---:|:---:|:---:|
|![dist-condlf](figures/dist-condlf.png)|![dist-condlf](figures/dist-condlr.png)|![dist-condrf](figures/dist-condrf.png)|![dist-condrr](figures/dist-condrr.png)|

`condtot` (*Float*): Total conductivity

### Yield
`yieldlf`, `yieldlr`, `yieldrf`, `yieldrr` (*Float*): Raw yield value for each teat

`ypmlf`, `ypmlr`, `ypmrf`, `ypmrr` (*Float*): Yield per minute for each teat. Engineered by dividing the raw yield by the milking duration.

### Flow
`flowlf`, `flowlf`, `flowrr`, `flowrf` (*Float*): Average flow during the milking. **Note**: This value is different from the yield per min. 

`peaklf`, `peaklr`, `peakrf`, `peakrr` (*Float*): Peak flow during the milking

### Blood
`bloodrf`, `bloodrr`, `bloodlf`, `bloodlr` (*Float*): Amount of blood detected in the milk

`bloodtot` (*Float*): Total amount of blood detected in milk from the 4 teats

### Others
`mdi` (*Float*): Mastitis Detection index (MDi) from the DeLaval milking robot

`teats_not_found`: Indicator for missing teats

`kicklf`, `kicklr`, `kickrf`, `kickrr`: Kickoff indicators

`spr`: Smart pulsation ratio

`incomplete`: Indicator for incompelete records