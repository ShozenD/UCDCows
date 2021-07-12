## Dairy Cow Data Code Book
A dictionary of variable names and their definition

### Cow Informaton
`id` (*Int*): Unique cow ID number

`groupid` (*Int*): Unique group ID

`lactnum` (*Int*): Lactation number
![dist-lactnum](figures/dist-lactnum.png)

### Time
`dinmilk` (*Int*): Days in milk
![dist-dinmilk](figures/dist-dinmilk.png)

`date` (*Date*): Date of the milking

`tbegin` (*Time*): Time when the milking began

`tend` (*Time*): Time when the milking ended

`milkdur` (*Time*): Milking duration

`mdurS` (*Int*): Milking duration in seconds
![dist-mdurS](figures/dist-mdurS.png)

`mdurM` (*Int*): Milking duration in minutes
![dist-mdurS](figures/dist-mdurM.png)

`interval` (*Float*): Time interval between the end time of the last milking and the begin time of the current milking
![dist-interval](figures/dist-interval.png)

`lastmilkint` (*Float*): Time interval between the begin time of the last milking and the begin time of the current milking
![dist-lastmilkint](figures/dist-lastmilkint.png)

### Conductivity
`condlf`, `condlr`, `condrf`, `condrr` (*Float*): Raw conductivity for each teat
|Left Front|Left Rear|
|:---:|:---:|
|![dist-condlf](figures/dist-condlf.png)|![dist-condlf](figures/dist-condlr.png)|

|Right Front|Right Rear|
|:---:|:---:|
|![dist-condrf](figures/dist-condrf.png)|![dist-condrr](figures/dist-condrr.png)|

`condtot` (*Float*): Total conductivity
![dist-condtot](figures/dist-condtot.png)

### Yield
`yieldlf`, `yieldlr`, `yieldrf`, `yieldrr` (*Float*): Raw yield value for each teat
|Left Front|Left Rear|
|:---:|:---:|
|![dist-yieldlf](figures/dist-yieldlf.png)|![dist-yieldlr](figures/dist-yieldlr.png)|

|Right Front|Right Rear|
|:---:|:---:|
|![dist-yieldrf](figures/dist-yieldrf.png)|![dist-yieldrr](figures/dist-yieldrr.png)|

`ypmlf`, `ypmlr`, `ypmrf`, `ypmrr` (*Float*): Yield per minute for each teat. Engineered by dividing the raw yield by the milking duration.
|Left Front|Left Rear|
|:---:|:---:|
|![dist-ypmlf](figures/dist-ypmlf.png)|![dist-ypmlr](figures/dist-ypmlr.png)|

|Right Front|Right Rear|
|:---:|:---:|
|![dist-ypmrf](figures/dist-ypmrf.png)|![dist-ypmrr](figures/dist-ypmrr.png)|

### Flow
`flowlf`, `flowlf`, `flowrr`, `flowrf` (*Float*): Average flow during the milking. **Note**: This value is different from the yield per min. 
|Left Front|Left Rear|
|:---:|:---:|
|![dist-flowlf](figures/dist-flowlf.png)|![dist-flowlr](figures/dist-flowlr.png)|

|Right Front|Right Rear|
|:---:|:---:|
|![dist-flowrf](figures/dist-flowrf.png)|![dist-flowrr](figures/dist-flowrr.png)|

`peaklf`, `peaklr`, `peakrf`, `peakrr` (*Float*): Peak flow during the milking
|Left Front|Left Rear|
|:---:|:---:|
|![dist-peaklf](figures/dist-flowlf.png)|![dist-peaklr](figures/dist-flowlr.png)|

|Right Front|Right Rear|
|:---:|:---:|
|![dist-peakrf](figures/dist-flowrf.png)|![dist-peakrr](figures/dist-flowrr.png)|

### Blood
`bloodrf`, `bloodrr`, `bloodlf`, `bloodlr` (*Float*): Amount of blood detected in the milk
|Left Front|Left Rear|
|:---:|:---:|
|![dist-bloodlf](figures/dist-bloodlf.png)|![dist-bloodlr](figures/dist-bloodlr.png)|

|Right Front|Right Rear|
|:---:|:---:|
|![dist-bloodrf](figures/dist-bloodrf.png)|![dist-bloodrr](figures/dist-bloodrr.png)|

`bloodtot` (*Float*): Total amount of blood detected in milk from the 4 teats
![dist-bloodtot](figures/dist-bloodtot.png)

### Others
`mdi` (*Float*): Mastitis Detection index (MDi) from the DeLaval milking robot
![dist-mdi](figures/dist-mdi.png)

`teats_not_found`: Indicator for missing teats

`kicklf`, `kicklr`, `kickrf`, `kickrr`: Kickoff indicators

`spr`: Smart pulsation ratio

`incomplete`: Indicator for incompelete records