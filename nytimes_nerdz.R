
library(data.table)
library(lubridate)
library(dplyr)
library(ggplot2)

##compare
# i manually combined the two output files from running the web scraping script with my credentials 
# and running the script on my girlfriend's credentials into a csv file "plots"
# i probably could have just had the script do this but it took a decent amount of time to run and 
# i am kind of lazy

dat.orig = fread("../Desktop/december/euler/plots.csv")
dat.orig[, date2 := as.Date(date,'%m/%d/%Y')]

# flag data points to remove, some puzzles have completion times of 0 or 1 second should be removed
# also some times puzzles take a while because one of us might forget we have it open, so I'm removing
# times of longer than 2 minutes, for a mini puzzle I think this is reasonable
dat.orig[, remove := ifelse(secs < 2, 1, ifelse(secs > 120, 1, 0))]

# only plot the non-outlier data
toplot = dat.orig[remove == 0]
# get weekdays so we can use them as a facet (will I be sluggish on mondays???)
toplot[, day := weekdays(date2)]
this_year = toplot[date2 >= '2016-10-01']

# and at long last, the moment we've all been waiting for!
ggplot(toplot, aes(x=date2, y = secs, color=id))+
  facet_wrap(~day)+
  geom_point()+
  geom_smooth(method='loess')


toplot %>% group_by(id,day) %>% summarize(mean(secs))


rose = toplot[id == 'rose']
jane = toplot[id == 'jane']

# let's misbehave slightly and assume a normal distribution because why not
t.test(jane$secs, rose$secs)
# p value of .05142 means she and I are pretty darn close


#rose 2018 recap
# everything below this is all pretty much a work in progress
# this is all random stuff I was doing with my friend danny's times, I am pretty bad at naming things
rose.orig = fread("../Desktop/december/euler/rose2.csv")
rose.orig[, secs := V4 + V3*10 + V1*60]

dates = seq.Date(as.Date('2018-07-01'), as.Date('2019-1-22'), by=1)

rose = cbind(rose.orig, dates)
rose[, remove := ifelse(secs < 2, 1, ifelse(secs > 2000, 1, 0))]

toplot = rose[remove == 0]
toplot[, id := 'danny']

ggplot(toplot, aes(dates, secs, color = id))+
  geom_point()+
  geom_smooth(method='loess')
  scale_color_manual(values = c('black'))

lm = lm(toplot$dates ~ toplot$secs)
summary(lm)

