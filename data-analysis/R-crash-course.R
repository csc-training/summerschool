x <- c(2, -6, 42.5, Inf)
y <- x + 1 # vectorized calculation with recycling
y
log(x[2:3],10)

zoo <- list(animals=c("zebra", "giraffe", "anteater"),
            location=c(60,24),
            area=22)
zoo$area*2.47

treatg <- sample(c("trt","ctrl"), 10, replace=TRUE)
summary(treatg) # not very informative

treatg <- factor(treatg,levels=c("trt","ctrl"))
summary(treatg) # understood as a categorical variable

data(iris) # a classic example data set
?iris # documentation for the data
summary(iris)
dim(iris)
iris[1:3,1:4] # array indexing works too

library(ggplot2)

ggplot(iris) + geom_point(aes(Sepal.Width,Sepal.Length,col=Species))

ggplot(iris) + 
  geom_point(aes(Sepal.Width,Sepal.Length,col=Species)) +
  geom_smooth(aes(Sepal.Width,Sepal.Length,col=Species))

ggplot(iris,aes(Sepal.Width,Sepal.Length,col=Species)) + 
  geom_point() +
  geom_smooth(method = "lm")


library(readr)
cards <- read_csv("cards.csv") # here everything is clean and easy

# but the data is not tidy

library(tidyr)
library(dplyr)

tidycards <- cards %>% 
  gather(key = "week",value = "N", starts_with("week"))

ggplot(tidycards)+geom_line(aes(x=week,y=N,color=card))
# this does not work!
# the reason is that week is not numeric

# note: piping is cool
tidycards <- tidycards %>% 
  # separate the text in week column in to two columns
  # after 4th character: 
  # (you could use regexps too)
  separate(col = week,into = c("foo","week"),sep = 4) %>% 
  # throw away the unneeded one:
  select(-foo) %>% 
  # make the column numeric
  mutate(week=as.numeric(week))

ggplot(tidycards)+
  geom_line(aes(x=week,y=N,color=card))
# now it works!

ggplot(tidycards)+
  geom_smooth(aes(x=week,y=N,color=card),method="lm",se=FALSE)

# find the weekly ranks for the cards
tidycards <- tidycards %>% 
  group_by(week) %>% 
  mutate(weekrank=rank(N)) %>% 
  ungroup()

# find the mean number of each card over the weeks
tidycards %>% 
  group_by(card) %>% 
  summarise(meanN=mean(N)) %>% 
  ungroup()
