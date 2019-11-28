library(tidyverse)
data = read.csv("C:/Users/glius/Google Drive/HOPE/accelerometer/fitbit_beiwe_minutewise.csv")

## create a new column to specify if there is walking happened in nearby intervals 
## k: window size
## h: the threshold to be considered as walking
nearby_walk = function(data,k,h){
  ## create an indicator ti show if the current minute is walking based on beiwe
  walk_all = as.numeric(data$step_infer>h)
  record_all = as.numeric(data$avtive_s>0)
  nearby = walk_all
  active = record_all
  for(id in unique(data$id)){
    index = data$id==id
    walk = walk_all[index]
    record = record_all[index]
    n = sum(index)
    for(i in 1:k){
      nearby[index] = nearby[index] + c(walk[(i+1):n],rep(0,i)) + c(rep(0,i),walk[1:(n-i)])
      active[index] = active[index] + c(record[(i+1):n],rep(0,i)) + c(rep(0,i),record[1:(n-i)])
    }
  }
  final = as.numeric(active>=1)
  final[active>=1 & nearby==0] = 1
  final[active>=1 & nearby>=1] = 2
  return(final)
}


## by person/by hour/by status (nearby walk or not)
tune_result = c()
for(k in c(10,20,30,40,50,60)){
  for(h in c(30,40,50,60)){
    s = nearby_walk(data,k,h)
    #data %>% filter(step_fitbit>0) %>%
    #  ggplot(aes(x=step_fitbit)) +
    #  geom_histogram(bins=100)+facet_wrap(~hour,nrow=6,ncol=4)
    
    hourly_walk_p=c();hourly_walk_c=c()
    for (i in 0:2){
      temp1=c();temp2=c()
      for (j in 0:23){
        subdata = subset(data,s==i & hour==j)
        m = nrow(subdata)
        temp1 = c(temp1,mean(subdata$step_fitbit>0,na.rm=T))
        temp2 = c(temp2,m)
      }
      hourly_walk_p = rbind(hourly_walk_p,temp1)
      hourly_walk_c = rbind(hourly_walk_c,temp2)
    }
    colnames(hourly_walk_p)=0:23
    rownames(hourly_walk_p)=c("no record","non_walk","walk")
    tune_result=rbind(tune_result,c(k,h,mean(abs(hourly_walk_p[3,]-hourly_walk_p[2,]),na.rm=T),min(hourly_walk_c[hourly_walk_c>0])))
  }
}

data$walk = nearby_walk(data,60,60)
hourly_walk_per=c()
for (i in 0:2){
  temp=c()
  for (j in 0:23){
    subdata = subset(data,walk==i & hour==j)
    temp = c(temp,mean(subdata$step_fitbit>0,na.rm=T))
  }
  hourly_walk_per = rbind(hourly_walk_per,temp)
}
colnames(hourly_walk_per)=0:23
rownames(hourly_walk_per)=c("no record","non_walk","walk")
hourly_walk_per