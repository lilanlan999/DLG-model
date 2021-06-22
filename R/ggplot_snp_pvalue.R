library(ggplot2)
library(reshape2)
temp <- read.csv("Chrome.csv")
temp_1<-melt(temp)
p <-ggplot(temp_1, aes(variable,SNPname)) + 
  geom_tile(aes(x = variable ,y = SNPname,fill = value),colour = "white") +
  labs(title = "Chr 11")+
  scale_fill_gradient(name="p-value", low = "blue",high = "white",breaks=c(0,0.05,1)) +
  theme(axis.text.x = element_text(vjust = 0.5, hjust = 0.5, angle = 90))+
  coord_fixed(ratio=1)
ggsave(p,filename = "value_11_SORL1.pdf",width = 12,height = 9,family = "Times")