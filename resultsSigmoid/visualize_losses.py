import numpy as np
import matplotlib.pyplot as plt
#import time

#from datetime import datetime

#data = np.genfromtxt('./val_loss.txt', delimiter=',')
#print(np.min(data))


#start = datetime.now()
#time.sleep(20)
#end = datetime.now()
#diff =end - start 
#print(diff)
#print(type(diff))

names= ["loss","mean_squared_error","mean_absolute_error","ssd_loss","mad_loss","val_loss","val_mean_squared_error","val_mean_absolute_error","val_ssd_loss","val_mad_loss"] 
my_data = []

for n in names:
    data = np.genfromtxt('../' + n + '.txt', delimiter=',')
    my_data.append(data[:data.shape[0]//3])

my_data = np.array(my_data)


print(my_data.shape)
x = np.arange(0, my_data.shape[1])
fig, axs = plt.subplots(5,2,constrained_layout=True)
for i in range(0, my_data.shape[0]//2):
    loss = my_data[i,:]
    axs[i][0].plot(x, loss)
    axs[i][0].set_title(names[i])
j = 0
for i in range(my_data.shape[0]//2, my_data.shape[0]):
    loss = my_data[i,:]
    axs[j][1].plot(x, loss)
    axs[j][1].set_title(names[i])
    j += 1
plt.show()
