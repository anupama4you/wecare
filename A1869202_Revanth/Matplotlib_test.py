# 1st enable I2C in raspberry pi -> sudo raspi-config (Interface -> I2C -> enable) 
# 2nd, then reboot the system -> sudo reboot
# 3rd, install required libraries -> sudo apt-get install i2c-tools
# 4th, check whether installed or not -> sudo i2cdetect -y 1 
# 5th, Installing git -> sudo apt-get install git
# 6th, Installing math libraries required -> sudo apt-get install python3-scipy python3-numpy python3-matplotlib


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
x = y = list(range(100))
idx = [1,5,77,86,97]
new_y = [y[i] for i in idx]
plt.plot(range(len(idx)), new_y, 'o-')
plt.xticks(range(len(idx)),idx)
plt.savefig('foo.png') 

# it is not showing image inline, only option is to save images and then open it.