import matplotlib.pyplot as plt
#import seaborn as sns; sns.set()
import  json
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(111)

train = json.load(open('logs/y_err.json', 'r'))['train']
pair = json.load(open('logs/verif_err.json', 'r'))['train']
train = np.array(train)
x = np.arange(0, len(train), 1)
mean = np.mean(train)
std = np.std(train)
print(mean)
print(std)
#df = pd.Series(train)
#data = pd.DataFrame(data=train, columns=['train'])
#data['epoch'] =  np.arange(0, len(train), 1)

#print(type(fmri))
#fmri['signal'] = train
#fmri['timepoint'] = np.arange(0, len(train), 1)
#print(data['train'])
#print(type(data))
#sns.lineplot(x="timepoint", y="signal", data=fmri)
#ax = sns.lineplot(x='timepoint')
ax.plot(x, train)
ax.plot(x, pair)
#ax.fill_between(x, train - 0.03, train + 0.030, color='b', alpha=0.3)
ax.set_xlim(0, 50)



plt.show()