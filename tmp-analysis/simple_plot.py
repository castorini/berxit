import matplotlib.pyplot as plt

data = []
with open('tmp_loss') as fin:
    for line in fin:
        data.append(float(line))

plt.plot(data)
plt.show()
