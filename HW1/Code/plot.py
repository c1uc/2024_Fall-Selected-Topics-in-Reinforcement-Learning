import seaborn as sns

def plot(file_name):
    with open(file_name) as f:
        lines = f.readlines()
        y = [float(line.strip()) for line in lines]
        y_mean_1000 = [sum(y[i:i+1000])/1000 for i in range(0, len(y), 1000)]
        y = y_mean_1000
        x = [_ * 1000 for _ in range(len(y))]
        g = sns.lineplot(x=x, y=y)
        g.set(xlabel='Iteration', ylabel='Mean Reward')
        g.get_figure().savefig('value.png')


if __name__ == '__main__':
    plot('../value.txt')
