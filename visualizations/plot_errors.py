import os, pickle, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def read_log(path):
    lines = open(path).readlines()[1:]
    values = []
    for line in lines:
        values.append( float(line) )
    return values

def find_min_length(logs):
    min_length = float('inf')
    for item in logs['albedo']:
        if len(item) < min_length:
            min_length = len(item)
    return min_length 

def truncate_logs(logs, min_length):
    new = {}
    for key in logs:
        new[key] = []
        for item in logs[key]:
            new[key].append( item[:min_length] )
    return new

def extract_stats(label_logs):
    zipped = zip(*label_logs)
    means = [np.mean(i) for i in zipped]
    std = [np.std(i) for i in zipped]
    return means, std

def make_color(vals):
    return [i/255. for i in vals]

def plot_logs(prefix, ax, labels):
    logs = {label: [] for label in labels}
    for ind in range(2, num_logs+1):
        folder = os.path.join(log_folder, prefix + str(ind))
        for label in labels:
            filepath = os.path.join(folder, label + '_err.log')
            errors = read_log(filepath)
            logs[label].append(errors)

    min_length = find_min_length(logs)
    print logs
    print min_length
    logs = truncate_logs(logs, min_length)

    print logs

    x = range(min_length)
    for label in labels:
        print label
        means, std = extract_stats(logs[label])
        lower = [means[i]-std[i] for i in range(len(means))]
        upper = [means[i]+std[i] for i in range(len(means))]
        print x
        print means

        ax.plot(x, means, linewidth=3, color=colors[label], label=fig_labels[label])
        ax.fill_between(x, lower, upper, color=colors[label], alpha=0.25)

    if 'normals' not in labels:
        ax.plot(x, [-0.005 for i in x], linewidth=3, color=colors['normals'], label='Shape')

    ax.tick_params(
        axis='both',
        which='both',
        top='off',
        bottom='off',
        left='off',
        right='off')
    # ax = plt.subplot(111)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

log_folder = '../skip/'
labels = ['albedo', 'normals', 'shading', 'render']
comparison_labels = ['albedo', 'shading', 'render']
blue = [55,52,200]
red = [204,70,211]
green=[80,200,100]
cyan=[112,173,246]
colors = {'albedo': make_color(red), 'normals': make_color(green), 'shading': make_color(blue), 'render': make_color(cyan)}
fig_labels = {'albedo': 'Reflectance', 'normals': 'Shape', 'shading': 'Shading', 'render': 'Rendered'}

folder = 'saved_logs_normed/log_bottle_car_10:0,1,0,5_0.001_'
comparison = 'saved_alternate/log_bottle_car_10:1,1,1_0.0001_'

num_logs = 5

fig, (ax1,ax2) = plt.subplots(1,2, sharey=True)

ax1.set_title('with shader', fontsize=14)
ax2.set_title('without shader', fontsize=14)
ax1.set_ylabel('Error  ($L^2$norm)', fontsize=12)
ax1.set_xlabel('Epoch', fontsize=12)
ax2.set_xlabel('Epoch', fontsize=12)

sup = fig.suptitle('Transfer Learning', fontsize=18, fontweight='bold')
plt.subplots_adjust(top=0.8)

for ax in [ax1,ax2]:
    # ax.axis([0,30,.0005,.014])
    ax.set_xlim([0,20])
    ax.set_ylim([0, 0.025])
    plt.sca(ax)
    plt.yticks(np.arange(0, 0.03, 0.005))
    plt.xticks(np.arange(0,30,10))


plot_logs(folder, ax1, labels)
plot_logs(comparison, ax2, comparison_labels)


handles, labels = ax.get_legend_handles_labels()
h = [handles[i] for i in [0, 3, 1, 2]]
l = [labels[i] for i in [0, 3, 1, 2]]
lgd = plt.legend(h, l, bbox_to_anchor=(1.2, -.225), fontsize=12, ncol=4, frameon=False)
fig.set_size_inches(6,4)
plt.savefig('plots/errors_normed.png', bbox_extra_artists=(sup,lgd,), bbox_inches='tight')
# plt.savefig('plots/errors.png', bbox_inches='tight')
