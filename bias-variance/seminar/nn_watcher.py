import matplotlib.pyplot as plt
import numpy as np
import os.path as osp

__all__ = [
  'NNWatcher',
  'SNNWatcher'
]

from IPython import display

class NNWatcher(object):
  limit = 2 ** 15

  def __init__(self, title, labels=('loss', ), colors=('blue', ), mode='full',
               figsize=(9, 6), save_dir='./', plot_mode='inline'):
    self.save_dir = save_dir
    self.plot_mode = plot_mode
    self.mode = mode

    self.fig = plt.figure(figsize=figsize)
    self.ax = self.fig.add_subplot(111)

    self.ax.set_xlim([0.0, 1.0])
    self.ax.set_ylim([0.0, 1.0])

    self.mean_lines = []
    self.lines = []

    self.fig.suptitle(title)
    self.title = title

    for label, color in zip(labels, colors):
      self.mean_lines.append(
        self.ax.plot([], [], label=label, color=color)[0]
      )

      if mode is 'full':
        self.lines.append(
          self.ax.plot([], [], alpha=0.5, color=color)[0]
        )
      elif mode is 'mean':
        self.lines.append(None)

    self.ax.legend()

  @classmethod
  def _get_ylim(cls, data):
    trends = [np.mean(d, axis=1) for d in data]

    min_trend = np.min([np.min(trend) for trend in trends])
    max_trend = np.max([np.max(trend) for trend in trends])
    s_trend = 0.05 * (max_trend - min_trend)

    s = np.max([np.std(d - trend[:, None]) for d, trend in zip(data, trends)])
    min_data = np.min([np.percentile(d, q=2) for d in data])
    max_data = np.max([np.percentile(d, q=98) for d in data])

    lower_bound = np.min([min_data - s, min_trend - s_trend])
    upper_bound = np.max([max_data + s, max_trend + s_trend])

    return lower_bound, upper_bound


  def draw(self, *data):
    def crop(d):
      epoch_size = np.prod(d.shape[1:])
      lim = self.limit // epoch_size

      return d[-lim:]

    if self.plot_mode == 'inline':
      display.clear_output(wait=True)

    data = [ crop(d) for d in data ]

    x_lim = np.max([d.shape[0] for d in data])
    self.ax.set_xlim(0.0, x_lim)

    y_lower, y_upper = self._get_ylim(data)
    self.ax.set_ylim([y_lower, y_upper])

    for d, line, mean_line in zip(data, self.lines, self.mean_lines):
      trend = np.mean(d, axis=1)

      mean_line.set_xdata(np.arange(d.shape[0]) + 0.5)
      mean_line.set_ydata(trend)

      if self.mode == 'full':
        xs = np.linspace(0, d.shape[0], num=int(np.prod(d.shape)))
        line.set_xdata(xs)
        line.set_ydata(d)

    if self.plot_mode == 'inline':
      display.display(self.fig)
    else:
      self.fig.canvas.draw()

    self.fig.savefig(osp.join(self.save_dir, '%s.png' % self.title), dpi=420)

class SNNWatcher(object):
  limit = 2 ** 15

  def __init__(self, title, labels=('loss', ), colors=('blue', ), mode='full',
               figsize=(9, 6), save_dir='./', plot_mode='inline'):
    self.save_dir = save_dir
    self.plot_mode = plot_mode
    self.mode = mode

    self.fig = plt.figure(figsize=figsize)
    self.ax = self.fig.add_subplot(111)

    self.ax.set_xlim([0.0, 1.0])
    self.ax.set_ylim([0.0, 1.0])

    self.fig.suptitle(title)
    self.title = title

    self.colors = colors
    self.labels = labels
    self.drawn = False

  @classmethod
  def _get_ylim(cls, data):
    trends = [np.mean(d, axis=1) for d in data]

    min_trend = np.min([np.min(trend) for trend in trends])
    max_trend = np.max([np.max(trend) for trend in trends])
    s_trend = 0.05 * (max_trend - min_trend)

    s = np.max([np.std(d - trend[:, None]) for d, trend in zip(data, trends)])
    min_data = np.min([np.percentile(d, q=5) for d in data])
    max_data = np.max([np.percentile(d, q=95) for d in data])

    lower_bound = np.min([min_data - s, min_trend - s_trend])
    upper_bound = np.max([max_data + s, max_trend + s_trend])

    return lower_bound, upper_bound


  def draw(self, *data):
    def crop(d):
      if self.mode == 'full':
        epoch_size = np.prod(d.shape[1:])
        lim = self.limit // epoch_size
        return d[-lim:]
      else:
        return d

    self.ax.clear()

    if self.plot_mode == 'inline':
      display.clear_output(wait=True)

    data = [ crop(d) for d in data ]

    x_lim = np.max([d.shape[0] for d in data])
    self.ax.set_xlim(0.0, x_lim)

    y_lower, y_upper = self._get_ylim(data)
    self.ax.set_ylim([y_lower, y_upper])

    for d, color, label in zip(data, self.colors, self.labels):
      trend = np.mean(d, axis=1)

      iters = np.arange(d.shape[0]) + 0.5
      self.ax.plot(iters, trend, label=label, color=color)

      if self.mode == 'full':
        xs = np.linspace(0, d.shape[0], num=int(np.prod(d.shape)))
        self.ax.plot(xs, d.ravel(), color=color, alpha=0.5)

      if self.mode == 'fill':
        lower1 = np.percentile(d, q=20, axis=1)
        upper1 = np.percentile(d, q=80, axis=1)
        lower2 = np.percentile(d, q=10, axis=1)
        upper2 = np.percentile(d, q=90, axis=1)
        self.ax.fill_between(iters, lower1, upper1, alpha=0.2, color=color)
        self.ax.fill_between(iters, lower2, upper2, alpha=0.1, color=color)

      if self.mode == 'avg' or self.mode == 'mean':
        pass

    if not self.drawn:
      self.ax.legend()
      self.drawn = False

    if self.plot_mode == 'inline':
      display.display(self.fig)
    else:
      self.fig.canvas.draw()

    self.fig.savefig(osp.join(self.save_dir, '%s.png' % self.title), dpi=420)