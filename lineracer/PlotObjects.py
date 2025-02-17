
# external imports
import matplotlib.pyplot as plt

'''
PlotObject: A container for matplotblib objects.

Created on: Mar 28 2024

@author: Jonas Hall

@details: This container class can be used to store matplotlib objects to 
    simplify their removal later on.
'''
class PlotObject:
    def __init__(self, *args) -> None:
        self._objs = []
        self.add(*args)

    def add(self, *args) -> None:
        for obj in args:
            if isinstance(obj, PlotObject):
                self.add(obj._objs)
            else:
                try:
                    for o in obj:
                        self.add(o)
                except:
                    self._objs.append(obj)

    def remove(self) -> None:
        for obj in self._objs:
            obj.remove()
        self._objs.clear()


'''
getAxes: Get the current axis.

Created on: Apr 1 2024

@author: Jonas Hall

@details: Returns the current axis if ax is None, otherwise returns ax.
'''
def getAxes(ax : plt.Axes = None) -> plt.Axes:
    if ax is None:
        ax = plt.gca()
    if not isinstance(ax, plt.Axes):
        raise ValueError("Expected ax to be of type matplotlib.pyplot.Axes.")
    return ax
