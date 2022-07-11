import matplotlib.pyplot as plt
from IPython.display import Image, display
import numpy as np
import time
import uuid
import os
import warnings
class NotebookFigure():
    """An extension of matplotlib figure to work with jupyter notebook's display module.
    Works by storing the figure as an image on disk and loading it to display on updating.
    """
    def __init__(self,image_path=None,nrows=1,ncols=1,decorate_fn=None,**subplot_kwargs):
        """
        image_path: the filename of the stored figure image
        decorate_fn: A function which takes in a 2d array of axes, and should perform
            basic plot decorations like labeling, title etc. It is called once.
            
        subplot_kwargs are passed to subplots' kwargs
        """
        
        self.nrows=nrows
        self.ncols=ncols
        self.fig, self.axes = plt.subplots(nrows,ncols,**subplot_kwargs)
        
        self.axes=np.array([self.axes]).reshape(nrows,ncols)
        
        if decorate_fn is not None:
            decorate_fn(self.axes)
        
        plt.close(self.fig)
        
        
        self.image_path = image_path
        
            
        self.disp = None
        
        self.xlims = [[(None,None) for j in range(ncols)] for i in range(nrows)]
        self.ylims = [[(None,None) for j in range(ncols)] for i in range(nrows)]
        
        if self.image_path is None:
            self.temp_image_path = str(uuid.uuid4()) + "you_should_not_have_this_name.png"
        self.save_fig()

        self.plot = self.getAxis().plot
        self.imshow = self.getAxis().imshow

    def __del__(self):
        temp_image_path = getattr(self, 'temp_image_path', None)
        if temp_image_path is not None:
            if os.path.isfile(temp_image_path):
                os.remove(temp_image_path)
    
     
    def set_image_path(self, image_path):
        """Sets the path for saving the figure image. This path must be set(either in init
        or using this function) before display can be called."""
        
        self.image_path = image_path
        self.fig.savefig(image_path, bbox_inches='tight')
    
    def update_lims(self):
        """
        Updates lims of all axes
        """
        for i in range(self.nrows):
            for j in range(self.ncols):
                ax = self.axes[i][j]
                ax.relim()
#                 ax.autoscale_view()
                ax.autoscale()
                ax.set_xlim(self.xlims[i][j])
                ax.set_ylim(self.ylims[i][j])
                
    def set_xlim(self,xlim,axis_num=0):
        """Set the xlims of the axes indexed row major wise starting from 0.
        Effect will be visible only after update.
        Setting either limit to None will make that limit update automatically."""
        row = axis_num // self.ncols
        col = axis_num % self.ncols
        self.xlims[row][col]=xlim
        self.update()

    def set_ylim(self,ylim,axis_num=0):
        """Set the ylims of the axes indexed row major wise, starting from 0.
        Effect will be visible only after update.
        Setting either limit to None will make that limit update automatically."""
        row = axis_num // self.ncols
        col = axis_num % self.ncols
        self.ylims[row][col]=ylim
        self.update()
        
    def display(self):
        """Create a display of the figure"""
        image_path = self.image_path if self.image_path is not None else self.temp_image_path
        self.disp = display(Image(image_path),display_id=f"{id(self)}_{time.time()}")
    
    def update(self, update_lims=True):
        """Update the lims(if set to True) and update all display instances"""
        if update_lims:
            self.update_lims()
        if self.disp is None:
            warnings.warn("The figure has not been displayed. Please display before update")
            return
        image_path = self.image_path if self.image_path is not None else self.temp_image_path
        self.fig.savefig(image_path, bbox_inches='tight')
        self.disp.update(Image(image_path))
    
    def getAxis(self,axis_num=0):
        """Get the axis in the subplot indexed row major wise starting from 0."""
        row = axis_num // self.ncols
        col = axis_num % self.ncols
        return self.axes[row][col]
    
    def save_fig(self):
        image_path = self.image_path if self.image_path is not None else self.temp_image_path
        self.fig.savefig(image_path, bbox_inches='tight')

    def show(self):
        self.display()
        self.update()

    def clear(self):
        for i in range(self.nrows):
            for j in range(self.ncols):
                self.axes[i, j].clear()
