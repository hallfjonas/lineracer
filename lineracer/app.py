import tkinter

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

from Race import *

class App:
    def __init__(self):           
        self.root = tkinter.Tk()
        self.root.wm_title("Embedding in Tk")

        self.fig, self.ax = plt.subplots()
        self.fig.tight_layout()
        self.ax.set_aspect('equal')
        self.ax.set_axis_off()
    
        # tool tip to update control
        self.predicted, = self.ax.plot([], [], '-', alpha=0.8)
        self.predicted_uncontrolled, = self.ax.plot([], [], '--', alpha=1.0)
        
        # generate canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)

        self.canvas.mpl_connect("button_release_event", self.on_mouse_release)
        self.canvas.mpl_connect("motion_notify_event", self.on_mouse_hover)

        self.quit_button = tkinter.Button(master=self.root, text="Quit", command=self._quit)
        self.quit_button.pack(side=tkinter.BOTTOM)

        self.regenerate_button = tkinter.Button(master=self.root, text="Regenerate", command=self.reset_racetrack)
        self.regenerate_button.pack(side=tkinter.BOTTOM)

        self.track_plot = None
        self.initialize_race()

    def initialize_race(self):
        self.race = Race(n_vehicles=2)
        self.vehicle_plot_data = {}
        for v in self.race.vehicles: 
            vpd = self.vehicle_plot_data[v] = {}
            vpd['pos'] = self.ax.plot(v.position[0], v.position[1], v.marker, color=v.color)[0]
            vpd['hp'] = self.ax.plot(v.position[0], v.position[1], '-', color=v.color)[0]
            vpd['x_history'] = [v.position[0]]
            vpd['y_history'] = [v.position[1]]
            vpd['progress'] = tkinter.Label(self.root,text='Progress: 0%',fg=v.color)
            vpd['progress'].pack(fill=tkinter.BOTH)

        self.reset_racetrack()
        
    def reset_racetrack(self):
        if self.track_plot is not None:
            self.track_plot.remove()

        # regenerate and plot track 
        self.race.set_track(RaceTrack.generate_random_track())
        self.track_plot, = self.race.track.plot_track(ax=self.ax)
        lims = self.race.track.get_limits()
        self.ax.set_xlim(lims[0])
        self.ax.set_ylim(lims[1])

        # reset the vehicles
        for v in self.race.vehicles: 
            v.reset()
            vpd = self.vehicle_plot_data[v]
            vpd['pos'].set_data([v.position[0]], [v.position[1]])
            vpd['hp'].set_data([v.position[0]], [v.position[1]])
            vpd['x_history'] = [v.position[0]]
            vpd['y_history'] = [v.position[1]]        
        self.predicted.set_data([], [])
        self.canvas.draw()

    def update_control(self, event):
        cv = self.race.get_cv()
        next_p = cv.position + cv.velocity
        # find the feasible control action that leads to the closest point to the mouse
        min_dist = np.inf
        cv.u = None
        for control in cv.get_feasible_controls():
            p = next_p + control
            dist = np.linalg.norm(np.array(p) - np.array([event.xdata, event.ydata]))
            if dist < min_dist:
                min_dist = dist
                cv.u = control

    def on_mouse_release(self, event):
        if event.xdata is None:
            return
        cv = self.race.get_cv()
        if cv.u is not None:
            cv.update()
            vpd = self.vehicle_plot_data[cv]
            vpd['pos'].set_data([cv.position[0]], [cv.position[1]])
            vpd['x_history'].append(cv.position[0])
            vpd['y_history'].append(cv.position[1])
            vpd['hp'].set_data(self.vehicle_plot_data[cv]['x_history'], self.vehicle_plot_data[cv]['y_history'])
            vpd['progress'].config(text=f"Progress: {round(cv.track.lap_progress(cv.position)*100)}%")
            self.canvas.draw()
            cv.u = None
            new_cv = self.race.next_cv()
            np = new_cv.position
            npu = new_cv.position + new_cv.velocity
            self.predicted_uncontrolled.set_data([np[0], npu[0]], [np[1], npu[1]])
            self.predicted_uncontrolled.set_color(new_cv.color)
            self.canvas.draw()

    def on_mouse_hover(self, event):
        if event.xdata is None:
            return
        cv = self.race.get_cv()
        self.update_control(event)
        next_p = cv.position + cv.velocity + cv.u
        self.predicted.set_data([cv.position[0], next_p[0]], [cv.position[1], next_p[1]])
        self.predicted.set_color(cv.color)
        self.canvas.draw()

    def _quit(self):
        self.root.quit()
        self.root.destroy()

if __name__ == '__main__':
    app = App()
    tkinter.mainloop()
    