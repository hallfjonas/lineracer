"""The main application for the line racer."""

# external imports
import tkinter
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np

# internal imports
from lineracer.Race import Race
from lineracer.RaceTrack import RaceTrack
from lineracer.PlotObjects import PlotObject

class App:
    def __init__(self):
        self.root = tkinter.Tk()
        self.root.wm_title("Embedding in Tk")

        self.fig, self.ax = plt.subplots()
        self.fig.tight_layout()
        self.ax.set_aspect('equal')
        self.ax.set_axis_off()

        # tool tip to update control
        self.predicted: PlotObject = PlotObject(self.ax.plot([], [], '-', alpha=0.8))
        self.predicted_uncontrolled: PlotObject = PlotObject(self.ax.plot([], [], '--', alpha=1.0))

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
        """Initialize the race."""
        self.race = Race()
        self.vehicle_plot_data = {}
        for v in self.race.vehicles:
            vpd = self.vehicle_plot_data[v] = {}
            vpd['progress'] = tkinter.Label(self.root,text='Progress: 0%',fg=v.color)
            vpd['progress'].pack(fill=tkinter.BOTH)
        self.reset_racetrack()

    def reset_racetrack(self):
        """Reset the racetrack and vehicles."""
        if self.track_plot is not None:
            self.track_plot.remove()
            for v in self.race.vehicles:
                vpd = self.vehicle_plot_data[v]
                if 'pos' in vpd and vpd['pos'] is not None:
                    vpd['pos'].remove()
                    vpd['hp'].remove()

        # regenerate and plot track
        self.race.set_track(RaceTrack.generate_random_track())
        self.track_plot: PlotObject = self.race.track.plot_track(ax=self.ax)
        lims = self.race.track.get_limits()
        self.ax.set_xlim(lims[0])
        self.ax.set_ylim(lims[1])

        # reset the vehicles
        for v in self.race.vehicles:
            v.reset()
            vpd = self.vehicle_plot_data[v]
            vpd['pos'] = PlotObject(self.ax.plot(v.position[0], v.position[1], v.marker, color=v.color))
            vpd['hp'] = PlotObject(self.ax.plot(v.trajectory[0, :], v.trajectory[1, :], '-', color=v.color))
            vpd['progress'].config(text=f"Progress: {round(v.track.lap_progress(v.position)*100)}%")


        self.predicted.first().set_data([], [])
        self.predicted_uncontrolled.first().set_data([], [])
        self.canvas.draw()

    def update_control(self, event):
        """Update the control action based on the mouse position.

        Args:
            event: The mouse event.
        """
        cv = self.race.get_cv()
        next_p = cv.position + cv.velocity
        # find the feasible control action that leads to the closest point to the mouse
        min_dist = np.inf
        for control in cv.get_feasible_controls():
            p = next_p + control
            dist = np.linalg.norm(np.array(p) - np.array([event.xdata, event.ydata]))
            if dist < min_dist:
                min_dist = dist
                cv.controller.set_control(control)

    def update_vehicle(self, vehicle):
        vehicle.update()
        vpd = self.vehicle_plot_data[vehicle]
        vpd['pos'].first().set_data([vehicle.position[0]], [vehicle.position[1]])
        vpd['hp'].first().set_data(vehicle.trajectory[0,:], vehicle.trajectory[1,:])
        progress = round(vehicle.track.lap_progress(vehicle.position)*100)
        vpd['progress'].config(text=f"Progress: {progress}%")
        self.canvas.draw()

    def on_mouse_release(self, event):
        """Update vehicle based on the current control action.

        Args:
            event: The release event.
        """
        if event.xdata is None:
            return

        # handle current player vehicle
        self.update_vehicle(self.race.get_cv())

        # handle all NPC vehicles until we reach the next player vehicle
        while True:
            new_cv = self.race.next_cv()
            if new_cv.is_player:
                break

            # otherwise update NPC action
            new_cv.controller.compute_control(new_cv.position, new_cv.velocity)
            self.update_vehicle(new_cv)

        # update the predicted uncontrolled vehicle
        np = new_cv.position
        npu = new_cv.position + new_cv.velocity
        self.predicted_uncontrolled.first().set_data([np[0], npu[0]], [np[1], npu[1]])
        self.predicted_uncontrolled.first().set_color(new_cv.color)
        self.canvas.draw()

    def on_mouse_hover(self, event):
        """Update the current vehicle control action based on the mouse position.

        Args:
            event: The hover event.
        """
        if event.xdata is None:
            return
        cv = self.race.get_cv()
        self.update_control(event)
        next_p = cv.position + cv.velocity + cv.controller.get_control()
        self.predicted.first().set_data([cv.position[0], next_p[0]], [cv.position[1], next_p[1]])
        self.predicted.first().set_color(cv.color)
        self.canvas.draw()

    def _quit(self):
        """Quit the application."""
        self.root.quit()
        self.root.destroy()

if __name__ == '__main__':
    app = App()
    tkinter.mainloop()
