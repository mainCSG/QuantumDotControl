import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
from matplotlib.widgets import Button, Slider, TextBox, RangeSlider

import numpy as np

import os

class Plotter:
    def __init__(self) -> None:
        e, h = 1.602176634e-19, 6.62607015e-34 
        self.G0 = 2 * e**2 / h # Siemans (S)

    def interactive2D(self,x,y,data, title='', xlabel='', ylabel=''):
        # Create initial figure and axis
        fig, ax = plt.subplots()
        # fig.dpi = 250
        fig.set_size_inches(7,5)
        plt.subplots_adjust(bottom=0.5)  # Adjust the bottom to make room for the slider
        # Add a slider to control the value parameter
        ax_slider = plt.axes([0.205, 0.1, 0.60, 0.03])
        slider_x_ax = fig.add_axes([0.205, 0.2, 0.60, 0.03])
        slider_y_ax = fig.add_axes([0.205, 0.3, 0.60, 0.03])

        global x_start, x_end, y_start, y_end
        x_start, x_end, y_start, y_end = x[0], x[-1], y[0], y[-1]
        
        slider_x = RangeSlider(slider_x_ax, "VST (mV)", x[0], x[-1], valstep=1, valinit=(x[0],x[-1]), color='black')
        slider_y = RangeSlider(slider_y_ax, "VSD (mV)", y[0], y[-1],valstep=0.1, valinit=(y[0],y[-1]), color='black')
        slider = Slider(ax_slider, r'$G_{filter} = 10^{x}G_0$  ', -2, 7, valstep=0.1, valinit=np.log10(1/self.G0), color='black', initcolor='white')

        data,_ = self.crop_data_by_values(data,x,y,x_start,x_end,y_start,y_end)
        im = ax.imshow(data,
                   origin='lower',
                   aspect='auto',
                   cmap='binary',
                   extent=[x_start, x_end, y_start, y_end])
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        global points, click_counter, slopes, lever_arms
        click_counter = 0
        points = []
        slopes = []
        lever_arms = []
        
        # Define a function to update the data based on the slider value
        def update(val):
            global y_start, y_end
            y_start, y_end = slider_y.val
            global x_start, x_end
            x_start, x_end = slider_x.val

            new_data,_ = self.crop_data_by_values(data,x,y,x_start,x_end,y_start,y_end)

            value = (10**(slider.val)) * self.G0
            updated_data = self.filter_conductance(new_data, value)
            im.set_data(updated_data)
            im.set_extent([x_start, x_end, y_start, y_end])

            ax.set_xlim(x_start,x_end)
            ax.set_ylim(y_start,y_end)

            fig.canvas.draw_idle()

        def on_pick(event):
            global points, click_counter, slopes, lever_arms
            # if (event.inaxes != ax_slider) and (event.inaxes != slider_x_ax) and (event.inaxes != slider_y_ax):
            if event.inaxes == ax:
                if event.button is MouseButton.LEFT:
                    click_counter += 1
                if click_counter > 2:
                    click_counter = 1
                    points = [event.xdata, event.ydata]
                else:
                    points.extend([event.xdata, event.ydata])
                    if click_counter == 2:
                        slope = calculate_slope()
                        slopes.append(slope)
                        text,line_data = plot_line(points, slope)
                        points = []
                        click_counter = 0

                    if len(slopes) == 2:
                        lever_arm = 1/(np.abs(1/slopes[0]) + np.abs(1/slopes[1]))
                        lever_arms.append(lever_arm)

                        text.append(f'\u03B1{len(lever_arms)} -> {round(lever_arm,3)} (eV/V)')
                        self._pprint(text)
                        
                        click_counter = 0
                        slopes = []
                        points = []

                        os.system('cls' if os.name == 'nt' else 'clear')

                        statistics = [
                            f'\u03B1avg -> {round(np.mean(lever_arms),3)} (eV/V)',
                            f'\u03C3(\u03B1) -> {round(np.std(lever_arms),3)} (eV/V)',
                        ]
                        self._pprint(statistics)
            else:
                pass

        def plot_line(points, slope):
            global x_start, x_end, y_start, y_end
            x1, y1, x2, y2 = points
    
            ax.plot([x1, x2], [y1, y2], 'r-' if slope > 0 else 'b-', alpha=0.5, marker='o',markersize=2)
            fig.canvas.draw_idle()
            slope_label = 'ms' if slope > 0 else 'md'

            data = [x1,y1,x2,y2,slope]
            text = [
                f"P1: ({round(x1,3)} (mV), {round(y1,3)} (uV))",
                f"P2: ({round(x2,3)} (mV), {round(y2,3)} (uV))",
                f"{slope_label}: {round(slope,3)} (V/V)",
            ]
            return text, data

        def calculate_slope():
            x1, y1, x2, y2 = points
            slope = (1e-3*(y2 - y1)) / (1e-3*(x2 - x1))
            return slope

        # Connect the slider's event to the update function
        slider.on_changed(update)
        slider_x.on_changed(update)
        slider_y.on_changed(update)
        plt.connect('button_press_event', on_pick)

        plt.show()

          
    def _pprint(self,lines, alignment_char=None):
    
        max_line_length = max(len(line) for line in lines)
        box_width = max_line_length + 4  # Adjust the width based on the longest line and padding
        horizontal_line = '┌' + '─' * (box_width - 2) + '┐'
        bottom_line = '└' + '─' * (box_width - 2) + '┘'

        print(horizontal_line)
        for line in lines:
            padding = ' ' * (max_line_length - len(line))
            print(f'│ {line}{padding} │')
        print(bottom_line)

    def crop_data_by_values(self, data, x, y, x_start, x_end, y_start, y_end):

        # Create boolean masks for the window
        x_mask = (x >= x_start) & (x <= x_end)
        y_mask = (y >= y_start) & (y <= y_end)

        x_crop_mask = x[x_mask]
        y_crop_mask = y[y_mask]

        x_min_index = np.where(x == x_crop_mask[0])[0][0]
        x_max_index = np.where(x == x_crop_mask[-1])[0][0]
        y_min_index = np.where(y == y_crop_mask[0])[0][0]
        y_max_index = np.where(y == y_crop_mask[-1])[0][0]

        data_crop = data[x_min_index:x_max_index, y_min_index:y_max_index]
        return data_crop, [x_min_index, x_max_index, y_min_index, y_max_index]
    
    def filter_conductance(self, data, A=None):
        return np.sign(data) * np.log((np.abs(data)/A) + 1)
    