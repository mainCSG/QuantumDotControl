import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
from matplotlib.widgets import Button, Slider, RangeSlider

import numpy as np
import scipy 
import os, csv

class Plotter:
    '''
    Class responsible for plotting all of the figures in the charge noise measurement software.
    '''
    def __init__(self) -> None:
        e, h = 1.602176634e-19, 6.62607015e-34 
        self.G0 = 2 * e**2 / h # Siemans (S)

    def plot_coulomb_oscillations(self, VST: np.array, ISD: np.array):
            '''
            Function responsible for plotting coulomb oscillations along with the interactive environment
            that allows thresholding the conductance.
            '''
            # Need to globalize these variables in order for the sliders to function properly.
            global threshold, results_sorted

            # Filter data as it is pretty noisy.
            ISD_filtered = scipy.ndimage.gaussian_filter1d(ISD, sigma=2)

            # For some reason there is an offset with the current, future fix?
            offset = np.abs(min(ISD_filtered))

            # Update the current to disregard the offset
            ISD += offset
            ISD_filtered += offset

            # Calculate conductance w.r.t VST
            G_raw = np.gradient(ISD, 1e-3 * (VST[1] - VST[0]))
            G = np.gradient(ISD_filtered, 1e-3 * (VST[1] - VST[0]))
            G_filtered = scipy.ndimage.gaussian_filter1d(G, sigma=2)

            # Initial threshold value
            threshold = max(G_filtered)

            # This bit of code is in charge of calculating all local and global maxima above a certain threshold
            maxima = scipy.signal.argrelextrema(G_filtered, np.greater)[0]
            maxima = maxima[G_filtered[maxima] >= threshold]
            if len(maxima) != 0:
                results = dict(zip(VST[maxima], G_filtered[maxima]))
                # sorted from lowest to highest conductance and store in dict for future use
                results_sorted = dict(sorted(results.items(), key=lambda item: item[1]))

            # Create figure and axes
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

            # Create sliders
            threshold_slider_ax = plt.axes([0.205, 0.1, 0.60, 0.03])
            current_sigma_slider_ax = plt.axes([0.205, 0.15, 0.60, 0.03])
            G_sigma_slider_ax = plt.axes([0.205, 0.2, 0.60, 0.03])
            save_button_ax = plt.axes([0.47, 0.01, 0.1, 0.05])
            
            # Adjust the bottom to make room for the sliders
            plt.subplots_adjust(bottom=0.3) 

            # Create buttons
            save_button = Button(save_button_ax, 'Save Data')
            threshold_slider = Slider(threshold_slider_ax, r'$G_{threshold} = 10^{x}G_0$  ', -5, 0, valinit=np.log10(max(G_filtered)/self.G0), valstep=0.01, color='black', initcolor='white')
            current_sigma_slider = Slider(current_sigma_slider_ax, r'$I_{SD}$ Filter', 0.1, 5.0, valinit=2.0, valstep=0.5, color='black', initcolor='white')
            G_sigma_slider = Slider(G_sigma_slider_ax, r'$G_{SD}$ Filter', 0.1, 5.0, valinit=2.0, valstep=0.5, color='black', initcolor='white')

            # Plot for I_SD vs V_ST
            ax1.set_title(r"$I_{SD}$")
            ax1.set_ylabel(r"$I_{SD}\ (A)$")
            ax1.plot(VST, ISD, 'k-', alpha=0.3,linewidth=0.75)
            ax1.plot(VST, ISD_filtered, 'k-', linewidth=1.5)

            # Plot for G vs V_ST
            ax2.set_title(r"$G_{SD}$")
            ax2.set_ylabel(r"$G_{SD}\ (S)$")
            ax2.set_xlabel(r"$V_{P}\ (mV)$")
            ax2.plot(VST, G_raw, 'k-', alpha=0.3, linewidth=0.75)
            ax2.plot(VST, G_filtered, 'k-', linewidth=1.5)
            ax2.hlines(y=threshold, xmin=VST[0], xmax=VST[-1], color='black', linestyle='--', linewidth=1.5)

            # Plot all the points of interest according to their sensitivity
            if len(maxima) > 0:
                for i in maxima:
                    index = list(results_sorted.keys()).index(VST[i])
                    if index == len(results_sorted)-1:
                        label = "High Sensitivity"
                        color = 'b'
                    elif index == 0: 
                        label = "Low Sensitivity"
                        color = 'r'
                    else:
                        label = "Medium Sensitivity"
                        color = 'g'
                    ax1.text(VST[i], ISD_filtered[i], f'{VST[i]:.2f} mV', color='black', fontsize=6, fontweight=750, ha='right', va='bottom', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5', alpha=0.25))
                    ax2.scatter(VST[i], G_filtered[i], c=color, label=label)
                    ax1.scatter(VST[i], ISD_filtered[i], c=color, label=label)
                    ax1.legend(loc='best')
                    self.legend_without_duplicate_labels(ax1)

            # Function to update the plot based on the slider values
            def update(val):

                global threshold, results_sorted

                # Read slider values
                threshold = 10**(threshold_slider.val) * self.G0
                current_sigma = current_sigma_slider.val
                G_sigma = G_sigma_slider.val

                # Update data according to new values and re-calculate maxima.
                ISD_filtered = scipy.ndimage.gaussian_filter1d(ISD, sigma=current_sigma)
                offset = min(ISD_filtered)
                ISD_filtered += np.abs(offset)
                G_filtered = scipy.ndimage.gaussian_filter1d(np.gradient(ISD_filtered, 1e-3 * (VST[1] - VST[0])), sigma=G_sigma)

                maxima = scipy.signal.argrelextrema(G_filtered, np.greater)[0]
                maxima = maxima[G_filtered[maxima] >= threshold]

                if len(maxima) != 0:
                    results = dict(zip(VST[maxima], G_filtered[maxima]))
                    # sorted from lowest to highest conductance
                    results_sorted = dict(sorted(results.items(), key=lambda item: item[1]))

                # Clear plots and re-draw
                ax1.clear()
                ax1.set_title(r"$I_{SD}$")
                ax1.set_ylabel(r"$I_{SD}\ (A)$")
                ax1.plot(VST, ISD, 'k--', alpha=0.3,linewidth=0.75)
                ax1.plot(VST, ISD_filtered, 'k-', linewidth=1.5)

                ax2.clear()
                ax2.set_title(r"$G_{SD}$")
                ax2.set_ylabel(r"$G_{SD}\ (S)$")
                ax2.set_xlabel(r"$V_{P}\ (mV)$")
                ax2.plot(VST, G_raw, 'k-', alpha=0.3, linewidth=0.75)
                ax2.plot(VST, G_filtered, 'k-', linewidth=1.5)
                ax2.hlines(y=threshold, xmin=VST[0], xmax=VST[-1], color='black', linestyle='--', linewidth=1.5)

                # plot flank points again 
                if len(maxima) > 0:
                    for i in maxima:
                        index = list(results_sorted.keys()).index(VST[i])
                        if index == len(results_sorted)-1:
                            label = "High Sensitivity"
                            color = 'b'
                        elif index == 0: 
                            label = "Low Sensitivity"
                            color = 'r'
                        else:
                            label = "Medium Sensitivity"
                            color = 'g'
                        ax1.text(VST[i], ISD_filtered[i], f'{VST[i]:.2f} mV', color='black', fontsize=6, fontweight=750, ha='right', va='bottom', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5', alpha=0.25))
                        ax2.scatter(VST[i], G_filtered[i], c=color, label=label)
                        ax1.scatter(VST[i], ISD_filtered[i], c=color, label=label)
                        ax1.legend(loc='best')
                        self.legend_without_duplicate_labels(ax1)
            
            def save_data(event):
                global results_sorted
                # Get the filename using a file dialog
                file_path = "./coulomb_oscillations.csv"

                # Write the data to the CSV file
                with open(file_path, 'w', newline='') as csvfile:
                    csv_writer = csv.writer(csvfile)
                    csv_writer.writerow(["VST (mV)", "G (S)"])  # Add headers if needed

                    # Write the data from results_sorted
                    for voltage, conductance in results_sorted.items():
                        csv_writer.writerow([round(voltage,3), "{:0.3e}".format(float(conductance))])

                print(f"Data saved to {file_path}")

            # Attach the update function to the sliders
            threshold_slider.on_changed(update)
            current_sigma_slider.on_changed(update)
            G_sigma_slider.on_changed(update)
            save_button.on_clicked(save_data)

            plt.show()

    def plot_lb_rb_scan(self, lb: np.array, rb: np.array, current: np.array):
        '''
        Very simple just plots barrier barrier scan if needed.
        '''
        fig, ax = plt.subplots(1, sharex=False)

        # Plot raw data
        ax.imshow(current, extent=[rb[0],rb[-1], lb[0], lb[-1]])
        # ax1.set_extent()
        ax.set_ylabel(r'LB (mV)')
        ax.set_title(r'BB Scan')
        ax.set_xlabel(r'RB (mV)')
        
        plt.tight_layout()  # Adjust layout to prevent overlap
        plt.show()

    def plot_current_and_spectrum(self, t: np.array, data: dict, sampling_rate: float):
        '''
        Plots the current and current spectrum for various sensitivities.
        '''
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=False)

        # Plot raw data
        for key, d in data.items():
            ax1.plot(t,d[1],label=d[2])
        ax1.set_ylabel(r'$I_{SD}\ (A)$')
        ax1.set_title(r'$I_{SD}\ (t)$')
        ax1.set_xlabel(r'$t$ (s)')
        
        # Plot noise spectrum
        for key, d in data.items():
            f, Pxx = scipy.signal.periodogram(d[1], fs=sampling_rate, scaling='density')
            ax2.loglog(f, Pxx, label=d[2])
        ax2.set_xlabel(r'$\omega$ (Hz)')
        ax2.set_ylabel(r'$S_I$ (A$^2$/Hz)')
        ax2.set_title(r'$S_I$')

        plt.tight_layout()  # Adjust layout to prevent overlap
        plt.legend(loc='best')
        plt.show()
  
    def plot_charge_noise_spectrum(self, t, data, lever_arm_G_dict, sampling_rate):
        '''
        Plots charge noise spectrum based on the formula given in https://doi.org/10.1038/s41467-023-36951-w.
        '''
        # Plot noise spectrum
        for key, d in data.items():
            f, Pxx = scipy.signal.periodogram(d[1], fs=sampling_rate, scaling='density')
            lever_arm, G_max = lever_arm_G_dict[d[2]]
            plt.loglog(f, (lever_arm / G_max)**2 * Pxx, label=d[2])
        plt.xlabel(r'$f$ (Hz)')
        plt.ylabel(r'$S_E$ (eV$^2$/Hz)')
        plt.title(r'$S_E$')

        plt.tight_layout()  # Adjust layout to prevent overlap
        plt.legend(loc='best')
        plt.show()

    def interactive2D(self,x,y,data, title='', xlabel='', ylabel=''):
        '''
        Function purely for the interactive lever arm measurements.
        '''
        global x_start, x_end, y_start, y_end, points, click_counter, slopes, lever_arms


        # Create initial figure and axis
        fig, ax = plt.subplots()

        fig.set_size_inches(7,5)
        plt.subplots_adjust(bottom=0.5)  # Adjust the bottom to make room for the slider

        ax_slider = plt.axes([0.205, 0.1, 0.60, 0.03])
        slider_x_ax = fig.add_axes([0.205, 0.2, 0.60, 0.03])
        slider_y_ax = fig.add_axes([0.205, 0.3, 0.60, 0.03])

        # Initialize start and end crop
        x_start, x_end, y_start, y_end = x[0], x[-1], y[0], y[-1]
        
        # Create sliders
        slider_x = RangeSlider(slider_x_ax, "VST (mV)", x[0], x[-1], valstep=1, valinit=(x[0],x[-1]), color='black')
        slider_y = RangeSlider(slider_y_ax, "VSD (mV)", y[0], y[-1],valstep=0.1, valinit=(y[0],y[-1]), color='black')
        slider = Slider(ax_slider, r'$G_{filter} = 10^{x}G_0$  ', -2, 7, valstep=0.1, valinit=np.log10(1/self.G0), color='black', initcolor='white')

        # Crop and show data
        data,_ = self.crop_data_by_values(data,x,y,x_start,x_end,y_start,y_end)
        im = ax.imshow(data,
                   origin='lower',
                   aspect='auto',
                   cmap='binary',
                   extent=[x_start, x_end, y_start, y_end]
                )
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        # Initialize data structures 
        click_counter = 0
        points = []
        slopes = []
        lever_arms = []
        
        # Function to update the data based on the slider value
        def update(val):
            global y_start, y_end, x_start, x_end
            y_start, y_end = slider_y.val
            x_start, x_end = slider_x.val

            new_data,_ = self.crop_data_by_values(data,x,y,x_start,x_end,y_start,y_end)

            value = (10**(slider.val)) * self.G0
            updated_data = self.filter_conductance(new_data, value)
            im.set_data(updated_data)
            im.set_extent([x_start, x_end, y_start, y_end])

            ax.set_xlim(x_start,x_end)
            ax.set_ylim(y_start,y_end)

            fig.canvas.draw_idle()

        # Function for when points are clicked on the axes.
        def on_pick(event):
            global points, click_counter, slopes, lever_arms
            lines = []

            if event.inaxes == ax:

                if event.button is MouseButton.LEFT:
                    click_counter += 1

                if click_counter > 2:
                    click_counter = 1
                    points = [event.xdata, event.ydata]

                else:
                    points.extend([event.xdata, event.ydata])

                    if click_counter == 2:
                        # User has clicked twice, so calculate the slope between those two points
                        slope = calculate_slope()
                        slopes.append(slope)
                        text,line_data = plot_line(points, slope)
                        lines.append(line_data)
                        points = []
                        click_counter = 0

                    if len(slopes) == 2:
                        # Two slopes have been calculate, now just calculate the lever arm
                        lever_arm = 1/(np.abs(1/slopes[0]) + np.abs(1/slopes[1]))
                        lever_arms.append(lever_arm)

                        # display on the axes
                        ax.text(1, 1, f'\u03B1 -> {round(np.mean(lever_arms), 3)} \u00B1 {round(np.std(lever_arms), 3)} (eV/V)',
                                color='black', fontsize=10, fontweight=500, ha='right', va='top',
                                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'),
                                transform=ax.transAxes)

                        # reset for next round
                        click_counter = 0
                        slopes = []
                        points = []
                        line_data = []

                        # Print the statisitics to the user
                        os.system('cls' if os.name == 'nt' else 'clear')
                        statistics = [
                            f'\u03B1avg -> {round(np.mean(lever_arms),3)} (eV/V)',
                            f'\u03C3(\u03B1) -> {round(np.std(lever_arms),3)} (eV/V)',
                        ]

                        self.pprint(statistics)
            else:
                pass

        def plot_line(points, slope):
            # Plots a line between the two points
            x1, y1, x2, y2 = points
    
            ax.plot([x1, x2], [y1, y2], 'r-' if slope > 0 else 'b-', alpha=0.5, marker='o',markersize=5, linewidth=3)
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
            # Calculates the slope between the global variable of points
            x1, y1, x2, y2 = points
            slope = (y2 - y1) / (x2 - x1)
            return slope

        # Connect the slider's event to the update function
        slider.on_changed(update)
        slider_x.on_changed(update)
        slider_y.on_changed(update)
        plt.connect('button_press_event', on_pick)

        plt.show()
    
    def pprint(self,lines: list, alignment_char=None):
        '''
        Prints a list of strings in a nice way.
        '''
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
        '''
        Function that crops a 2D dataset within a provided set of bounds.
        '''
        # Create boolean masks for the window
        x_mask = (x >= x_start) & (x <= x_end)
        y_mask = (y >= y_start) & (y <= y_end)

        x_crop_mask = x[x_mask]
        y_crop_mask = y[y_mask]

        x_min_index = np.where(x == x_crop_mask[0])[0][0]
        x_max_index = np.where(x == x_crop_mask[-1])[0][0]
        y_min_index = np.where(y == y_crop_mask[0])[0][0]
        y_max_index = np.where(y == y_crop_mask[-1])[0][0]

        data_crop = data[y_min_index:y_max_index, x_min_index:x_max_index]
        return data_crop, [x_min_index, x_max_index, y_min_index, y_max_index]
    
    def filter_conductance(self, data, A=None):
        '''
        Filters conductance to bring out the diamond features.
        '''
        return np.sign(data) * np.log((np.abs(data)/A) + 1)
    
    def legend_without_duplicate_labels(self, ax):
        # Helper function to prevent duplicates in the legend.
        handles, labels = ax.get_legend_handles_labels()
        unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
        ax.legend(*zip(*unique))