import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
from matplotlib.widgets import Button, Slider, TextBox, RangeSlider

import os

from scipy.optimize import curve_fit
import skimage

import sympy as sp
import scipy 

class ChargeNoiseExtractor:
    def __init__(self) -> None:
        e, h = 1.602176634e-19, 6.62607015e-34 
        self.G0 = 2 * e**2 / h # Siemans (S)

    def get_VST_for_Gmax(self, VST: np.array, ISD: np.array, VSD: float, plot=False):
        ISD_filtered = scipy.ndimage.gaussian_filter1d(ISD, sigma=1)
        G = np.gradient(ISD_filtered, 1e-3 *(VST[1] - VST[0]))
        G_filtered = scipy.ndimage.gaussian_filter1d(G, sigma=1)

        maxima = scipy.signal.argrelextrema(G_filtered, np.greater)[0]

        maxima = maxima[G_filtered[maxima] >= 1e-7 * self.G0]
        max_VST_index = maxima[np.argmax(G_filtered[maxima])]
        
        max_VST = VST[max_VST_index]
        max_G = G[max_VST_index]

        if plot:
            plt.figure(dpi=200)
            plt.title(r"$I_{SD}$ v.s. $V_{ST}$ @ $V_{SD} = "+f"{round(VSD,1)}"+r"\ mV$")
            plt.ylabel(r"$I_{SD}\ (A)$")
            plt.xlabel(r"$V_{P}\ (mV)$")
            plt.plot(VST,ISD, 'k-')
            plt.scatter(VST[maxima], ISD[maxima], c='r', label='Local Maxima')
            plt.scatter(max_VST, ISD[max_VST_index], c='b', label='Global Maxima')
            plt.legend(loc='best')
            plt.show()

            plt.figure(dpi=200)
            plt.title(r"$G$ v.s. $V_{ST}$ @ $V_{SD} = "+f"{round(VSD,1)}"+r"\ mV$")
            plt.ylabel(r"$G_{SD}\ (S)$")
            plt.xlabel(r"$V_{P}\ (mV)$")
            plt.plot(VST, G, 'k-')
            plt.scatter(VST[maxima], G[maxima], c='r', label='Local Maxima')
            plt.scatter(max_VST, G[max_VST_index], c='b', label='Global Maxima')
            plt.legend(loc='best')
            plt.show()

        return max_VST, max_G
    
    def measure_current_time_trace(self, VST_crit: float, VSD: float, sampling_rate: int, plot=False):
        # set VST and VSD accordingly, and record for a given amount of time.
        memory = 100,000 # traces, double check?
        t_trace = memory / sampling_rate
        pass

    def get_current_noise_spectrum(self, current_noise: np.array, sampling_rate: int, plot=False):
        SI = scipy.signal.periodogram(current_noise, sampling_rate, scaling='density')

    def measure_coulomb_diamonds(self, plot=False):
        # write code to do the measurement
        pass

    def get_lever_arms(self,
                       VST_sweep: np.array, 
                       VSD_sweep: np.array, 
                       ISD_2D: np.array, 
                       VST_window: tuple, 
                       VSD_window: tuple, 
                       plot=False,
                       automated=False):

        VST_min, VST_max = VST_window[0], VST_window[1] 
        VSD_min, VSD_max = VSD_window[0], VSD_window[1] 

        ISD_crop, indices = self._crop_data_by_values(ISD_2D, VST_sweep, VSD_sweep, VST_min, VST_max, VSD_min, VSD_max)
        VST_min_index, VST_max_index, VSD_min_index, VSD_max_index = indices

        VSD_crop = VSD_sweep[VSD_min_index:VSD_max_index]
        VST_crop = VST_sweep[VSD_min_index:VSD_max_index]

        G_crop = np.gradient(ISD_crop)[1]
        G = np.gradient(ISD_2D)[1]

        self._make_interactive(VST_sweep, VSD_sweep, G, title=r"$G(V_{SD}, V_{ST})$", xlabel=r'$V_{ST}\ (mV)$', ylabel=r'$V_{SD}\ (uV)$')

        # Filter for Hough transform
        G_adjusted = self._adjust_data(G_crop,self.G0)
        if automated:
            # Get edges
            G_edges = skimage.feature.canny(G_adjusted, sigma=3)
            tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint=False)
            h, theta, d = skimage.transform.hough_line(G_edges, theta=tested_angles)

            plt.imshow(ISD_crop,origin='lower')
            lines_dict = {}
            for _, angle, dist in zip(*skimage.transform.hough_line_peaks(h, theta, d)):
                slope = np.tan(angle + np.pi/2)
                if np.abs(slope) > 5 or np.abs(slope) < 0.7:
                    continue
                (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])

                plt.axline((x0, y0), slope=np.tan(angle + np.pi/2),c='b')

                x_int_index = round(self._x_intercept([x0,y0],slope))
                x_int_coord = (x_int_index, 0)

                P = [round(VSD_max_index/slope + x_int_index), VSD_max_index]

                plt.scatter([x_int_index], [0])
                plt.scatter([round(VSD_max_index/ slope + x_int_index)], [VSD_max_index])

                true_slope = (1e-6*(VSD_sweep[P[1]] - VSD_sweep[0] ))/ (1e-3*(VST_sweep[VST_min_index + P[0]] - VST_sweep[VST_min_index + x_int_coord[0]]))
                
                # max_slope_mag, min_slope_mag = -1, -1
                # if np.abs(true_slope) > max_slope_mag or np.abs(true_slope) < min_slope_mag:
                #     continue

                lines_dict[x_int_index] = true_slope 

            plt.show()

            result, inverse_reciprocal_sums = self._cluster_x_intercepts(lines_dict)
            counter = 1
            
            for pair, lever_arm in zip(result, inverse_reciprocal_sums):
                text = [f"PAIR {counter}:", 
                    f"(VST, ms) -> ({VST_crop[pair[0][0]]} (mV),{pair[0][1]} (V/V))",
                    f"(VST, md) -> ({VST_crop[pair[1][0]]} (mV),{pair[1][1]}) (V/V))",
                    f'\u03B1 -> {round(lever_arm,3)} (eV/V)']
                self._pprint(text)
                counter += 1   

    def get_charge_noise_spectrum(self, 
                                  current_trace: np.array, 
                                  fs: int, 
                                  VST_sweep: np.array, 
                                  VSD_sweep: np.array, 
                                  ISD_1D: np.array,
                                  ISD_2D: np.array, 
                                  plot=False):
        SI = self.get_current_noise_spectrum(current_trace, fs)
        alpha = self.get_lever_arms(VST_sweep, VSD_sweep, ISD_2D, plot=False)
        _, max_G = self.get_VST_for_Gmax(VST_sweep, ISD_1D, np.NaN)

        SE = (alpha/np.abs(max_G))**2 * SI
        return SE

    def calculate_T2(self) -> float:
        pass

    def _parse_bias_spec_data(self, data: str):
        data_df = pd.read_csv(data, skiprows=[0,2], sep='\t')
        data_df.columns = data_df.columns.str.replace('[#, ,"]','',regex=True)
        
        self.V_SD = np.unique(np.array(data_df['V_SD']))
        self.V_ST = np.unique(np.array(data_df['SDP']))
        self.I_SD = np.array(data_df['Isd_DC']).reshape(len(self.V_ST), len(self.V_SD))
        return data_df

    def _parse_current_trace_data(self, data: str):
        data_df = pd.read_csv(data, skiprows=[0,2], sep='\t')
        data_df.columns = data_df.columns.str.replace('[#, ,"]','',regex=True)
        
        self.t = np.array(data_df['t'])
        self.I_trace = np.array(data_df['Isd_DC'])
        return data_df

    def _U(self, x,y):
        sigX, sigY = 5,5
        return (1/(2 * np.pi * sigX * sigY)) * np.exp(- 0.5* ((x/sigX)**2 + (y/sigY)**2))

    def _adjust_data(self, data, A=None):
        return np.sign(data) * np.log((np.abs(data)/A) + 1)
    
    def _filter_data(self, data):
        U = self._U(xx,yy)
        return (data - scipy.convolve(data,U)) / np.sqrt(scipy.convolve(data,U)**2 + self.G0**2)

    def _x_intercept(self, point, slope):
        x_0, y_0 = point
        x_intercept_value = x_0 - y_0 / slope
        return x_intercept_value
    
    def _calculate_inverse_reciprocal_sum(self,pair):
        x1, slope1 = pair[0]
        x2, slope2 = pair[1]

        # Calculate the inverse reciprocal sum
        inverse_reciprocal_sum = np.abs(1 / slope1) + np.abs(1 / slope2)

        return 1/inverse_reciprocal_sum

    def _cluster_x_intercepts(self,lines_dict):
        intercepts = list(lines_dict.keys())
        slopes = list(lines_dict.values())
        
        intercept_pairs = []

        for i in range(len(intercepts)):
            for j in range(i + 1, len(intercepts)):
                x_intercept1, x_intercept2 = intercepts[i], intercepts[j]
                slope1, slope2 = slopes[i], slopes[j]
                intercept_pairs.append(((x_intercept1, round(slope1,3)), (x_intercept2, round(slope2,3))))

        # Sort the pairs based on the difference in x-intercepts
        intercept_pairs.sort(key=lambda pair: abs(pair[0][0] - pair[1][0]))

        # Take the first N pairs (N = number of clusters)
        num_clusters = len(intercepts) // 2
        closest_pairs = intercept_pairs[:num_clusters]

        # Calculate the inverse reciprocal sum for each pair
        inverse_reciprocal_sums = [self._calculate_inverse_reciprocal_sum(pair) for pair in closest_pairs]

        return closest_pairs, inverse_reciprocal_sums

    def _make_interactive(self,x,y,data, title='', xlabel='', ylabel=''):
        # Create initial figure and axis
        fig, ax = plt.subplots()
        fig.dpi = 250
        fig.set_size_inches(7,5)
        plt.subplots_adjust(bottom=0.5)  # Adjust the bottom to make room for the slider
        # Add a slider to control the value parameter
        ax_slider = plt.axes([0.205, 0.1, 0.60, 0.03])
        slider_x_ax = fig.add_axes([0.205, 0.2, 0.60, 0.03])
        slider_y_ax = fig.add_axes([0.205, 0.3, 0.60, 0.03])

        global x_start, x_end, y_start, y_end
        x_start, x_end, y_start, y_end = x[0], x[-1], y[0], y[-1]
        
        slider_x = RangeSlider(slider_x_ax, "VST (mV)", x[0], x[-1], valstep=1, valinit=(x[0],x[-1]), color='black')
        slider_y = RangeSlider(slider_y_ax, "VSD (uV)", y[0], y[-1],valstep=10, valinit=(y[0],y[-1]), color='black')
        slider = Slider(ax_slider, r'$G_{filter} = 10^{x}G_0$  ', -2, 7, valstep=0.1, valinit=np.log10(1/self.G0), color='black', initcolor='white')

        data,_ = self._crop_data_by_values(data,x,y,x_start,x_end,y_start,y_end)
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

            new_data,_ = self._crop_data_by_values(data,x,y,x_start,x_end,y_start,y_end)

            value = (10**(slider.val)) * self.G0
            updated_data = self._adjust_data(new_data, value)
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
            slope = (1e-6*(y2 - y1)) / (1e-3*(x2 - x1))
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

    def _crop_data_by_values(self, data, x, y, x_start, x_end, y_start, y_end):

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