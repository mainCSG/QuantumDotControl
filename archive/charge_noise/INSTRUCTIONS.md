# Charge Noise Data Viewer

This is the *alpha* state of the charge noise measurement software. Right now, it's capabilities are limited to data viewing and processing. This is a tool for researchers to use along side their measurement set-up to gather critical information such as max conductance peaks, lever arms, etc.

Next update: 
1. Integration of qcodes measurement protocols to allow the user to directly measure and display the data all through the app.
2. Automated extraction of lever arms using Hough transform.
3. Organize all auxiliary information into neat database for user information. This includes spectator gate values that are only really used for data tracking.

## Important Information

- Script is written assuming the data is in *.db format from qcodes. *.dat is *not* supported as it is the legacy format of qcodes and will be replaced soon.
- All voltages are assumed to be in *mV*.
- Code doesn't store any of the extra information that is not used in the plots (e.g. VSD, LB, C, RB values for coulomb oscillations). Please track this yourself for now.

## How to run the program

To run the program,

1. Make sure your environment has the following packages: matplotlib, numpy, scipy, csv, os, and qcodes. If you are missing a package use "python -m pip install" followed by the missing package.
2. Run charge_noise/main.py and a window should pop up.
3. Choose your desired data analysis from the menu bar.
4. Input the fields you desire and press the button.