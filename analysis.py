import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from scipy.optimize import curve_fit

def hydrogen_ground_state(r, A, a):
    """
    Hydrogen atom ground state radial probability density
    A: amplitude factor
    a: effective Bohr radius (scaling factor)
    """
    return A * (r/a)**2 * np.exp(-r/a)

def calculate_droplet_separation(df):
    # Read the CSV file
    data = pd.read_csv(df)
    
    # Group data by frame
    frame_groups = data.groupby('frame')
    
    # Initialize lists to store results
    frames_with_two_droplets = []
    separations = []
    times = []
    
    # Analyze each frame
    for frame, group in frame_groups:
        # Check if frame has exactly 2 droplets
        if len(group) == 2:
            # Get coordinates of both droplets
            droplet1 = group.iloc[0]
            droplet2 = group.iloc[1]
            
            # Calculate Euclidean distance between droplets
            separation = sqrt(
                (droplet1['x'] - droplet2['x'])**2 + 
                (droplet1['y'] - droplet2['y'])**2
            )
            
            frames_with_two_droplets.append(frame)
            separations.append(separation)
            times.append(droplet1['time_sec'])
    
    # Create histogram plot with improved aesthetics
    plt.figure(figsize=(10, 6))
    
    # Create histogram with customized style
    n, bins, patches = plt.hist(separations, 
                              bins=40,
                              color='#5DADE2',  # Lighter blue color
                              alpha=0.7,
                              rwidth=1.0,
                              label='Separation frequency',
                              density=True,  # Normalize for fitting
                              edgecolor='none')
    
    # Add outer outline using step plot
    plt.step(bins, np.append(n, n[-1]), where='post', color='black', linewidth=1.5)
    
    # Fit hydrogen ground state function
    bin_centers = (bins[:-1] + bins[1:]) / 2
    try:
        popt, _ = curve_fit(hydrogen_ground_state, bin_centers, n, 
                           p0=[np.max(n), np.mean(separations)],
                           bounds=([0, 0], [np.inf, np.inf]))
        
        # Plot the fitted curve
        x_fit = np.linspace(min(separations), max(separations), 200)
        y_fit = hydrogen_ground_state(x_fit, *popt)
        plt.plot(x_fit, y_fit, 'r-', linewidth=2, 
                label='Hydrogen atom ground state electron wave function fit')
    except RuntimeError:
        print("Warning: Could not fit the hydrogen ground state function")
    
    # Customize appearance
    plt.ylabel('Normalized Frequency', fontsize=12)  # Updated y-label
    plt.xlabel('Droplet Separation (pixels)', fontsize=12)
    plt.title('Separation between pair of walkers in trial t09', fontsize=14, pad=15)
    
    # Customize grid
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Add legend
    plt.legend(frameon=True, facecolor='white', framealpha=1)
    
    # Customize ticks
    plt.tick_params(axis='both', which='major', labelsize=10)
    
    # Tight layout to prevent label cutoff
    plt.tight_layout()
    
    # Save and show the plot
    plt.savefig('droplet_separation_histogram.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    return frames_with_two_droplets, separations, times

if __name__ == "__main__":
    # Replace 'your_file.csv' with your actual CSV file name
    frames, seps, times = calculate_droplet_separation('ring_droplet_detection_20250303_075901/ring_droplet_data.csv')
    
    # Print results
    print("\nFrames with exactly two droplets:")
    for frame, sep, time in zip(frames, seps, times):
        print(f"Frame {frame} (t={time:.3f}s): Separation = {sep:.2f} pixels")
