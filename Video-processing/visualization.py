import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.animation as animation

print("v1.0")

def load_particle_data(csv_file):
    """Load particle data from CSV file"""
    return pd.read_csv(csv_file)

def create_frame_plot(frame_data, ax):
    """Create a plot for a single frame of particle data"""
    ax.clear()
    
    # Plot each particle as a circle
    for _, particle in frame_data.iterrows():
        circle = Circle((particle['x'], particle['y']), 
                       radius=particle['radius'],
                       fill=True,
                       alpha=0.6)
        ax.add_patch(circle)
    
    # Set plot limits and aspects
    ax.set_aspect('equal')
    ax.set_xlim(0, 1000)  # Adjust these limits based on your data
    ax.set_ylim(0, 1000)  # Adjust these limits based on your data
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title(f'Frame {frame_data.iloc[0]["frame"]}')

def visualize_particles(csv_file):
    """Create visualization of particles across frames"""
    # Load data
    df = load_particle_data(csv_file)
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Get unique frames
    frames = sorted(df['frame'].unique())
    
    def update(frame_num):
        """Update function for animation"""
        frame = frames[frame_num]
        frame_data = df[df['frame'] == frame]
        create_frame_plot(frame_data, ax)
        return ax,
    
    # Create animation
    anim = animation.FuncAnimation(
        fig, 
        update,
        frames=len(frames),
        interval=200,  # 200ms between frames
        blit=True
    )
    
    plt.show()

def plot_single_frame(csv_file, frame_number):
    """Plot a single frame of particle data"""
    # Load data
    df = load_particle_data(csv_file)
    
    # Filter data for the specified frame
    frame_data = df[df['frame'] == frame_number]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 10))
    create_frame_plot(frame_data, ax)
    plt.show()

if __name__ == "__main__":
    csv_file = "ring_droplet_detection_20250303_075901/ring_droplet_data.csv"  # Replace with your CSV file path
    
    # To visualize all frames as an animation:
    visualize_particles(csv_file)
    
    # To plot a single frame, uncomment the following line:
    # plot_single_frame(csv_file, frame_number=0)
