
import json
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.signal import find_peaks
from sklearn.linear_model import LinearRegression
from scipy.interpolate import interp1d
from io import BytesIO
import base64

class BPlot:
    def __init__(self, filename):
        self.filename = "alphapose-results-"+filename+".json"
        self.category = 0
        self.jsondata = {}
        self.distances = []
        self.slopes = []
        self.avg = 0

    def convert(self):
        with open(self.filename,'rb') as fh:
            data = json.load(fh)

        mmskeleton_data = {
            "data": {
                "keypoint": []
            },
            "frame_dir": "video_name",  # Set video name here
            "img_shape": [720, 1280],  # Example: height and width of the video
            "original_shape": [720, 1280],
            "total_frames": len(data)
        }


        bounding = []
        for frame_data in data:
            keypoints = frame_data["keypoints"]
            frame_keypoints = []
            # Group keypoints in sets of 3 (x, y, confidence)
            for i in range(0, len(keypoints), 3):
                frame_keypoints.append([keypoints[i], keypoints[i + 1], keypoints[i + 2]])
            # Append to the keypoint list in MMSkeleton format
            mmskeleton_data["data"]["keypoint"].append([frame_keypoints])
            bounding.append(frame_data["box"])

        # Example label for the video, adjust accordingly
        mmskeleton_data["data"]["label"] = 0

        # Convert to JSON
        self.jsondata = json.dumps(mmskeleton_data, indent=4)
        """ print(json_data)
        print(bounding) """
        return self.jsondata

    def plot_nx_graph(self):
        json_data = json.loads(self.jsondata)

        """ skeleton_edges = [
            (47, 48), (48, 49), (49, 50), (50, 51), 
            (47, 52), (52, 53), (53, 54), (54, 55),
            (47, 56), (56, 57), (57, 58), (58, 59),
            (47, 60), (60, 61), (61, 62), (62, 63),
            (47, 64), (64, 65), (65, 66), (66, 67),
        ] """
        skeleton_edges = [
            (0, 1), (18, 17), (18, 19), (18, 25), (19, 20), (20, 21), 
            (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 9), (9, 10), (10, 11), (11, 12),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 17), (1, 2), (2, 3), (3, 4),
        ]
        hand_parts = [47, 64, 65, 66, 67, 60, 61, 62, 63, 56, 57, 58, 59, 52, 53, 54, 55, 48, 49, 50, 51]

        """ skeleton_edges = [
            (15, 27), (27, 28), (28, 29), (29, 30), 
            (15, 31), (31, 32), (32, 33), (33, 34),
            (15, 35), (35, 36), (36, 37), (37, 38),
            (15, 39), (39, 40), (40, 41), (41, 42),
            (15, 43), (43, 44), (44, 45), (45, 46),
        ] """
        frames = json_data["data"]["keypoint"]
        main_kp = []
        # Iterate through frames and plot every 20th frame
        for frame_index in range(len(frames)):
                if frame_index % 10 == 0:
                    frame_keypoints = frames[frame_index][0]  # Get keypoints for the current frame

                    # Extract (x, y) positions, ignoring confidence scores
                    keypoints = [(frame_keypoints[kp][0], frame_keypoints[kp][1]) for kp in range(len(frame_keypoints)) if kp in hand_parts]
                    main_kp.append(keypoints)
                    # Create a graph to plot keypoints and connections
                    G = nx.Graph()
                    for idx, (x, y) in enumerate(keypoints):
                        G.add_node(idx, pos=(x, y))

                    # Add edges based on the skeleton structure
                    for edge in skeleton_edges:
                        if edge[0] < len(keypoints) and edge[1] < len(keypoints):  # Check if keypoints exist
                            G.add_edge(edge[0], edge[1])

                    # Plotting
                    plt.figure(figsize=(8, 8))
                    pos = nx.get_node_attributes(G, 'pos')
                    nx.draw(G, pos, node_size=400, node_color='lightblue', with_labels=True)
                    nx.draw_networkx_edges(G, pos, width=2, edge_color='orange')  # Draw edges
                    plt.title(f"Skeleton Plot for Frame {frame_index}")
                    plt.gca().invert_yaxis()
                    #plt.show()

    def plot_amplitude(self):
        json_data = self.jsondata
        if isinstance(json_data, str):
            json_data = json.loads(json_data)

        # Prepare a list to store distances for each frame
        self.distances = []

        # Get all frames from the JSON data
        frames = json_data["data"]["keypoint"]

        def calculate_distance(x1, y1, x2, y2):
            return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        # Iterate through frames and calculate the distance for each
        for frame_index in range(len(frames)):
            frame = frames[frame_index][0]
            
            # Extract (x55, y55, c55) and (x51, y51, c51) for the current frame
            x55, y55, c55 = frame[55]  # 55th keypoint (index 54)
            x51, y51, c51 = frame[51]  # 51st keypoint (index 50)
            
            # Calculate the distance between the 51st and 55th keypoints
            distance = calculate_distance(x55, y55, x51, y51)
            
            # Store the calculated distance
            self.distances.append(distance)

        # Convert distances to a numpy array
        distances_array = np.array(self.distances)
        summ = 0
        for i in self.distances:
            summ += i
        self.avg = summ/len(self.distances)
        # Find only local maxima (peaks) and local minima (troughs) separately
        peaks_indices, _ = find_peaks(distances_array, distance=5, prominence=20)  # Adjust prominence for clearer peak detection
        troughs_indices, _ = find_peaks(-distances_array, distance=5, prominence=10)  # Detect troughs by finding peaks in the negative array

        # Extract the highest points (distances) at the found peaks and lowest points at the troughs
        highest_points = distances_array[peaks_indices]
        lowest_points = distances_array[troughs_indices]


        # Separate plot for the highest points (amplitudes over time)
        figure, ax = plt.subplots()
        ax.set_ylim(0, 1000)
        ax.plot(peaks_indices + 1, highest_points, marker='o', color='green', linestyle='None')
        ax.set_xlabel("Frame")
        ax.set_ylabel("Distance between the thumb and forefinger")
        ax.set_title('Highest Amplitudes over time')
        plt.grid(True)
        for i in [150, 300]:
            ax.axvline(x=i, color='orange', linestyle='--')

        # Add one point between every two points
        for i in range(len(peaks_indices) - 1):
            x_mid = (peaks_indices[i] + peaks_indices[i + 1]) / 2 + 1
            y_mid = (highest_points[i] + highest_points[i + 1]) / 2
            ax.plot(x_mid, y_mid, marker='x', color='blue', linestyle='None')

        # Calculate regression lines for each section
        sections = [(0, 150), (150, 300), (300, 450)]
        colors = ['black', 'black', 'black']
        self.slopes = []
        for i, (start, end) in enumerate(sections):
            section_indices = [j for j, x in enumerate(peaks_indices + 1) if start <= x <= end]
            section_x = np.array([peaks_indices[j] + 1 for j in section_indices])
            section_y = np.array([highest_points[j] for j in section_indices])
            coeffs = np.polyfit(section_x, section_y, 1)
            slope = coeffs[0]  # slope is the first coefficient
            self.slopes.append(slope)
            regression_line = np.poly1d(coeffs)
            x_range = np.linspace(start, end, 100)
            ax.plot(x_range, regression_line(x_range), color=colors[i], label=f'Section {i+1} (Slope: {slope:.2f})')

        ax.legend()
        #plt.show()

        return self.export_plot_to_base64(plt)

    def plot_speed(self):
        frame_rate = 30  # 30 frames per second
        time_interval = 1 / frame_rate  # Time difference between frames

        # Calculate the speed as the change in distance over the time interval between frames
        speeds = np.abs(np.diff(self.distances)) / time_interval  # Use absolute value to treat reverse movements as positive

        # Create time points for the speed plot (since the speed is calculated between frames)
        time_points = np.arange(1, len(speeds) + 1)

        # Plot the speed of movements
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(time_points, speeds, marker='o', color='orange', label='Speed')

        # Fit a linear regression line to the speed data
        X = time_points.reshape(-1, 1)  # Reshape for sklearn
        model = LinearRegression()
        model.fit(X, speeds)
        regression_line = model.predict(X)

        # Plot the regression line
        ax.plot(time_points, regression_line, color='blue', linestyle='-', label='Trend Line')

        # Set labels and title
        ax.set_xlabel("Frame")
        ax.set_ylabel("Speed between 55th and 51st Keypoints (units/second)")
        ax.set_title("Speed of Movements between Keypoints Over Time (All Positive Speeds)")

        # Add grid and legend
        plt.grid(True)
        plt.legend()

        # Show plot
        #plt.show()
        return self.export_plot_to_base64(plt)

    def determine_severity(self):
        a = self.slopes[0]
        b = self.slopes[1]
        c = self.slopes[2]
        if a <= -1:
            self.category = 3
        elif b <= -1:
            self.category = 2
        elif c <= -1:
            self.category = 1
        else:
            if self.avg > 200:
                self.category = 0
            else:
                self.category = 4
        return self.category

    def export_plot_to_base64(self, plt):
        """Converts plot to base64 to be used in HTML display."""
        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        buf.close()
        return image_base64



