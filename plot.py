import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
from datetime import datetime
import math
import subprocess

# Initialize lists for each emotion

def start_plot():
    plt.ioff()
    plt.close('all')
    emotions = {'happy': [], 'sad': [], 'anger': [],  'fear': [], 'surprise': [], 'neutral': []}
    x = []
    index = 0
    n = 0
    temp_timestamp = datetime.strptime("10:10:10","%H:%M:%S")
    avg = 0
    repeat = True
    first_duration =datetime.strptime("10:10:10","%H:%M:%S")
    last_duration = datetime.strptime("10:10:10","%H:%M:%S")
    duration = 0
    file = "test.txt"
    with open(file) as d:
        for l in d.readlines():
            c = l.split(' ')
            timestamp = datetime.strptime(c[0], '%H:%M:%S')
            if n == 0:
                first_duration = timestamp
                n += 1
            last_duration = timestamp
        duration = last_duration - first_duration
        duration = duration.total_seconds() * 0.1
        duration = math.ceil(duration)
        n = 0

    with open(file) as d:
        for l in d.readlines():
            if repeat == True:
                repeat = False
                # Initialize counts for each emotion
                for emotion in emotions:
                    emotions[emotion].append(0)
            avg += 1
            s = l.split(' ')
            # Increment count based on the detected emotion
            emotions[s[1]][index] += 0.99  # Accumulate emotion values
            if temp_timestamp != s[0]:
                n += 1
            temp_timestamp = s[0]
            if n == duration:
                x.append(s[0])  # Use the timestamp of the last line in the 10-second interval
                for emotion in emotions:
                    emotions[emotion][index] /= avg  # Calculate average emotion value
                index += 1
                n = 0
                avg = 0
                repeat = True

        # If there are remaining lines
        if n != 0:
            x.append(s[0])
            for emotion in emotions:
                emotions[emotion][index] /= avg  # Calculate average emotion value

    # Plot line graph
    plt.figure(figsize=(10, 6))
    for emotion, data in emotions.items():
        plt.plot(x, data, label=emotion)

    plt.legend()

    plt.savefig("plot/plt1.png",bbox_inches="tight")
    # Plot doughnut chart
    plt.figure(figsize=(6, 6))
    labels = list(emotions.keys())
    sizes = [sum(emotions[emotion]) for emotion in labels]

    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, wedgeprops=dict(width=0.3))
    plt.gca().add_artist(plt.Circle((0, 0), 0.2, fc='white'))

    plt.title('Emotion Distribution')
    plt.savefig("plot/plt2.png")
    subprocess.run(['xdg-open',"plot/plt1.png"])
