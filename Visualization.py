import matplotlib.pyplot as plt
from IPython import display

# Stil belirle
plt.style.use('ggplot')  # ggplot stilini kullan

plt.ion()

def plotSetter(scores, mean_scores):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()

    # Çizgi grafiği
    plt.subplot(1, 2, 1)  # 1 satır, 2 sütun, 1. grafik
    plt.title('Line Plot')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores, color='blue', label='Scores')  # Mavi renkte çizgi
    plt.plot(mean_scores, color='red', label='Mean Scores')  # Kırmızı renkte çizgi
    plt.ylim(ymin=0)
    plt.legend()

    # Histogram
    plt.subplot(1, 2, 2)  # 1 satır, 2 sütun, 2. grafik
    plt.title('Histogram')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.hist(scores, color='green', alpha=0.5, label='Scores')  # Yeşil renkte histogram
    plt.hist(mean_scores, color='orange', alpha=0.5, label='Mean Scores')  # Turuncu renkte histogram
    plt.legend()

    plt.show(block=False)
    plt.pause(.1)
