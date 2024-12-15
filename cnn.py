import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Circle
from matplotlib import rcParams
from PIL import Image, ImageTk
import os
from sklearn.model_selection import train_test_split

class TrainerApp:
    def __init__(self, master, x_train, y_train, conv, pool, full, lr=0.01, epochs=10):
        rcParams.update({
            "axes.facecolor": "#333230",
            "axes.edgecolor": "white",
            "axes.labelcolor": "white",
            "xtick.color": "white",
            "ytick.color": "white",
            "figure.facecolor": "#333230",
            "figure.edgecolor": "#333230",
            "grid.color": "gray",
        })

        self.master = master
        self.master.title("CNN Training")
        self.master.geometry("1250x850")

        self.x_train = x_train
        self.y_train = y_train
        self.conv = conv
        self.pool = pool
        self.full = full
        self.lr = lr
        self.epochs = epochs
        self.current_epoch = 0
        self.current_image_index = 0  # Track the current image being trained on

        # Initialize loss and accuracy tracking
        self.losses = []
        self.accuracies = []

        # Style configuration
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TFrame', background='#333230')
        style.configure('TLabel', background='#333230', foreground='white', font=('Arial', 12))
        style.configure('TButton', background='#333230', foreground='white', font=('Arial', 14, 'bold'), relief="raised")
        style.map('TButton', background=[('active', '#444240')], relief=[('active', 'raised')])

        # Main layout
        main_frame = ttk.Frame(self.master, style='TFrame')
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Top bar for controls and information
        top_bar = ttk.Frame(main_frame, style='TFrame', height=80)
        top_bar.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        # Status Label on the left
        self.status_label = ttk.Label(
            top_bar, text="Status: Ready", style="TLabel", anchor="center", wraplength=180
        )
        self.status_label.pack(side=tk.LEFT, padx=10)

        # Metrics on the left
        self.metrics_label = ttk.Label(
            top_bar, text="Loss: -\nAccuracy: -%", style="TLabel", anchor="center", justify=tk.LEFT
        )
        self.metrics_label.pack(side=tk.LEFT, padx=10)

        # Start Button on the left (rounded)
        self.start_button = ttk.Button(
            top_bar, text="Start Training", command=self.start_training, style="TButton"
        )
        self.start_button.pack(side=tk.LEFT, padx=10)

        # Label for training image
        self.training_image_label = ttk.Label(
            top_bar, text="Currently Training Image", style="TLabel", anchor="center"
        )
        self.training_image_label.pack(side=tk.RIGHT, padx=10)

        # Canvas to display current training image on the right
        self.image_canvas = tk.Canvas(top_bar, width=80, height=80, bg="#333230", highlightthickness=0)
        self.image_canvas.pack(side=tk.RIGHT, padx=10)

        # Plotting area
        plot_frame = ttk.Frame(main_frame, style='TFrame')
        plot_frame.pack(fill=tk.BOTH, expand=True)

        # Grid layout for plots
        self.fig, self.axes = plt.subplots(1, 2, figsize=(10, 6))
        self.fig.patch.set_facecolor("#333230")  # Set background color
        self.fig.tight_layout()

        # Network Diagram
        self.ax_network = self.axes[0]
        self.ax_network.set_title("CNN Network", fontsize=16, color="white", fontname='Arial')
        self.ax_network.axis("off")  # Hide axes
        self.ax_network.set_facecolor("#333230")  # Set background color

        # Draw Layers
        self.input_positions = self.draw_layer(self.ax_network, x=1, num_nodes=5, color='white', y_offset=0)
        self.conv_positions = self.draw_layer(self.ax_network, x=2, num_nodes=self.conv.num_filters, color='white', y_offset=1)
        self.pool_positions = self.draw_layer(self.ax_network, x=3, num_nodes=self.conv.num_filters, color='white', y_offset=0)
        self.fc_positions = self.draw_layer(self.ax_network, x=4, num_nodes=self.full.output_size, color='white', y_offset=1)
        self.connect_layers(self.ax_network, self.input_positions, self.conv_positions)
        self.connect_layers(self.ax_network, self.conv_positions, self.pool_positions)
        self.connect_layers(self.ax_network, self.pool_positions, self.fc_positions)

        self.network_canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.network_canvas.draw()
        self.network_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Training Plots (Loss and Accuracy)
        self.ax_plots = self.axes[1]
        self.ax_plots.set_facecolor("#333230")
        self.loss_line, = self.ax_plots.plot([], [], label="Loss", color="red", linewidth=2)
        self.accuracy_line, = self.ax_plots.plot([], [], label="Accuracy", color="white", linewidth=2)
        self.ax_plots.set_title("Training Metrics", fontsize=16, color="white", fontname='Arial')
        self.ax_plots.set_xlabel("Epochs", fontsize=12, color="white", fontname='Arial')
        self.ax_plots.set_ylabel("Values", fontsize=12, color="white", fontname='Arial')
        legend = self.ax_plots.legend(facecolor="black", edgecolor="white")
        for text in legend.get_texts():
            text.set_color("white")  # Make legend text white
        self.ax_plots.grid(True, linestyle="--", alpha=0.5)

    def draw_layer(self, ax, x, num_nodes, color='blue', y_offset=0, spacing=0.7):
        positions = []
        for i in range(num_nodes):
            y = i * spacing + y_offset
            circle = Circle((x, y), 0.1, color=color, fill=True)
            ax.add_patch(circle)
            positions.append((x, y))
        return positions

    def connect_layers(self, ax, from_positions, to_positions):
        for (x1, y1) in from_positions:
            for (x2, y2) in to_positions:
                ax.plot([x1, x2], [y1, y2], color='gray', linewidth=0.5)

    def update_training_image(self):
        """Update the displayed training image."""
        current_image = self.x_train[self.current_image_index].squeeze()  # Remove extra dimensions
        img_normalized = (current_image * 255).astype(np.uint8)  # Normalize to 0-255
        img = Image.fromarray(img_normalized, mode="L").resize((80, 80))  # Resize for display
        self.tk_image = ImageTk.PhotoImage(img)
        self.image_canvas.create_image(0, 0, anchor="nw", image=self.tk_image)

    def start_training(self):
        self.start_button.config(state="disabled")
        self.status_label.config(text="Training Started...")
        self.run_training_step()

    def run_training_step(self):
        """Simulate one training step."""
        # Display the current image being trained on
        self.update_training_image()

        epoch_loss = np.random.uniform(0.1, 1.0)
        epoch_accuracy = np.random.uniform(70, 100)
        self.losses.append(epoch_loss)
        self.accuracies.append(epoch_accuracy)

        self.loss_line.set_data(range(len(self.losses)), self.losses)
        self.accuracy_line.set_data(range(len(self.accuracies)), self.accuracies)
        self.ax_plots.relim()
        self.ax_plots.autoscale_view()
        self.network_canvas.draw()

        self.status_label.config(text=f"Epoch {self.current_epoch + 1} in progress...")
        self.metrics_label.config(text=f"Loss: {epoch_loss:.4f}\nAccuracy: {epoch_accuracy:.2f}%")

        self.current_image_index = (self.current_image_index + 1) % len(self.x_train)  # Loop through images
        self.current_epoch += 1

        if self.current_epoch < self.epochs:
            self.master.after(1000, self.run_training_step)
        else:
            self.status_label.config(text="Training Complete!")
            self.start_button.config(state="normal")



def load_dataset(dataset_path, image_size=(64, 64)):
    data, labels = [], []
    class_mapping = {"Parasitized": 0, "Uninfected": 1}

    for class_name, label in class_mapping.items():
        class_path = os.path.join(dataset_path, class_name)
        for img_name in os.listdir(class_path):
            if img_name.endswith((".jpg", ".png")):
                img_path = os.path.join(class_path, img_name)
                img = Image.open(img_path).convert('L')
                img = img.resize(image_size)
                img_array = np.array(img) / 255.0
                data.append(img_array)
                labels.append(label)

    data = np.array(data).reshape(-1, image_size[0], image_size[1], 1)
    labels = np.eye(len(class_mapping))[labels]
    return train_test_split(data, labels, test_size=0.2, random_state=42)


class Convolution:
    def __init__(self, input_shape, filter_size, num_filters):
        input_height, input_width = input_shape
        self.num_filters = num_filters
        self.num_filters = num_filters
        self.filter_size = filter_size
        if filter_size > input_height or filter_size > input_width:
            raise ValueError("Filter size too large.")
        self.output_shape = (num_filters, input_height - filter_size + 1, input_width - filter_size + 1)
        if self.output_shape[1] <= 0 or self.output_shape[2] <= 0:
            raise ValueError("Invalid output dimensions.")
        self.filters = np.random.randn(num_filters, filter_size, filter_size) * np.sqrt(2 / (filter_size * filter_size))
        self.biases = np.zeros(self.num_filters)

    def forward(self, input_data):
        self.input_data = input_data
        self.output_height = input_data.shape[0] - self.filter_size + 1
        self.output_width = input_data.shape[1] - self.filter_size + 1
        num_channels = input_data.shape[2]

        output = np.zeros((self.output_height, self.output_width, self.num_filters))
        for f in range(self.num_filters):
            for i in range(self.output_height):
                for j in range(self.output_width):
                    input_patch = input_data[i:(i + self.filter_size), j:(j + self.filter_size), :]
                    output[i, j, f] = np.sum(input_patch * self.filters[f]) + self.biases[f]
        return output

    def backward(self, dL_dout, lr):
        dL_dinput = np.zeros_like(self.input_data)
        dL_dfilters = np.zeros_like(self.filters)
        for f in range(self.num_filters):
            for i in range(dL_dout.shape[1]):
                for j in range(dL_dout.shape[2]):
                    patch = self.input_data[i:i + self.filter_size, j:j + self.filter_size, 0]
                    dL_dfilters[f] += patch * dL_dout[f, i, j]
                    dL_dinput[i:i + self.filter_size, j:j + self.filter_size, 0] += self.filters[f] * dL_dout[f, i, j]

        self.filters -= lr * dL_dfilters
        self.biases -= lr * np.sum(dL_dout, axis=(1, 2))
        return dL_dinput


class MaxPool:
    def __init__(self, pool_size):
        self.pool_size = pool_size

    def forward(self, input_data):
        self.input_data = input_data
        self.num_channels, self.input_height, self.input_width = input_data.shape
        self.output_height = self.input_height // self.pool_size
        self.output_width = self.input_width // self.pool_size
        self.output = np.zeros((self.num_channels, self.output_height, self.output_width))
        for c in range(self.num_channels):
            for i in range(self.output_height):
                for j in range(self.output_width):
                    start_i = i * self.pool_size
                    start_j = j * self.pool_size
                    end_i = start_i + self.pool_size
                    end_j = start_j + self.pool_size
                    patch = input_data[c, start_i:end_i, start_j:end_j]
                    self.output[c, i, j] = np.max(patch)
        return self.output

    def backward(self, dL_dout, lr):
        dL_dinput = np.zeros_like(self.input_data)
        for c in range(self.num_channels):
            for i in range(self.output_height):
                for j in range(self.output_width):
                    start_i = i * self.pool_size
                    start_j = j * self.pool_size
                    end_i = start_i + self.pool_size
                    end_j = start_j + self.pool_size
                    patch = self.input_data[c, start_i:end_i, start_j:end_j]
                    mask = patch == np.max(patch)
                    dL_dinput[c, start_i:end_i, start_j:end_j] = dL_dout[c, i, j] * mask
        return dL_dinput


class Fully_Connected:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.randn(output_size, input_size) * np.sqrt(2 / input_size)
        self.biases = np.zeros((output_size, 1))

    def softmax(self, z):
        shifted_z = z - np.max(z)
        exp_values = np.exp(shifted_z)
        sum_exp_values = np.sum(exp_values, axis=0, keepdims=True)
        probabilities = exp_values / sum_exp_values
        return probabilities

    def forward(self, input_data):
        self.input_data = input_data
        flattened_input = input_data.flatten().reshape(1, -1)
        self.z = np.dot(self.weights, flattened_input.T) + self.biases
        self.output = self.softmax(self.z)
        return self.output

    def backward(self, dL_dout, lr):
        dL_dy = dL_dout
        dL_dw = np.dot(dL_dy, self.input_data.flatten().reshape(1, -1))
        dL_db = dL_dy
        dL_dinput = np.dot(self.weights.T, dL_dy)
        dL_dinput = dL_dinput.reshape(self.input_data.shape)

        self.weights -= lr * dL_dw
        self.biases -= lr * dL_db
        return dL_dinput


def train_model(x_train, y_train, conv, pool, full, lr, epochs):
    for epoch in range(epochs):
        total_loss, correct_predictions = 0, 0
        for i in range(len(x_train)):
            x = x_train[i]
            y = y_train[i]

            conv_out = conv.forward(x)
            pool_out = pool.forward(conv_out)
            output = full.forward(pool_out)

            loss = cross_entropy_loss(output, y)
            total_loss += loss

            if np.argmax(output) == np.argmax(y):
                correct_predictions += 1

            dL_dout = cross_entropy_loss_gradient(y, output)
            dL_dfc = full.backward(dL_dout, lr)
            dL_dpool = pool.backward(dL_dfc, lr)
            conv.backward(dL_dpool, lr)

        accuracy = correct_predictions / len(x_train)
        print(f"Epoch {epoch + 1}/{epochs}: Loss = {total_loss:.4f}, Accuracy = {accuracy * 100:.2f}%")


def cross_entropy_loss(predictions, targets):
    epsilon = 1e-7
    predictions = np.clip(predictions, epsilon, 1 - epsilon)
    return -np.sum(targets * np.log(predictions)) / targets.shape[0]


def cross_entropy_loss_gradient(actual_labels, predicted_probs):
    epsilon = 1e-7
    predicted_probs = np.clip(predicted_probs, epsilon, 1 - epsilon)
    return -(actual_labels / predicted_probs) / actual_labels.shape[0]


def leaky_relu_derivative(x, alpha=0.01):
    grad = np.ones_like(x)
    grad[x <= 0] = alpha
    return grad


def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)


if __name__ == "__main__":
    # Dummy dataset and model components for testing UI
    dataset_path = "dataset/cell_images/cell_images"
    x_train, x_test, y_train, y_test = load_dataset(dataset_path)

    conv = Convolution((64, 64), filter_size=3, num_filters=6)
    pool = MaxPool(pool_size=2)
    fully_connected_size = conv.num_filters * (64 // 2) * (64 // 2)
    full = Fully_Connected(input_size=fully_connected_size, output_size=2)

    # Initialize and run the TrainerApp
    root = tk.Tk()
    app = TrainerApp(root, x_train, y_train, conv, pool, full, lr=0.01, epochs=100)
    root.mainloop()