import os

# --- Path Configuration ---
BASE_DIR = os.getcwd() 
DATA_DIR = os.path.join(BASE_DIR, "dataset")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
RESULTS_FILE = os.path.join(BASE_DIR, "results.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Dataset Parameters ---
IMG_HEIGHT = 128
IMG_WIDTH = 128
CHANNELS = 3
NUM_CLASSES = 6

# --- The 15-Minute Hyperparameter Sweep ---
OPTIMIZERS = ['adam', 'sgd']  # Dropped rmsprop
LEARNING_RATES = [0.001]      # Locked to best standard LR
BATCH_SIZES = [32]            # Locked to 32 for maximum GPU utilization
EPOCHS = [5]                  # 5 epochs is enough to show architectural differences