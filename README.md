# TickNet: Efficient Lightweight CNNs for Image Classification

## Overview

TickNets are a novel family of Lightweight Convolutional Neural Networks (LWCNNs) designed for image classification, particularly on resource-constrained devices such as mobile and embedded systems. The primary purpose of TickNets is to balance performance and efficiency, achieving accuracy comparable to larger models while requiring fewer parameters and computations.

Key innovations for efficiency include:
*   **Full-Residual Point-Depth-Point (FR-PDP) Perceptron:** Streamlined feature extraction using pointwise and depthwise convolutions. This preserves crucial spatial information through the Full-Residual (FR) mechanism, enabling skip connections even during downsampling.
*   **"Tick-Shape" Backbone:** Characterized by periods of channel expansion and contraction, inspired by the shape of a checkmark. This optimizes resource usage without compromising performance, setting TickNets apart from conventional CNNs with their steady increase in channels.

## Architecture

### Core Concepts

TickNets are built upon two key innovations:

*   **Full-Residual Point-Depth-Point (FR-PDP) Perceptron:** The FR-PDP block is a fundamental component of TickNets, designed to enhance network efficiency and performance. It streamlines feature extraction using pointwise and depthwise convolutions. A key aspect is the Full-Residual (FR) mechanism, which preserves crucial spatial information by enabling skip connections even during downsampling, a feature often limited in other lightweight models. The FR-PDP block consists of Pointwise Convolution (Pw), Depthwise Convolution (Dw), a second Pointwise Convolution (Pw2), a Pointwise Residual Convolution (PwR) for dimension adjustment in residual connections, and a Squeeze-and-Excitation (SE) Block to recalibrate feature maps.
*   **"Tick-Shape" Backbone:** Unlike conventional Lightweight Convolutional Neural Networks (LWCNNs) that often feature a "plumb" backbone with a consistent increase in output channels, TickNets introduce a "tick-shape" backbone. This design is characterized by channel elasticity, with periods of channel expansion and contraction that resemble a checkmark. This approach allows for more efficient parameter usage, avoiding rapid growth in model size, especially in later layers, by strategically placing FR-PDP blocks with varying strides to control the channel dimensions.

### TickNet Variants

TickNet offers a family of architectures, allowing for flexibility based on the task's complexity:

*   **TickNet-Basic:** This is the simplest design, making it efficient for less complex image classification tasks.
*   **TickNet-Small:** This variant has increased complexity and depth compared to the Basic version, making it suitable for moderately complex tasks.
*   **TickNet-Large:** This is the most complex and deepest architecture in the TickNet family, designed for handling intricate patterns in highly complex image classification scenarios.

### S-TickNet (Enhancement): Enhanced Spatial Feature Learning

To further improve performance, TickNets can be enhanced with "Spread-learned spatial features," leading to an architecture referred to as S-TickNet. The primary aim of this enhancement is to enrich feature representation and classification accuracy. This is achieved by:
*   Extracting and processing features to capture complex spatial relationships within images.
*   Spreading these learned spatial features across different layers of the network.
*   A key modification in S-TickNet is the addition of a 256-channel FR-PDP block at the beginning of the TickNet architecture. This is intended to boost early spatial feature capture and improve the model's ability to understand complex spatial relationships, leading to enriched feature representation, better class distinction, and potentially reduced overfitting.

## Prerequisites and Installation

### Prerequisites

This project is written in Python 3.x. The main dependencies are:

*   **Python** (3.x recommended)
*   **PyTorch:** The core deep learning framework.
*   **Torchvision:** Provides datasets, model architectures, and image transformations for PyTorch.
*   **SciPy:** Used for loading `.mat` files, particularly for the StanfordDogs dataset.
*   **Argparse:** For parsing command-line arguments when running training scripts (part of the Python standard library).

Other standard Python libraries used include `os`, `sys`, `time`, `random`, `shutil`, and `warnings`.

### Installation

1.  **Set up a virtual environment (recommended):**
    ```bash
    python -m venv ticknet_env
    source ticknet_env/bin/activate  # On Windows use: ticknet_env\Scripts\activate
    ```

2.  **Install PyTorch and Torchvision:**
    The specific command for PyTorch can vary depending on your operating system and CUDA version (if you plan to use a GPU). It is highly recommended to install PyTorch by following the instructions on the official website to ensure compatibility.
    Visit [pytorch.org](https://pytorch.org) for the correct installation command.

    A typical command might look like:
    ```bash
    # Example for Linux with CUDA 11.x - verify on PyTorch website!
    pip install torch torchvision torchaudio
    ```

3.  **Install SciPy:**
    ```bash
    pip install scipy
    ```

Once these prerequisites are installed, you should be able to run the provided scripts. Make sure to activate your virtual environment before running any scripts if you created one.

## Dataset Preparation

The TickNet models are trained and evaluated on the following standard image classification datasets:

*   **Stanford Dogs:** (http://vision.stanford.edu/aditya86/ImageNetDogs/)
*   **ImageNet-1k:** (https://www.image-net.org/)
*   **Places365:** (http://places2.csail.mit.edu/index.html)

You will need to download these datasets from their respective official websites or other legitimate sources. Ensure you have the appropriate directory structure for each dataset as expected by the training scripts (usually a `train` and `val` or `test` subdirectory structure).

**Important: Configuring Dataset Paths**

As noted in the original project documentation: "Subject to your system, modify these training files (*.py) to have the right path to datasets".

The Python training scripts (`TickNet_Dogs.py`, `TickNet_ImageNet.py`, `TickNet_Places365.py`) require you to specify the root directory for each dataset. You will need to modify these scripts to point to the location where you have stored the datasets on your local system.

Look for command-line arguments or variables within each script that define the dataset path. These are typically named `data_root`, `data`, or similar.

For example:

*   In `TickNet_Dogs.py`, the path is often set via a command-line argument like:
    ```python
    parser.add_argument('-r', '--data-root', type=str, default='../../../datasets/StanfordDogs', help='Dataset root path.')
    ```
    You can either change the `default` value in the script or provide the path when running the script:
    ```bash
    python TickNet_Dogs.py --data-root /path/to/your/StanfordDogs
    ```

*   Similarly, in `TickNet_ImageNet.py` and `TickNet_Places365.py`, look for an argument like:
    ```python
    parser.add_argument('-r', '--data', type=str, default='../../../datasets/ImageNet', help='path to dataset')
    # or
    parser.add_argument('-r', '--data', type=str, default='../../../datasets/places365standard/places365_standard', help='path to dataset')
    ```
    Update the `default` path in the script or pass the correct path via the command line:
    ```bash
    python TickNet_ImageNet.py --data /path/to/your/ImageNet
    python TickNet_Places365.py --data /path/to/your/Places365
    ```

Ensure the path you provide points to the root directory of the respective dataset. The `TickNet_Dogs.py` script also includes a `--download` argument which might attempt to download the Stanford Dogs dataset if the files are not found in the specified `data_root`, but it's generally more reliable to download datasets manually.

## Usage

This section describes how to train and evaluate TickNet and S-TickNet models. Remember to adjust dataset paths as described in the "Dataset Preparation" section.

### Training

*   **Stanford Dogs:**
    ```bash
    python TickNet_Dogs.py
    ```
    Note: The `TickNet_Dogs.py` script, in its current version, is configured to train the **TickNet-Small** model by default. The script iterates through a list `arr_typesize` which is currently set to `['small']`. To train other variants (e.g., basic, large) with this script, you would need to modify the `arr_typesize` list within `TickNet_Dogs.py` to include `'basic'` or `'large'`.

*   **ImageNet-1k:**
    To train TickNet-Small:
    ```bash
    python TickNet_ImageNet.py -a small
    ```
    To train TickNet-Large:
    ```bash
    python TickNet_ImageNet.py -a large
    ```

*   **Places365:**
    To train TickNet-Small:
    ```bash
    python TickNet_Places365.py -a small
    ```
    To train TickNet-Large:
    ```bash
    python TickNet_Places365.py -a large
    ```

**Architecture Flags:**
*   `-a small`: Use this flag with `TickNet_ImageNet.py` and `TickNet_Places365.py` to train or evaluate the TickNet-Small architecture.
*   `-a large`: Use this flag with `TickNet_ImageNet.py` and `TickNet_Places365.py` to train or evaluate the TickNet-Large architecture.
*   For `TickNet_Dogs.py`, the "basic" architecture is not explicitly selected via a flag in the current script version. Training the "basic" model would require modifying the `arr_typesize` list in the script.

### Validation

To validate a trained model, use the `--evaluate` flag. This flag switches the script from training mode to evaluation mode, loading a pre-trained model to measure its performance on the validation set.

*   **Stanford Dogs (TickNet-Small):**
    ```bash
    python TickNet_Dogs.py --evaluate
    ```
    This command will evaluate the TickNet-Small model by default, expecting the checkpoint at `./checkpoints/StanfordDogs/small/model_best.pth`. If you've trained other variants (by modifying the script) and want to evaluate them, ensure the checkpoint path in the script corresponds to the model you wish to evaluate.

*   **ImageNet-1k:**
    To validate TickNet-Small:
    ```bash
    python TickNet_ImageNet.py -a small --evaluate
    ```
    To validate TickNet-Large:
    ```bash
    python TickNet_ImageNet.py -a large --evaluate
    ```

*   **Places365:**
    To validate TickNet-Small:
    ```bash
    python TickNet_Places365.py -a small --evaluate
    ```
    To validate TickNet-Large:
    ```bash
    python TickNet_Places365.py -a large --evaluate
    ```

### Using Pre-trained Models

The project provides pre-trained models for the TickNet-Small variant on several datasets. You can download them using the links below:

*   **Places365 (TickNet-Small):** [Click here for Places365](https://drive.google.com/drive/folders/1EdlA3tuOutBJMR23B-fcSOKKB69hAQ5R?usp=sharing)
*   **ImageNet-1k (TickNet-Small):** [Click here for ImageNet-1k](https://drive.google.com/drive/folders/1t1M_QJwCmcaTgKBsJBmzrU-kabQeOPDT?usp=sharing)
*   **Stanford Dogs (TickNet-Small):** [Click here for Stanford Dogs](https://drive.google.com/drive/folders/1RGglukdrd5xDrGSo6ONmHTCZNZ-YwpZb?usp=sharing)

After downloading, locate the model file (e.g., `model_best.pth.tar` or `model_best.pth`) inside the `./checkpoints/[name_dataset]/small` directory structure. For example:
*   For Places365 TickNet-Small: `./checkpoints/Places365/small/model_best.pth.tar`
*   For ImageNet-1k TickNet-Small: `./checkpoints/ImageNet1k/small/model_best.pth.tar`
*   For Stanford Dogs TickNet-Small: `./checkpoints/StanfordDogs/small/model_best.pth`

The validation scripts are generally configured to load models from these paths.

### S-TickNet Usage

S-TickNet refers to TickNet enhanced with "Spread-learned spatial features". The provided script `S_TickNet_Dogs.py` is specifically for training/evaluating S-TickNet on the Stanford Dogs dataset.

The command to run it (as found in `S-TickNet.md`) is:
```bash
python ./S_TickNet_Dogs.py --download \
    --base-dir='./' --data-root='./datasets/StanfordDogs' \
    --architecture-types='basic' -g 0
```

**Command Explanation:**
*   `./S_TickNet_Dogs.py`: The Python script for S-TickNet operations on the Stanford Dogs dataset.
*   `--download`: This flag likely attempts to download the Stanford Dogs dataset if not found, or potentially a pre-trained S-TickNet model (behavior should be verified by checking the script's argument parser or documentation if available).
*   `--base-dir='./'`: Specifies the base directory for the project.
*   `--data-root='./datasets/StanfordDogs'`: Sets the path to the Stanford Dogs dataset.
*   `--architecture-types='basic'`: Selects the 'basic' variant of the S-TickNet architecture. Other options might be 'small' or 'large', depending on the script's implementation.
*   `-g 0`: Specifies the GPU ID to use for the operation (e.g., `0` for the first GPU). Set to `-1` for CPU.

This command is likely used for **training** S-TickNet, as evaluation usually involves an `--evaluate` flag and pre-trained model paths. However, without further details on `S_TickNet_Dogs.py`'s argument parsing for evaluation, this is an assumption. You may need to inspect the script for evaluation-specific arguments.

## File Structure Overview

Here's an overview of the most important files and directories in this project:

*   `README.md`: This file, providing a comprehensive guide to the TickNet project.
*   `TickNet_Dogs.py`: Main script for training and evaluating TickNet models on the Stanford Dogs dataset.
*   `TickNet_ImageNet.py`: Main script for training and evaluating TickNet models on the ImageNet-1k dataset.
*   `TickNet_Places365.py`: Main script for training and evaluating TickNet models on the Places365 dataset.
*   `S_TickNet_Dogs.py`: Script specifically for training and evaluating S-TickNet (TickNet with spatial feature enhancements) on the Stanford Dogs dataset.
*   `models/`: This directory contains the core definitions for the neural network architectures.
    *   `models/TickNet.py`: Contains the Python implementation of the TickNet and S-TickNet architectures (Basic, Small, Large variants), including the FR-PDP blocks.
    *   `models/SE_Attention.py`: Implements the Squeeze-and-Excitation (SE) attention mechanism used within the TickNet models.
    *   `models/common.py`: Provides common utility functions, such as activation functions and convolutional block wrappers, used by the models.
    *   `models/datasets.py`: Contains custom dataset loading logic, specifically for the StanfordDogs dataset, including handling of `.mat` files.
*   `checkpoints/`: This directory is the default location for storing and loading model checkpoints. Pre-trained models should be placed here (e.g., `checkpoints/StanfordDogs/small/model_best.pth`). Training scripts will also save their progress here.
    *   `checkpoints/README.md`: Provides a brief explanation of the `checkpoints` directory.
*   `log/`: Intended for storing log files generated during training or other processes.
    *   `log/kaggle/spatial_ticket_batch_64_e100_configa.log`: An example log file, possibly from a Kaggle experiment.
*   `notebook/`: Contains Jupyter notebooks, which are likely used for experimentation, examples, or visualization.
    *   `notebook/Colab_Ticknet.ipynb`: A notebook potentially set up for running TickNet on Google Colab.
    *   `notebook/Kaggle_Ticknet.ipynb`: A notebook potentially set up for running TickNet on Kaggle.
*   `checkmodel.py`: A utility script likely used for inspecting model architecture, parameters, or FLOPs (e.g., using `torchsummary`).
*   `writeLogAcc.py`: A utility script for writing accuracy and loss logs to files during training.
*   `.gitignore`: Specifies intentionally untracked files that Git should ignore.
*   `S-TickNet.md`, `SlideContent.txt`, `TickNetArchitecture.md`: These are supporting documents from the original project, such as specific notes on S-TickNet, content for a presentation, and an older architecture markdown. They are part of the project artifacts. The original `README.md` has been replaced by this comprehensive guide.

## Performance / Results

TickNets are designed to provide a balance of computational efficiency (fewer parameters and FLOPs) and strong performance for image classification tasks, particularly on resource-constrained devices.

**Quantitative Results:**

For detailed quantitative results, performance comparisons, and ablation studies, please refer to the original research paper:

```
@article{neucoTickNetNguyen23,
  author       = {Thanh Tuan Nguyen and Thanh Phuong Nguyen},
  title        = {Efficient tick-shape networks of full-residual point-depth-point blocks for image classification},
  journal      = {Neurocomputing},
  note         = {(submitted in 2023)}
}
```

The paper and its accompanying materials will provide benchmark scores on datasets like ImageNet-1k, Stanford Dogs, and Places365.

**Generating Your Own Benchmarks:**

You can obtain performance metrics (e.g., accuracy, loss) by running the evaluation scripts provided in this repository. Ensure you have downloaded the relevant datasets and pre-trained models, or trained the models yourself.

For example, to evaluate a TickNet-Small model on ImageNet-1k:
```bash
python TickNet_ImageNet.py -a small --evaluate --data /path/to/your/ImageNet
```
(Ensure the model checkpoint exists in the expected location, typically `./checkpoints/ImageNet1k/small/model_best.pth.tar`, or modify the script to point to your checkpoint.)

Similar commands can be used for other datasets and model variants as described in the "Usage" section. The scripts will output metrics like top-1 and top-5 accuracy.

**Qualitative Observations from Experiments (Stanford Dogs Dataset):**

Based on initial experiments (as noted in Slide 10 of `SlideContent.txt`) conducted on the Stanford Dogs dataset using Google Colab:
*   **Training Behavior:** Training loss consistently decreased, and training accuracy steadily increased, indicating that the models were effectively learning from the training data.
*   **Validation Behavior:** Validation loss showed fluctuations, and validation accuracy tended to plateau or fluctuate. This suggests that, under the specific experimental conditions (e.g., limited computational resources on Colab), the models might have experienced some difficulty in generalizing to unseen data or exhibited signs of potential overfitting.

These qualitative observations highlight the importance of hyperparameter tuning, regularization techniques, and potentially more extensive computational resources for optimizing generalization performance. For the most comprehensive and up-to-date results, please consult the official publication.

## Citation

If you use TickNet or any materials from this project in your research, please cite the following work:

```bibtex
@article{neucoTickNetNguyen23,
  author       = {Thanh Tuan Nguyen and Thanh Phuong Nguyen},
  title        = {Efficient tick-shape networks of full-residual point-depth-point blocks for image classification},
  journal      = {Neurocomputing},
  note         = {(submitted in 2023)}
}
```
