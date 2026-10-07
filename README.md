# Handwritten Letter Recognition with CNN

A deep learning project that recognises handwritten English letters using a Convolutional Neural Network (CNN).

The project covers data cleaning, image preprocessing, CNN experiments, model evaluation and an interactive Gradio drawing demo.

## Repository Files

| File | Description |
| --- | --- |
| [EMNIST CNN.ipynb](EMNIST%20CNN.ipynb) | Notebook containing preprocessing, training, evaluation and the Gradio demo. |
| [Handwritten Letter Recognition.pptx](Handwritten%20Letter%20Recognition.pptx) | Presentation summarising the experiments and results. |

The notebook also requires `emnist-letters.csv`. Training generates `CA1_PartA_final_model.weights.h5`, which is used to reload the model and run the demo.

## Dataset

The notebook uses an EMNIST letters CSV with:

- **99,040 original rows**
- **26 letter classes**, A–Z
- **28 × 28 grayscale images**
- One label column and 784 pixel columns

Data inspection identifies invalid labels, duplicate rows and an invalid pixel value.

Cleaning keeps labels from **1 to 26** and pixel values from **0 to 255**, leaving **88,800 images**. Duplicate rows are inspected but not all removed.

## Image Preprocessing

1. Reshape the 784 pixels into a 28 × 28 image.
2. Transpose each image to correct its orientation.
3. Normalise pixel values to the range `0–1`.
4. Add a channel dimension to produce shape `(28, 28, 1)`.
5. Convert class labels from `1–26` to `0–25`.

The transpose is an orientation correction, separate from data augmentation.

## Data Split

A stratified split uses seed **88** to preserve class proportions.

| Set | Proportion | Images |
| --- | ---: | ---: |
| Training | 70% | 62,160 |
| Validation | 10% | 8,880 |
| Test | 20% | 17,760 |

## Model Experiments

The notebook compares CNN configurations using:

- Different convolution filters and dense-layer sizes
- ReLU and LeakyReLU
- Batch Normalisation
- Dropout
- Early stopping
- Static training-image augmentation

### Validation Results

| Model | Validation Accuracy | Validation Loss |
| --- | ---: | ---: |
| Baseline CNN | 92.97% | 0.2119 |
| Model 1 | 93.98% | 0.2064 |
| Model 2 | 93.89% | 0.2156 |
| Model 3 | 93.85% | 0.2170 |
| Model 4 | 94.34% | 0.1889 |
| Static augmentation experiment | 94.13% | 0.1948 |
| Dropout-tuned model | 94.43% | 0.1798 |

Static augmentation does not improve validation performance in this experiment and is not selected for the final model.

## Selected CNN Architecture

- Conv2D: 32 filters, 3 × 3 kernel, same padding
- Batch Normalisation and LeakyReLU
- Max pooling
- Conv2D: 64 filters, 3 × 3 kernel, same padding
- Batch Normalisation and LeakyReLU
- Max pooling
- Dropout: `0.35`
- Flatten
- Dense: 384 units
- Batch Normalisation and LeakyReLU
- Dropout: `0.40`
- Dense: 26 units with softmax

Training uses **Adam** and **sparse categorical cross-entropy**.

Early stopping monitors validation loss with `patience=3` and restores the best weights. The training limit is **30 epochs**, with a batch size of **200**.

## Final Test Results

The selected architecture is rebuilt and retrained before final evaluation.

| Metric | Test Result |
| --- | ---: |
| Accuracy | 94.16% |
| Error rate | 5.84% |
| Loss | 0.1763 |
| Macro precision | 0.9423 |
| Macro recall | 0.9416 |
| Macro F1 score | 0.9418 |
| Weighted F1 score | 0.9418 |

Of **17,760 test images**, the model classifies **16,723 correctly** and **1,037 incorrectly**.

These results come from the notebook's saved outputs.

### Common Classification Errors

Similar handwritten shapes are the main challenge.

| True Letter | Predicted Letter | Errors |
| --- | --- | ---: |
| I | L | 167 |
| L | I | 132 |
| G | Q | 66 |
| Q | G | 60 |
| U | V | 32 |
| V | U | 21 |

The lowest class recalls are **73.72% for I** and **79.65% for L**.

## Gradio Drawing Demo

The notebook includes an interface where users can draw one letter and click **Predict**.

The demo displays:

- Predicted letter
- Processed 28 × 28 input image
- Top five model confidence scores

Drawing preprocessing converts the image to grayscale, adjusts its polarity, crops the letter, resizes it and centres it on a 28 × 28 canvas.

The demo reloads the saved CNN weights without retraining.

## Tools Used

- Python
- TensorFlow and Keras
- pandas and NumPy
- scikit-learn
- Matplotlib
- Gradio
- Pillow
- Jupyter Notebook

## How to Run

1. Download or clone the repository.
2. Install the required packages:

   ```bash
   python -m pip install pandas numpy matplotlib tensorflow scikit-learn gradio pillow notebook
   ```

3. Place `emnist-letters.csv` beside the notebook.
4. Start Jupyter:

   ```bash
   jupyter notebook
   ```

5. Open `EMNIST CNN.ipynb` and run the cells in order.
6. Run the training and weight-saving cells to generate:

   ```text
   CA1_PartA_final_model.weights.h5
   ```

7. Run the Gradio demo cells and open the local URL shown in the output.

Training time depends on the available hardware. A compatible GPU can speed up the CNN experiments.

## Limitations and Future Improvements

- Similar letters remain difficult to distinguish.
- Performance on drawings made in the demo may differ from performance on the test dataset.
- Fixed seeds do not guarantee identical results across hardware and package versions.
- Future work could compare other augmentation methods, inspect duplicate overlap across splits and test more handwriting styles.

## Author

**Yan Myoe Naing**  
Applied AI and Analytics, Singapore Polytechnic
