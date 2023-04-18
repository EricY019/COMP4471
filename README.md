# Dynamic CAPTCHA Recognition
We collected a dynamic textbased CAPTCHA dataset (DyCap), containing 10,000 CAPTCHAs with 250,000 frames. All dynamic CAPTCHAs include moving characters and interference characters, and are annotated with corresponding ground truths. Furthermore, we present our proposed network, which utilizes features extracted by a convolutional neural network (CNN) between neighbouring frames in a duplet network.

```./Code_CAPTCHA_generator```stores the code of our dataset generator.

```./Code_ResNet-2D``` stores the code of our baseline network, specifically, utilizing ResNet as feature extractor.

```./Code_network``` stores the code of our proposed network.

```./Example_dataset``` presents some examples of our dataset.

# Dataset
Dataset-10000 can be accessed via https://drive.google.com/file/d/1oQjYaxnyYwyObrHv7yPa3A3FRMBC8_6c/view?usp=sharing

Dataset-2000 can be accesssed via https://drive.google.com/file/d/1Q1w5Ej0YnQxLAbydYPcE1e4222lgqA-8/view?usp=sharing

# Training Checkpoint
Our reported training checkpoint can be accessed via https://drive.google.com/file/d/1qCt2LGcMiNRTJawm67MJVXWISFnOy_eF/view?usp=sharing

# Final Report
Our final report can be accessed via https://drive.google.com/file/d/1AhgziPbrOHoIIWc7BVmBfYFlAUJUe3z-/view?usp=share_link
