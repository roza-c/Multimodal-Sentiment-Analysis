Interpreting User Sentiment in Instagram Posts: A Multimodal Deep Learning Approach

This repository contains the code and documentation for the University of Toronto course APS360 project, "Interpreting User Sentiment in Instagram Posts." The project develops a deep learning pipeline to classify the sentiment (positive, negative, or neutral) of Instagram posts by analyzing both the image and its associated text caption.

For a complete overview of the project's methodology, architecture, results, and discussion, please see the Final_Report.pdf.


Project Overview

The sentiment behind social media posts, particularly on a platform like Instagram, is often complex and determined by the interplay between visual and textual elements. This project tackles this challenge by implementing a multimodal deep learning model that processes an image and its caption in parallel to determine the overall sentiment. This approach provides more nuanced and accurate predictions than analyzing either modality alone.


Key Features

    Multimodal Analysis: Integrates two parallel deep learning networks to process both images and text.

    Image Sentiment Model: A fine-tuned ResNet-50 Convolutional Neural Network (CNN) classifies the sentiment conveyed by the image.

    Text Sentiment Model: A fine-tuned RoBERTa-based model, specifically cardiffnlp/twitter-roberta-base-sentiment, analyzes the sentiment of the post's caption.

    Fusion Network: A Weighted Average Neural Network (WANN) acts as a trainable attention mechanism, learning the optimal weights to combine the predictions from the image and text models into a final, unified sentiment classification.


Repository Structure

    Final_Report.pdf: The comprehensive final report detailing the project's background, methodology, results, and ethical considerations.

    Notebooks/: Contains all Jupyter Notebooks used for the project, from data preprocessing and augmentation to model training and evaluation.

    Documents/: Includes supplementary materials, primarily visualizations of model training and validation curves.

    Test-Data/: Contains the manually labeled captions from our custom Instagram test set and the final predictions from our model pipeline.


Datasets

Due to their large size, the primary training datasets are not included in this repository. The models were trained on the following:

    Image Model: The Crowdflower Image Sentiment Polarity dataset, augmented to create a balanced dataset.

    Text Model: A Twitter Sentiments Dataset from Kaggle.

    WANN Model: The Flickr dataset by Borth et al. which contains paired images and captions.

A custom test set of 240 image-caption pairs was manually collected from Instagram and labeled by the team to evaluate the final model's performance in a real-world scenario. The text and prediction data for this set can be found in the Test-Data/ folder.
Getting Started

To explore this project, you can follow the steps outlined in the various notebooks.

    Data Preprocessing: See Notebooks/APS360_ Datapreprocessing.ipynb and Notebooks/Augmenting the data.ipynb to understand how the datasets were prepared.

    Individual Model Training:

        Image Model: Notebooks/Pretrained Model CNN.ipynb

        Text Model: Notebooks/Pretrained Text model.ipynb

    Combined Model: Notebooks/Trainable Weight 2.0.ipynb shows how the WANN was trained to combine the outputs of the two models.

    Evaluation: Notebooks/Model for Demo Testing.ipynb can be used to run the final pipeline on new data.


Dependencies

To run the notebooks, you will need a Python environment with the following major libraries installed:

- PyTorch
- Transformers
- TensorFlow
- Keras
- scikit-learn
- pandas
- NumPy
- Matplotlib


Results

The final multimodal pipeline achieved a 57% accuracy on our custom-collected Instagram test set. This result outperformed the individual models, demonstrating the effectiveness of the ensemble approach. The WANN was particularly effective at leveraging the strengths of both the image and text models to correct their individual weaknesses, especially in cases where their predictions conflicted.

For detailed quantitative and qualitative results, including confusion matrices and an analysis of misclassified instances, please refer to the Final_Report.pdf.
Contributors

    Charlotte Vedrines

    Zeina Shaltout

    Roza Cicek