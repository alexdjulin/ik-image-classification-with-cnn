# Project I | Deep Learning: Image Classification with CNN

Build a Convolutional Neural Network (CNN) model to classify images from a given dataset into predefined categories/classes.

![gradio_demo](readme/gradio_demo.png)

[Task Descriptions and Project Instructions](https://github.com/ironhack-labs/project-1-deep-learning-image-classification-with-cnn)

## Repository Folders and Files

Here is a short description of the folder and files available on the repository.

### Documents
**Group2 - Image Classification with CNN - Final Report**  
Final PDF report (shared with group)  

**Group2 - Image Classification with CNN - Presentation Slides**  
Final Slides presentation (shared with group)  

### Notebooks  
- **animals10_dataset_exploration**: Explores the Animals10 dataset (not used)
- **cifar10_dataset_exploration**: Explores the Cifar10 dataset (used in project)
  
- **model1_training**: Training a minimalist CNN from scratch
- **model2_training**: Improving model 1 with a more complex structure
- **model3_training**: Improving model 2's hyperparameters and experimenting with optimizers
- **model4_training**: Improving model 3 and reducing overfitting
- **model5_training**: Trying to replicate VGG16 architecture and train it from scratch
- **model6_training**: Transfer learning on VGG16 - Failed
- **model7_training**: Transfer learning on ResNet50 - Best model
- **model8_training**: Fine-tuning model 7 - Failed
  
- **model_testing**: Load and test the model on unseen pictures
- **model_deployment**: Deploy a demo of our best model on Gradio

### Python Modules
**helpers.py:** Helper module used my all the notebooks with useful methods to load and preprocess the dataset, evaluate the model, plot curves and confusion matrix, and so on.

### Additional Folders
**test_images**: folder of unseen images to test on the model

## Installation
Use **requirements.txt** to install the required packages to run the notebooks. It is advised to use a virtual environment.
```bash
python -m venv .venv
.venv/Scripts/activate
pip install -r requirements.txt
```

## Acknowledgements
A big thank you to my colleagues Freddy and KK who worked with me all along this exciting and challenging project.