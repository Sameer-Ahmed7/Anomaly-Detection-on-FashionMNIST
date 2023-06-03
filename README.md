# Anomaly-Detection-on-FashionMNIST
In this repository, I am designing a model, who are able to find out anomalies in FashionMNIST data. Here I am designing 2 different approaches, One is a baseline approach and the other is State-of-the-art (SOTA) model.

<h1>Task Description:</h1>
Our task is to find out anomaly detection in FashionMNIST dataset. Anomaly detection
on FashionMNIST refers to the process of identifying unusual or anomalous patterns in
the FashionMNIST dataset. FashionMNIST is a popular benchmark dataset in the field
of computer vision, consisting of 60,000 labeled images of 10 different fashion
categories, such as T-shirts, dresses, shoes, and bags. Each image is a 28x28
grayscale picture.
Anomaly detection aims to find instances that deviate significantly from the normal
patterns present in the dataset. In the context of FashionMNIST, anomalies can be
images that do not belong to any of the predefined fashion categories or images that
exhibit unusual or unexpected characteristics compared to the majority of the dataset.
<h1>What is our approach:</h1>
Here to find out anomalies in FashionMNIST dataset, I am considering one
FashionMNIST class (label) as normal and rest of nine classes as anomaly.
<ol>
  <li>
    <b>Normal</b>
    <ul><li>Class 0 (T-shirt/top)</li></ul>
    </li>
 
<li><b>Anomaly</b></li>
  <ul>
<li>Class 1 (Trouser)</li>
<li>Class 2 (Pullover)</li>
<li>Class 3 (Dress)</li>
<li>Class 4 (Coat)</li>
<li>Class 5 (Sandal)</li>
<li>Class 6 (Shirt)</li>
<li>Class 7 (Sneaker)</li>
<li>Class 8 (Bag)</li>
<li>Class 9 (Ankle boot)</li>
    </ul>
</ol>
Here, I am taken two approaches,
One is the baseline approach, where the model is based on Normal GAN, and
Generator is based on autoencoder. On the other hand, The second approach is
based on a state-of-the-art (SOTA) model, where the model is based on DCGAN,
and Generator is based on autoencoder. This time we try to use different loss
functions according to the research paper.
https://paperswithcode.com/paper/gan-based-anomaly-detection-in-imbalance

<h1>Setup Environment<.h1>

<a href="https://github.com/Sameer-Ahmed7/Anomaly-Detection-on-FashionMNIST/blob/main/DL_FinalProject.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>
<h1>Baseline Model:</h1>
The first model is the baseline approach, where the model is based on traditional
GAN, and Generator is based on autoencoder. “I am taken the same
methodology given in the SOTA model research paper”. But here the
approach is traditional GAN instead of DCGAN. And here I am using a single
loss function, Mean Squared Error (MSE) Loss. No multiple losses mention in
the SOTA model paper. The traditional GAN approach and the use of MSE loss
provide a straightforward and easy-to-implement solution. This approach may be
suitable for initial exploratory research or as a baseline to compare against more
complex models.

<h1>Model Architecture:</h1>
<img src="Results/Model-1/model_architecture.jpg" alt="Model Architecture">

<h1>Model Result (Training Data)</h1>
<img src="Results/Model-1/img1.jpg" alt="Model Result (Training Data">

<h1>Model Results (Testing Data):</h1>
<img src="Results/Model-1/img2.jpg" alt="Model Results (Testing Data)">

<p align="center"><i><b>‘The model has some noise, It’s not generating an accurate image of Normal.’</b></i></p>

<h1>State-of-the-art (SOTA) Model</h1>
Here, I am using DCGAN instead of GAN. And the losses will be calculated according to the SOTA model paper.
<h1>Model Architecture:</h1>

<img src="Results/Model-2/model_architecture.jpg" alt="Model Architecture">

<h1>Model Result (Training Data)</h1>
<img src="Results/Model-2/img1.jpg" alt="Model Result (Training Data">

<h1>Model Results (Testing Data):</h1>
<img src="Results/Model-2/img2.jpg" alt="Model Results (Testing Data)">

<p align="center"><i><b>‘The model has not generated noise now, It’s generating approx the same result as a Normal image and changing the image for an anomaly image.’</b></i></p>

<h1> AUROC Result For Test Dataset</h1>


| Model                           | AUROC     |
|---------------------------------|-----------|
| Baseline Model                  | ≈ 87%    |
| State-of-the-art (SOTA) Model   | ≈ 90%    |

<h1>Conclusion:</h1>
<p>The anomaly detection task for the FashionMNIST dataset is that the SOTA model performs significantly better compared to the baseline model. The SOTA model, based on DCGAN architecture and utilizing specific loss functions designed for anomaly detection, demonstrates superior performance in accurately identifying anomalies within the dataset. By employing convolutional layers, the SOTA model captures spatial features and patterns in the images more effectively, resulting in higher-quality and more realistic image generation. Additionally, the adoption of specialized loss functions tailored for anomaly detection enhances the model's ability to distinguish between normal and anomalous instances, leading to improved AUROC results. Overall, the SOTA model presents a more advanced and effective approach for anomaly detection in the FashionMNIST dataset compared to the baseline model. The results suggest that utilizing DCGAN architecture and incorporating specific loss functions can significantly enhance the model's ability to identify anomalies, providing valuable insights for practical applications in the fashion domain.</p>

