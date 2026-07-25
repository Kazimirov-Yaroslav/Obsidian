---
title: "Extreme Solar Flare Prediction Using Residual Networks with HMI Magnetograms and Intensitygrams"
authors: ["Juyoung Yun", "Jungmin Shin"]
year: 2024
source_file: "Extreme Solar Flare Prediction Using Residual Networks with HMI Magnetograms and Intensitygrams.pdf"
tags: [paper/original, solar-flares, deep-learning, resnet, space-weather]
---
![[Extreme Solar Flare Prediction Using Residual Networks with HMI Magnetograms and Intensitygrams.pdf]]

## Abstract

Solar flares, especially C, M, and X class, pose significant risks to satellite operations, communication systems, and power grids. We present a novel approach for predicting extreme solar flares using HMI intensitygrams and magnetograms. By detecting sunspots from intensitygrams and extracting magnetic field patches from magnetograms, we train a Residual Network(ResNet) to classify extreme class flares. Our model demonstrates high accuracy, offering a robust tool for predicting extreme solar flares and improving space weather forecasting. Additionally, we show that HMI magnetograms provide more useful data for deep learning compared to other SDO AIA images by better capturing features critical for predicting flare magnitudes. This study underscores the importance of identifying magnetic fields in solar flare prediction, marking a significant advancement in solar activity prediction with practical implications for mitigating space weather impacts.

## 1 Introduction

Solar flares are intense bursts of radiation caused by the release of magnetic energy associated with sunspots. These events can significantly disrupt satellite communications, navigation systems, and power grids on Earth(Oceanic and Administration 2024; Schwenn 2006; Rao et al. 2009; Filjar and Kos 2006). The severe consequences of solar flares, especially extreme C, M, and X class events, underscore the importance of accurate prediction to protect infrastructure and mitigate their impacts(Pandey et al. 2023). Additionally, solar flares pose risks to space weather, affecting technological systems, astronauts, and high-altitude flights(Schwenn 2006; Rao et al. 2009). Effective prediction allows for timely warnings and protective measures, such as adjusting satellite orbits and safeguarding power grids, thereby minimizing potential damage and ensuring the continued operation of critical services(Filjar and Kos 2006).

The need for accurate and timely space weather forecasts is growing with our increasing reliance on satellite technology and high-altitude aviation. Solar activity, including flares and coronal mass ejections, can influence the Earth's magnetosphere and ionosphere, leading to geomagnetic storms that affect satellite operations, GPS accuracy, and ground-based technologies(Schwenn 2006; Rao et al. 2009). Recent advancements in solar observations from the Solar Dynamics Observatory(SDO) have provided high-resolution data, particularly from the Helioseismic and Magnetic Imager(HMI)(He et al. 2016). HMI captures intensitygrams, detailing sunspots, and magnetograms, mapping the Sun's magnetic field. These datasets are crucial for understanding the magnetic complexity leading to solar flares(Oceanic and Administration 2024; Schwenn 2006; Rao et al. 2009).

In this study, we propose a novel approach to predict extreme solar conditions, specifically C, M, and X class solar flares, by leveraging HMI intensitygrams(HMII) and magnetograms(HMIB). Our methodology involves detecting sunspots from intensitygrams and segmenting corresponding magnetic field patches from magnetograms. These patches are then used to train a Residual Network(ResNet)(He et al. 2016), a deep learning model known for its effectiveness in image classification. We focus on a binary classification task by grouping Quiet, A, and B class events together and C, M, and X class events together to predict extreme flare occurrences. This approach emphasizes the importance of magnetic field analysis for accurate solar flare prediction.

## 2 Related Works

In recent years, solar flare prediction has significantly advanced with the application of machine learning and deep learning techniques, leveraging data from the Solar Dynamics Observatory(SDO)(NASA 2024), particularly the Helioseismic and Magnetic Imager(HMI). Various studies have explored the utility of HMI data for predicting solar flares. Bobra and Couvidat used SHARP vector magnetic field data from HMI to calculate 25 physical parameters, including total flux and proxies for helicity and energy, training a machine learning algorithm to predict solar flares based on these parameters(Bobra et al. 2014). Nishizuka et al. developed the Deep Flare Net(DeFN), employing deep learning for operational solar flare prediction using 79 physics-based features extracted from HMI NRT data(Nishizuka et al. 2018). Additionally, Li et al. focused on enhancing the interpretability of predictions using attention mechanisms within deep neural networks(Pandey et al. 2023).

These approaches often rely on a broad range of features extracted from HMI data, combining both magnetograms and intensitygrams. Bobra and Couvidat's method emphasizes physical parameters derived from SHARP data, while Nishizuka et al.'s DeFN integrates a large number of physics-based features. Li et al.'s approach is distinct in its use of attention mechanisms to improve interpretability, allowing for better understanding of the decision-making process within the neural network.

Our study is different because it specifically targets the prediction of extreme C, M, and X class solar flares using a more focused approach based on magnetic field patches. Unlike previous methods that use a wide range of image, we concentrate on detecting sunspots from intensitygrams and segmenting the corresponding magnetic field patches from magnetograms. These patches are then used to train and predict a Residual Network(ResNet)(He et al. 2016), a deep learning model known for its effectiveness in image classification. By focusing on sunspot detection and magnetic field patch extraction, our method simplifies the feature selection process, potentially reducing the complexity and computational load compared to models that handle a broader set of features.

## 3 Methodology

Our approach involves a two-step process as shown in Figure 1: training a Residual Network(He et al. 2016) on the SDO dataset(NASA 2024) to predict solar flare occurrences and using computer vision techniques to detect sunspots on HMI intensitygrams and segment magnetic field patches from HMI magnetograms. These patches are then used with the trained model to predict extreme flare occurrences.

![[Figure 1.png]]
*Figure 1: The two-phase process for extreme solar flare prediction. Phase 1 involves training a deep neural network(ResNet) with HMI magnetogram(HMIB) data to classify extreme and non-extreme events. Phase 2 includes detecting sunspots on HMI intensitygram(HMII) images, extracting corresponding magnetic field patches from HMIB, and using the trained model to predict extreme solar flare occurrences.*

### 3.1 Training Deep Neural Networks with Magnetogram

We utilized the ResNet50 architecture(He et al. 2016) for training on HMI magnetogram patch images to predict extreme solar flares. Magnetogram patch images were sourced from the Solar Dynamics Observatory(SDO) and resized to 256 × 256 pixels. The images were annotated with flare classifications based on their intensity levels, grouped into two categories: Quiet, A, and B class events as QA, and C, M, and X class events as MX. This binary classification aids in predicting extreme flare occurrences more effectively. The ResNet50 model, with weights pre-trained on the ImageNet dataset(Deng et al. 2009), was used. Final Fc layer consists of 512 neurons and ReLU activation, and an output layer is with a softmax activation function to classify images into QA or MX.

### 3.2 Sunspot Detection for Patch Extraction

Our methodology for detecting sunspots using HMI intensitygrams involves key steps as shown in Figure 2. We start with real-time HMI intensitygram images, convert them to RGB color space, and apply a noise-canceling filter(Buades, Coll, and Morel 2005). The denoised images are then binarized using adaptive thresholding(Gonzalez and Woods 2006) to distinguish sunspots from the background. Contours on the binary image(Marr 1980) detect sunspots, which are filtered by location within the solar disk. Close bounding rectangles are merged. Using detected sunspots, we extract 256x256 patches from HMI magnetogram images centered on the sunspot locations. These patches are fed into a trained ResNet to predict extreme solar flare occurrences.

![[Figure 2.png]]
*Figure 2: The process of detecting sunspots using HMI intensitygrams. From left to right: the original HMI intensitygram (HMII), the binarized HMII, detected sunspots on the HMII, and the corresponding sunspot detection mapped onto the HMI magnetogram(HMIB).*

## 4 Results

### 4.1 Experimental Setting

We used a ResNet-50 model(He et al. 2016) pretrained on the ImageNet dataset(Deng et al. 2009) to train our solar flare prediction model. The training was conducted using the SDOBenchmark dataset(Bolzern and Aerni 2024), which consists of 256x256 pixel images. The model was trained for 20 epochs with a batch size of 32. The Adam optimizer(Kingma and Ba 2014) was employed with a learning rate of 0.0001. The experiments were performed on an Nvidia RTX 4080 GPU.

### 4.2 Training Results of Deep Neural Networks

The training results of our deep neural networks demonstrate that using HMI magnetogram(HMIB) patches for predicting extreme solar flares yields higher accuracy compared to other satellite image datasets. As shown in Figure 3, the test accuracy achieved with HMIB images is 0.755, significantly higher than the accuracies obtained with various AIA datasets, such as AIA-131(0.713), AIA-171(0.710), and AIA-335(0.712). This underscores the importance of magnetic field data captured in HMIB for accurate solar flare prediction.

![[Figure 3.png]]
*Figure 3: Comparison of test accuracy for different satellite image datasets using the SDOBenchmark dataset(Bolzern and Aerni 2024). The model used is ResNet-50(He et al. 2016) pre-trained on ImageNet(Deng et al. 2009) for predicting two flare classes, QA and MX.*

### 4.3 Prediction Results

Our model demonstrated ability in predicting MX(C, M, and X) class solar flares using HMI magnetograms and intensitygrams. Figure 4 shows the final prediction result, with the left panel showing the HMI magnetogram(HMIB) where the predicted flare region is highlighted, and the right panel displaying the corresponding AIA-131 image that confirms the flare occurrence.

![[Figure 4.png]]
*Figure 4: Flare Prediction Results.(Left) HMI magnetogram(HMIB) showing predicted MX class flare region.(Right) Corresponding AIA-131 image showing the flare. Timeline is 2022-01-16 00:00:00.*

## 5 Discussion

Our study highlights the potential of using HMI magnetograms and intensitygrams for the prediction of extreme solar flares. However, there are several limitations that need to be addressed in future research. One significant limitation is the overall scarcity of magnetogram training data, which constrains the robustness and generalizability of our model. To mitigate this, we propose the creation of a comprehensive solar image dataset specifically designed for research purposes. This dataset would provide a valuable resource for the scientific community, enabling more extensive and varied training of predictive models. Moreover, expanding our research with a richer magnetogram dataset could enhance the model's capability to capture the critical moments leading up to a flare. By focusing on the magnetogram data just before flare onset, we could develop models that not only predict the occurrence of solar flares but also provide more precise forecasts of the intense sunspot eruptions immediately preceding these events.

## 6 Conclusion

In this study, we proposed a novel approach for predicting extreme solar flares using HMI intensitygrams and magnetograms. By detecting sunspots from intensitygrams and extracting corresponding magnetic field patches from magnetograms, we trained a Residual Network(ResNet) to classify these flares with high accuracy. Our experimental results demonstrated that HMI magnetograms provide superior data for deep learning models compared to other SDO AIA images, due to their ability to capture critical magnetic field features necessary for accurate flare prediction. Our research underscores the importance of magnetic field data in forecasting solar flare magnitudes. This advancement in solar activity prediction has practical implications for mitigating the impacts of space weather.

## Acknowledgement

We would like to thank Roman Bolzern and Michael Aerni from the Institute for Data Science, FHNW, Switzerland for providing the SDOBenchmark dataset. We also acknowledge the SDO satellite mission and JSOC Stanford for providing the raw data.

## References

Bobra, M. G.; Sun, X.; Hoeksema, J. T.; Turmon, M.; Liu, Y.; Hayashi, K.; Barnes, G.; and Leka, K. 2014. The Helioseismic and Magnetic Imager(HMI) Vector Magnetic Field Pipeline: SHARPs–Space-Weather HMI Active Region Patches. Solar Physics, 289(9): 3549–3578.
Bolzern, R.; and Aerni, M. 2024. SDOBenchmark: A machine learning image dataset for the prediction of solar flares. https://i4ds.github.io/SDOBenchmark. Accessed: 2024-05-18.
Buades, A.; Coll, B.; and Morel, J. M. 2005. A non-local algorithm for image denoising. In 2005 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR'05), volume 2, 60–65.
Deng, J.; Dong, W.; Socher, R.; Li, L.-J.; Li, K.; and Fei-Fei, L. 2009. ImageNet: A large-scale hierarchical image database. In 2009 IEEE Conference on Computer Vision and Pattern Recognition, 248–255. Ieee.
Filjar, R.; and Kos, T. 2006. GPS Positioning Accuracy in Croatia during the Extreme Space Weather Conditions in September 2005. In European Navigation Conference ENC 2006. Manchester, UK.
Gonzalez, R. C.; and Woods, R. E. 2006. Digital Image Processing. Boston, MA: Pearson Education, 3rd edition.
He, K.; Zhang, X.; Ren, S.; and Sun, J. 2016. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, 770–778.
Kingma, D. P.; and Ba, J. 2014. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980.
Marr, D. 1980. Theory of edge detection. Proceedings of the Royal Society of London B: Biological Sciences, 207(1167): 187–217.
NASA. 2024. Solar Dynamics Observatory(SDO). Retrieved from https://sdo.gsfc.nasa.gov/.
Nishizuka, N.; Sugiura, K.; Kubo, Y.; Den, M.; Watari, S.; and Ishii, M. 2018. Operational solar flare prediction model using Deep Flare Net. Earth, Planets and Space, 70(1): 1–18.
Oceanic, N.; and Administration, A. 2024. Space weather. Retrieved from https://www.noaa.gov/education/resource-collections/space-weather.
Pandey, C.; Ji, A.; Angryk, R. A.; and Aydin, B. 2023. Towards Interpretable Solar Flare Prediction with Attention-based Deep Neural Networks. In 2023 IEEE Sixth International Conference on Artificial Intelligence and Knowledge Engineering(AIKE), 83–90. Los Alamitos, CA, USA: IEEE Computer Society.
Rao, P. V. S. R.; Krishna, S. G.; Prasad, J. V.; Prasad, S. N. V. S.; Prasad, D. S. V. V. D.; and Niranjan, K. 2009. Geomagnetic storm effects on GPS based navigation. Ann. Geophys., 27: 2101–2110.
Schwenn, R. 2006. Space Weather: The Solar Perspective. Living Reviews in Solar Physics, 3(2).