# Face2BMI

Face2BMI: Predicting Body Mass Index from Facial Images

A deep learning pipeline to estimate BMI from facial images using transfer learning and traditional regressors.
This project builds upon prior work (Kocabey et al.) and improves it using modern CNN backbones (EfficientNet, ResNet, FaceNet) and ensemble regression strategies to achieve non-invasive, image-based BMI prediction.

**Project Overview:**

Body Mass Index (BMI) is a key health metric, but public datasets often rely on self-reported values, introducing bias and inaccuracies. This project presents a computer vision–based approach to predict BMI from facial images—offering a scalable, objective, and non-invasive alternative.

We improve on earlier methods by:

Leveraging state-of-the-art CNN architectures\
Employing fine-tuned transfer learning\
Integrating traditional regressors and ensembles\
Building a real-time prediction UI for static images and video streams

**Reference Paper Replication:**

This project replicates and extends the work from: Kocabey et al. (2017) – Face-to-BMI: Using Computer Vision to Infer Body Mass Index on Social Media

Link to paper: https://arxiv.org/pdf/1703.03156

Baseline models: VGGFace and VGGNet (VGG16)\
Dataset: VisualBMI (Reddit facial image posts with self-reported BMI)\
We reproduced their methodology and benchmarked our models against the original results:

Model	Paper Pearson r (Overall / Male / Female)	Our Pearson r (Overall / Male / Female)\
VGGFace	0.65 / 0.71 / 0.57	0.64 / 0.65 / 0.63\
VGGNet	0.47 / 0.58 / 0.36	0.65 / 0.70 / 0.58

✅ Our models replicated the baseline paper results closely

🚀 We exceeded the paper’s performance using fine-tuned VGG19 and EfficientNetB3:

EfficientNetB3 achieved the best overall Pearson r = 0.67\
VGG19 produced the best male Pearson r = 0.70, matching the paper's highest value\
This validates the original approach while demonstrating the value of modern architectures and deeper optimization strategies.

**Dataset:**

Source: VisualBMI Project (Reddit-sourced facial images)

3,210 training images\
752 test images\
Includes: Facial image, gender, ground truth BMI\
Preprocessing: Manual cleaning, cropping, resizing\
Balanced across gender and BMI ranges

**Model Pipeline:**

🔍 Stage 1: Feature Extraction (CNNs)

Model	Description\
VGGFace	Pre-trained on facial ID tasks (fc6 embedding)\
VGG16/19	Classic CNNs with regression heads\
ResNet50	Fine-tuned top 25 layers, GAP + Dense\
EfficientNetB3	Best performing model (compound scaling)\
FaceNet	512D embeddings using MTCNN + Inception-ResNet

📊 Stage 2: Regression Modeling

Regressors Tested\
Ridge, SVR, Random Forest, KNN, MLP, XGBoost, LightGBM, CatBoost\
Ensembling: SVR + CatBoost + LightGBM yielded best results\
Gender concatenated as feature for all regressors

📈 Evaluation Metrics

MAE – Mean Absolute Error (lower is better)\
Pearson Correlation (r) – Measures prediction alignment with true BMI (closer to 1 is better)\
Metrics reported for overall, male, and female subgroups

**Key Results:**

Model	MAE	Pearson r (Overall)	Male r	Female r\
EfficientNetB3	4.72	0.67	0.69	0.65\
VGG19	4.99	0.65	0.70	0.58\
VGGFace	5.04	0.64	0.65	0.63\
FaceNet	5.52	0.58	0.62	0.53\
ResNet50	6.04	0.47	0.41	0.51

EfficientNetB3 beat all baselines, including reference paper’s VGGFace (r = 0.65) and VGGNet (r = 0.47)

Best Male r = 0.70 with VGG19\
Best Female r = 0.65 with EfficientNet

**Exploratory Data Analysis (EDA):**

Verified and cleaned missing/mismatched .bmp image entries\
Final dataset: 3,210 train / 752 test images\
Resized and normalized per model requirements:\
224×224 for VGGFace, ResNet, VGG19\
160×160 for FaceNet\
300×300 for EfficientNet

**Experimental Highlights:**

Fine-tuning improved performance in VGG19 and EfficientNet\
Classical regressors added interpretability and marginal gains\
Ensembling improved prediction stability and r score\
FaceNet underperformed when fine-tuned — pretrained weights proved better

**Implementation Details:**

Libraries: TensorFlow, Keras, OpenCV, MTCNN, scikit-learn, CatBoost, LightGBM, XGBoost\
Image augmentation: Flip, rotation, zoom, shift\
Feature extraction → regressor head → ensemble\
Optional: Live prediction UI (image or webcam input)

**Future Work:**

-Model & Data:

Add more data across ethnicities, ages, lighting conditions\
Try attention-based architectures like ViT or hybrid CNN-Transformer models\
Interpretability & Fairness\
Evaluate performance by skin tone, age, expression\
Integrate tools like SHAP for feature explainability

-Deployment & Application:

Mobile/web deployment for BMI estimation as a pre-screening tool\
Potential integration into telemedicine or public health screening
