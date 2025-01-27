# Explainable Human-centered Traits from Head Motion and Facial Expression Dynamics
This is the official repository for the paper [Multiview Attention Fusion for Explainable Body Language Behavior Recognition]
Body language behavior, including gestures and fine-grained movements not only reflects human emotions, but also serves as a versatile cue for enhancing emotional intelligence and creating responsive technologies. In this work, we explore the efficacy of multiview-multimodal cues for explainable prediction of bodily behavior. This paper proposes an attention fusion method that combines features extracted from (1) multiview videos termed “RGB”, (2) their multiview Discrete Cosine Transform representations termed “DCT” and (3) three stream skeleton features termed “Skeleton”, via a transformer-based approach. We evaluate our approach on the diverse BBSI and Drive&Act datasets. Empirical results confirm that the RGB, DCT and Skeleton features enable discovery of multiple class-specific behaviors resulting in explainable predictions. Our key findings are: (a) Multimodal approaches outperform unimodal counterparts in categorizing bodily behavioral classes; (b) Efficient class predictions and plausible explanations are achieved with both unimodal and multimodal approaches; and (c) Empirical results confirm the superiority of our approach compared to state-of-the-art methods on both datasets.

## Framework Overview:
![framework_overview](https://github.com/deepsurbhi8/Explainable_Human_Traits_Prediction/assets/79365852/9c3480ab-78c1-40ae-89bb-02b75503fa37)
*Overview of the proposed framework: Kinemes (elementary head motions), action units (atomic facial movements) and speech features employed for explainable trait prediction*

## Implementation Details:
**Preprocessing** -- We've used [Openface](https://github.com/TadasBaltrusaitis/OpenFace) to extract pose angles and action units for kineme and AU generation. For audio feature generation, we've used [Librosa](https://librosa.org/doc/latest/index.html) python package.

**Model Architecture** -- We've done Chunk level and video level analysis for both the datasets MIT and FICS. For more details, please refer to the paper.

**Requirements** -- This code was tested on Python 3.8, Tensorflow 2.7 and Keras 2.7. It is recommended to use the appropraite versions of each library to implement the code.

**Dataset Features** -- All the openface and librosa features for both the datasets are available at the following link: (https://drive.google.com/drive/folders/1gyPFH2kSGEHgpNxiGDVTV2w6eaN0wqsU?usp=sharing). Note: In case of any issue regarding data access, please write an email to: er.surbhimadan.2013@gmail.com

## Repository Structure:
<pre>
<code>
├── README.md    
├── Architectures           
│   ├── Attention_fusion_figure.pdf      # Additive attention fusion architecture overview
│   ├── Framework_overview.pdf           # Overview of the proposed framework
│   └── LSTM_arch.pdf                    # Trimodal feature fusion architecture
├── Codes         
│   ├── Bimodal Fusion                   # Bimodal implementation of different combination of features (AU, Kineme and Audio features)        
│   │   ├── Classification_Feature_au_kin_VL_MIT.py  # (Eg: contains the code to implement video-level classification appraoch over the MIT dataset using AU and Kineme matrices using Feature fusion)
│   ├── Feature Extraction               
│   │   ├── Action_units_data_prep.py    # Code to create Action Unit data matrix from the openface extracted au files
│   │   ├── Audio_chunk_formation.py     # Code to create chunks from the Audio data matrix 
│   │   ├── Kineme_data_prep.py          # Code to create kineme feature data matrix for train and test set files
│   ├── Trimodal Fusion                  # Different approaches (Decision, Feature and Attention-based fusion) of the three modalities (AU, Kineme, Audio features)
│   ├── Unimodal Approach                # Single modality code implementation over the two datasets
├── Data         
│   ├── FICS_dataset                  # Bimodal implementation of different combination of features (AU, Kineme and Audio features)        
│   │   ├── FICS_test_files.zip       # All features (Action Unit, Kineme and Audio) over the test set of FICS dataset
│   │   ├── FICS_train_files.zip      # All features over the train set of FICS dataset
│   │   ├── FICS_val_files.zip        # All features for the validation set of FICS dataset
│   ├── MIT_dataset    
│   │   ├── MIT_AU_features.zip       # Action Unit features extracted over all files of the MIT dataset
│   │   ├── MIT_kineme_features.zip   # Kineme representation for the head pose values over the MIT dataset
├── Presentation         
│   ├── Explainable_Human_Traits_Prediction.pdf        
</code>
</pre>
## Related Links:
[Head Matters Paper (ICMI 2021)](https://dl.acm.org/doi/10.1145/3462244.3479901)

[Head Matters Code](https://github.com/MonikaGahalawat11/Head-Matters--Code) 


