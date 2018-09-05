# Music-Genre-Verification
Are the genres of two music samples the same or different?

* This code contains an implementation of a Siamese-like convolutional neural network 
architecture to predict similarity of genre between two audio samples (processed as melspectrograms).



* Download the fma_small dataset from https://github.com/mdeff/fma
* Save the folders which are named as '000','001', etc containing the audio files in a folder called ./music_samples/ as the processing functions look for that folder.

* Run ``` python load_audio.py ``` to extract and save the melspectrogram and the audio features (we don't use this directly in our code.)

* Run the training and testing on the dataset:
```
python models.py
```
