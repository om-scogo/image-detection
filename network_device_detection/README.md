# File Descriptions:

1. binary-image-classification.ipynb - Just a tryout model giving accuracy 86%
2. ImageAPI.py - An API which classifies the image as device or rack. If its device then it tries to extract the MAC address.
3. device-vs-rack-1.ipynb - CNN model with accuracy approx 97% (Unseen data - 81.2%)
4. device-vs-rack-2.ipynb - CNN model combined with transfer learning VGG16 with accuracy approx 100% (Unseen data - 92.2%) Image size (128,128,3)
5. device-vs-rack-3.ipynb - CNN model combined with transfer learning VGG19 with accuracy approx 100% (Unseen data - 96.9%) Image size (256,256,3)
6. CSV files are just to support Unseen data accuracies of respective models.
