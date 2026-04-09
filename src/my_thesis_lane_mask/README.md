# Thesis Lane Mask Tool

This repository provides a self-contained pipeline for generating highly accurate driving lane masks from dashboard camera images.

## Structure
- `ai_segmentation.py`: Uses DeepLab V3 (a deep neural network) to identify raw boundaries of the road and context elements (cars, sidewalks).
- `crf_smoother.py`: Implements a Conditional Random Field (CRF) that strictly aligns the AI's probability mask to the real edges inside the image.
- `generate_lane_mask.py`: The entry script connecting all pieces together.

## Execution Requirements
The tool relies on `tensorflow` and `pydensecrf`. The dependencies are mapped inside `requirements.txt`.
Note: The parent folder must contain testing images (e.g. `test1.png`) and the DeepLab V3 `tar.gz` frozen graph file.

### Google Colab Instructions (Cloud Processing)
Because this tool natively requires a GPU to operate efficiently, it is best ran under Google Colab.

1. Zip the `my_thesis_lane_mask` folder (call it `my_thesis_lane_mask.zip`).
2. Upload the zip to your Google Drive. 
3. Open [Google Colab](https://colab.research.google.com/) and create a **New Notebook**.
4. In the menu, go to **Runtime > Change runtime type**, select **T4 GPU**, and save.
5. Paste the following block into a code cell and execute it:

```python
# 1. Mount your Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. Extract your zip into Colab workspace
!unzip /content/drive/MyDrive/my_thesis_lane_mask.zip -d /content/lane_tool

# 3. Download the heavy AI model directly to Colab
!wget http://download.tensorflow.org/models/deeplabv3_cityscapes_train_2018_02_06.tar.gz -O /content/deeplabv3_cityscapes_train_2018_02_06.tar.gz

# 4. Install requirements natively
%cd /content/lane_tool/my_thesis_lane_mask
!pip install -r requirements.txt

# 5. Execute on the testing sample (assuming test1.png was packaged inside)
!python generate_lane_mask.py -i test1.png -o output_mask.png

# 6. View the output
from IPython.display import Image, display
print("Lane Mask Output:")
display(Image("output_mask.png", width=600))
```
