# relation
## Environment Setup

To run the `feature_map_visualize` project, you need to configure your environment by following these steps:

1. **Install environment dependencies:**
   - Ensure you have conda installed. Install all necessary packages and dependencies using the conda environment file:
     ```bash
     conda env create -f feature_map_visualize/environment.yaml
     ```
   - Activate the newly created conda environment:
     ```bash
     conda activate <environment_name>
     ```

2. **Install additional Python packages:**
   - Install OpenCV for image processing features:
     ```bash
     pip install opencv-python
     ```
   - Install pycocotools version 2.0.7, a tool library for annotating and evaluating images in the COCO dataset:
     ```bash
     pip install pycocotools==2.0.7
     ```

Ensure all the steps above are completed before attempting to run any code examples from the project.
