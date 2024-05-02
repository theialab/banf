# BANF @ 2D Fitting

<video width="920" controls>
  <source src="vis_data/2d_fitting.mp4" type="video/mp4">
</video>

## Installation
Run 
`
pip install -r requirements.txt
`
to install all necessary dependencies.

## Data
You can load the training data and our results from the following link: [2D Fitting Data](https://drive.google.com/file/d/1sS2sfqkGdhKh-marO4wBIjM-A0jzVQ6e/view?usp=sharing). The data should be unzipped and placed in the current directory.

## Usage
Files `main_sinc.py` and `main_linear.py` contain the training and evaluation scripts for sinc and linter interpolations, respectively. To train a model, execute the following command:
```bash
python main_linear.py --image_id 0879
```