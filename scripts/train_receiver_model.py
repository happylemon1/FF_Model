import sys
import os

# Get the absolute path of the script's directory (i.e., .../FF_Model/scripts)
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the path of the parent directory (i.e., .../FF_Model)
project_root = os.path.dirname(script_dir)
# Add the project root to the Python path
sys.path.append(project_root)
from src.rookie_receiver import load_and_clean_receiver_data

def main(): 
    print('hi')
    college_wr_file = os.path.join('data', 'raw', 'official_college_receiver.csv')
    nfl_wr_file = os.path.join('data', 'raw', 'nfl_wr_data_2.csv')
    print(college_wr_file)
    print(nfl_wr_file)

    model_output_file = os.path.join('models', 'wr_pytorch_model.pth')

    wr_data = load_and_clean_receiver_data(college_wr_file, nfl_wr_file)
    # train_and_save_pytorch_model(wr_data, model_output_file)

if __name__ == '__main__':
    main()
