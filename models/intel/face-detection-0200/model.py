import sys
import argparse
import subprocess
import os

from pathlib import Path

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('input_dir', type=Path)
#     parser.add_argument('output_dir', type=Path)
#     args = parser.parse_args()
#     print(os.path.join(args.input_dir, "model", "mmcv-1.3.9"))
#     sys.path.extend(os.path.join(args.input_dir, "model", "mmcv-1.3.9"))
#     weights = os.path.join(args.input_dir, 'model', 'snapshot.pth')
#     subprocess.run(f'python3 {os.path.join(args.input_dir, "model", "export.py")} '
#                     f'--load_weights {weights} '
#                     f'--save_model_to {args.output_dir}',
#                     shell=True)


# if __name__ == '__main__':
#     main()

def get_onnx_model(work_dir, weights, onnx_path):
    print(f'{weights}')
    subprocess.run(f'python3 {os.path.join(work_dir, "export.py")} '
                    f'--load_weights {weights} '
                    f'--save_model_to {onnx_path}',
                    shell=True)
