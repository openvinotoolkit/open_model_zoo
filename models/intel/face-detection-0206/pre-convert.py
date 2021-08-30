import sys
import argparse
import subprocess # nosec - disable B404:import-subprocess check
import os
from contextlib import contextmanager
from pathlib import Path

@contextmanager
def cd(newdir):
    prevdir = os.getcwd()
    os.chdir(newdir)
    try:
        yield
    finally:
        os.chdir(prevdir)

def get_env(paths):
    for path in paths:
        sys.path.append(path)

    pyenv = ''
    for path in sys.path:
        pyenv += str(path) + os.pathsep
    env = os.environ.copy()
    env['PYTHONPATH'] = pyenv
    return env

def create_workdir(env):
    env['MODEL_TEMPLATE'] = os.path.realpath('./model_templates/face-detection/face-detection-0206/template.yaml')
    env['SNAPSHOT'] = 'snapshot.pth'
    tmp = os.path.basename(os.path.dirname(env['MODEL_TEMPLATE']))
    env['WORK_DIR'] = f'/tmp/my-{tmp}'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', type=Path)
    parser.add_argument('output_dir', type=Path)
    args = parser.parse_args()

    paths = [Path(args.input_dir / 'model' / 'lib'),
             Path(args.input_dir / 'model' / 'lib' / 'terminaltables-3.1.0'),
             Path(args.input_dir / 'model' / 'lib' / 'mmcv-full-1.3.0'),
             Path(args.input_dir / 'model' / 'training_extensions-develop' / 'external' / 'mmdetection-4856591efb5c60a1380e574a6c7f8566a9b31dfe'),
             Path(args.input_dir / 'model' / 'training_extensions-develop' / 'ote')]
    weights = Path(args.input_dir, 'model', 'ckpt', 'snapshot.pth')

    # setup.py for mmcv
    setup_dir = Path(args.input_dir / 'model' / 'lib' / 'mmcv-full-1.3.0')
    work_env = get_env(paths)
    with cd(setup_dir):
        subprocess.run([
            sys.executable, 'setup.py', 'develop'],
            check=True, env=work_env)
    
    # create work-dir tmp
    models_dir = Path(args.input_dir / 'model' / 'training_extensions-develop' / 'models' / 'object_detection')
    with cd(models_dir):
        create_workdir(work_env)
        subprocess.run([
            sys.executable, '../../tools/instantiate_template.py',
            f'{work_env["MODEL_TEMPLATE"]}', f'{work_env["WORK_DIR"]}', '--do-not-load-snapshot'],
            check=True, env=work_env)
    
    # convert to onnx
    work_dir = Path(work_env['WORK_DIR'])
    with cd(work_dir):
        subprocess.run([
            sys.executable, 'export.py',
            '--load-weights', str(weights),
            '--save-model-to', str(args.output_dir)
        ], check=True, env=work_env)


if __name__ == '__main__':
    main()
