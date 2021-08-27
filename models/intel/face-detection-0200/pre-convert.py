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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', type=Path)
    parser.add_argument('output_dir', type=Path)
    args = parser.parse_args()

    paths = [Path(args.input_dir / 'model'),
             Path(args.input_dir / 'model' / 'lib'),
             Path(args.input_dir / 'model' / 'lib' / 'terminaltables-3.1.0'),
             Path(args.input_dir / 'model' / 'lib' / 'mmcv-full-1.3.0'),
             Path(args.input_dir / 'model' / 'lib' / 'mmdetection-4856591efb5c60a1380e574a6c7f8566a9b31dfe')]
    weights = Path(args.input_dir, 'model', 'ckpt', 'snapshot.pth')

    # setup.py for mmcv
    work_dir = Path(args.input_dir / 'model' / 'lib' / 'mmcv-full-1.3.0')
    work_env = get_env(paths)
    with cd(work_dir):
        subprocess.run([
            sys.executable, 'setup.py', 'develop'],
            check=True, env=work_env)

    work_dir = Path(args.input_dir / 'model')
    with cd(work_dir):
        subprocess.run([
            sys.executable, 'export.py',
            '--load-weights', str(weights),
            '--save-model-to', str(args.output_dir)
        ], check=True, env=work_env)


if __name__ == '__main__':
    main()
