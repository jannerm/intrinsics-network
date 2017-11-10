#!/om/user/janner/anaconda2/bin/python

import os, argparse, subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--low',        default=0,              type=int, help='min image index')
parser.add_argument('--high',       default=100,            type=int, help='max image index')
parser.add_argument('--repeat',     default=2,              type=int, help='number of renderings per object')
parser.add_argument('--category',   default='car',          type=str, help='object category (from ShapeNet or primitive, see options in config.py)')
parser.add_argument('--output',     default='output/car/',  type=str, help='save folder')
parser.add_argument('--script',     default='render.py',    type=str, help='script run within blender')
parser.add_argument('--include',    default=None,                     help='directory to include in python path')
args = parser.parse_args()

def render(script, low, high, repeat, category, output):
    if args.include is None:
        working_dir = os.path.dirname(os.path.realpath(__file__))
        repo_folder = os.path.join(working_dir, '..')
    else:
        repo_folder = args.include

    command = ['/om/user/janner/blender-2.76b-linux-glibc211-x86_64/blender', '--background', '-noaudio', '--python', script, '--', '--include', repo_folder, \
        '--start', low, '--finish', high, '--repeat', repeat, '--category', category, '--output', output] #, \

    p = subprocess.call(command)


if __name__ == '__main__':
    print args
    render(args.script, str(args.low), str(args.high), str(args.repeat), args.category, args.output)



