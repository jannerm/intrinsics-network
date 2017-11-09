#!/om/user/janner/anaconda2/bin/python

import os, argparse, subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--low', default=0)
parser.add_argument('--high', default=10)
parser.add_argument('--repeat', default=1)
parser.add_argument('--category', default='car')
parser.add_argument('--output', default='output/car/')
parser.add_argument('--script', default='render.py')

args = parser.parse_args()

def render(script, low, high, repeat, category, output):
    working_dir = os.path.dirname(os.path.realpath(__file__))
    repo_folder = os.path.join(working_dir, '..')

    command = ['/om/user/janner/blender-2.76b-linux-glibc211-x86_64/blender', '--background', '-noaudio', '--python', script, '--', '--include', repo_folder, \
        '--start', low, '--finish', high, '--repeat', repeat, '--category', category, '--output', output] #, \

    p = subprocess.call(command)


if __name__ == '__main__':
    print args
    render(args.script, str(args.low), str(args.high), str(args.repeat), args.category, args.output)



