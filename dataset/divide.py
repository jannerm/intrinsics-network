import argparse, subprocess
import run

parser = argparse.ArgumentParser()
parser.add_argument('--low', default=0)
parser.add_argument('--high', default=5000)
parser.add_argument('--repeat', default=2)
parser.add_argument('--division', default=50)
parser.add_argument('--qos', default=True)
parser.add_argument('--jobs', default=None)
parser.add_argument('--script', default='run.py')
parser.add_argument('--mode', default='render.py')
parser.add_argument('--category', default='motorbike')
parser.add_argument('--output', default='output/motorbike_right/')
args = parser.parse_args()


def make_divisions(args):
    for start in range(args.low, args.high, args.division):
        end = start + args.division
        launch(args, start, end)


def use_divisions(args):
    for (start, end) in args.jobs:
        launch(args, start, end)

def launch(args, start, end):
    print 'Submitting: ', start, end
    command = [   'sbatch', '-c', '1', \
                            '-J', str(start)+'_'+str(end), '--time=1-12:0', \
                            '--mem=5G', args.script, '--script', args.mode, '--low', str(start), '--high', str(end), '--repeat', str(args.repeat), '--category', args.category, '--output', args.output]
    if args.qos:
        command.insert(1, '--qos=tenenbaum')
    p = subprocess.call( command )


if args.jobs:
    args.jobs = list(args.jobs)
    use_divisions(args)
else:
    make_divisions(args)