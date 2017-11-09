import os, argparse, subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--low',        default=0,                      type=int,   help='min image index')
parser.add_argument('--high',       default=10000,                    type=int,   help='max image index')
parser.add_argument('--repeat',     default=2,                      type=int,   help='number of renderings per object')
parser.add_argument('--division',   default=50,                     type=int,   help='number of images per SLURM job')
parser.add_argument('--qos',        default=True,                   type=bool,  help='whether to use quality-of-service queue')
parser.add_argument('--jobs',       default=None,                               help='comma-separated string of indices to render, overrides {low,high,repeat,division}')
parser.add_argument('--script',     default='run.py',               type=str,   help='script to launch blender job')
parser.add_argument('--mode',       default='render.py',            type=str,   help='script run within blender')
parser.add_argument('--category',   default='bottle',            type=str,   help='object category (from ShapeNet or primitive, see options in config.py)')
parser.add_argument('--output',     default='output/bottle_train/',    type=str,   help='save folder')
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
        
    working_dir = os.path.dirname(os.path.realpath(__file__))
    repo_folder = os.path.join(working_dir, '..')

    command = [   'sbatch', '-c', '1', \
                            '-J', str(start)+'_'+str(end), '--time=1-12:0', \
                            '--mem=5G', args.script, '--script', args.mode, '--low', str(start), '--high', str(end), \
                            '--repeat', str(args.repeat), '--category', args.category, '--output', args.output, '--include', repo_folder]
    if args.qos:
        command.insert(1, '--qos=tenenbaum')
    p = subprocess.call( command )


if args.jobs:
    args.jobs = list(args.jobs)
    use_divisions(args)
else:
    make_divisions(args)