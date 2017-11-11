## if obj.attr is a string of the form '[0,0,0]',
## converts it to a list ([0,0,0])
## (modifies the object)
def parse_attribute(obj, attr):
    param = getattr(obj, attr)
    if type(param) == str:
        setattr(obj, attr, eval(param))

## call parse_attribute on many attributes
## for one object
## (modifies the object)
def parse_attributes(obj, *args):
    for attr in args:
        parse_attribute(obj, attr)

render_parameters = {
    'chair':        {'scale_low': 2.0, 'scale_high': 4.5, 'pos_low': [0, 0, 0], 'pos_high': [0, 0, 0], 'theta_low': [60, -45, 0], 'theta_high': [120, 45, 360]},
    'sofa':         {'scale_low': 1.5, 'scale_high': 4.0, 'pos_low': [0, 0, 0], 'pos_high': [0, 0, 0], 'theta_low': [60, -45, 0], 'theta_high': [120, 45, 360]},
    'tower':        {'scale_low': 2.5, 'scale_high': 3.5, 'pos_low': [0, 0, 0], 'pos_high': [0, 0, 0], 'theta_low': [60, -45, 0], 'theta_high': [120, 45, 360]},
    'bottle':       {'scale_low': 2.5, 'scale_high': 3.5, 'pos_low': [0, 0, 0], 'pos_high': [0, 0, 0], 'theta_low': [60, -45, 0], 'theta_high': [120, 45, 360]},
    'guitar':       {'scale_low': 0.5, 'scale_high': .75, 'pos_low': [0, 0, 0], 'pos_high': [0, 0, 0], 'theta_low': [60, -45, 0], 'theta_high': [120, 45, 360]},
    'airplane':     {'scale_low': 6.0, 'scale_high': 9.0, 'pos_low': [0, 0, 0], 'pos_high': [0, 0, 0], 'theta_low': [60, -45, 0], 'theta_high': [120, 45, 360]},
    'car':          {'scale_low': 6.0, 'scale_high': 9.0, 'pos_low': [0, 0, 0], 'pos_high': [0, 0, 0], 'theta_low': [60, -45, 0], 'theta_high': [120, 45, 360]},
    'boat':         {'scale_low': 6.0, 'scale_high': 9.0, 'pos_low': [0, 0, 0], 'pos_high': [0, 0, 0], 'theta_low': [60, -45, 0], 'theta_high': [120, 45, 360]},
    'motorbike':    {'scale_low': 6.0, 'scale_high': 9.0, 'pos_low': [0, 0, 0], 'pos_high': [0, 0, 0], 'theta_low': [60, -45, 0], 'theta_high': [120, 45, 360]},
    'flowerpot':    {'scale_low': 6.0, 'scale_high': 9.0, 'pos_low': [0, 0, 0], 'pos_high': [0, 0, 0], 'theta_low': [60, -45, 0], 'theta_high': [120, 45, 360]},
    'piano':        {'scale_low': 6.0, 'scale_high': 9.0, 'pos_low': [0, 0, 0], 'pos_high': [0, 0, 0], 'theta_low': [60, -45, 0], 'theta_high': [120, 45, 360]},
    'train':        {'scale_low': 6.0, 'scale_high': 9.0, 'pos_low': [0, 0, 0], 'pos_high': [0, 0, 0], 'theta_low': [60, -45, 0], 'theta_high': [120, 45, 360]},
    'cube':         {'scale_low': 3.0, 'scale_high': 5.0, 'pos_low': [-2,0,-2], 'pos_high': [2, 0, 2], 'theta_low': [60, -45, 0], 'theta_high': [120, 45, 360]},
    'sphere':       {'scale_low': 3.0, 'scale_high': 5.0, 'pos_low': [-2,0,-2], 'pos_high': [2, 0, 2], 'theta_low': [60, -45, 0], 'theta_high': [120, 45, 360]},
    'cylinder':     {'scale_low': 3.0, 'scale_high': 5.0, 'pos_low': [-2,0,-2], 'pos_high': [2, 0, 2], 'theta_low': [60, -45, 0], 'theta_high': [120, 45, 360]},
    'cone':         {'scale_low': 3.0, 'scale_high': 5.0, 'pos_low': [-2,0,-2], 'pos_high': [2, 0, 2], 'theta_low': [60, -45, 0], 'theta_high': [120, 45, 360]},
    'torus':        {'scale_low': 3.0, 'scale_high': 5.0, 'pos_low': [-2,0,-2], 'pos_high': [2, 0, 2], 'theta_low': [60, -45, 0], 'theta_high': [120, 45, 360]},
    'suzanne':      {'scale_low': 7.0, 'scale_high': 7.5, 'pos_low': [0, 0, 0], 'pos_high': [0, 0, 0], 'theta_low': [-30, 0,-65], 'theta_high': [-15,  0,  65]},
    'bunny':        {'scale_low': 4.0, 'scale_high': 6.0, 'pos_low': [0, 0,-2], 'pos_high': [0, 0,-2], 'theta_low': [0,   0,-65], 'theta_high': [0,    0,  65]},
    'teapot':       {'scale_low': 5.0, 'scale_high': 7.5, 'pos_low': [0, 0,-2], 'pos_high': [0, 0,-2], 'theta_low': [0,   0,-65], 'theta_high': [0,    0,  65]}
}
