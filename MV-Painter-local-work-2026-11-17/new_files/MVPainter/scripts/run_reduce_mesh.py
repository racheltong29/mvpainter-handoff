import os
import sys
import argparse

if "--" in sys.argv:
    sys.argv = [sys.argv[0]] + sys.argv[sys.argv.index("--") + 1:]

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.append(_THIS_DIR)

from remesh_reduce_blender_script import reduce_mesh

parser = argparse.ArgumentParser()
parser.add_argument("--input_obj", required=True)
parser.add_argument("--output_obj", required=True)
parser.add_argument("--target_vertices", type=int, default=30000)
args = parser.parse_args()

reduce_mesh(args.input_obj, target_vertices=args.target_vertices, output_path=args.output_obj)
