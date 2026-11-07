"""Run the applications through `python -m snake_v9 COMMAND`."""

import argparse
import importlib
import sys


def main():
    parser = argparse.ArgumentParser(description='SnakeAI v9')
    parser.add_argument('command', choices=['play', 'evaluate', 'train', 'dashboard', 'diagnose_losses'])
    args = parser.parse_args(sys.argv[1:2])
    sys.argv = [f'snake-{args.command}', *sys.argv[2:]]
    importlib.import_module(f'snake_v9.cli.{args.command}').main()


if __name__ == '__main__':
    main()
