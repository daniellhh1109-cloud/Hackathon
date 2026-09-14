"""Entry point for the guide pipeline. Implementation has not started."""
import argparse

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/base.yaml', help='Planned pipeline configuration')
    parser.parse_args()
    parser.exit(2, 'Pipeline not implemented yet. See README.md for the development order.\n')

if __name__ == '__main__':
    main()
