"""Legacy import/CLI alias. Canonical entrypoint is scripts.run."""
from scripts.run import SyntheticDataset, SmokeRegressor, smoke_components, main

if __name__ == '__main__':
    main()
