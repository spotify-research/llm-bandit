# llm-bandit

Companion code for the paper:

> Nicolò Felicioni, Lucas Maystre, Sina Ghiassian, Kamil Ciosek.
> [On the Importance of Uncertainty in Decision-Making with Large Language Models
](#). TMLR.

This repository contains a reference implementation of the algorithms presented in the paper.

The paper investigates the role of uncertainty in decision-making problems with natural language as input. It focuses on the contextual bandit framework where the context information consists of text.
The paper compares the greedy policy to LLM bandits that make active use of uncertainty estimation by integrating the uncertainty in a Thompson Sampling policy, employing different techniques for uncertainty estimation, such as Last layer Laplace Approximation, Diagonal Laplace Approximation, Dropout, and Epinets.
<!-- For an accessible overview of the main idea, you can read our [blog
post](#).-->

## Getting Started

To get started, follow these steps:

- Clone the repo locally with: `git clone llm-bandit`
- Move to the repository: `cd llm-bandit`
- Install the dependencies: `pip install -r requirements.txt`
- Create the datasets: `python prepare_data.py --data DATASET_NAME`, where DATASET_NAME can be "hate", "imdb", "toxic", or "offensive".
- Edit the configuration file called `bandit_config.py`
- Run the main script: `python main.py --ts TS_VARIANT`, where TS_VARIANT can be "last_la", "la", "dropout", or "epinet". 

Our codebase was tested with Python 3.10 and Cuda 11.8.

## Support

Create a [new issue](https://github.com/spotify-research/llm-bandit/issues/new)

## Contributing

We feel that a welcoming community is important and we ask that you follow Spotify's
[Open Source Code of Conduct](https://github.com/spotify/code-of-conduct/blob/main/code-of-conduct.md)
in all interactions with the community.

## Authors

- [Nicolò Felicioni](mailto:nicolo.felicioni@polimi.com)
- [Lucas Maystre](mailto:lucas@maystre.ch)
- [Sina Ghiassian](mailto:sinag@spotify.com)
- [Kamil Ciosek](mailto:kamilc@spotify.com)

Follow [@SpotifyResearch](https://twitter.com/SpotifyResearch) on Twitter for updates.

## License

Copyright 2024 Spotify, Inc.

Licensed under the Apache License, Version 2.0: https://www.apache.org/licenses/LICENSE-2.0

## Security Issues?

Please report sensitive security issues via Spotify's bug-bounty program (https://hackerone.com/spotify) rather than GitHub.
