## Project Details
In this project an agent is trained to navigate (and collect bananas!) in a large, square world.

![environment](docs/banana.gif "Environment")

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. Thus, the goal of the agent is to collect as many yellow bananas as possible while avoiding blue bananas.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction. Given this information, the agent learns how to best select actions. Four discrete actions are available, corresponding to:

* 0 - move forward.
* 1 - move backward.
* 2 - turn left.
* 3 - turn right.

The task is episodic, and in order to solve the environment, the agent must get an average score of +13 over 100 consecutive episodes.

## Getting Started
This project requires Python 3.6+, pytorch, torchvision, matplotlib and numpy to work.

For this project, you will need to download the environment from one of the links below. You need only select the environment that matches your operating system:
* Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
* Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
* Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
* Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

Then, place the file in the root folder, and unzip (or decompress) the file.

## Instructions
To run the project navigate in your terminal to the root folder and execute `python3.6 -m navigation.main --environment path_to_your_environment`. For example, if you use Linux (64 bits) and placed the environment in the root folder following the instructions, the concrete instruction would be `python3.6 -m navigation.main --environment Banana_Linux/Banana.x86_64`. To use a trained model, include the option `--trained`: `python3.6 -m navigation.main --trained --environment Banana_Linux/Banana.x86_64`