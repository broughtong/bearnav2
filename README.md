# Bearnav2

## Overview

This ROS package allows a robot to be taught a path through an environment using a camera.
The robot can then retrace this path at a future point in time, correcting any error.
The theory can be found in the [linked paper.](http://eprints.lincoln.ac.uk/12501/7/surfnav_2010_JFR.pdf)

## Installation

Clone the repository into a ROS workspace and build.


## Usage

Once built, to run the system use `roslaunch bearnav2 bearnav2-gui.launch`.
You can optionally run `roslaunch bearnav2 bearnav2-no-gui.launch` if these aren't required.

Don't forget to source your workspace!

Once the package is running, you can begin mapping by publishing a message to the mapmaker module:

`rostopic pub /bearnav2/mapmaker/goal [tab][tab]`

Make sure you have the mapname (filename) set, for loading the map later!
Publish the message with start set to `True` to begin, and then publish another with `False` when you have finished your path.

To replay a map, run:

`rostopic pub /bearnav2/repeater/goal [tab][tab]`

Simply fill in the mapname field and your robot will begin to re-trace the path.
