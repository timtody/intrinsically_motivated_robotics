from pyrep import PyRep
from pyrep.robots.arms.panda import Panda
from pyrep.robots.end_effectors.panda_gripper import PandaGripper

pr = PyRep()
# Launch the application with a scene file that contains a robot
pr.launch('scenes/test.ttt', headless=True)
pr.start()  # Start the simulation

arm = Panda()  # Get the panda from the scene
#gripper = PandaGripper()  # Get the panda gripper from the scene

#velocities = [.1, .2, .3, .4, .5, .6, .7]
#arm.set_joint_target_velocities(velocities)
pr.step()  # Step physics simulation

done = False
# Open the gripper halfway at a velocity of 0.04.
while not done:
    # done = gripper.actuate(0.5, velocity=0.04)
    pr.step()
    
pr.stop()  # Stop the simulation
pr.shutdown()  # Close the application