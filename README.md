# linemod_pose_est
Dependency

  OpenCV3
  cv_bridge3
  object_recognition_core
  ork_linemod
  ork_renderer

How to install the dependencies?

  OpenCV3
    use command: sudo apt-get install ros-indigo-opencv3
  cv_bridge3
    catkin_make package from soucre. Please use the source code provided in the birl_vision/linemod folder
  object_recognition_core
    sudo apt-get install ros-indigo-object-recognition-core 
  ork_linemod
    catkin_make package from soucre. Please use the source code provided in the birl_vision/linemod folder
  ork_renderer
    catkin_make package from soucre. Please use the source code provided in the birl_vision/linemod folder

How to use?

  Tranining 
    Open linemod_renderer.launch: 
        1 edit camera_focal_length_x and camera_focal_length_y according to your camera intrisic parameters
        2 enter the path of the stl file of the object you want to detect. The stl file is used to generated multi-view templates during the training. Pay attension that the coordinate of the object shoule be placed in the geometry center. You can use blender to open the stl file, check if the the coordinate is in the center and if not, modify it in Blender. Here we can use coke can object: set model_STL_file_path=$(find linemod_pose_estimation)/config/stl/coke/stl.
        3 enter the ouput path of linemod_template_output_path and linemod_renderer_params_output_path. These two files are used to load the LINEMOD detector and recover detected object pose.
    
    run command: roslaunch linemod_pose_estimation linemod_renderer.launch

  Detection
    Open linemod_detect.launch: edit linemod_template_input_path, linemod_renderer_params_input_path and model_STL_file_path according to your linemod_renderer.launch.
    Plug in xtion camera
    run command: roslaunch linemod_detect.launch
