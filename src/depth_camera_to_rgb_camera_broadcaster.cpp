//ros
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <sensor_msgs/CameraInfo.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>
#include <tf_conversions/tf_eigen.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <message_filters/synchronizer.h>
#include <message_filters/subscriber.h>

//std
#include <iostream>

//Eigen
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>
#include <eigen3/Eigen/Core>

int main(int argc,char** argv)
{
    ros::init(argc,argv,"depth_to_rgb_broadcaster");
    ros::NodeHandle nh;

    double x = atof(argv[1]);
    double y = atof(argv[2]);
    double z = atof(argv[3]);
    double roll = atof(argv[4]);
    double pitch = atof(argv[5]);
    double yaw = atof(argv[6]);

    ros::Duration sleeper(atof(argv[7])/1000.0);

    Eigen::Affine3d pose_rgbTdep_;
    Eigen::Affine3d pose_depTrgb_;
    //Translation
    pose_rgbTdep_.translation()<< x/1000.0,y/1000.0,z/1000.0;
    //Rotation
        //Z-Y-X euler ---> yaw-pitch-roll
    Eigen::Matrix3d rotation_matrix;
    rotation_matrix = Eigen::AngleAxisd(yaw/180.0*M_PI,Eigen::Vector3d(0.0,0.0,1.0)) *
                      Eigen::AngleAxisd(pitch/180.0*M_PI,Eigen::Vector3d(0.0,1.0,0.0)) *
                      Eigen::AngleAxisd(roll/180.0*M_PI,Eigen::Vector3d(1.0,0.0,0.0));
    pose_rgbTdep_.linear()=rotation_matrix;
    //Inverse
    pose_depTrgb_ = pose_rgbTdep_.inverse();


    tf::Transform pose_tf;
    tf::poseEigenToTF(pose_depTrgb_,pose_tf);
    tf::TransformBroadcaster tf_broadcaster;

    while(ros::ok())
    {
        tf_broadcaster.sendTransform (tf::StampedTransform(pose_tf,ros::Time::now(),"camera_link","rgb_camera_link"));
        sleeper.sleep();
    }

    return 0;

}
