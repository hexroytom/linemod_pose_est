//ros
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/CameraInfo.h>
#include <geometry_msgs/PoseStamped.h>

//#include <cv_bridge/cv_bridge.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>

//Eigen
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>
#include <eigen3/Eigen/Core>

//opencv
#include <opencv2/core/core.hpp>
#include <opencv2/rgbd/rgbd.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

//tf
#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>
#include <tf_conversions/tf_eigen.h>

//ork
#include <object_recognition_renderer/renderer3d.h>
#include <object_recognition_renderer/utils.h>

//pcl
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/io/ply_io.h>
#include <pcl/common/transforms.h>

// Functions to store detector and templates in single XML/YAML file
static cv::Ptr<cv::linemod::Detector> readLinemod(const std::string& filename)
{
  //cv::Ptr<cv::linemod::Detector> detector = cv::makePtr<cv::linemod::Detector>();
  cv::Ptr<cv::linemod::Detector> detector(new cv::linemod::Detector);
  cv::FileStorage fs(filename, cv::FileStorage::READ);
  detector->read(fs.root());

  cv::FileNode fn = fs["classes"];
  for (cv::FileNodeIterator i = fn.begin(), iend = fn.end(); i != iend; ++i)
    detector->readClass(*i);

  return detector;
}

static void writeLinemod(const cv::Ptr<cv::linemod::Detector>& detector, const std::string& filename)
{
  cv::FileStorage fs(filename, cv::FileStorage::WRITE);
  detector->write(fs);

  std::vector<cv::String> ids = detector->classIds();
  fs << "classes" << "[";
  for (int i = 0; i < (int)ids.size(); ++i)
  {
    fs << "{";
    detector->writeClass(ids[i], fs);
    fs << "}"; // current class
  }
  fs << "]"; // classes
}

static void writeLinemodTemplateParams(std::string fileName,
                                       std::vector<cv::Mat>& Rs,
                                       std::vector<cv::Mat>& Ts,
                                       std::vector<double>& distances,
                                       std::vector<double>& obj_origin_dists,
                                       std::vector<cv::Mat>& Ks,
                                       std::vector<cv::Rect>& Rects,
                                       int& renderer_n_points,
                                       int& renderer_angle_step,
                                       double& renderer_radius_min,
                                       double& renderer_radius_max,
                                       double& renderer_radius_step,
                                       int& renderer_width,
                                       int& renderer_height,
                                       double& renderer_focal_length_x,
                                       double& renderer_focal_length_y,
                                       double& renderer_near,
                                       double& renderer_far)
{
    cv::FileStorage fs(fileName,cv::FileStorage::WRITE);
    int num_templates=Rs.size ();
    for(int i=0;i<num_templates;++i)
    {
        std::stringstream ss;
        std::string a;
        ss<<i;
        a="Template ";
        a+=ss.str ();
        fs<<a<<"{";
        fs<<"ID"<<i;
        fs<<"R"<<Rs[i];
        fs<<"T"<<Ts[i];
        fs<<"K"<<Ks[i];
        fs<<"D"<<distances[i];
        fs<<"Ori_dist"<<obj_origin_dists[i];
        fs<<"Rect"<<Rects[i];
        fs<<"}";
    }
    //fs<<"K Intrinsic Matrix"<<cv::Mat(K_matrix);
    fs<<"renderer_n_points"<<renderer_n_points;
    fs<<"renderer_angle_step"<<renderer_angle_step;
    fs<<"renderer_radius_min"<<renderer_radius_min;
    fs<<"renderer_radius_max"<<renderer_radius_max;
    fs<<"renderer_radius_step"<<renderer_radius_step;
    fs<<"renderer_width"<<renderer_width;
    fs<<"renderer_height"<<renderer_height;
    fs<<"renderer_focal_length_x"<<renderer_focal_length_x;
    fs<<"renderer_focal_length_y"<<renderer_focal_length_y;
    fs<<"renderer_near"<<renderer_near;
    fs<<"renderer_far"<<renderer_far;
    fs.release ();
}
static void writeLinemodRender(std::string& fileName,
                                       std::vector<cv::Mat>& depth_img,
                                       std::vector<cv::Mat>& mask,
                                       std::vector<cv::Rect>& rect)
{
    cv::FileStorage fs(fileName,cv::FileStorage::WRITE);
    int num_templates=depth_img.size ();
    for(int i=0;i<num_templates;++i)
    {
        std::stringstream ss;
        std::string a;
        ss<<i;
        a="Template ";
        a+=ss.str ();
        fs<<a<<"{";
        fs<<"ID"<<i;
        fs<<"Depth"<<depth_img[i];
        fs<<"Mask"<<mask[i];
        fs<<"Rect"<<rect[i];
        fs<<"}";
    }
    fs.release ();
}

void depth_to_3d(double renderer_focal_length_x, double renderer_focal_length_y, int U0, int V0, cv::Mat depth, pcl::PointCloud<pcl::PointXYZ>::Ptr pc)
{
  for(int i=0;i<depth.rows;++i)
  {
    for(int j=0;j<depth.cols;++j)
    {
      if(depth.at<ushort>(i,j)>0)
        {
        double z = depth.at<ushort>(i,j);
        double x = z*(j-U0)/renderer_focal_length_x;
        double y = z*(i-V0)/renderer_focal_length_y;

        z /= 1000;
        x /= 1000;
        y /= 1000;

        pc->points.push_back(pcl::PointXYZ(x,y,z));
      }
    }
  }
}

int main(int argc,char** argv)
{
    //Ros
    ros::init(argc,argv,"renderer_node");
    ros::NodeHandle nh;
    ros::Publisher pointcloud_pub = nh.advertise<sensor_msgs::PointCloud2>("obj25d_pointcloud_viz",10);
    ros::Publisher pointcloud_pub2 = nh.advertise<sensor_msgs::PointCloud2>("obj3d_pointcloud_viz",10);
    ros::Publisher pose_pub = nh.advertise<geometry_msgs::PoseStamped>("object_pose",10);

    std::vector< cv::Ptr<cv::linemod::Modality> > modalities;
    //cv::Ptr<cv::linemod::ColorGradient> color_grad(new cv::linemod::ColorGradient);
    //cv::Ptr<cv::linemod::DepthNormal> depth_grad(new cv::linemod::DepthNormal);
    modalities.push_back(cv::Ptr<cv::linemod::ColorGradient>(new cv::linemod::ColorGradient));
    modalities.push_back(cv::Ptr<cv::linemod::DepthNormal>(new cv::linemod::DepthNormal));
    std::vector<int> ensenso_T;
    ensenso_T.push_back(5);
    ensenso_T.push_back(8);
    cv::Ptr<cv::linemod::Detector> detector_(new cv::linemod::Detector(modalities,ensenso_T));

    int renderer_n_points_;
    int renderer_angle_step_;
    double renderer_radius_min_;
    double renderer_radius_max_;
    double renderer_radius_step_;
    int renderer_width_;
    int renderer_height_;
    double renderer_near_;
    double renderer_far_ ;
    double renderer_focal_length_x_;//Kinect ;//xtion 570.342;
    double renderer_focal_length_y_;//Kinect //xtion 570.342;
    std::string stl_file;
    std::string template_output_path;
    std::string renderer_params_output_path;
    std::string renderer_depth_output_path;
    std::string ply_file_path;

    if(argc<10)
        {
        renderer_n_points_ = 150;
        renderer_angle_step_ = 10;
        renderer_radius_min_ = 0.7;
        renderer_radius_max_ = 0.8;
        renderer_radius_step_ = 0.1;
        renderer_width_ = 640; //ensenso 752
        renderer_height_ = 480;
        renderer_near_ = 0.1;
        renderer_far_ = 1000.0;
        renderer_focal_length_x_ = 535.566011;//Kinect ;//carmine 535.566011; //dataset 571.9737
        renderer_focal_length_y_ = 537.168115;//Kinect //carmine 537.168115;  //dataset 571.0073
        stl_file="/home/yake/catkin_ws/src/linemod_pose_est/config/stl/triangle_board.STL";
        template_output_path="/home/yake/catkin_ws/src/linemod_pose_est/config/data/triangle_board_linemod_carmine_templates.yml";
        renderer_params_output_path="/home/yake/catkin_ws/src/linemod_pose_est/config/data/triangle_board_linemod_carmine_renderer_params.yml";
        ply_file_path="/home/yake/catkin_ws/src/linemod_pose_est/config/stl/triangle_board.ply";

    }else{
     renderer_n_points_ = 150;
     renderer_angle_step_ = 10;
     renderer_radius_min_ = atof(argv[8]);
     renderer_radius_max_ = atof(argv[9]);
     renderer_radius_step_ = atof(argv[10]);
     renderer_width_ = atoi(argv[3]);
     renderer_height_ = atoi(argv[4]);
     renderer_near_ = 0.1;
     renderer_far_ = 1000.0;
     renderer_focal_length_x_ = atof(argv[1]);//Kinect ;//xtion 570.342;
     renderer_focal_length_y_ = atof(argv[2]);//Kinect //xtion 570.342;
     stl_file=argv[5];
     template_output_path=argv[6];
     renderer_params_output_path=argv[7];
    }

    Renderer3d render=Renderer3d(stl_file);
    render.set_parameters (renderer_width_, renderer_height_, renderer_focal_length_x_,
                           renderer_focal_length_y_, renderer_near_, renderer_far_);
    RendererIterator renderer_iterator=RendererIterator(&render,renderer_n_points_);
    renderer_iterator.angle_step_ = renderer_angle_step_;
    renderer_iterator.radius_min_ = float(renderer_radius_min_);
    renderer_iterator.radius_max_ = float(renderer_radius_max_);
    renderer_iterator.radius_step_ = float(renderer_radius_step_);

    cv::Mat image, depth, mask,flip_mask,flip_depth,flip_image;
    cv::Matx33d R;
    cv::Matx33d R_cam;
    cv::Vec3d T;
    cv::Matx33f K;
    std::vector<cv::Mat> Rs_;
    std::vector<cv::Mat> Ts_;
    std::vector<double> distances_;
    std::vector<double> Origin_dists_;
    std::vector<cv::Mat> Ks_;
    std::vector<cv::Mat> depth_img;
    std::vector<cv::Mat> masks;
    std::vector<cv::Rect> rects;

    for (size_t i = 0; !renderer_iterator.isDone(); ++i, ++renderer_iterator)
    {
      std::stringstream status;
      status << "Loading images " << (i+1) << "/"
          << renderer_iterator.n_templates();
      std::cout << status.str();

      cv::Rect rect;
      bool is_restricted=true;
      bool is_image_valid;
      renderer_iterator.render(image, depth, mask, rect,is_restricted,is_image_valid);
      cv::flip(mask,flip_mask,0);
      cv::flip(depth,flip_depth,0);
      cv::flip(image,flip_image,0);
      if(is_image_valid)
      {
          R = renderer_iterator.R_obj();   //obj pose
          T = renderer_iterator.T();
          R_cam = renderer_iterator.R_cam();

          //D_obj distance from camera to object origin
          double distance = renderer_iterator.D_obj() - double(depth.at<ushort>(depth.rows/2.0f, depth.cols/2.0f)/1000.0f);
          double obj_origin_dist=renderer_iterator.D_obj();
          K = cv::Matx33f(float(renderer_focal_length_x_), 0.0f, float(depth.cols)/2.0f, 0.0f, float(renderer_focal_length_y_), float(depth.rows)/2.0f, 0.0f, 0.0f, 1.0f);

          std::vector<cv::Mat> sources(2);
          sources[0] = image;
          sources[1] = depth;

          // Display the rendered image
          if (true)
          {
              cv::namedWindow("Rendering RGB");
              cv::namedWindow("Rendering Depth");
              cv::namedWindow("Rendering Mask");
              if (!image.empty()) {
                  cv::imshow("Rendering RGB", image);
                  cv::imshow("Rendering Depth", depth);
                  cv::imshow("Rendering Mask", mask);
                  cv::imshow("Rendering Flip Mask", flip_mask);
                  cv::imshow("Rendering Flip RGB", flip_image);

              }
          }

          cv::Mat pc_cv;
          pcl::PointCloud<pcl::PointXYZ>::Ptr model_pc(new pcl::PointCloud<pcl::PointXYZ>);
          cv::Mat K_matrix=(cv::Mat_<double>(3,3)<<renderer_focal_length_x_,0.0,depth.cols/2,
                                        0.0,renderer_focal_length_y_,depth.rows/2,
                                        0.0,0.0,1.0);
          Eigen::Affine3d obj_pose,obj_pose2;
          obj_pose = Eigen::Affine3d::Identity();
          //cv orientation ---> eigen orientation
          Eigen::Matrix<double,3,3> R_eig;
          cv::cv2eigen(cv::Mat(R),R_eig);

          //!!!Attention: modify object's coordinate system. Object Specific!
//          Eigen::Vector3d rot_axis(0.0,0.0,1.0);
//          obj_pose.linear()=R_eig*Eigen::AngleAxisd(3.14,rot_axis);
//          obj_pose.translation()<<0.0,0.0,renderer_iterator.D_obj();
          cv::Vec3d t_obj_pCamera = (-1)*(R_cam.t())*(-1)*(T);
          cv::Matx33d R_obj_pCamera = R_cam.t();
          Eigen::Matrix<double,3,3> R_obj_pCamera_eig;
          cv::cv2eigen(cv::Mat(R_obj_pCamera),R_obj_pCamera_eig);
          obj_pose.linear() = R_eig;
          obj_pose.translation()<<0, 0, -renderer_iterator.D_obj();
          obj_pose2.linear() = R_obj_pCamera_eig;
          obj_pose2.translation()<<t_obj_pCamera(0), t_obj_pCamera(1), t_obj_pCamera(2);
          Eigen::Affine3d cam_pose = obj_pose.inverse();
          Eigen::Affine3d cam_pose2 = obj_pose2.inverse();

          model_pc->header.frame_id="/obj_link";

          cv::depthTo3d(flip_depth,K_matrix,pc_cv);    //mm ---> m
          for(int ii=0;ii<pc_cv.rows;++ii)
          {
              double* row_ptr=pc_cv.ptr<double>(ii);
              for(int jj=0;jj<pc_cv.cols;++jj)
              {
                  if(flip_mask.at<uchar>(ii,jj)>0)
                  {
                      double* data =row_ptr+jj*3;
//                      //Desire result
//                      double desire_x,desire_y,desire_z;
//                      desire_x=data[2]*(jj-320)/renderer_focal_length_x_;
//                      desire_y=data[2]*(ii-240)/renderer_focal_length_y_;
//                      desire_z=data[2];
//                      cout<<"Desire coordinates: "<<desire_x<<" "<<desire_y<<" "<<desire_z<<endl;
//                      cout<<"Actual coordinates: "<<data[0]<<" "<<data[1]<<" "<<data[2]<<endl;
                      model_pc->points.push_back(pcl::PointXYZ(data[0],data[1],data[2]));
                  }
              }
          }
          model_pc->height=1;
          model_pc->width=model_pc->points.size();
          pcl::transformPointCloud(*model_pc,*model_pc,cam_pose2.cast<float>());

          tf::TransformBroadcaster tf_broadcaster;
          tf::Transform pose_tf;
          tf::poseEigenToTF(cam_pose2,pose_tf);
          //tf_broadcaster.sendTransform (tf::StampedTransform(pose_tf,ros::Time::now(),"camera_link","object_frame"));

          geometry_msgs::Pose pose_msg;
          tf::poseTFToMsg(pose_tf,pose_msg);
          geometry_msgs::PoseStamped pose_stamped_msg;
          pose_stamped_msg.pose=pose_msg;
          pose_stamped_msg.header.frame_id="/obj_link";
          pose_stamped_msg.header.stamp=ros::Time::now();
          pose_pub.publish(pose_stamped_msg);

          sensor_msgs::PointCloud2 ros_pointcloud;
          ros_pointcloud.header.stamp = ros::Time::now();
          ros_pointcloud.header.frame_id = "/obj_link";
          pcl::toROSMsg(*model_pc,ros_pointcloud);
          pointcloud_pub.publish(ros_pointcloud);

          //Full obj pointcloud
          sensor_msgs::PointCloud2 ros_pointcloud2;
          pcl::PointCloud<pcl::PointXYZ>::Ptr model_pc2(new pcl::PointCloud<pcl::PointXYZ>);
          pcl::io::loadPLYFile(ply_file_path,*model_pc2);
          pcl::toROSMsg(*model_pc2,ros_pointcloud2);
          ros_pointcloud2.header.stamp = ros::Time::now();
          ros_pointcloud2.header.frame_id = "/obj_link";
          pointcloud_pub2.publish(ros_pointcloud2);
          cv::waitKey(0);

          int template_in = detector_->addTemplate(sources, "obj", mask);
          if (template_in == -1)
          {
              // Delete the status
              for (size_t j = 0; j < status.str().size(); ++j)
                  std::cout << '\b';
              continue;
          }

          // Also store the pose of each template
          Rs_.push_back(cv::Mat(R));
          Ts_.push_back(cv::Mat(T));
          distances_.push_back(distance);
          Ks_.push_back(cv::Mat(K));
          Origin_dists_.push_back (obj_origin_dist);
          rects.push_back(rect);

          //Store depth image, mask and rect
          //      depth_img.push_back(depth);
          //      masks.push_back(mask);
          //      rects.push_back(rect);

          // Delete the status
          for (size_t j = 0; j < status.str().size(); ++j)
              std::cout << '\b';
      }
    }

    writeLinemod (detector_,template_output_path);
    writeLinemodTemplateParams (renderer_params_output_path,
                                Rs_,
                                Ts_,
                                distances_,
                                Origin_dists_,
                                Ks_,
                                rects,
                                renderer_n_points_,
                                renderer_angle_step_,
                                renderer_radius_min_,
                                renderer_radius_max_,
                                renderer_radius_step_,
                                renderer_width_,
                                renderer_height_,
                                renderer_focal_length_x_,
                                renderer_focal_length_y_,
                                renderer_near_,
                                renderer_far_);
    //writeLinemodRender(renderer_depth_output_path,depth_img,masks,rects);

      return 0;

}
