#include <iostream>
#include <opencv2/opencv.hpp>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>

using namespace std;
using namespace cv;

int main()
{
  string rgb_path="/home/yake/catkin_ws/src/ensenso/pcd/1482665846_rgb.jpg";
  string pc_path="/home/yake/catkin_ws/src/ensenso/pcd/1482665846_pc.pcd";

  Mat rgb_img=imread(rgb_path,IMREAD_COLOR);

  pcl::PointCloud<pcl::PointXYZ>::Ptr pc(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::io::loadPCDFile(pc_path,*pc);
  Mat depth_img(pc->height,pc->width,CV_64FC1);
  for(int i=0;i<depth_img.rows;++i)
  {
    for(int j=0;j<depth_img.cols;++j)
    {
      depth_img.at<double>(i,j) = pc->at(j,i).z;
    }
  }

  imshow("RGB",rgb_img);
  imshow("Depth",depth_img);
  cv::waitKey(0);
}
