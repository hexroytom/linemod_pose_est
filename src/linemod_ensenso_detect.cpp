//ros
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/CameraInfo.h>

//tf
#include <tf/transform_broadcaster.h>

//#include <cv_bridge/cv_bridge.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <message_filters/synchronizer.h>
#include <message_filters/subscriber.h>

//ork
#include "linemod_icp.h"
#include "linemod_pointcloud.h"
#include "db_linemod.h"
#include <object_recognition_renderer/renderer3d.h>
#include <object_recognition_renderer/utils.h>
//#include <object_recognition_core/common/pose_result.h>
//#include <object_recognition_core/db/ModelReader.h>

//opencv
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/rgbd/rgbd.hpp>
#include <opencv2/opencv.hpp>

//PCL
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/common/common.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/median_filter.h>

//ensenso
#include <ensenso/RegistImage.h>
#include <ensenso/CaptureSinglePointCloud.h>

//boost
#include <boost/foreach.hpp>

//std
#include <math.h>

//time synchronize
#define APPROXIMATE

#ifdef EXACT
#include <message_filters/sync_policies/exact_time.h>
#endif
#ifdef APPROXIMATE
#include <message_filters/sync_policies/approximate_time.h>
#endif

#ifdef EXACT
typedef message_filters::sync_policies::ExactTime<sensor_msgs::Image, sensor_msgs::Image> SyncPolicy;
#endif
#ifdef APPROXIMATE
typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> SyncPolicy;
#endif

typedef pcl::PointCloud<pcl::PointXYZ> PointCloudXYZ;

//namespace
using namespace cv;
using namespace std;

struct LinemodData{
  LinemodData(
          std::vector<cv::Vec3f> _pts_ref,
          std::vector<cv::Vec3f> _pts_model,
          std::string _match_class,
          int _match_id,
          const float _match_sim,
          const cv::Point location_,
          const float _icp_distance,
          const cv::Matx33f _r,
          const cv::Vec3f _t){
    pts_ref = _pts_ref;
    pts_model = _pts_model;
    match_class = _match_class;
    match_id=_match_id;
    match_sim = _match_sim;
    location=location_;
    icp_distance = _icp_distance;
    r = _r;
    t = _t;
    check_done = false;
  }
  std::vector<cv::Vec3f> pts_ref;
  std::vector<cv::Vec3f> pts_model;
  std::string match_class;
  int match_id;
  float match_sim;
  float icp_distance;
  cv::Matx33f r;
  cv::Vec3f t;
  cv::Point location;
  bool check_done;
};

class linemod_detect
{
    ros::NodeHandle nh;
    image_transport::ImageTransport it;
    ros::Subscriber sub_cam_info;
    //image_transport::Subscriber sub_color_;
    //image_transport::Subscriber sub_depth;
    //image_transport::Publisher pub_color_;
    //image_transport::Publisher pub_depth;
    message_filters::Subscriber<sensor_msgs::Image> sub_color;
    message_filters::Subscriber<sensor_msgs::Image> sub_depth;
    message_filters::Synchronizer<SyncPolicy> sync;
    ros::Publisher pc_rgb_pub_;
    ros::Publisher extract_pc_pub;

    //voting space
    unsigned int* accumulator;

public:
    Ptr<linemod::Detector> detector;
    float threshold;
    bool is_K_read;
    cv::Vec3f T_ref;

    vector<Mat> Rs_,Ts_;
    vector<float> Distances_;
    vector<float> Obj_origin_dists;
    vector<Mat> Ks_;
    Mat K_depth;
    Matx33f R_diag;
    int renderer_n_points;
    int renderer_angle_step;
    double renderer_radius_min;
    double renderer_radius_max;
    double renderer_radius_step;
    int renderer_width;
    int renderer_height;
    double renderer_focal_length_x;
    double renderer_focal_length_y;
    double renderer_near;
    double renderer_far;
    Renderer3d *renderer_;
    RendererIterator *renderer_iterator_;

    float px_match_min_;
    float icp_dist_min_;
    float th_obj_dist_;

    LinemodPointcloud *pci_real_icpin_ref;
    LinemodPointcloud *pci_real_icpin_model;
    LinemodPointcloud *pci_real_nonICP_model;
    LinemodPointcloud *pci_real_condition_filter_model;
    std::string depth_frame_id_;

    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;

    //std::vector <object_recognition_core::db::ObjData> objs_;
    std::vector<LinemodData> objs_;//pose result after clustering process
    std::vector<LinemodData> final_poses;//pose result after clustering process

    tf::TransformBroadcaster tf_broadcaster;
    //Service client
    ros::ServiceClient ensenso_registImg_client;
    ros::ServiceClient ensenso_singlePc_client;


public:
        linemod_detect(std::string template_file_name,std::string renderer_params_name,std::string mesh_path,float detect_score_threshold,float clustering_threshold,int icp_max_iter,float icp_tr_epsilon,float icp_fitness_threshold):
            it(nh),
            sub_color(nh,"/camera/rgb/image_rect_color",1),
            sub_depth(nh,"/camera/depth_registered/image_raw",1),
            depth_frame_id_("camera_link"),
            sync(SyncPolicy(1), sub_color, sub_depth),
            px_match_min_(0.25f),
            icp_dist_min_(0.06f)
        {
            //Publisher
            //pub_color_=it.advertise ("/sync_rgb",2);
            //pub_depth=it.advertise ("/sync_depth",2);
            pc_rgb_pub_=nh.advertise<sensor_msgs::PointCloud2>("ensenso_rgb_pc",1);
            extract_pc_pub = nh.advertise<sensor_msgs::PointCloud2>("/render_pc",1);

            //the intrinsic matrix
            //sub_cam_info=nh.subscribe("/camera/depth/camera_info",1,&linemod_detect::read_cam_info,this);

            //ork default param
            threshold=detect_score_threshold;
            th_obj_dist_=clustering_threshold;

            //read the saved linemod detecor
            detector=readLinemod (template_file_name);

            //read the poses of templates
            readLinemodTemplateParams (renderer_params_name,Rs_,Ts_,Distances_,Obj_origin_dists,
                                       Ks_,renderer_n_points,renderer_angle_step,
                                       renderer_radius_min,renderer_radius_max,
                                       renderer_radius_step,renderer_width,renderer_height,
                                       renderer_focal_length_x,renderer_focal_length_y,
                                       renderer_near,renderer_far);


            //load the stl model to GL renderer
            renderer_ = new Renderer3d(mesh_path);
            renderer_->set_parameters(renderer_width, renderer_height, renderer_focal_length_x, renderer_focal_length_y, renderer_near, renderer_far);
            renderer_iterator_ = new RendererIterator(renderer_, renderer_n_points);
            renderer_iterator_->angle_step_ = renderer_angle_step;
            renderer_iterator_->radius_min_ = float(renderer_radius_min);
            renderer_iterator_->radius_max_ = float(renderer_radius_max);
            renderer_iterator_->radius_step_ = float(renderer_radius_step);

            pci_real_icpin_model = new LinemodPointcloud(nh, "real_icpin_model", depth_frame_id_);
            pci_real_icpin_ref = new LinemodPointcloud(nh, "real_icpin_ref", depth_frame_id_);
            pci_real_condition_filter_model=new LinemodPointcloud(nh, "real_condition_filter", depth_frame_id_);
            //pci_real_nonICP_model= new LinemodPointcloud(nh, "real_nonICP_model", depth_frame_id_);
            //pci_real_1stICP_model= new LinemodPointcloud(nh, "real_1stICP_model", depth_frame_id_);

            icp.setMaximumIterations (icp_max_iter);
            icp.setMaxCorrespondenceDistance (0.05);
            icp.setTransformationEpsilon (icp_tr_epsilon);
            icp.setEuclideanFitnessEpsilon (icp_fitness_threshold);

            R_diag=Matx<float,3,3>(1.0,0.0,0.0,
                                  0.0,1.0,0.0,
                                  0.0,0.0,1.0);

            //Service client
            ensenso_registImg_client=nh.serviceClient<ensenso::RegistImage>("grab_registered_image");
            ensenso_singlePc_client=nh.serviceClient<ensenso::CaptureSinglePointCloud>("capture_single_point_cloud");


        }

        virtual ~linemod_detect()
        {
            if(accumulator)
                free(accumulator);
        }

        void detect_cb(const sensor_msgs::Image& msg_rgb,sensor_msgs::PointCloud2 pc,bool is_rgb)
        {
            //Convert image mgs to OpenCV
            Mat mat_rgb;
            Mat mat_grey;
            //if the image comes from monocular camera
            if(is_rgb)
            {
                cv_bridge::CvImagePtr img_ptr=cv_bridge::toCvCopy(msg_rgb,sensor_msgs::image_encodings::BGR8);
                img_ptr->image.copyTo(mat_rgb);
            }else //if the image comes from left camera of the stereo camera
                {
                cv_bridge::CvImagePtr img_ptr=cv_bridge::toCvCopy(msg_rgb,sensor_msgs::image_encodings::MONO8);
                img_ptr->image.copyTo(mat_grey);
                mat_rgb.create(mat_grey.rows,mat_grey.cols,CV_8UC3);
                int from_to[]={0,0,0,1,0,2};
                mixChannels(&mat_grey,1,&mat_rgb,1,from_to,3);

                //imshow("grey",mat_grey);
                imshow("conbined_gray",mat_rgb);
                waitKey(0);
            }

            //Convert pointcloud2 msg to PCL
            pcl::PointCloud<pcl::PointXYZ>::Ptr pc_ptr (new pcl::PointCloud<pcl::PointXYZ>);
            pcl::PointCloud<pcl::PointXYZ>::Ptr pc_median_ptr (new pcl::PointCloud<pcl::PointXYZ>);
            pcl::fromROSMsg(pc,*pc_ptr);
                //Option: Median filter for PC smoothing
//            pcl::MedianFilter<pcl::PointXYZ> median_filter;
//            median_filter.setWindowSize(11);
//            median_filter.setInputCloud(pc_ptr);
//            median_filter.applyFilter(*pc_median_ptr);

            if(detector->classIds ().empty ())
            {
                ROS_INFO("Linemod detector is empty");
                return;
            }

            //Convert point cloud to depth image
            Mat mat_depth;
            pc2depth(pc_ptr,mat_depth);

            //Crop the image
            cv::Rect crop(56,0,640,480);
            Mat mat_rgb_crop=mat_rgb(crop);
            Mat display=mat_rgb_crop;   //image for displaying results
            Mat mat_depth_crop=mat_depth(crop);

            //Perform the detection
            std::vector<Mat> sources;
            sources.push_back (mat_rgb_crop);
            //sources.push_back (mat_depth_crop);
            std::vector<linemod::Match> matches;
            double t;
            t=cv::getTickCount ();
            detector->match (sources,threshold,matches,std::vector<String>(),noArray());
            t=(cv::getTickCount ()-t)/cv::getTickFrequency ();

            //Display all the results
            for(std::vector<linemod::Match>::iterator it= matches.begin();it != matches.end();it++){
                std::vector<cv::linemod::Template> templates=detector->getTemplates(it->class_id, it->template_id);
                drawResponse(templates, 1, display,cv::Point(it->x,it->y), 2);
            }
            imshow("display",display);
            cv::waitKey (0);

            //Clustering based on Row Col Depth
            std::map<std::vector<int>, std::vector<linemod::Match> > map_match;
            int vote_row_col_step=4;
            double vote_depth_step=renderer_radius_step;
            int voting_height_cells,voting_width_cells;
            rcd_voting(vote_row_col_step, vote_depth_step, matches,map_match, voting_height_cells, voting_width_cells);

            //Filter based on size of clusters
            uchar thresh=10;
            cluster_filter(map_match,thresh);

            //Compute criteria for each cluster
            t=cv::getTickCount ();
            cluster_scoring(map_match,mat_rgb_crop,mat_depth_crop);
            t=(cv::getTickCount ()-t)/cv::getTickFrequency ();
            cout<<"Time consumed by scroing: "<<t<<endl;
            int p=0;

        }

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

        void drawResponse(const std::vector<cv::linemod::Template>& templates, int num_modalities, cv::Mat& dst, cv::Point offset,
                     int T)
        {
          static const cv::Scalar COLORS[5] =
          { CV_RGB(0, 0, 255), CV_RGB(0, 255, 0), CV_RGB(255, 255, 0), CV_RGB(255, 140, 0), CV_RGB(255, 0, 0) };
          if (dst.channels() == 1)
            cv::cvtColor(dst, dst, cv::COLOR_GRAY2BGR);

          //cv::circle(dst, cv::Point(offset.x + 20, offset.y + 20), T / 2, COLORS[4]);
          if (num_modalities > 5)
            num_modalities = 5;
          for (int m = 0; m < num_modalities; ++m)
          {
        // NOTE: Original demo recalculated max response for each feature in the TxT
        // box around it and chose the display color based on that response. Here
        // the display color just depends on the modality.
            cv::Scalar color = COLORS[m+2];

            for (int i = 0; i < (int) templates[m].features.size(); ++i)
            {
              cv::linemod::Feature f = templates[m].features[i];
              cv::Point pt(f.x + offset.x, f.y + offset.y);
              cv::circle(dst, pt, T / 2, color);
            }
          }
        }

         void readLinemodTemplateParams(const std::string fileName,
                                        std::vector<cv::Mat>& Rs,
                                        std::vector<cv::Mat>& Ts,
                                        std::vector<float>& Distances,
                                        std::vector<float>& Obj_origin_dists,
                                        std::vector<cv::Mat>& Ks,
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
            FileStorage fs(fileName,FileStorage::READ);

            for(int i=0;;++i)
            {
                std::stringstream ss;
                std::string s;
                s="Template ";
                ss<<i;
                s+=ss.str ();
                FileNode templates = fs[s];
                if(!templates.empty ())
                {
                    Mat R_tmp,T_tmp,K_tmp;
                    float D_tmp,obj_dist_tmp;
                    templates["R"]>>R_tmp;
                    Rs.push_back (R_tmp);
                    templates["T"]>>T_tmp;
                    Ts.push_back (T_tmp);
                    templates["K"]>>K_tmp;
                    Ks.push_back (K_tmp);
                    templates["D"]>>D_tmp;
                    Distances.push_back (D_tmp);
                    templates["Ori_dist"] >>obj_dist_tmp;
                    Obj_origin_dists.push_back (obj_dist_tmp);

                }
                else
                {
                    //fs["K Intrinsic Matrix"]>>K_matrix;
                    //std::cout<<K_matrix<<std::endl;
                    fs["renderer_n_points"]>>renderer_n_points;
                    fs["renderer_angle_step"]>>renderer_angle_step;
                    fs["renderer_radius_min"]>>renderer_radius_min;
                    fs["renderer_radius_max"]>>renderer_radius_max;
                    fs["renderer_radius_step"]>>renderer_radius_step;
                    fs["renderer_width"]>>renderer_width;
                    fs["renderer_height"]>>renderer_height;
                    fs["renderer_focal_length_x"]>>renderer_focal_length_x;
                    fs["renderer_focal_length_y"]>>renderer_focal_length_y;
                    fs["renderer_near"]>>renderer_near;
                    fs["renderer_far"]>>renderer_far;
                    break;
                }
            }

            fs.release ();
        }

         void read_cam_info(const sensor_msgs::CameraInfoConstPtr& infoMsg)
         {
             K_depth = (cv::Mat_<float>(3, 3) <<
                                  infoMsg->K[0], infoMsg->K[1], infoMsg->K[2],
                                  infoMsg->K[3], infoMsg->K[4], infoMsg->K[5],
                                  infoMsg->K[6], infoMsg->K[7], infoMsg->K[8]);

             Mat distCoeffs = (cv::Mat_<double>(1, 5) <<
                                   infoMsg->D[0],
                                   infoMsg->D[1],
                                   infoMsg->D[2],
                                   infoMsg->D[3],
                                   infoMsg->D[4]);
             sub_cam_info.shutdown ();
             is_K_read=true;
         }

         //Extract PointXYZ by indices
         void extractPointsByIndices(pcl::PointIndices::Ptr indices, const PointCloudXYZ::Ptr ref_pts, PointCloudXYZ::Ptr extracted_pts, bool is_negative,bool is_organised)
         {
             pcl::ExtractIndices<pcl::PointXYZ> tmp_extractor;
             tmp_extractor.setKeepOrganized(is_organised);
             tmp_extractor.setInputCloud(ref_pts);
             tmp_extractor.setNegative(is_negative);
             tmp_extractor.setIndices(indices);
             tmp_extractor.filter(*extracted_pts);
         }

         void alignment_test(PointCloudXYZ::Ptr pc, Mat img)
         {
             pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcRgb_ptr(new pcl::PointCloud<pcl::PointXYZRGB>);
             pcl::copyPointCloud(*pc,*pcRgb_ptr);

             for(int i=0;i<pcRgb_ptr->height;++i)
                 {
                 for(int j=0;j<pcRgb_ptr->width;++j)
                     {
                     pcRgb_ptr->at(j,i).r=img.at<cv::Vec3b>(i,j)[2];
                     pcRgb_ptr->at(j,i).g=img.at<cv::Vec3b>(i,j)[1];
                     pcRgb_ptr->at(j,i).b=img.at<cv::Vec3b>(i,j)[0];
                 }
             }

             sensor_msgs::PointCloud2 pc_rgb_msg;
             pcl::toROSMsg(*pcRgb_ptr,pc_rgb_msg);
             pc_rgb_pub_.publish(pc_rgb_msg);
             return;
         }

         bool pc2depth(PointCloudXYZ::ConstPtr pc,Mat& mat_depth)
         {
             //Convert point cloud to depth image
             Mat mat_depth_m;
             CV_Assert(pc->empty() == false);
             if(!pc->empty()){
                 int height=pc->height;
                 int width=pc->width;
                 mat_depth_m.create(height,width,CV_32FC1);
                 for(int i=0;i<height;i++)
                     for(int j=0;j<width;j++)
                         mat_depth_m.at<float>(i,j)=pc->at(j,i).z;
                 //imshow("depth",mat_depth_m);
                 //waitKey(0);
                 //Convert m to mm
                 mat_depth_m.convertTo(mat_depth,CV_16UC1,1000.0);
                 return true;
             }else{
                 ROS_ERROR("Empty pointcloud! Detection aborting!");
                 return false;
             }
         }

         void rcd_voting(const int& vote_row_col_step,const double& renderer_radius_step_,const vector<linemod::Match>& matches,std::map<std::vector<int>, std::vector<linemod::Match> >& map_match,int& voting_height_cells,int& voting_width_cells)
         {
             //----------------3D Voting-----------------------------------------------------//

             int position_voting_width=640; //640 or 752
             int position_voting_height=480; //480
             float position_voting_depth=renderer_radius_max-renderer_radius_min;

             int voting_width_step=vote_row_col_step; //Unit: pixel
             int voting_height_step=vote_row_col_step; //Unit: pixel, width step and height step suppose to be the same
             float voting_depth_step=renderer_radius_step_;//Unit: m

             voting_width_cells=(int)(position_voting_width/voting_width_step);
             voting_height_cells=(int)(position_voting_height/voting_height_step);
             int voting_depth_cells=(int)(position_voting_depth/voting_depth_step)+1;

             int max_index=0;
             int max_vote=0;

             BOOST_FOREACH(const linemod::Match& match,matches){
                 //Get height(row), width(cols), depth index
                 int height_index=match.y/voting_height_step;
                 int width_index=match.x/voting_width_step;
                 float depth = Obj_origin_dists[match.template_id];//the distance from the object origin to the camera origin
                 int depth_index=(int)((depth-renderer_radius_min)/voting_depth_step);

                 //fill the index
                 vector<int> index(3);
                 index[0]=height_index;
                 index[1]=width_index;
                 index[2]=depth_index;

                 //Use MAP to store matches of the same index
                 if(map_match.find (index)==map_match.end ())
                 {
                     std::vector<linemod::Match> temp;
                     temp.push_back (match);
                     map_match.insert(pair<std::vector<int>,std::vector<linemod::Match> >(index,temp));
                 }
                 else
                 {
                     map_match[index].push_back(match);
                 }

             }
         }

         void cluster_filter(std::map<std::vector<int>, std::vector<linemod::Match> >& map_match,int thresh)
         {
             assert(map_match.size() != 0);
             std::map<std::vector<int>, std::vector<linemod::Match> > map_filtered;
             std::map<std::vector<int>, std::vector<linemod::Match> >::iterator it= map_match.begin();
             for(;it!=map_match.end();++it)
             {
                if(it->second.size()>thresh)
                    map_filtered.insert(*it);
             }

             map_match.clear();
             map_match=map_filtered;
             //Copy
         }

         void cluster_scoring(std::map<std::vector<int>, std::vector<linemod::Match> >& map_match, Mat rgb_img, Mat depth_img)
         {
             //test for depth diff
             std::map<std::vector<int>, std::vector<linemod::Match> >::iterator it_map= map_match.begin();
             for(;it_map!=map_match.end();++it_map)
             {
                 //Perform depth difference computation and normal difference computation
                 double score=depth_normal_diff_calc(it_map->second,depth_img);

             }
         }

         // compute depth and normal diff for 1 cluster
         double depth_normal_diff_calc(std::vector<linemod::Match> match_cluster, Mat& depth_img)
         {
             double sum_depth_diff=0.0;
             std::vector<linemod::Match>::iterator it_match=match_cluster.begin();
             for(;it_match!=match_cluster.end();++it_match)
              {
                 //get mask during rendering
                 //get the pose
                 cv::Matx33d R_match = Rs_[it_match->template_id].clone();// rotation of the object w.r.t to the view point
                 cv::Vec3d T_match = Ts_[it_match->template_id].clone();//the translation of the camera with respect to the current view point

                 //get the point cloud of the rendered object model
                 cv::Mat template_mask;
                 cv::Rect rect;
                 cv::Matx33d R_temp(R_match.inv());//rotation of the viewpoint w.r.t the object
                 cv::Vec3d up(-R_temp(0,1), -R_temp(1,1), -R_temp(2,1));//the negative direction of y axis of viewpoint w.r.t object frame
                 cv::Mat depth_template;
                 renderer_iterator_->renderDepthOnly(depth_template, template_mask, rect, -T_match, up);//up?

                 //Compute depth diff for each match
                 sum_depth_diff+=depth_diff(depth_template,template_mask,rect);
                 int p=0;

                 //Compute normal diff for each match
                    //...
             }
             sum_depth_diff=sum_depth_diff/match_cluster.size();

         }

         double depth_diff(cv::Mat& depth_img,cv::Mat& template_mask,cv::Mat& rect)
         {
             //prepare the bounding box for the model and reference point clouds
             rect.x = it_match->x;
             rect.y = it_match->y;
             Mat depth_roi=depth_img(rect);
             Mat depth_mask;
             depth_roi.convertTo(depth_mask,CV_8UC1,1,0);
             Mat mask;
             bitwise_and(template_mask,depth_mask,mask);
             //             imshow("1",template_mask);
             //             imshow("2",mask);
             //             waitKey(0);

             //Perform subtraction and accumulate differences
             //-------- Method A for computing depth diff----//
             Mat subtraction(depth_template.size(),CV_16SC1);
             subtraction=depth_template-depth_roi;
             MatIterator_<uchar> it_mask=mask.begin<uchar>();
             MatIterator_<short> it_subs=subtraction.begin<short>();
             double sum=0.0;
             int num=0;
             for(;it_mask!=mask.end<uchar>();++it_mask,++it_subs)
             {
                 if(*it_mask>0)
                 {
                     sum=sum+(double)(1/(abs(*it_subs)+1));
                     num++;
                 }

             }
             sum=sum/num;
             //-------Method B for computing depth diff----//
//                 t1=cv::getTickCount();
//                 Mat subtraction=Mat::zeros(depth_template.size(),CV_16SC1);
//                 subtract(depth_template,depth_roi,subtraction,mask);
//                 Mat abs_sub=cv::abs(subtraction);
//                 Scalar sumOfDiff=sum(abs_sub);
//                 sum_depth_diff+=sumOfDiff[0];
//                 t1=(cv::getTickCount()-t1)/cv::getTickFrequency();
//                 cout<<"Time consumed by compute depth diff for 1 template: "<<t1<<endl;
             return sum;
         }

         double normal_diff()
         {

         }
};

int main(int argc,char** argv)
{
    ros::init (argc,argv,"linemod_detect");
    std::string linemod_template_path;
    std::string renderer_param_path;
    std::string model_stl_path;
    float detect_score_th;
    float clustering_th;
    int icp_max_iter;
    float icp_tr_epsilon;
    float icp_fitness_th;
    if(argc<8)
    {
        linemod_template_path="/home/yake/catkin_ws/src/linemod_pose_est/config/data/pipe_linemod_ensenso_templates.yml";
        renderer_param_path="/home/yake/catkin_ws/src/linemod_pose_est/config/data/pipe_linemod_ensenso_renderer_params.yml";
        model_stl_path="/home/yake/catkin_ws/src/linemod_pose_est/config/stl/pipe_connector.stl";
        detect_score_th=96.0;
        clustering_th=0.02;
        icp_max_iter=25;
        icp_tr_epsilon=0.0001;
        icp_fitness_th=0.0002;
    }
    else
    {
    linemod_template_path=argv[1];
    renderer_param_path=argv[2];
    model_stl_path=argv[3];
    detect_score_th=atof(argv[4]);
    clustering_th=atof(argv[5]);
    icp_max_iter=atoi(argv[6]);
    icp_tr_epsilon=atof(argv[7]);
    icp_fitness_th=atof(argv[8]);
    }
    linemod_detect detector(linemod_template_path,renderer_param_path,model_stl_path,
                            detect_score_th,clustering_th,icp_max_iter,icp_tr_epsilon,icp_fitness_th);

    ensenso::RegistImage srv;
    srv.request.is_rgb=true;
    ros::Rate loop(1);

    ros::Time now =ros::Time::now();
    Mat cv_img=imread("/home/yake/catkin_ws/src/ensenso/pcd/1481939332_rgb.jpg",IMREAD_COLOR);
    cv_bridge::CvImagePtr bridge_img_ptr(new cv_bridge::CvImage);
    bridge_img_ptr->image=cv_img;
    bridge_img_ptr->encoding="bgr8";
    bridge_img_ptr->header.stamp=now;
    srv.response.image = *bridge_img_ptr->toImageMsg();

    PointCloudXYZ::Ptr pc(new PointCloudXYZ);
    pcl::io::loadPCDFile("/home/yake/catkin_ws/src/ensenso/pcd/1481939332_pc.pcd",*pc);
    pcl::toROSMsg(*pc,srv.response.pointcloud);
    srv.response.pointcloud.header.frame_id="/camera_link";
    srv.response.pointcloud.header.stamp=now;

    while(ros::ok())
    {
        detector.detect_cb(srv.response.image,srv.response.pointcloud,srv.request.is_rgb);
        loop.sleep();
    }

    ros::spin ();
}
