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

//Eigen
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>
#include <eigen3/Eigen/Core>

//opencv
#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
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
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/statistical_outlier_removal.h>

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

//Score based on evaluation for 1 cluster
struct ClusterData{

    ClusterData():
        model_pc(new PointCloudXYZ)
    {
        index.resize(3);
        is_checked=false;
        dist=0.0;

    }

    ClusterData(const vector<int> index_,double score_):
        index(index_),
        score(score_),
        is_checked(false),
        dist(0.0),
        model_pc(new PointCloudXYZ)
    {

    }

    vector<linemod::Match> matches;
    vector<int> index;
    double score;
    bool is_checked;
    Rect rect;
    cv::Matx33d orientation;
    cv::Vec3d position;
    cv::Vec3d T_match;
    cv::Mat mask;
    cv::Mat K_matrix;
    double dist;
    PointCloudXYZ::Ptr model_pc;
    Eigen::Affine3d pose;

};

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

bool sortOrienCluster(const vector<Eigen::Matrix3d>& cluster1,const vector<Eigen::Matrix3d>& cluster2)
{
    return(cluster1.size()>cluster2.size());
}

bool sortIdCluster(const vector<int>& cluster1,const vector<int>& cluster2)
{
    return(cluster1.size()>cluster2.size());
}

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
    uchar clustering_step_;

public:
    Ptr<linemod::Detector> detector;
    float threshold;
    bool is_K_read;
    cv::Vec3f T_ref;

    vector<Mat> Rs_,Ts_;
    vector<double> Distances_;
    vector<double> Obj_origin_dists;
    vector<Mat> Ks_;
    vector<Rect> Rects_;
    Mat K_depth;
    Matx33d K_rgb;
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
    float orientation_clustering_th_;

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
        linemod_detect(std::string template_file_name,std::string renderer_params_name,std::string mesh_path,float detect_score_threshold,int icp_max_iter,float icp_tr_epsilon,float icp_fitness_threshold, float icp_maxCorresDist, uchar clustering_step,float orientation_clustering_th):
            it(nh),
            sub_color(nh,"/camera/rgb/image_rect_color",1),
            sub_depth(nh,"/camera/depth_registered/image_raw",1),
            depth_frame_id_("camera_link"),
            sync(SyncPolicy(1), sub_color, sub_depth),
            px_match_min_(0.25f),
            icp_dist_min_(0.06f),
            clustering_step_(clustering_step)
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

            //read the saved linemod detecor
            detector=readLinemod (template_file_name);

            //read the poses of templates
            readLinemodTemplateParams (renderer_params_name,Rs_,Ts_,Distances_,Obj_origin_dists,
                                       Ks_,Rects_,renderer_n_points,renderer_angle_step,
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
            icp.setMaxCorrespondenceDistance (icp_maxCorresDist);
            icp.setTransformationEpsilon (icp_tr_epsilon);
            icp.setEuclideanFitnessEpsilon (icp_fitness_threshold);


            R_diag=Matx<float,3,3>(1.0,0.0,0.0,
                                  0.0,1.0,0.0,
                                  0.0,0.0,1.0);
            K_rgb=Matx<double,3,3>(844.5966796875,0.0,338.907012939453125,
                                   0.0,844.5966796875,232.793670654296875,
                                   0.0,0.0,1.0);

            //Service client
            ensenso_registImg_client=nh.serviceClient<ensenso::RegistImage>("grab_registered_image");
            ensenso_singlePc_client=nh.serviceClient<ensenso::CaptureSinglePointCloud>("capture_single_point_cloud");

            //Orientation Clustering threshold
            orientation_clustering_th_=orientation_clustering_th;

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
//            for(std::vector<linemod::Match>::iterator it= matches.begin();it != matches.end();it++){
//                std::vector<cv::linemod::Template> templates=detector->getTemplates(it->class_id, it->template_id);
//                drawResponse(templates, 1, display,cv::Point(it->x,it->y), 2);
//            }

            //Clustering based on Row Col Depth
            std::map<std::vector<int>, std::vector<linemod::Match> > map_match;
            int vote_row_col_step=clustering_step_;
            double vote_depth_step=renderer_radius_step;
            int voting_height_cells,voting_width_cells;
            rcd_voting(vote_row_col_step, vote_depth_step, matches,map_match, voting_height_cells, voting_width_cells);

            //Filter based on size of clusters
            uchar thresh=10;
            cluster_filter(map_match,thresh);

            //Compute criteria for each cluster
                //Output: Vecotor of ClusterData, each element of which contains index, score, flag of checking.
            vector<ClusterData> cluster_data;
            t=cv::getTickCount ();
            cluster_scoring(map_match,mat_depth_crop,cluster_data);
            t=(cv::getTickCount ()-t)/cv::getTickFrequency ();
            cout<<"Time consumed by scroing: "<<t<<endl;
            int p=0;

            //Non-maxima suppression
            nonMaximaSuppression(cluster_data,10,map_match);

            //Pose average
            t=cv::getTickCount ();
            getRoughPoseByClustering(cluster_data,pc_ptr);
            t=(cv::getTickCount ()-t)/cv::getTickFrequency ();
            cout<<"Time consumed by pose clustering: "<<t<<endl;

            vizResultPclViewer(cluster_data,pc_ptr);

            //Pose refinement
            t=cv::getTickCount ();
            icpPoseRefine(cluster_data,pc_ptr);
            t=(cv::getTickCount ()-t)/cv::getTickFrequency ();
            cout<<"Time consumed by pose refinement: "<<t<<endl;

            //Display all the bounding box
            for(int ii=0;ii<cluster_data.size();++ii)
            {
                rectangle(display,cluster_data[ii].rect,Scalar(0,0,255),2);
            }

            imshow("display",display);
            cv::waitKey (0);

            //Viz in point cloud
            vizResultPclViewer(cluster_data,pc_ptr);

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
                                        std::vector<double>& Distances,
                                        std::vector<double>& Obj_origin_dists,
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
                    Rect rect_tmp;
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
                    templates["Rect"]>>rect_tmp;
                    Rects.push_back(rect_tmp);

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

         void cluster_scoring(std::map<std::vector<int>, std::vector<linemod::Match> >& map_match,Mat& depth_img,std::vector<ClusterData>& cluster_data)
         {             
             std::map<std::vector<int>, std::vector<linemod::Match> >::iterator it_map= map_match.begin();
             for(;it_map!=map_match.end();++it_map)
             {
                 //Perform depth difference computation and normal difference computation
                 //double score=depth_normal_diff_calc(it_map->second,depth_img);
                 //Options: similairy score computation
                 double score=similarity_score_calc(it_map->second);
                 cluster_data.push_back(ClusterData(it_map->first,score));

             }
         }

         // compute depth and normal diff for 1 cluster
         double depth_normal_diff_calc(std::vector<linemod::Match> match_cluster, Mat& depth_img)
         {
             double sum_depth_diff=0.0;
             double sum_normal_diff=0.0;
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
                 renderer_iterator_->renderDepthOnly(depth_template,template_mask, rect, -T_match, up);//up?
                 rect.x = it_match->x;
                 rect.y = it_match->y;

                 //Compute depth diff for each match
                 sum_depth_diff+=depth_diff(depth_img,depth_template,template_mask,rect);

                 //Compute normal diff for each match
                 sum_normal_diff+=normal_diff(depth_img,depth_template,template_mask,rect);
             }
             sum_depth_diff=sum_depth_diff/match_cluster.size();
             sum_normal_diff=sum_normal_diff/match_cluster.size();
             int p=0;
             return (getClusterScore(sum_depth_diff,sum_normal_diff));
         }

         double depth_diff(Mat& depth_img,Mat& depth_template,cv::Mat& template_mask,cv::Rect& rect)
         {
             //Crop ROI in depth image according to the rect
             Mat depth_roi=depth_img(rect);
             //Convert ROI into a mask. NaN point of depth image will become zero in mask image
             Mat depth_mask;
             depth_roi.convertTo(depth_mask,CV_8UC1,1,0);
             //And operation. Only valid points in both images will be considered.
             Mat mask;
             bitwise_and(template_mask,depth_mask,mask);

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

         double normal_diff(cv::Mat& depth_img,Mat& depth_template,cv::Mat& template_mask,cv::Rect& rect)
         {
             //Crop ROI in depth image according to the rect
             Mat depth_roi=depth_img(rect);
             //Convert ROI into a mask. NaN point of depth image will become zero in mask image
             Mat depth_mask;
             depth_roi.convertTo(depth_mask,CV_8UC1,1,0);
             //And operation. Only valid points in both images will be considered.
             Mat mask;
             bitwise_and(template_mask,depth_mask,mask);

             //Normal estimation for both depth images
            RgbdNormals normal_est(depth_roi.rows,depth_roi.cols,CV_64F,K_rgb,5,RgbdNormals::RGBD_NORMALS_METHOD_LINEMOD);
            Mat template_normal,roi_normal;
            normal_est(depth_roi,roi_normal);
            normal_est(depth_template,template_normal);

            //Compute normal diff
            Mat subtraction=roi_normal-template_normal;
            MatIterator_<uchar> it_mask=mask.begin<uchar>();
            MatIterator_<Vec3d> it_subs=subtraction.begin<Vec3d>();
            double sum=0.0;
            int num=0;
            for(;it_mask!=mask.end<uchar>();++it_mask,++it_subs)
            {
                if(*it_mask>0)
                {
                    if(isValidDepth((*it_subs)[0]) && isValidDepth((*it_subs)[1]) && isValidDepth((*it_subs)[2]))
                    {
                        double abs_diff=abs((*it_subs)[0])+abs((*it_subs)[1])+abs((*it_subs)[2]);
                        sum+=1/(abs_diff+1);
                        num++;
                    }

                }

            }
            sum/=num;
            return sum;

         }

         //Computer average similarity score for one cluster
         double similarity_score_calc(std::vector<linemod::Match> match_cluster)
         {
             double sum_score=0.0;
             int num=0;
             std::vector<linemod::Match>::iterator it_match=match_cluster.begin();
             for(;it_match!=match_cluster.end();++it_match)
             {
                 sum_score+=it_match->similarity;
                 num++;
             }
             sum_score/=num;
             return sum_score;
         }

         double getClusterScore(const double& depth_diff_score,const double& normal_diff_score)
         {
             //Simply add two scores
             return(depth_diff_score+normal_diff_score);

         }

         void nonMaximaSuppression(vector<ClusterData>& cluster_data,const double& neighborSize, std::map<std::vector<int>, std::vector<linemod::Match> >& map_match)
         {
             vector<ClusterData> nms_cluster_data;
             std::map<std::vector<int>, std::vector<linemod::Match> > nms_map_match;
             vector<ClusterData>::iterator it1=cluster_data.begin();
             for(;it1!=cluster_data.end();++it1)
             {
                 if(!it1->is_checked)
                 {
                     ClusterData* best_cluster=&(*it1);
                     vector<ClusterData>::iterator it2=it1;
                     it2++;
                     //Look for local maxima
                     for(;it2!=cluster_data.end();++it2)
                     {
                         if(!it2->is_checked)
                         {

                             double dist=sqrt((best_cluster->index[0]-it2->index[0]) * (best_cluster->index[0]-it2->index[0]) + (best_cluster->index[1]-it2->index[1]) * (best_cluster->index[1]-it2->index[1]));
                             if(dist < neighborSize)
                             {
                                 it2->is_checked=true;
                                 if(it2->score > best_cluster->score)
                                 {
                                     best_cluster=&(*it2);
                                 }
                             }
                         }
                     }
                     nms_cluster_data.push_back(*best_cluster);
                 }
             }

             cluster_data.clear();
             cluster_data=nms_cluster_data;

             //Add matches to cluster data
             it1=cluster_data.begin();
             for(;it1!=cluster_data.end();++it1)
             {
                 std::map<std::vector<int>, std::vector<linemod::Match> >::iterator it2;
                 it2=map_match.find(it1->index);
                 nms_map_match.insert(*it2);
                 it1->matches=it2->second;
             }

             //Compute bounding box for each cluster
             it1=cluster_data.begin();
             for(;it1!=cluster_data.end();++it1)
             {
                 int X=0; int Y=0; int WIDTH=0; int HEIGHT=0;
                 std::vector<linemod::Match>::iterator it2=it1->matches.begin();
                 for(;it2!=it1->matches.end();++it2)
                 {
                     Rect tmp=Rects_[it2->template_id];
                     X+=it2->x;
                     Y+=it2->y;
                     WIDTH+=tmp.width;
                     HEIGHT+=tmp.height;
                 }
                 X/=it1->matches.size();
                 Y/=it1->matches.size();
                 WIDTH/=it1->matches.size();
                 HEIGHT/=it1->matches.size();

                 it1->rect=Rect(X,Y,WIDTH,HEIGHT);

             }

             map_match.clear();
             map_match=nms_map_match;
             int p=0;
         }

         //get rough pose by pose averaging
//         void getRoughPoseByPoseAver(vector<ClusterData>& cluster_data,PointCloudXYZ::Ptr pc)
//         {
//             //For each cluster
//             for(vector<ClusterData>::iterator it = cluster_data.begin();it!=cluster_data.end();++it)
//             {
//                 //For each match
//                 Eigen::Vector3d T_average(0.0,0.0,0.0);
//                 Eigen::Vector4d R_average(0.0,0.0,0.0,0.0);
//                 double D_average=0.0;

//                 for(vector<linemod::Match>::iterator it2=it->matches.begin();it2!=it->matches.end();++it2)
//                 {
//                     //Get translation
//                     Mat T_mat=Ts_[it2->template_id];
//                     Eigen::Matrix<double,3,1> T;
//                     cv2eigen(T_mat,T);
//                     T_average+=T;

//                     //Get rotation
//                     Mat R_mat=Rs_[it2->template_id];
//                     Eigen::Matrix<double,3,3> R;
//                     cv2eigen(R_mat,R);
//                     R_average+=Eigen::Quaterniond(R).coeffs();

//                     //Get distance from match surface center to object center
//                     D_average+=Distances_[it2->template_id];
//                 }

//                 //Averaging
//                 T_average/=it->matches.size();
//                 R_average/=it->matches.size();
//                 D_average/=it->matches.size();

//                 //Don't forget to normalize quaternion
//                    //Ref: "On Averaging Rotations" by CLAUS GRAMKOW
//                 it->rotation=Eigen::Quaterniond(R_average).normalized();
//                 it->dist=D_average;

//                 //Update translation
//                 int x=it->rect.x+it->rect.width/2;
//                 int y=it->rect.y+it->rect.height/2;
//                    //Add offset to x due to previous cropping operation
//                 x+=56;
//                 pcl::PointXYZ pt = pc->at(x,y);
//                    //Notice
//                 pt.z+=D_average;
//                 it->translation=Eigen::Translation3d(pt.x,pt.y,pt.z);

//                 //Render pointcloud and update its position
//                    //Render
//                 cv::Mat R_mat;
//                 Eigen::Matrix3d R_eig;
//                 R_eig=it->rotation.toRotationMatrix();
//                 eigen2cv(R_eig,R_mat);
//                 cv::Vec3d T_match;//the translation of the camera with respect to the current view point
//                 T_match[0]=T_average[0];
//                 T_match[1]=T_average[1];
//                 T_match[2]=T_average[2];
//                 cv::Mat K_matrix= Ks_[it->matches[0].template_id].clone();
//                 //get the point cloud of the rendered object model
//                 cv::Mat mask;
//                 cv::Rect rect;
//                 cv::Matx33d R_match(R_mat);
//                 cv::Matx33d R_temp(R_match.inv());//rotation of the viewpoint w.r.t the object
//                 cv::Vec3d up(-R_temp(0,1), -R_temp(1,1), -R_temp(2,1));//the negative direction of y axis of viewpoint w.r.t object frame
//                 cv::Mat depth_ref_;
//                 renderer_iterator_->renderDepthOnly(depth_ref_, mask, rect, -T_match, up);
//                 imshow("depth",mask);
//                 waitKey(0);

//             }

//         }

         //get rough pose by clustering
         void getRoughPoseByClustering(vector<ClusterData>& cluster_data,PointCloudXYZ::Ptr pc)
         {
             //For each cluster
             for(vector<ClusterData>::iterator it = cluster_data.begin();it!=cluster_data.end();++it)
             {
                 //Perform clustering
                 vector<vector<Eigen::Matrix3d> > orienClusters;
                 vector<vector<int> > idClusters;
                 for(vector<linemod::Match>::iterator it2=it->matches.begin();it2!=it->matches.end();++it2)
                 {
                     //Get rotation
                     Mat R_mat=Rs_[it2->template_id];
                     Eigen::Matrix<double,3,3> R;
                     cv2eigen(R_mat,R);

                     bool found_cluster=false;

                     //Compare current orientation with existing cluster
                     for(int i=0;i<orienClusters.size();++i)
                     {
                         if(orientationCompare(R,orienClusters[i].front(),orientation_clustering_th_))
                         {
                             found_cluster=true;
                             orienClusters[i].push_back(R);
                             idClusters[i].push_back(it2->template_id);
                             break;
                         }
                     }

                     //If current orientation is not assigned to any cluster, create a new one for it.
                     if(found_cluster == false)
                     {
                         vector<Eigen::Matrix3d> new_cluster;
                         new_cluster.push_back(R);
                         orienClusters.push_back(new_cluster);

                         vector<int> new_cluster_;
                         new_cluster_.push_back(it2->template_id);
                         idClusters.push_back(new_cluster_);
                     }
                 }
                 //Sort cluster according to the number of poses
                 std::sort(orienClusters.begin(),orienClusters.end(),sortOrienCluster);
                 std::sort(idClusters.begin(),idClusters.end(),sortIdCluster);

                 //Test display all the poses in 1st cluster
//                 for(int i=0;i<idClusters[0].size();++i)
//                 {
//                     int ID=idClusters[0][i];

//                     //get the pose
//                     cv::Matx33d R_match = Rs_[ID].clone();// rotation of the object w.r.t to the view point
//                     cv::Vec3d T_match = Ts_[ID].clone();//the translation of the camera with respect to the current view point
//                     cv::Mat K_matrix= Ks_[ID].clone();
//                     cv::Mat mask;
//                     cv::Rect rect;
//                     cv::Matx33d R_temp(R_match.inv());//rotation of the viewpoint w.r.t the object
//                     cv::Vec3d up(-R_temp(0,1), -R_temp(1,1), -R_temp(2,1));//the negative direction of y axis of viewpoint w.r.t object frame
//                     cv::Mat depth_ref_;
//                     renderer_iterator_->renderDepthOnly(depth_ref_, mask, rect, -T_match, up);
//                     imshow("mask",mask);
//                     waitKey(0);

//                 }

                //Test average all poses in 1st cluster
                 Eigen::Vector3d T_aver(0.0,0.0,0.0);
                 Eigen::Vector4d R_aver(0.0,0.0,0.0,0.0);
                 double D_aver=0.0;
                 vector<Eigen::Matrix3d>::iterator iter=orienClusters[0].begin();
                 for(int i=0;iter!=orienClusters[0].end();++iter,++i)
                 {
                     //get rotation
                        //Matrix33d ---> Quaternion ---> Vector4d
                     R_aver+=Eigen::Quaterniond(*iter).coeffs();
                     //get translation
                     Mat T_mat=Ts_[idClusters[0][i]];
                     Eigen::Matrix<double,3,1> T;
                     cv2eigen(T_mat,T);
                     T_aver+=T;
                     //Get distance
                     D_aver+=Distances_[idClusters[0][i]];

                 }
                //Averaging operation
                 R_aver/=orienClusters[0].size();
                 T_aver/=orienClusters[0].size();
                 D_aver/=orienClusters[0].size();
                //normalize the averaged quaternion
                 Eigen::Quaterniond quat=Eigen::Quaterniond(R_aver).normalized();
                 cv::Mat R_mat;
                 Eigen::Matrix3d R_eig;
                 R_eig=quat.toRotationMatrix();
                 eigen2cv(R_eig,R_mat);

                 cv::Mat mask;
                 cv::Rect rect;
                 cv::Matx33d R_match(R_mat);
                 cv::Matx33d R_temp(R_match.inv());//rotation of the viewpoint w.r.t the object
                 cv::Vec3d up(-R_temp(0,1), -R_temp(1,1), -R_temp(2,1));//the negative direction of y axis of viewpoint w.r.t object frame
                 cv::Mat depth_render;
                 cv::Vec3d T_match;
                 T_match[0]=T_aver[0];
                 T_match[1]=T_aver[1];
                 T_match[2]=T_aver[2];
                 renderer_iterator_->renderDepthOnly(depth_render, mask, rect, -T_match, up);
//                 imshow("mask",mask);
//                 waitKey(0);

                 //Save orientation, T_match and K_matrix
                 it->orientation=R_match;
                 it->T_match=T_match;
                 it->dist=D_aver;
                 it->K_matrix=(Mat_<double>(3,3)<<renderer_focal_length_x,0.0,rect.width/2,
                                               0.0,renderer_focal_length_y,rect.height/2,
                                               0.0,0.0,1.0);
                 it->pose.linear()=R_eig;

                 //Save position
                 int x=it->rect.x+it->rect.width/2;
                 int y=it->rect.y+it->rect.height/2;
                 //Add offset to x due to previous cropping operation
                 x+=56;
                 pcl::PointXYZ pt =pc->at(x,y);
                    //Notice
                 pt.z+=D_aver;
                 it->position=cv::Vec3d(pt.x,pt.y,pt.z);
                 it->pose.translation()<< pt.x,pt.y,pt.z;

                 //Save mask
                 it->mask=mask;

//                 //Get render point cloud
                 Mat pc_cv;
                 cv::depthTo3d(depth_render,it->K_matrix,pc_cv);    //mm ---> m


                 pcl::PointCloud<pcl::PointXYZ> pts;
                 it->model_pc->width=pc_cv.cols;
                 it->model_pc->height=pc_cv.rows;
                 it->model_pc->resize(pc_cv.cols*pc_cv.rows);
                 it->model_pc->header.frame_id="/camera_link";

                 for(int ii=0;ii<pc_cv.rows;++ii)
                 {
                     double* row_ptr=pc_cv.ptr<double>(ii);
                     for(int jj=0;jj<pc_cv.cols;++jj)
                     {
                            double* data =row_ptr+jj*3;
                            //cout<<" "<<data[0]<<" "<<data[1]<<" "<<data[2]<<endl;                            
                            it->model_pc->at(jj,ii).x=data[0];
                            it->model_pc->at(jj,ii).y=data[1];
                            it->model_pc->at(jj,ii).z=data[2];

                     }
                 }

                 //Transform the pc                 
                 pcl::PointXYZ pt_scene =pc->at(x,y);
                 pcl::PointXYZ pt_model=it->model_pc->at(it->model_pc->width/2,it->model_pc->height/2);
                 pcl::PointXYZ translation(pt_scene.x -pt_model.x,pt_scene.y -pt_model.y,pt_scene.z -pt_model.z);
                 Eigen::Affine3d transform =Eigen::Affine3d::Identity();
                 transform.translation()<<translation.x, translation.y, translation.z;

                 PointCloudXYZ::Ptr transformed_cloud (new PointCloudXYZ ());
                 pcl::transformPointCloud(*it->model_pc,*transformed_cloud,transform);

                 //it->model_pc.swap(transformed_cloud);

                 pcl::copyPointCloud(*transformed_cloud,*(it->model_pc));
             }
         }

         void icpPoseRefine(vector<ClusterData>& cluster_data,PointCloudXYZ::Ptr pc)
         {
             for(vector<ClusterData>::iterator it = cluster_data.begin();it!=cluster_data.end();++it)
             {
                //Get point cloud indices
                 pcl::PointIndices::Ptr indices(new pcl::PointIndices);
                 indices=getPointCloudIndices(it);
                //Extract pc according to indices
                 PointCloudXYZ::Ptr scene_pc(new PointCloudXYZ);
                 extractPointsByIndices(indices,pc,scene_pc,false,false);

                 //Viz for test
//                 pcl::visualization::PCLVisualizer v("view_test");
//                 v.addPointCloud(scene_pc,"scene");
//                 v.spin();

                 //Remove Nan points
                 vector<int> index;
                 it->model_pc->is_dense=false;
                 pcl::removeNaNFromPointCloud(*(it->model_pc),*(it->model_pc),index);
                 pcl::removeNaNFromPointCloud(*scene_pc,*scene_pc,index);

                 //Statistical outlier removal
                 statisticalOutlierRemoval(scene_pc,50,1.0);

                 //Viz for test
//                 v.updatePointCloud(scene_pc,"scene");
//                 v.spin();

                 //Coarse alignment
                 icp.setInputSource (it->model_pc);
                 icp.setInputTarget (scene_pc);
                 icp.align (*(it->model_pc));
                 //Update pose
                 Eigen::Matrix4f tf_mat = icp.getFinalTransformation();
                 Eigen::Matrix4d tf_mat_d=tf_mat.cast<double>();
                 Eigen::Affine3d tf(tf_mat_d);
                 it->pose=tf*it->pose;

                 //Fine alignment
                 icp.setMaximumIterations(30);
                 icp.setMaxCorrespondenceDistance(0.01);
                 icp.setInputSource (it->model_pc);
                 icp.setInputTarget (scene_pc);
                 icp.align (*(it->model_pc));
                 //Update pose
                 tf_mat = icp.getFinalTransformation();
                 tf_mat_d=tf_mat.cast<double>();
                 tf.matrix()=tf_mat_d;
                 it->pose=tf*it->pose;

                 int p=0;
             }
         }

        bool orientationCompare(Eigen::Matrix3d& orien1,Eigen::Matrix3d& orien2,double thresh)
        {
            //lazyProduct and eval reference: https://eigen.tuxfamily.org/dox/TopicLazyEvaluation.html
            Eigen::AngleAxisd rotation_diff_mat (orien1.inverse ().lazyProduct (orien2).eval());
            double angle_diff = (double)(fabs(rotation_diff_mat.angle())/M_PI*180.0);
            //thresh unit: degree
            if(angle_diff<thresh)
            {
                return true;
            }
            else
            {
                return false;
            }

        }

        pcl::PointIndices::Ptr getPointCloudIndices(const cv::Rect& rect)
        {
            int row_offset=rect.y;
            int col_offset=rect.x;
            pcl::PointIndices::Ptr indices(new pcl::PointIndices);
            for(int i=0;i<rect.height;++i)
            {
                for(int j=0;j<rect.width;++j)
                {
                    //Notice: the coordinate of input params "rect" is w.r.t cropped image, so the offset is needed to transform coordinate
                    int x_cropped=j+col_offset;
                    int y_cropped=i+row_offset;
                    int x_uncropped=x_cropped+56;
                    int y_uncropped=y_cropped;
                    //Attention: image width of ensenso: 752, image height of ensenso: 480
                    int index=y_uncropped*752+x_uncropped;
                    indices->indices.push_back(index);

                }
            }
            return indices;

        }

        pcl::PointIndices::Ptr getPointCloudIndices(vector<ClusterData>::iterator& it)
        {
            int x_cropped=0;
            int y_cropped=0;
            pcl::PointIndices::Ptr indices(new pcl::PointIndices);
            for(int i=0;i<it->mask.rows;++i)
            {
                const uchar* row_data=it->mask.ptr<uchar>(i);
                for(int j=0;j<it->mask.cols;++j)
                {
                    //Notice: the coordinate of input params "rect" is w.r.t cropped image, so the offset is needed to transform coordinate
                    if(row_data[j]>0)
                    {
                        x_cropped=j+it->rect.x;
                        y_cropped=i+it->rect.y;
                        //Attention: image width of ensenso: 752, image height of ensenso: 480
                        int index=y_cropped*752+x_cropped+56;
                        indices->indices.push_back(index);
                    }
                }
            }
            return indices;

        }

        void projectPtsToImage()
        {
            //            //Project pose into image plane
            //                //Define camera matrix
            //            Mat k_mtx=(Mat_<double>(3,3)<<renderer_focal_length_x,0.0,640/2,
            //                                          0.0,renderer_focal_length_y,480/2,
            //                                          0.0,0.0,1.0);
            //            cout<<k_mtx<<endl;
            //                //Define distortion coefficients
            //            vector<double> distCoeffs(5,0.0);
            //            distCoeffs[0]=-0.0932546108961105347;
            //                //Define points in object coordinate
            //            vector<Point3d> pts(4);
            //            pts[0]=Point3d(10.0,0.0,0.0);
            //            pts[1]=Point3d(0.0,10.0,0.0);
            //            pts[2]=Point3d(0.0,0.0,10.0);
            //            pts[3]=Point3d(0.0,0.0,0.0);
            //            for(int ii=0;ii<cluster_data.size();++ii)
            //            {
            //                //Get rodrigues representation
            //                Matx31d rod;
            //                Rodrigues(cluster_data[ii].orientation,rod);

            //                //Output pts
            //                vector<Point2d> image_pts;

            //                //Translation vector
            //                vector<double> trans(3);
            //                trans[0]=cluster_data[ii].position[0]*1000;
            //                trans[1]=cluster_data[ii].position[1]*1000;
            //                trans[2]=cluster_data[ii].position[2]*1000;

            //                projectPoints(pts,rod,trans,k_mtx,distCoeffs,image_pts);
            //                for(int jj=0;jj<image_pts.size();++jj)
            //                {
            //                    Point pt_offset=Point(image_pts[jj].x,image_pts[jj].y);
            //                    circle(display,pt_offset,3,Scalar(255,0,0),2);
            //                 }
            //                imshow("circle",display);
            //                waitKey(0);

            //                int p=0;

            //            }
        }

        void vizResultPclViewer(const vector<ClusterData>& cluster_data,PointCloudXYZ::Ptr pc_ptr)
        {
            pcl::visualization::PCLVisualizer view("v");
            view.addPointCloud(pc_ptr,"scene");
            for(int ii=0;ii<cluster_data.size();++ii)
            {
                pcl::visualization::PointCloudColorHandlerRandom<pcl::PointXYZ> color(cluster_data[ii].model_pc);
                string str="model ";
                stringstream ss;
                ss<<ii;
                str+=ss.str();
                view.addPointCloud(cluster_data[ii].model_pc,color,str);

                //Get eigen tf
//                Eigen::Affine3d obj_pose;
//                Eigen::Matrix3d rot;
//                cv2eigen(cluster_data[ii].orientation,rot);
//                obj_pose.linear()=rot;
//                obj_pose.translation()<<cluster_data[ii].position[0],cluster_data[ii].position[1],cluster_data[ii].position[2];

                Eigen::Affine3f obj_pose_f=cluster_data[ii].pose.cast<float>();
                //Eigen::Affine3f obj_pose_f=obj_pose.cast<float>();
                view.addCoordinateSystem(0.08,obj_pose_f);
            }
            view.spin();
        }

        void euclidianClustering(PointCloudXYZ::Ptr pts,float dist)
        {
            pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
            tree->setInputCloud (pts);
            std::vector<int> index1;
            pcl::removeNaNFromPointCloud(*pts,*pts,index1);

            std::vector<pcl::PointIndices> cluster_indices;
            pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
            ec.setClusterTolerance (dist); // 1cm
            ec.setMinClusterSize (50);
            ec.setMaxClusterSize (25000);
            ec.setSearchMethod (tree);
            ec.setInputCloud (pts);
            ec.extract (cluster_indices);
            PointCloudXYZ::Ptr pts_filtered(new PointCloudXYZ);
            extractPointsByIndices(boost::make_shared<pcl::PointIndices>(cluster_indices[0]),pts,pts_filtered,false,false);
            //pts.swap(pts_filtered);
            pcl::copyPointCloud(*pts_filtered,*pts);
        }

        void statisticalOutlierRemoval(PointCloudXYZ::Ptr pts, int num_neighbor,float stdDevMulThresh)
        {
            PointCloudXYZ::Ptr pts_filtered(new PointCloudXYZ);
            pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
            sor.setInputCloud (pts);
            sor.setMeanK (num_neighbor);
            sor.setStddevMulThresh (stdDevMulThresh);
            sor.filter (*pts_filtered);
            //pts.swap(pts_filtered);
            pcl::copyPointCloud(*pts_filtered,*pts);
        }

};

int main(int argc,char** argv)
{
    ros::init (argc,argv,"linemod_detect");
    std::string linemod_template_path;
    std::string renderer_param_path;
    std::string model_stl_path;
    float detect_score_th;
    int icp_max_iter;
    float icp_tr_epsilon;
    float icp_fitness_th;
    float icp_maxCorresDist;
    uchar clustering_step;
    float orientation_clustering_step;

    linemod_template_path=argv[1];
    renderer_param_path=argv[2];
    model_stl_path=argv[3];
    detect_score_th=atof(argv[4]);
    icp_max_iter=atoi(argv[5]);
    icp_tr_epsilon=atof(argv[6]);
    icp_fitness_th=atof(argv[7]);
    icp_maxCorresDist=atof(argv[8]);
    clustering_step=atoi(argv[9]);
    orientation_clustering_step=atof(argv[10]);

    linemod_detect detector(linemod_template_path,renderer_param_path,model_stl_path,
                            detect_score_th,icp_max_iter,icp_tr_epsilon,icp_fitness_th,icp_maxCorresDist,clustering_step,orientation_clustering_step);

    ensenso::RegistImage srv;
    srv.request.is_rgb=true;
    ros::Rate loop(1);

    string img_path=argv[11];
    string pc_path=argv[12];
    ros::Time now =ros::Time::now();
    Mat cv_img=imread(img_path,IMREAD_COLOR);
    cv_bridge::CvImagePtr bridge_img_ptr(new cv_bridge::CvImage);
    bridge_img_ptr->image=cv_img;
    bridge_img_ptr->encoding="bgr8";
    bridge_img_ptr->header.stamp=now;
    srv.response.image = *bridge_img_ptr->toImageMsg();

    PointCloudXYZ::Ptr pc(new PointCloudXYZ);
    pcl::io::loadPCDFile(pc_path,*pc);
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
