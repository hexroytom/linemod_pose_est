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
#include <pcl/registration/icp_nl.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/common/common.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/median_filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/octree/octree.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/ply/ply.h>

//ensenso
#include <ensenso/RegistImage.h>
#include <ensenso/CaptureSinglePointCloud.h>

//boost
#include <boost/foreach.hpp>

//std
#include <math.h>
#include <fstream>

#include <linemod_pose_estimation/rgbdDetector.h>

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

//Global vairable
string gr_file_serial;
string serial_number;

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


string getGRfileSerial(string img_path)
{
    //Ref: home/yake/catkin_ws/src/linemod_pose_est/dataset/RGB/img_330.png
    string::iterator it_str=img_path.end();
    it_str-=5;
    string::iterator last_number=it_str;
    while(*it_str != '_')
    {
        it_str--;
    }
    string::iterator first_num=it_str+1;
    string num_str(first_num,last_number+1);
    if(num_str[0] == '0')
    {
        num_str.erase(num_str.begin());
    }

    return num_str;
}

bool sortClusterBasedOnScore(const ClusterData& cluster1,const ClusterData& cluster2)
{
    return(cluster1.score>cluster2.score);
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
    float linemod_match_threshold;
    float collision_rate_threshold;
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

    //Offset for compensating cropped image
    int bias_x;
    rgbdDetector::IMAGE_WIDTH image_width;


    //File path for ground truth
    std::string gr_prefix;

    //rgbd detector test
    rgbdDetector rgbd_detector;

    //model point cloud
    PointCloudXYZ::Ptr model_pc;

    //final results
    vector<ClusterData> cluster_data;
    double nms_radius;


public:
        linemod_detect(std::string template_file_name,std::string renderer_params_name,std::string mesh_path,float detect_score_threshold,int icp_max_iter,float icp_tr_epsilon,float icp_fitness_threshold, float icp_maxCorresDist, uchar clustering_step,float orientation_clustering_th):
            it(nh),
            gr_prefix("/home/yake/catkin_ws/src/linemod_pose_est/dataset/Annotation/"),
            sub_color(nh,"/camera/rgb/image_rect_color",1),
            sub_depth(nh,"/camera/depth_registered/image_raw",1),
            depth_frame_id_("camera_link"),
            sync(SyncPolicy(1), sub_color, sub_depth),
            px_match_min_(0.25f),
            icp_dist_min_(0.06f),
            clustering_step_(clustering_step),
            bias_x(0),
            image_width(rgbdDetector::CARMINE),
            model_pc(new PointCloudXYZ),
            collision_rate_threshold(0.4)
        {
            //Publisher
            //pub_color_=it.advertise ("/sync_rgb",2);
            //pub_depth=it.advertise ("/sync_depth",2);
            pc_rgb_pub_=nh.advertise<sensor_msgs::PointCloud2>("ensenso_rgb_pc",1);
            extract_pc_pub = nh.advertise<sensor_msgs::PointCloud2>("/render_pc",1);

            //the intrinsic matrix
            //sub_cam_info=nh.subscribe("/camera/depth/camera_info",1,&linemod_detect::read_cam_info,this);

            //ork default param
            linemod_match_threshold=detect_score_threshold;

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

        void detect_cb(const Mat& rgb_img,const Mat& depth_img)
        {
            cluster_data.clear();
            //This program aims for detect objects using dataset of paper "Latent-Class Forests for 3D Object Detection and Pose Estimation"
            if(detector->classIds ().empty ())
            {
                ROS_INFO("Linemod detector is empty");
                return;
            }

            //Read camera intrinsic params. Camera intinsic can be found on the dataset website
            Mat K_rgb;
            K_rgb = (cv::Mat_<float>(3, 3) <<
                                 571.9737, 0.0, 319.5000,
                                 0.0, 571.0073, 239.5000,
                                 0.0, 0.0, 1.0);

            //Convert depth image to pc2
            Mat pc_cv, depth_m;
            depth_img.convertTo(depth_m,CV_64FC1,0.001);
            cv::depthTo3d(depth_m,K_rgb,pc_cv);    //mm ---> m
            PointCloudXYZ::Ptr pc_ptr(new PointCloudXYZ);
            pc_ptr->resize(depth_img.cols*depth_img.rows);
            pc_ptr->height=depth_img.rows;
            pc_ptr->width=depth_img.cols;
            pc_ptr->header.frame_id="/camera_link";
            for(int ii=0;ii<pc_cv.rows;++ii)
            {
                double* row_ptr=pc_cv.ptr<double>(ii);
                for(int jj=0;jj<pc_cv.cols;++jj)
                {
                       double* data =row_ptr+jj*3;
                       //cout<<" "<<data[0]<<" "<<data[1]<<" "<<data[2]<<endl;
                       pc_ptr->at(jj,ii).x=data[0];
                       pc_ptr->at(jj,ii).y=data[1];
                       pc_ptr->at(jj,ii).z=data[2];

                }
            }

            //Get LINEMOD source image
            vector<Mat> sources(2);
            sources[0]=rgb_img;
            sources[1]=depth_img;
            Mat mat_rgb,mat_depth;
            mat_rgb=sources[0];
            mat_depth=sources[1];

            //Image for displaying detection
            Mat initial_img=mat_rgb.clone();
            Mat display=mat_rgb.clone();
            Mat final=mat_rgb.clone();
            Mat cluster_img=mat_rgb.clone();
            Mat nms_img = mat_rgb.clone();

            //Perform the LINEMOD detection
            std::vector<linemod::Match> matches;
            double t=cv::getTickCount ();
            rgbd_detector.linemod_detection(detector,sources,linemod_match_threshold,matches);
            t=(cv::getTickCount ()-t)/cv::getTickFrequency ();
            //cout<<"Time consumed by template matching: "<<t<<" s"<<endl;

            //Display all the results
//            for(std::vector<linemod::Match>::iterator it= matches.begin();it != matches.end();it++){
//                std::vector<cv::linemod::Template> templates=detector->getTemplates(it->class_id, it->template_id);
//                drawResponse(templates, 1, display,cv::Point(it->x,it->y), 2);
//            }
//            imshow("intial results",display);
//            waitKey(1);
            for(std::vector<linemod::Match>::iterator it= matches.begin();it != matches.end();it++)
            {
                cv::Rect rect_tmp= Rects_[it->template_id];
                rect_tmp.x=it->x;
                rect_tmp.y=it->y;
                rectangle(initial_img,rect_tmp,Scalar(0,0,255),2);
            }

            //Clustering based on Row Col Depth
            std::map<std::vector<int>, std::vector<linemod::Match> > map_match;
            t=cv::getTickCount ();
            //rcd_voting(vote_row_col_step, vote_depth_step, matches,map_match, voting_height_cells, voting_width_cells);
            rgbd_detector.rcd_voting(Obj_origin_dists,renderer_radius_min,clustering_step_,renderer_radius_step,matches,map_match);
            t=(cv::getTickCount ()-t)/cv::getTickFrequency ();
            //cout<<"Time consumed by rcd voting: "<<t<<" s"<<endl;
            //Display all the results
//            RNG rng(0xFFFFFFFF);
//            for(std::map<std::vector<int>, std::vector<linemod::Match> >::iterator it= map_match.begin();it != map_match.end();it++)
//            {
//                Scalar color(rng.uniform(0,255),rng.uniform(0,255),rng.uniform(0,255));
//                for(std::vector<linemod::Match>::iterator it_v= it->second.begin();it_v != it->second.end();it_v++)
//                    {
//                    cv::Rect rect_tmp= Rects_[it_v->template_id];
//                    rect_tmp.x=it_v->x;
//                    rect_tmp.y=it_v->y;
//                    rectangle(cluster_img,rect_tmp,color,2);
//                }
//            }
//            imshow("cluster",cluster_img);
//            cv::waitKey (0);

            //Filter based on size of clusters
            uchar thresh=1;
            t=cv::getTickCount ();
            rgbd_detector.cluster_filter(map_match,thresh);
            t=(cv::getTickCount ()-t)/cv::getTickFrequency ();
            //cout<<"Time consumed by cluster filter: "<<t<<" s"<<endl;

            //Display
//            std::map<std::vector<int>, std::vector<linemod::Match> >::iterator it_map=map_match.begin();
//            for(;it_map!=map_match.end();++it_map)
//            {
//                for(std::vector<linemod::Match>::iterator it_vec= it_map->second.begin();it_vec != it_map->second.end();it_vec++){
//                    std::vector<cv::linemod::Template> templates=detector->getTemplates(it_vec->class_id, it_vec->template_id);
//                    drawResponse(templates, 1, display,cv::Point(it_vec->x,it_vec->y), 2);
//                }
//            }
//            imshow("template clustering",display);
//            waitKey(1);

            t=cv::getTickCount ();
            //Use match similarity score as evaluation
            //rgbd_detector.cluster_scoring(map_match,mat_depth,cluster_data);
            Matx33d K_tmp=Matx<double,3,3>(571.9737, 0.0, 319.5000,
                                           0.0, 571.0073, 239.5000,
                                           0.0, 0.0, 1.0);
            Mat depth_tmp=depth_img.clone();
            rgbd_detector.cluster_scoring(renderer_iterator_,K_tmp,Rs_,Ts_,map_match,depth_tmp,cluster_data);
            t=(cv::getTickCount ()-t)/cv::getTickFrequency ();
            //cout<<"Time consumed by scroing: "<<t<<endl;

            //Non-maxima suppression
            t=cv::getTickCount ();
            //nonMaximaSuppression(cluster_data,5,map_match);
            rgbd_detector.nonMaximaSuppression(cluster_data,nms_radius,Rects_,map_match);
            //nonMaximaSuppression(cluster_data,nms_radius,Rects_,map_match);
            t=(cv::getTickCount ()-t)/cv::getTickFrequency ();
            //cout<<"Time consumed by non-maxima suppression: "<<t<<endl;

            //Display
            for(vector<ClusterData>::iterator iter=cluster_data.begin();iter!=cluster_data.end();++iter)
            {
                rectangle(nms_img,iter->rect,Scalar(0,0,255),2);
            }

            //Pose average
            t=cv::getTickCount ();
            rgbd_detector.getRoughPoseByClustering(cluster_data,pc_ptr,Rs_,Ts_,Distances_,Obj_origin_dists,orientation_clustering_th_,renderer_iterator_,renderer_focal_length_x,renderer_focal_length_y,image_width,bias_x);
            t=(cv::getTickCount ()-t)/cv::getTickFrequency ();
            //cout<<"Time consumed by rough pose estimation : "<<t<<endl;

            //vizResultPclViewer(cluster_data,pc_ptr);

            //Pose refinement
            t=cv::getTickCount ();
            //icpPoseRefine(cluster_data,pc_ptr,false);
            rgbd_detector.icpPoseRefine(cluster_data,icp,pc_ptr,image_width,bias_x,false);
            t=(cv::getTickCount ()-t)/cv::getTickFrequency ();
            cout<<"Time consumed by pose refinement: "<<t<<endl;

            //Hypothesis verification
            t=cv::getTickCount ();
            //hypothesisVerification(cluster_data,0.002,0.15);
            rgbd_detector.hypothesisVerification(cluster_data,0.004,collision_rate_threshold,false);
            t=(cv::getTickCount ()-t)/cv::getTickFrequency ();
            //cout<<"Time consumed by hypothesis verification: "<<t<<endl;

            //Display all the bounding box
            for(int ii=0;ii<cluster_data.size();++ii)
            {
                rectangle(final,cluster_data[ii].rect,Scalar(0,0,255),2);
            }

            //Viz
            imshow("original image",mat_rgb);
            imshow("intial results",initial_img);
            imshow("Non-maxima suppression",nms_img);
            imshow("final result",final);
            cv::waitKey (1);

            //vizResultPclViewer(cluster_data,pc_ptr);

        }

        void nonMaximaSuppression(vector<ClusterData>& cluster_data,const double& neighborSize)
        {
            vector<ClusterData> nms_cluster_data;            
            vector<ClusterData>::iterator it1=cluster_data.begin();
            double radius=neighborSize;

            pcl::PointCloud<pcl::PointXYZ> cluster_pc;
            for(;it1!=cluster_data.end();++it1)
            {
                pcl::PointXYZ point;
                point.x=it1->index[0];
                point.y=it1->index[1];
                point.z=0.0;
                cluster_pc.points.push_back(point);
                it1->is_checked=false;
            }
            //Create kd tree
//            pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr tree(new pcl::KdTreeFLANN<pcl::PointXYZ>);
//            tree->setInputCloud(cluster_pc.makeShared());
            //Initiate an octree for nearest neighbour searching
            pcl::octree::OctreePointCloudSearch<pcl::PointXYZ> tree (1);
            tree.setInputCloud(cluster_pc.makeShared());
            tree.addPointsFromInputCloud();

            //Search NN
            vector<pcl::PointXYZ,Eigen::aligned_allocator<pcl::PointXYZ> >::iterator it_pc =cluster_pc.points.begin();
            for(int i=0;it_pc != cluster_pc.points.end();++it_pc,++i)
            {
                //check if the cluster has been checked
                if(!cluster_data[i].is_checked)
                {
                    std::vector<int> pointIdxVec;
                    std::vector<float> squaredDistance;
                    pcl::PointXYZ search_point = *it_pc;
                    tree.radiusSearch(search_point,radius,pointIdxVec,squaredDistance);

                    //Extract neighbors using indices
                    vector<ClusterData> tmp_cluster;
                    for(std::vector<int>::iterator it_idx = pointIdxVec.begin();it_idx!=pointIdxVec.end();++it_idx)
                    {
                        if(!cluster_data[*it_idx].is_checked)
                        {
                            tmp_cluster.push_back(cluster_data[*it_idx]);
                        }

                        int p =0;
                    }

                    //Find out the one with maximum score
                    std::sort(tmp_cluster.begin(),tmp_cluster.end(),sortClusterBasedOnScore);

                    //Save the local maximum
                    nms_cluster_data.push_back(tmp_cluster[0]);


                    //Mark all of the NN as checked
                    for(std::vector<int>::iterator it_idx = pointIdxVec.begin();it_idx!=pointIdxVec.end();++it_idx)
                    {
                        cluster_data[*it_idx].is_checked=true;
                    }
                }

            }

            cluster_data.clear();
            cluster_data=nms_cluster_data;
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
                cout<<obj_pose_f.matrix()<<endl;
            }
            view.spin();
        }


        void poseComparision(const vector<ClusterData>& cluster_data)
        {
            //Open ground truth file
            string gr_file_path=gr_prefix + "poses" + gr_file_serial + ".txt";
            PointCloudXYZ groundTruth_position;
            openGroundTruthFile(gr_file_path,groundTruth_position);

            //Initiate an octree for nearest neighbour searching
            pcl::octree::OctreePointCloudSearch<pcl::PointXYZ> octree (0.001);
            octree.setInputCloud(groundTruth_position.makeShared());
            octree.addPointsFromInputCloud();

            //Due to the corresponding ground truth is unkown, we have to use octree to search it
            for(vector<ClusterData>::const_iterator it = cluster_data.begin();it!=cluster_data.end();++it)
            {
                //cout<<it->pose.matrix()<<endl;
                Eigen::Affine3d transform=Eigen::Affine3d::Identity();
                //In my setting, object coordinate is fixed to the center of bounding box, but in the setting of the dataset, the coordinates is fixed a certained point.
                    //So i need to transform the origin coordinate to coincide with the one from dataset.
                transform.translation()<<-0.000329264,0.000714317,0.0119547;
                Eigen::Affine3d fix_pose=it->pose*transform;

                pcl::PointXYZ position(fix_pose.translation()[0],fix_pose.translation()[1],fix_pose.translation()[2]);
                //pcl::PointXYZ position(it->pose.translation()[0],it->pose.translation()[1],it->pose.translation()[2]);
                std::vector<int> pointIdxVec;
                std::vector<float> squaredDistance;
                int k=1;
                octree.nearestKSearch(position,k,pointIdxVec,squaredDistance);
                double x_err=fabs(position.x-groundTruth_position.points[pointIdxVec[0]].x);
                double y_err=fabs(position.y-groundTruth_position.points[pointIdxVec[0]].y);
                double z_err=fabs(position.z-groundTruth_position.points[pointIdxVec[0]].z);
                double position_error=sqrt(squaredDistance[0]);
                //double position_error=fabs(sqrt(squaredDistance[0])-0.0119547);
                cout<<"Position X Error: "<<x_err<<endl;
                cout<<"Position Y Error: "<<y_err<<endl;
                cout<<"Position Z Error: "<<z_err<<endl;
                cout<<"Position Euclidean Error: "<<position_error<<endl;

                int p=0;

            }
        }

        void posePRAnalysis(string gr_file_path,double coeff, double model_diameter,int& TP,int& FP,int& FN,int& num_objs)
        {
            //Save ground truth to an eigen vector
            vector<Eigen::Affine3d> gr_poses;            
            openGroundTruthFile(gr_file_path,gr_poses,num_objs);

            //Save position of ground truth to a kd tree for indexing
            pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr position_tree(new pcl::KdTreeFLANN<pcl::PointXYZ>);
            saveToKdTree(gr_poses,position_tree);

            //For each estimated pose, assign the gr_poses: pair(estimated pose, ground truth)
            vector<pair<Eigen::Affine3d,Eigen::Affine3d> > pose_set;
            assignGRPose(cluster_data, gr_poses, position_tree, -0.000301369, -0.00420908, -0.00425946, pose_set);

            //trasnform model point cloud using estimated pose and ground truth pose
            vector<pair<PointCloudXYZ::Ptr,PointCloudXYZ::Ptr> > pc_set;
            for(int i=0;i<pose_set.size();++i)
            {
                PointCloudXYZ::Ptr est_pc(new PointCloudXYZ);
                PointCloudXYZ::Ptr gr_pc(new PointCloudXYZ);
                pcl::transformPointCloud(*model_pc, *est_pc, pose_set[i].first);
                pcl::transformPointCloud(*model_pc, *gr_pc, pose_set[i].second);

                pc_set.push_back(pair<PointCloudXYZ::Ptr,PointCloudXYZ::Ptr>(est_pc,gr_pc));
            }

            bool is_symmetry = true;

            if(is_symmetry)
            {
                //For each hypothese
                for(int i=0;i<pc_set.size();++i)
                {
                    //save ground truth point cloud to kd tree
                    pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr pc_tree(new pcl::KdTreeFLANN<pcl::PointXYZ>);
                    pc_tree->setInputCloud(pc_set[i].second);

                    //search nearest neighbor in estimated point cloud using kd tree
                    double avg_distance = 0.0;
                    for(vector<pcl::PointXYZ,Eigen::aligned_allocator<pcl::PointXYZ> >::iterator it = (pc_set[i].first)->points.begin(); it!= (pc_set[i].first)->points.end();++it)
                    {
                        pcl::PointXYZ est_point = *it;
                        std::vector<int> pointIdxVec;
                        std::vector<float> squaredDistance;
                        int k=1;
                        pc_tree->nearestKSearch(est_point,k,pointIdxVec,squaredDistance);
                        pcl::PointXYZ gr_point = pc_set[i].second->points[pointIdxVec[0]];

                        double tmp_distance = sqrt((est_point.x - gr_point.x)*(est_point.x - gr_point.x)
                                                   + (est_point.y - gr_point.y)*(est_point.y - gr_point.y)
                                                   +(est_point.z - gr_point.z)*(est_point.z - gr_point.z));

                        avg_distance += tmp_distance;
                    }
                    avg_distance /= pc_set[i].first->points.size();

                    //Evaluate the distance (TP? FP? FN?)
                    if(avg_distance < coeff*model_diameter)
                    {
                        TP++;
                    }
                    else
                    {
                        FP++;
                    }
                }

                //Normally tp would not be greater than 2
                if(TP>num_objs)
                {
                    FN=0;
                }
                else
                {
                    FN=num_objs-TP;
                }

            }
            else // non-symmetry object
            {
                //For each hypothese
                for(int i=0;i<pc_set.size();++i)
                {
                    vector<pcl::PointXYZ,Eigen::aligned_allocator<pcl::PointXYZ> >::iterator it_est = (pc_set[i].first)->points.begin();
                    vector<pcl::PointXYZ,Eigen::aligned_allocator<pcl::PointXYZ> >::iterator it_gr = (pc_set[i].second)->points.begin();
                    double avg_distance = 0.0;
                    for(; it_est!= (pc_set[i].first)->points.end();++it_est,++it_gr)
                    {
                        pcl::PointXYZ est_point = *it_est;
                        pcl::PointXYZ gr_point = *it_gr;

                        double tmp_distance = sqrt((est_point.x - gr_point.x)*(est_point.x - gr_point.x)
                                                   + (est_point.y - gr_point.y)*(est_point.y - gr_point.y)
                                                   +(est_point.z - gr_point.z)*(est_point.z - gr_point.z));

                        avg_distance += tmp_distance;
                    }
                    avg_distance /= pc_set[i].first->points.size();

                    //Evaluate the distance (TP? FP? FN?)
                    if(avg_distance < coeff*model_diameter)
                    {
                        TP++;
                    }
                    else
                    {
                        FP++;
                    }
                }


                //Normally tp would not be greater than 3
                if(TP>num_objs)
                {
                    FN=0;
                }
                else
                {
                    FN=num_objs-TP;
                }
            }

        }

        void assignGRPose(const vector<ClusterData>& cluster_data,const vector<Eigen::Affine3d>& gr_poses,pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr tree,double x_refinement,double y_refinement, double z_refinement,vector<pair<Eigen::Affine3d,Eigen::Affine3d> >& pose_set)
        {
            for(vector<ClusterData>::const_iterator it = cluster_data.begin();it!=cluster_data.end();++it)
            {
                Eigen::Affine3d transform=Eigen::Affine3d::Identity();
                //In my setting, object coordinate is fixed to the center of bounding box, but in the setting of the dataset, the coordinates is fixed a certained point.
                    //So i need to transform the origin coordinate to coincide with the one from dataset.
                transform.translation()<<x_refinement,y_refinement,z_refinement;
                Eigen::Affine3d refined_pose=it->pose*transform;

                //kd tree search nearest neighbor
                pcl::PointXYZ position(refined_pose.translation()[0],refined_pose.translation()[1],refined_pose.translation()[2]);
                //pcl::PointXYZ position(it->pose.translation()[0],it->pose.translation()[1],it->pose.translation()[2]);
                std::vector<int> pointIdxVec;
                std::vector<float> squaredDistance;
                int k=1;
                tree->nearestKSearch(position,k,pointIdxVec,squaredDistance);
                Eigen::Affine3d gr_pose = gr_poses[pointIdxVec[0]];
                int p=0;
                pose_set.push_back(pair<Eigen::Affine3d,Eigen::Affine3d>(refined_pose,gr_pose));
            }
        }

        void saveToKdTree(vector<Eigen::Affine3d>& gr_poses, pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr tree)
        {
            PointCloudXYZ::Ptr pc(new PointCloudXYZ);

            for(int i=0;i<gr_poses.size();++i)
            {
                pcl::PointXYZ tmp_pc(gr_poses[i].translation()[0],gr_poses[i].translation()[1],gr_poses[i].translation()[2]);
                pc->points.push_back(tmp_pc);
            }
            tree->setInputCloud(pc);
        }

        bool openGroundTruthFile(string gr_file_path, PointCloudXYZ& position)
        {
            //Initiate position of ground truth
            position.clear();

            ifstream file;
            file.open(gr_file_path.data());
            if(!file.is_open())
                return false;

            string str;
            int num_line=0;
            while(getline(file,str))
            {
                num_line++;
                if(num_line == 2 || num_line == 6 || num_line == 10)
                {
                    string::iterator it=str.end();
                    while(*it != ' ')
                        it--;
                    string x_str(it+1,str.end());

                    //Read y
                    getline(file,str);
                    num_line++;
                    it=str.end();
                    while(*it != ' ')
                        it--;
                    string y_str(it+1,str.end());

                    //Read z
                    getline(file,str);
                    num_line++;
                    it=str.end();
                    while(*it != ' ')
                        it--;
                    string z_str(it+1,str.end());

                    double x=0.0; double y=0.0; double z=0.0;
                    x=atof(x_str.data());
                    y=atof(y_str.data());
                    z=atof(z_str.data());

                    position.push_back(pcl::PointXYZ(x,y,z));


                }


            }

            return true;
        }

        bool checkGroundTruthFile(string gr_file_path)
        {
            ifstream file;
            file.open(gr_file_path.data());
            if(!file.is_open())
                {
                return false;
            }else
                {
                return true;
            }
        }

        bool openGroundTruthFile(string gr_file_path, vector<Eigen::Affine3d>& gr_poses,int& num_objs)
        {
            //Initiate position of ground truth
            gr_poses.clear();

            ifstream file;
            file.open(gr_file_path.data());
            if(!file.is_open())
                return false;

            string str,num_poses;
            int num_line=0;
            int count=0;


            while(getline(file,str))
            {
                num_line++;
                if(num_line == 1)
                    {
                    num_poses = str;
                    num_objs=atoi(str.data());
                }
                if(num_poses == "1")
                {
                    if(num_line == 1) //|| num_line == 9
                    {
                        Eigen::Affine3d tmp_pose = Eigen::Affine3d::Identity();
                        for(int i=0;i<3;++i)
                        {
                            getline(file,str);
                            num_line++;
                            string::iterator it=str.begin();
                            string::iterator it_end;
                            for(int j=0;j<4;++j)
                            {
                                while(*it == ' ')
                                    it++;

                                it_end = it;

                                if(j != 3)
                                {
                                    while(*it_end != ' ')
                                        it_end++;

                                }
                                else
                                {
                                    it_end=str.end();
                                }
                                string tmp_str =string(it,it_end);
                                tmp_pose.matrix()(i,j) = atof(tmp_str.data());
                                it=it_end;
                            }
                        }
                        gr_poses.push_back(tmp_pose);

                    }
                }

                if(num_poses == "2")
                {
                    if(num_line == 1 || num_line == 5) //|| num_line == 9
                    {
                        Eigen::Affine3d tmp_pose = Eigen::Affine3d::Identity();
                        for(int i=0;i<3;++i)
                        {
                            getline(file,str);
                            num_line++;
                            string::iterator it=str.begin();
                            string::iterator it_end;
                            for(int j=0;j<4;++j)
                            {
                                while(*it == ' ')
                                    it++;

                                it_end = it;

                                if(j != 3)
                                {
                                    while(*it_end != ' ')
                                        it_end++;

                                }
                                else
                                {
                                    it_end=str.end();
                                }
                                string tmp_str =string(it,it_end);
                                tmp_pose.matrix()(i,j) = atof(tmp_str.data());
                                it=it_end;
                            }
                        }
                        gr_poses.push_back(tmp_pose);

                    }
                }

                if(num_poses == "3")
                {
                    if(num_line == 1 || num_line == 5 || num_line == 9) //|| num_line == 9
                    {
                        Eigen::Affine3d tmp_pose = Eigen::Affine3d::Identity();
                        for(int i=0;i<3;++i)
                        {
                            getline(file,str);
                            num_line++;
                            string::iterator it=str.begin();
                            string::iterator it_end;
                            for(int j=0;j<4;++j)
                            {
                                while(*it == ' ')
                                    it++;

                                it_end = it;

                                if(j != 3)
                                {
                                    while(*it_end != ' ')
                                        it_end++;

                                }
                                else
                                {
                                    it_end=str.end();
                                }
                                string tmp_str =string(it,it_end);
                                tmp_pose.matrix()(i,j) = atof(tmp_str.data());
                                it=it_end;
                            }
                        }
                        gr_poses.push_back(tmp_pose);

                    }
                }

            }

            return true;
        }

        void loadModelPC(string path)
        {
            pcl::io::loadPLYFile(path,*model_pc);
        }

        void setTemplateMatchingThreshold(float threshold_)
        {
            linemod_match_threshold = threshold_;
        }

        void setNonMaximumSuppressionRadisu(double radius)\
        {
            nms_radius=radius;
        }

        void setHypothesisVerficationThreshold(float threshold_)
        {
            collision_rate_threshold = threshold_;
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
    double nms_radius;

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
    nms_radius=atof(argv[11]);

    linemod_detect detector(linemod_template_path,renderer_param_path,model_stl_path,
                            detect_score_th,icp_max_iter,icp_tr_epsilon,icp_fitness_th,icp_maxCorresDist,clustering_step,orientation_clustering_step);

    ensenso::RegistImage srv;
    srv.request.is_rgb=true;
    ros::Rate loop(1);

    //Read rgb and depth image
    serial_number=argv[12];

    string model_path = "/home/tom/ros/catkin_ws/src/linemod_pose_est/dataset/camera/PLY/camera_plain.ply";
    string dataset_prefix="/home/tom/ros/catkin_ws/src/linemod_pose_est/dataset/camera/";

    //Extract file number for later use
    //gr_file_serial=getGRfileSerial(img_path);

    //Load model pc
    detector.loadModelPC(model_path);
    detector.setNonMaximumSuppressionRadisu(nms_radius);


    //static const int arr[] = {95,93,91,89,86,83,80,75};
//    static const float arr[] = {0.0};
//    vector<float> thresh_set (arr, arr + sizeof(arr) / sizeof(arr[0]) );
    vector<float> thresh_set;
    double hv_thresh=0.3;
    thresh_set.push_back(hv_thresh);
    for(int i=0;i<10;++i)
    {
        hv_thresh+=0.05;
        thresh_set.push_back(hv_thresh);
    }

    //Create a txt file for data storage
    ofstream outfile;
    outfile.open("/home/tom/camera_HV_3_8_thresh90.txt");
    if(!outfile)
        return -1;

    //Loop over all images
    vector<string> rgb_filenames;
    vector<string> depth_filenames;
    cv::glob(dataset_prefix+"RGB/*.png",rgb_filenames);
    cv::glob(dataset_prefix+"Depth/*.png",depth_filenames);


    //int total_TP=0; int total_FP=0; int total_FN=0; int total_objs=0;
    for(int i=0;i<thresh_set.size();++i)
    {
        int total_TP=0; int total_FP=0; int total_FN=0; int total_objs=0;
        detector.setHypothesisVerficationThreshold(thresh_set[i]);
        for(int j=0;j<400;++j)
        {
            //Check if ground truth file exists
            gr_file_serial=getGRfileSerial(rgb_filenames[j]);
            string gr_file_path=dataset_prefix+"Annotation/"+"poses"+gr_file_serial+".txt";
            cout<<"Testing image: "<<gr_file_serial<<endl;
            if(detector.checkGroundTruthFile(gr_file_path))
            {

                cout<<"Image order: "<<j<<endl;
                Mat rgb_img=imread(rgb_filenames[j],IMREAD_COLOR);
                Mat depth_img=imread(depth_filenames[j],IMREAD_UNCHANGED);

                detector.detect_cb(rgb_img,depth_img);

                //Result analysis
                int TP=0; int FP=0; int FN=0; int num_objs=0;
                detector.posePRAnalysis(gr_file_path,0.15,0.1438541511,TP,FP,FN,num_objs);
                total_TP+=TP;
                total_FP+=FP;
                total_FN+=FN;
                total_objs+=num_objs;

                cout<<"TP: "<<TP<<"  FP: "<<FP<<"  FN: "<<FN<<"  threshold: "<<thresh_set[i]<<endl;
                cout<<"==================================================="<<endl;

            }
        }
        //Comptute Precision and Recall and F1
        double precision, recall, f1;
        precision=(double)total_TP/(double)(total_TP+total_FP);
        recall=(double)total_TP/(double)(total_TP+total_FN);
        f1 = 2*precision*recall/(precision+recall);

        //save to txt
        outfile<<thresh_set[i]<<" "<<total_objs<<" "<<total_TP<<" "<<total_FP<<" "<<total_FN<<" "<<precision<<" "<<recall<<" "<<f1;
        outfile<<"\n";        
        cout<<"Precision: "<<precision<<" Recall: "<<recall<<" F1: "<<f1<<endl;
    }

    outfile.close();
    ros::shutdown();
}
