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
#include <Eigen/Dense>

//opencv
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/rgbd/rgbd.hpp>

//PCL
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/filters/filter.h>

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

struct match_xyz{
    match_xyz(const cv::linemod::Match& match,const cv::Vec3f& position)
    {
        match_=match;
        position_=position;
    }
    cv::linemod::Match match_;
    cv::Vec3f position_;
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

    int vote_thresh;

public:
        linemod_detect(std::string template_file_name,std::string renderer_params_name,std::string mesh_path,float detect_score_threshold,float clustering_threshold,int icp_max_iter,float icp_tr_epsilon,float icp_fitness_threshold):
            it(nh),
            sub_color(nh,"/camera/rgb/image_rect_color",1),
            sub_depth(nh,"/camera/depth_registered/image_raw",1),
            depth_frame_id_("camera_rgb_optical_frame"),
            sync(SyncPolicy(1), sub_color, sub_depth),
            px_match_min_(0.25f),
            icp_dist_min_(0.06f),
            vote_thresh(6)
        {
            //pub test
            //pub_color_=it.advertise ("/sync_rgb",2);
            //pub_depth=it.advertise ("/sync_depth",2);

            //the intrinsic matrix
            sub_cam_info=nh.subscribe("/camera/depth/camera_info",1,&linemod_detect::read_cam_info,this);

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

            sync.registerCallback(boost::bind(&linemod_detect::detect_cb,this,_1,_2));
        }

        void detect_cb(const sensor_msgs::ImageConstPtr& msg_rgb , const sensor_msgs::ImageConstPtr& msg_depth)
        {
            //Read camera intrinsic params
            if(!is_K_read)
                return;
            //If LINEMOD detector is not loaded correctly, return
            if(detector->classIds ().empty ())
            {
                ROS_INFO("Linemod detector is empty");
                return;
            }

            //Get LINEMOD source image
            vector<Mat> sources;
            rosMsgs_to_linemodSources(msg_rgb,msg_depth,sources);
            Mat mat_rgb,mat_depth;
            mat_rgb=sources[0];
            mat_depth=sources[1];

            //Perform the LINEMOD detection
            std::vector<linemod::Match> matches;
            double t=cv::getTickCount ();
            detector->match (sources,threshold,matches,std::vector<String>(),noArray());
            t=(cv::getTickCount ()-t)/cv::getTickFrequency ();
            cout<<"Time consumed by template matching: "<<t<<" s"<<endl;
            cout<<"(1) LINEMOD Matching Result: "<< matches.size()<<endl;

            //Viz all the detected result
            Mat display=sources[0];
            Mat display_remaind_match;
            display.convertTo(display_remaind_match,CV_8UC3);

             //If no match is found, return
             if(matches.size()==0)
                 return;

//----------------------Distance inconsisitency filter------------------------------//
//             double average_distance=0.0;
//             BOOST_FOREACH(const linemod::Match& match,matches){
//                 float template_distance=(Obj_origin_dists[match.template_id]-Distances_[match.template_id])*1000;
//                 double scene_distance=mat_depth.at<ushort>(match.y,match.x);
//                 double depth_inconsist=abs(template_distance-scene_distance);
//                 average_distance+=depth_inconsist;
//             }
//             average_distance=average_distance/matches.size();
//             cout<<average_distance<<endl;

             pci_real_icpin_model->clear();
             pci_real_icpin_ref->clear();

//----------------3D Voting-----------------------------------------------------//
             std::map<std::vector<int>, std::vector<linemod::Match> > map_match;
             int vote_row_col_step=4;
             double vote_depth_step=renderer_radius_step;
             int voting_height_cells,voting_width_cells;
             rcd_voting(vote_row_col_step, vote_depth_step, matches,map_match, voting_height_cells, voting_width_cells);

             //Filter the clusters with votes fewer than [vote_thresh]
             std::map<std::vector<int>, std::vector<linemod::Match> > map_filter_match;
             //create a map for non-maxima suppression
             Mat NMS_map=Mat::zeros(voting_height_cells,voting_width_cells,CV_8UC1);
             std::vector<std::pair<int,int> > NMS_row_col_index;
             Mat NMS_viz_before=cv::Mat::zeros(NMS_map.rows,NMS_map.cols,CV_8UC1);
             Mat NMS_viz_after=cv::Mat::zeros(NMS_map.rows,NMS_map.cols,CV_8UC1);
             for(std::map<std::vector<int>, std::vector<linemod::Match> >::iterator it =map_match.begin();it!=map_match.end();++it)
             {
                 if(it->second.size()>vote_thresh)
                 {
                     map_filter_match.insert(*it);
                     int row_index=(it->first)[0];
                     int col_index=(it->first)[1];
                     NMS_map.at<uchar>(row_index,col_index)=NMS_map.at<uchar>(row_index,col_index)+it->second.size();

                     //for visualization
                     NMS_viz_before.at<uchar>(row_index,col_index)=255;
                     NMS_viz_after.at<uchar>(row_index,col_index)=255;

                     NMS_row_col_index.push_back(std::pair<int,int>(row_index,col_index));
                 }
             }

             //Viz the result after vote thresholding
             for(std::map<std::vector<int>, std::vector<linemod::Match> >::iterator it1 = map_filter_match.begin();it1 != map_filter_match.end();++it1)
             {
                 for(std::vector<linemod::Match>::iterator it2= it1->second.begin();it2 != it1->second.end();it2++)
                 {
                     std::vector<cv::linemod::Template> templates=detector->getTemplates(it2->class_id, it2->template_id);
                     drawResponse(templates, 1, display,cv::Point(it2->x,it2->y), 2,3);
                 }
             }
             imshow("display",display);
             std::cout<<"(2) RCD clustering suppression result: "<< map_filter_match.size()<<std::endl;

             int NMS_neigh=6;
             //Implement non-maxima suppression
             for(std::vector<std::pair<int,int> >::iterator it = NMS_row_col_index.begin();it!=NMS_row_col_index.end();++it)
                {
                 int i=it->first;
                 int j=it->second;
                 int max_vote=NMS_map.at<uchar>(i,j);
                 for(int m=-NMS_neigh;m<=NMS_neigh;m++)
                 {
                    bool is_break=false;
                    for(int n=-NMS_neigh;n<=NMS_neigh;n++)
                       {
                          if(i+m >= 0 && i+m < NMS_map.rows && j+n >= 0 && j+n < NMS_map.cols)
                            {
                              if((NMS_map.at<uchar>(i+m,j+n)>=max_vote))
                                {
                                   if(i+m == i && j+n ==j){}
                                   else{
                                        NMS_map.at<uchar>(i,j)=0;
                                        NMS_viz_after.at<uchar>(i,j)=0;
                                        is_break=true;
                                        }
//                                    break;
                                }
                            }
                       }
//                    if(is_break)
//                        break;
                  }


                 }

             imshow("before",NMS_viz_before);
             imshow("after",NMS_viz_after);
             //cv::waitKey(0);

             //Filter the non-maxima clusters
                //loop over the NMS map
             std::map<std::vector<int>, std::vector<linemod::Match> > map_nms_match;
             for(int i=0;i<NMS_map.rows;++i)
             {
                 for(int j=0;j<NMS_map.cols;++j)
                 {
                     int vote= NMS_map.at<uchar>(i,j);
                     if(vote>0)
                    {
                     int NMS_row_index=i;
                     int NMS_col_index=j;
                     for(std::map<std::vector<int>, std::vector<linemod::Match> >::iterator it2=map_filter_match.begin();it2!=map_filter_match.end();++it2)
                         {
                         int row_index=(it2->first)[0];
                         int col_index=(it2->first)[1];;
                         if(row_index == NMS_row_index && col_index == NMS_col_index)
                         {
                             map_nms_match.insert(*it2);
                         }
                     }
                    }

                 }

             }



             //Viz the remain matches
             for(std::map<std::vector<int>, std::vector<linemod::Match> >::iterator it1 = map_nms_match.begin();it1 != map_nms_match.end();++it1)
             {
                 for(std::vector<linemod::Match>::iterator it2= it1->second.begin();it2 != it1->second.end();it2++)
                 {
                     std::vector<cv::linemod::Template> templates=detector->getTemplates(it2->class_id, it2->template_id);
                     drawResponse(templates, 1, display_remaind_match,cv::Point(it2->x,it2->y), 2,4);
                 }
             }
             std::cout<<"(3) Non-maxima suppression result: "<< map_nms_match.size()<<std::endl;
             imshow("display_remain",display_remaind_match);
             cv::waitKey (0);

             //XYZ space voting
                //convert the depth image to 3d pointcloud
             cv::Mat_<cv::Vec3f> depth_real_ref_raw;
             cv::depthTo3d(mat_depth, K_depth, depth_real_ref_raw);

             float xyz_voting_step=0.005; //Unit: m
             for(std::map<std::vector<int>, std::vector<linemod::Match> >::iterator it1 = map_filter_match.begin();it1 != map_filter_match.end();++it1)
             {
                 //Map for storing vector of match and its XYZ. Note that size of the vector denotes the votes for the cell
                 std::map<vector<int>,vector<match_xyz> > xyz_accumulator;
                 for(std::vector<linemod::Match>::iterator it2= it1->second.begin();it2 != it1->second.end();it2++)
                 {
                     cv::Vec3f template_center=depth_real_ref_raw.at<cv::Vec3f>(it2->y,it2->x);
                     //If it is a nan point, continue
                     if(!cv::checkRange(template_center))
                         continue;
                     cv::Vec3f obj_center=template_center+cv::Vec3f(0,0,Distances_[it2->template_id]);
                     vector<int> index(3);
                     for(int i=0;i<3;++i)
                        index[i]=obj_center[i]/xyz_voting_step;
                     //Store the index to the map
                     if(xyz_accumulator.find(index) == xyz_accumulator.end())
                     {
                         vector<match_xyz> tmp_vec;
                         tmp_vec.push_back(match_xyz(*it2,obj_center));
                         xyz_accumulator.insert(pair<vector<int>,vector<match_xyz> >(index,tmp_vec));
                     }
                     else
                     {
                         xyz_accumulator[index].push_back(match_xyz(*it2,obj_center));
                     }


                 }
                 int p=0;
             }

             return;

             //get matches according to the index
                //access the 1st element just for compling...
             std::vector<linemod::Match> match_result;
             std::map<std::vector<int>, std::vector<linemod::Match> >::iterator iter =map_match.begin();
             match_result=iter->second;

             //only withdraw the first match of the matches
             linemod::Match match=match_result[0];

//-----------------------------------------------------------------------------//

//-----------------------------2D Voting---------------------------------------//
//             unsigned int *accumulator =(unsigned int*)calloc (64*48,sizeof(unsigned int));
//             std::map<unsigned int, std::vector<linemod::Match> > map_match;
//             int max_index_xy=0;
//             int max_vote_xy=0;
//             BOOST_FOREACH(const linemod::Match& match,matches){
//                              int row_index=match.x/10;//10  step
//                              int col_index=match.y/10;
//                              int index=row_index*64+col_index;
//                              accumulator[index]++;
//                              if(map_match.find (index)==map_match.end ())
//                              {
//                                  std::vector<linemod::Match> temp;
//                                  temp.push_back (match);
//                                  map_match.insert(pair<unsigned int,std::vector<linemod::Match> >(index,temp));
//                              }
//                              else
//                              {
//                                  map_match[index].push_back(match);
//                              }

//                          }

//             for(int i=0;i<48*64;++i)
//             {
//                 if(accumulator[i]>max_vote_xy)
//                 {
//                     max_vote_xy=accumulator[i];
//                     max_index_xy=i;
//                 }

//             }

//             //get matches according to the index
//             std::vector<linemod::Match> xyVote_result;
//             xyVote_result=map_match[max_index_xy];

//            unsigned int *dist_accumulator =(unsigned int*)calloc (100,sizeof(unsigned int));
//            std::map<unsigned int, std::vector<linemod::Match> > map_dist_match;
//            int max_index_dist=0;
//            int max_vote_dist=0;
//            BOOST_FOREACH(const linemod::Match& match, xyVote_result){
//                float D_match=Distances_[match.template_id];
//                int dist_index=D_match/10;//10 distance step
//                dist_accumulator[dist_index]++;

//                if(map_dist_match.find (dist_index)==map_dist_match.end ())
//                {
//                    std::vector<linemod::Match> temp;
//                    temp.push_back (match);
//                    map_dist_match.insert(pair<unsigned int,std::vector<linemod::Match> >(dist_index,temp));
//                }
//                else
//                {
//                    map_dist_match[dist_index].push_back(match);
//                }
//            }

//            for(int i=0;i<100;++i)
//            {
//                if(dist_accumulator[i]>max_vote_dist)
//                {
//                    max_vote_dist=dist_accumulator[i];
//                    max_index_dist=i;
//                }

//            }

//            //get matches according to the index
//            std::vector<linemod::Match> distVote_result;
//            distVote_result=map_dist_match[max_index_dist];

//            //only withdraw the first match of the matches
//            linemod::Match match=distVote_result[0];

//------------------------------------------------------------------------------------//
             //deal with the candidates from the above linemod match
                 pci_real_condition_filter_model->clear ();
                 //get the pose
                 cv::Matx33d R_match = Rs_[match.template_id].clone();// rotation of the object w.r.t to the view point
                 cv::Vec3d T_match = Ts_[match.template_id].clone();//the translation of the camera with respect to the current view point
                 cv::Mat K_matrix= Ks_[match.template_id].clone();
                 float D_match = Distances_[match.template_id];//the distance from the center of object surface to the camera origin

                 //get the point cloud of the rendered object model
                 cv::Mat mask;
                 cv::Rect rect;
                 cv::Matx33d R_temp(R_match.inv());//rotation of the viewpoint w.r.t the object
                 cv::Vec3d up(-R_temp(0,1), -R_temp(1,1), -R_temp(2,1));//the negative direction of y axis of viewpoint w.r.t object frame
                 cv::Mat depth_ref_;
                 renderer_iterator_->renderDepthOnly(depth_ref_, mask, rect, -T_match, up);//up?

                 cv::Mat_<cv::Vec3f> depth_real_model_raw;
                 cv::depthTo3d(depth_ref_, K_matrix, depth_real_model_raw);//unit mm to m

                 //prepare the bounding box for the model and reference point clouds
                 cv::Rect_<int> rect_model(0, 0, depth_real_model_raw.cols, depth_real_model_raw.rows);

                 //prepare the bounding box for the reference point cloud: add the offset
                 cv::Rect_<int> rect_ref(rect_model);
                 rect_ref.x += match.x;
                 rect_ref.y += match.y;

                 //compute the intersection between template and real scene image
                 rect_ref = rect_ref & cv::Rect(0, 0, depth_real_ref_raw.cols, depth_real_ref_raw.rows);
                 if ((rect_ref.width < 5) || (rect_ref.height < 5))
                   return;
                 //adjust both rectangles to be equal to the smallest among them
                 if (rect_ref.width > rect_model.width)
                   rect_ref.width = rect_model.width;
                 if (rect_ref.height > rect_model.height)
                   rect_ref.height = rect_model.height;
                 if (rect_model.width > rect_ref.width)
                   rect_model.width = rect_ref.width;
                 if (rect_model.height > rect_ref.height)
                   rect_model.height = rect_ref.height;

                 //prepare the reference data: from the sensor : crop images
                 cv::Mat_<cv::Vec3f> depth_real_ref = depth_real_ref_raw(rect_ref);
                 //prepare the model data: from the match
                 cv::Mat_<cv::Vec3f> depth_real_model = depth_real_model_raw(rect_model);

                 //initialize the translation based on reference data
                 cv::Vec3f T_crop = depth_real_ref(depth_real_ref.rows / 2.0f, depth_real_ref.cols / 2.0f);
                 T_ref=T_crop;

               //first filter step
                 //reject if the depth of the templates exceed the ref depth to a certain extent
//                 float center_depth_model=(depth_real_model(depth_real_model.rows/2,depth_real_model.cols/2))[2];
//                 float center_depth_ref=(depth_real_ref(depth_real_ref.rows / 2.0f, depth_real_ref.cols / 2.0f))[2];
//                 //float depth_th=(renderer_radius_step/1000)*;
//                 float depth_th=center_depth_ref*0.2;
//                 if(fabs(center_depth_model-center_depth_ref)>depth_th)
//                     return;

              //second filter step
                 //using depth to filter scene points: only those points with depth within a range will survive
                    //calculate Z range
//                 float Z_th=T_crop(2);
//                 cv::Mat_<cv::Vec3f>::iterator it_real_model=depth_real_model.begin ();
//                 float Z_min=10000;
//                 float Z_max=0.0;
//                 float Z_range=0.0;
//                 for(;it_real_model!=depth_real_model.end ();++it_real_model)
//                 {
//                     if(!cv::checkRange (*it_real_model))
//                         continue;
//                     //cv::Vec3f tmp=*it_real_model;
//                     if((*it_real_model)[2]<Z_min)
//                         Z_min=(*it_real_model)[2];
//                     if((*it_real_model)[2]>Z_max)
//                        Z_max=(*it_real_model)[2];
//                 }
//                 Z_range=Z_max-Z_min;
//                    //start filtering
//                 cv::Mat_<cv::Vec3f>::iterator it_real_ref= depth_real_ref.begin ();
//                 cv::Mat_<cv::Vec3f> depth_real_ref_filter;
//                 for(;it_real_ref!=depth_real_ref.end ();++it_real_ref)
//                 {
//                     cv::Vec3f ref_tmp=*it_real_ref;
//                     if(cv::checkRange (ref_tmp))
//                     {
//                         if(fabs(ref_tmp[2]-Z_th)<Z_range)
//                             depth_real_ref_filter.push_back(ref_tmp);
//                     }

//                 }

                 //add the object's depth
                 T_crop(2) += D_match;
                 if (!cv::checkRange(T_crop))
                   return;
                 cv::Vec3f T_real_icp(T_crop);

                 //initialize the rotation based on model data
                 //R_match orientation of obj w.r.t view frame(camera frame)
                 if (!cv::checkRange(R_match))
                   return;
                 cv::Matx33f R_real_icp(R_match);

                 //get the point clouds (for both reference and model)
                 std::vector<cv::Vec3f> pts_real_model_temp;
                 std::vector<cv::Vec3f> pts_real_ref_temp;
                 float px_ratio_missing = matToVec(depth_real_ref, depth_real_model, pts_real_ref_temp, pts_real_model_temp);

                 //before using ICP
//                 pci_real_nonICP_model->fill (pts_real_model_temp,cv::Vec3b(255,0,0));
//                 pci_real_nonICP_model->publish ();

                //third filter step
                 //reject if the points of model are too less regard to scene points
                 if (px_ratio_missing > (1.0f-px_match_min_))
                   return;

                 //add the Translation vector according to the position of point cloud of interest(ref points)

                 cv::Vec3f T_vect;
                 //Origin is set in the geometry center
                 cv::Vec3f T_real_model=depth_real_model(depth_real_model.rows/2,depth_real_model.cols/2);

                 T_vect[0]=T_ref[0]-T_real_model[0];
                 T_vect[1]=T_ref[1]-T_real_model[1];
                 T_vect[2]=T_ref[2]-T_real_model[2];
                 transformPoints(pts_real_model_temp, pts_real_model_temp, R_diag, T_vect);
                 //update the position
                 T_real_model[2]+=D_match; //attention! if Origin is in the center,please uncomment this
                 T_real_model+=T_vect;
                 T_real_icp=T_real_model;

                 pci_real_condition_filter_model->fill (pts_real_model_temp,cv::Vec3b(255,0,0));
                 pci_real_condition_filter_model->publish ();
                 //transform the format
                 objs_.push_back(LinemodData(pts_real_ref_temp, pts_real_model_temp, match.class_id,match.template_id, match.similarity,cv::Point(match.x,match.y), 0.0, R_real_icp, T_real_icp));

             //ICP refinement
             LinemodData *o_match = &(objs_[0]);
             pcl::PointCloud<pcl::PointXYZ>::Ptr ref_pc(new pcl::PointCloud<pcl::PointXYZ>);
             pcl::PointCloud<pcl::PointXYZ>::Ptr model_pc(new pcl::PointCloud<pcl::PointXYZ>);
             //cv::Vec3f T_vect;
             Matx33f R_tmp;
             ref_pc->points.resize (o_match->pts_ref.size ());
             model_pc->points.resize(o_match->pts_model.size());

             for(int i=0;i<o_match->pts_ref.size();++i)
             {
                 ref_pc->points[i].x=o_match->pts_ref[i][0];
                 ref_pc->points[i].y=o_match->pts_ref[i][1];
                 ref_pc->points[i].z=o_match->pts_ref[i][2];
             }

             for(int i=0;i<o_match->pts_model.size();++i)
             {
                 model_pc->points[i].x=o_match->pts_model[i][0];
                 model_pc->points[i].y=o_match->pts_model[i][1];
                 model_pc->points[i].z=o_match->pts_model[i][2];
             }

             //perform icp
             //std::vector<int> index;
             icp.setInputSource (model_pc);
             icp.setInputTarget (ref_pc);
             //pcl::removeNaNFromPointCloud(*ref_pc,*ref_pc,index);
             //pcl::removeNaNFromPointCloud(*model_pc,*model_pc,index);
             icp.align (*model_pc);
             float icp_distance_=(icp.getFitnessScore ())/(o_match->pts_model.size());

             if (icp.hasConverged ())
             {
               Eigen::Matrix4d transformation_matrix = icp.getFinalTransformation ().cast<double>();
               for(int i=0;i<3;++i)
               {
                   for(int j=0;j<3;++j)
                   {
                       R_tmp(i,j)=transformation_matrix(i,j);
                   }
               }
               T_vect[0]=transformation_matrix(0,3);
               T_vect[1]=transformation_matrix(1,3);
               T_vect[2]=transformation_matrix(2,3);

               //update the pointcloud
               transformPoints(o_match->pts_model, o_match->pts_model, R_tmp, T_vect);
               //update the translation vector
               o_match->t = R_tmp * o_match->t;
               cv::add(T_vect, o_match->t, o_match->t);
               //update the rotation matrix
               o_match->r = R_tmp * o_match->r;
               //update the icp dist
               o_match->icp_distance=icp_distance_;

             //add points to the clouds
             pci_real_icpin_model->fill(o_match->pts_model, cv::Vec3b(0,255,0));
             pci_real_icpin_ref->fill(o_match->pts_ref, cv::Vec3b(0,0,255));
             pci_real_icpin_model->publish();
             pci_real_icpin_ref->publish();

             final_poses.push_back (*o_match);
             //std::cout<<"get final result"<<std::endl;

              }
             else
                 return;


             // broadcast TFs
             std::vector<LinemodData>::iterator it_pose=final_poses.begin ();
             //LinemodData best_match=*it_pose;
             for(;it_pose!=final_poses.end();++it_pose)
             {
                 cv::Vec3f position=it_pose->t;
                 cv::Matx33f orientation=it_pose->r;
                 tf::Transform transform;
                 tf::Matrix3x3 orientation3x3_tf;
                 transform.setOrigin (tf::Vector3(position[0],position[1],position[2]));
                 orientation3x3_tf.setValue (orientation(0,0),orientation(0,1),orientation(0,2),
                                          orientation(1,0),orientation(1,1),orientation(1,2),
                                          orientation(2,0),orientation(2,1),orientation(2,2));
                 tf::Quaternion orientationQua_tf;
                 orientation3x3_tf.getRotation (orientationQua_tf);
                 transform.setRotation (orientationQua_tf);
                 tf_broadcaster.sendTransform (tf::StampedTransform(transform,ros::Time::now(),"camera_rgb_optical_frame","coke_frame"));

             }

             //std::cout << "rough matches:" << matches.size()<<std::endl;
             objs_.clear ();
             final_poses.clear ();
             //publish the point clouds

        }

        void rosMsgs_to_linemodSources(const sensor_msgs::ImageConstPtr& msg_rgb , const sensor_msgs::ImageConstPtr& msg_depth,vector<Mat>& sources)
        {
            //Convert ros msg to OpenCV mat
            cv_bridge::CvImagePtr img_ptr_rgb;
            cv_bridge::CvImagePtr img_ptr_depth;
            Mat mat_rgb;
            Mat mat_depth;

            try{
                 img_ptr_depth = cv_bridge::toCvCopy(*msg_depth);
             }
             catch (cv_bridge::Exception& e)
             {
                 ROS_ERROR("cv_bridge exception:  %s", e.what());
                 return;
             }
             try{
                 img_ptr_rgb = cv_bridge::toCvCopy(*msg_rgb, sensor_msgs::image_encodings::BGR8);
             }
             catch (cv_bridge::Exception& e)
             {
                 ROS_ERROR("cv_bridge exception:  %s", e.what());
                 return;
             }

             if(img_ptr_depth->image.depth ()==CV_32F)
             {
                img_ptr_depth->image.convertTo (mat_depth,CV_16UC1,1000.0);
             }
             else
             {
                 img_ptr_depth->image.copyTo(mat_depth);
             }


             if(img_ptr_rgb->image.rows>960)
             {
                 cv::pyrDown (img_ptr_rgb->image.rowRange (0,960),mat_rgb);
             }
             else{
                  mat_rgb = img_ptr_rgb->image;
             }


             //test
//             cout<<"Before conversion"<<endl;
//             MatIterator_<float> it =img_ptr_depth->image.begin<float>();
//             for(;it!=img_ptr_depth->image.end<float>();++it)
//                 cout<<*it<<endl;

//             cout<<"After conversion"<<endl;
//             MatIterator_<ushort> it2 =mat_depth.begin<ushort>();
//             for(;it2!=mat_depth.end<ushort>();++it2)
//                 cout<<*it2<<endl;

            //set ROI
//             cv::Mat Mask=cv::Mat::zeros (480,640,CV_8UC1);
//             cv::Rect roi_rect=cv::Rect(125,0,391,326);//baxter table 0,55,635,239
//             Mask(roi_rect)=255;
//             std::vector<cv::Mat> masks;
//             masks.push_back (Mask);
//             masks.push_back (Mask);
             sources.push_back(mat_rgb);
             sources.push_back(mat_depth);
        }

        void rcd_voting(const int& vote_row_col_step,const double& renderer_radius_step_,const vector<linemod::Match>& matches,std::map<std::vector<int>, std::vector<linemod::Match> >& map_match,int& voting_height_cells,int& voting_width_cells)
        {
            //----------------3D Voting-----------------------------------------------------//

            int position_voting_width=renderer_width; //640 or 752
            int position_voting_height=renderer_height; //480
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

        //Get corresponding rotation matrix with the input of clusters of linemod match
        void get_rotation_clusters(vector<vector<linemod::Match> >& match_clusters,vector<vector<Eigen::Matrix3d> >& eigen_rot_mat_clusters)
        {
            for(vector<vector<linemod::Match> >::iterator it1 = match_clusters.begin();it1 != match_clusters.end();++it1)
            {
                vector<Eigen::Matrix3d> tmp_matries;
                for(std::vector<linemod::Match>::iterator it2= it1->begin();it2 != it1->end();it2++)
                {
                    Eigen::Matrix3d tmp_matrix;
                    cv2eigen(Rs_[it2->template_id],tmp_matrix);
                    tmp_matries.push_back(tmp_matrix);
                }
                eigen_rot_mat_clusters.push_back(tmp_matries);
            }

            //Convert rotation matrix to Z-Y-X euler angles if needed
            vector<vector<Eigen::Vector3d> > eigen_euler_clusters;
            for(vector<vector<Eigen::Matrix3d> >::iterator it1 = eigen_rot_mat_clusters.begin();it1 != eigen_rot_mat_clusters.end();++it1)
            {
                vector<Eigen::Vector3d> tmp_euler_angles;
                for(std::vector<Eigen::Matrix3d>::iterator it2= it1->begin();it2 != it1->end();it2++)
                {
                    Eigen::Vector3d tmp_euler;
                    tmp_euler=it2->eulerAngles(2,1,0);
                    tmp_euler_angles.push_back(tmp_euler);
                }
                eigen_euler_clusters.push_back(tmp_euler_angles);
            }

            //Extract z axis component of rotation matrix
//             vector<vector<Eigen::Vector3d> > eigen_zAxis_pose_clusters;
//             for(vector<vector<Eigen::Matrix3d> >::iterator it1 = eigen_rot_mat_clusters.begin();it1 != eigen_rot_mat_clusters.end();++it1)
//             {
//                 vector<Eigen::Vector3d> tmp_zAxis;
//                 for(std::vector<Eigen::Matrix3d>::iterator it2= it1->begin();it2 != it1->end();it2++)
//                 {
//                     Eigen::Vector3d tmp_z;
//                     if((*it2)(1,2)<0.0)
//                     {
//                         tmp_z[0]=-(*it2)(0,2);
//                         tmp_z[1]=-(*it2)(1,2);
//                         tmp_z[2]=-(*it2)(2,2);
//                     }else
//                         {
//                         tmp_z[0]=(*it2)(0,2);
//                         tmp_z[1]=(*it2)(1,2);
//                         tmp_z[2]=(*it2)(2,2);
//                     }
//                     tmp_zAxis.push_back(tmp_z);
//                 }
//                 eigen_zAxis_pose_clusters.push_back(tmp_zAxis);
//             }

            //Construct ZYX voting space
//             double orientation_voting_range=2*CV_PI; //-pi to pi
//             double angle_step=60.0/180.0*CV_PI;
//             for(vector<vector<Eigen::Vector3d> >::iterator it1 = eigen_euler_clusters.begin();it1 != eigen_euler_clusters.end();++it1)
//             {
//                 std::map<vector<int>, std::vector<Eigen::Vector3d> > map_ZYX_euler;
//                 for(std::vector<Eigen::Vector3d>::iterator it2= it1->begin();it2 != it1->end();it2++)
//                 {
//                     vector<int> euler_index(3);
//                     euler_index[0]=(int)((*it2)[0]/angle_step);
//                     euler_index[1]=(int)((*it2)[1]/angle_step);
//                     euler_index[2]=(int)((*it2)[2]/angle_step);
//                     if(map_ZYX_euler.find(euler_index)==map_ZYX_euler.end())
//                     {
//                        vector<Eigen::Vector3d> temp;
//                        temp.push_back(*it2);
//                        map_ZYX_euler.insert(pair<vector<int>,vector<Eigen::Vector3d> >(euler_index,temp));
//                     }
//                     else
//                     {
//                        map_ZYX_euler[euler_index].push_back(*it2);
//                     }

//                 }

//                 int p=0;
//             }
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
                     int T,int color_select)
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
            cv::Scalar color = COLORS[m+color_select];

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
        linemod_template_path="/home/yake/catkin_ws/src/linemod_pose_est/config/data/pipe_linemod_carmine_templates.yml";
        renderer_param_path="/home/yake/catkin_ws/src/linemod_pose_est/config/data/pipe_linemod_carmine_renderer_params.yml";
        model_stl_path="/home/yake/catkin_ws/src/linemod_pose_est/config/stl/pipe_connector.stl";
        detect_score_th=89.0;
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
    linemod_detect detector(linemod_template_path,
                            renderer_param_path,
                            model_stl_path,
                            detect_score_th,
                            clustering_th,
                            icp_max_iter,icp_tr_epsilon,icp_fitness_th);
    ros::spin ();
}
