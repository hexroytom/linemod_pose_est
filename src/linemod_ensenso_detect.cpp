#include <linemod_pose_estimation/rgbdDetector.h>
#include <pcl/visualization/pcl_visualizer.h>
//ros
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/CameraInfo.h>
#include <tf/transform_broadcaster.h>
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

    //Offset for compensating cropped image
    int bias_x;
    rgbdDetector::IMAGE_WIDTH image_width;

    //Wrapper for key methods
    rgbdDetector rgbd_detector;


public:
    linemod_detect(std::string template_file_name,std::string renderer_params_name,std::string mesh_path,float detect_score_threshold,int icp_max_iter,float icp_tr_epsilon,float icp_fitness_threshold, float icp_maxCorresDist, uchar clustering_step,float orientation_clustering_th):
        it(nh),
        sub_color(nh,"/camera/rgb/image_rect_color",1),
        sub_depth(nh,"/camera/depth_registered/image_raw",1),
        depth_frame_id_("camera_link"),
        sync(SyncPolicy(1), sub_color, sub_depth),
        px_match_min_(0.25f),
        icp_dist_min_(0.06f),
        bias_x(56),
        image_width(rgbdDetector::ENSENSO),
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
        //Publisher for visualize pointcloud in Rviz
        pointcloud_publisher scene_pc_pub(nh,string("/rgbDetect/scene"));
        pointcloud_publisher model_pc_pub(nh,string("/rgbDetect/pick_object"));
        pointcloud_publisher scene_cropped_pc_pub(nh,string("/rgbDetect/scene_roi"));

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
        cv::Rect crop(bias_x,0,640,480);
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
        rgbd_detector.linemod_detection(detector,sources,threshold,matches);
        t=(cv::getTickCount ()-t)/cv::getTickFrequency ();

        //Display all the results
        //            for(std::vector<linemod::Match>::iterator it= matches.begin();it != matches.end();it++){
        //                std::vector<cv::linemod::Template> templates=detector->getTemplates(it->class_id, it->template_id);
        //                drawResponse(templates, 1, display,cv::Point(it->x,it->y), 2);
        //            }

        //Clustering based on Row Col Depth
        std::map<std::vector<int>, std::vector<linemod::Match> > map_match;
        rgbd_detector.rcd_voting(Obj_origin_dists,renderer_radius_min,clustering_step_,renderer_radius_step,matches,map_match);

        //Filter based on size of clusters
        uchar thresh=5;
        rgbd_detector.cluster_filter(map_match,thresh);

        //Compute criteria for each cluster
        //Output: Vecotor of ClusterData, each element of which contains index, score, flag of checking.
        vector<ClusterData> cluster_data;
        t=cv::getTickCount ();
        rgbd_detector.cluster_scoring(map_match,mat_depth,cluster_data);
        t=(cv::getTickCount ()-t)/cv::getTickFrequency ();
        cout<<"Time consumed by scroing: "<<t<<endl;

        //Non-maxima suppression
        rgbd_detector.nonMaximaSuppression(cluster_data,10,Rects_,map_match);

        //Pose average
        t=cv::getTickCount ();
        rgbd_detector.getRoughPoseByClustering(cluster_data,pc_ptr,Rs_,Ts_,Distances_,Obj_origin_dists,orientation_clustering_th_,renderer_iterator_,renderer_focal_length_x,renderer_focal_length_y,image_width,bias_x);
        t=(cv::getTickCount ()-t)/cv::getTickFrequency ();
        cout<<"Time consumed by pose clustering: "<<t<<endl;

        //Pose refinement
        t=cv::getTickCount ();
        rgbd_detector.icpPoseRefine(cluster_data,icp,pc_ptr,image_width,bias_x,false);
        t=(cv::getTickCount ()-t)/cv::getTickFrequency ();
        cout<<"Time consumed by pose refinement: "<<t<<endl;

        //Hypothesis verification
        rgbd_detector.hypothesisVerification(cluster_data,0.002,0.17);

        //Display all the bounding box
        scene_pc_pub.publish (pc_ptr);
        for(vector<ClusterData>::iterator it = cluster_data.begin();it!=cluster_data.end();++it)
        {
            rectangle(display,it->rect,Scalar(0,0,255),2);
            model_pc_pub.publish (it->model_pc,it->pose,cv::Scalar(255,0,0));
            scene_cropped_pc_pub.publish(it->scene_pc,it->pose,cv::Scalar(0,255,0));
            imshow("display",display);
            cv::waitKey (0);
        }



        //Viz in point cloud
        //vizResultPclViewer(cluster_data,pc_ptr);

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

    ros::NodeHandle& getNodeHandle()
    {
        return nh;
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
