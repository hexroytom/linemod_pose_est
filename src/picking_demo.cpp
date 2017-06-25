#include <linemod_pose_estimation/rgbdDetector.h>

//ork
#include <object_recognition_renderer/renderer3d.h>
#include <object_recognition_renderer/utils.h>

//ensenso
#include <ensenso/RegistImage.h>
#include <ensenso/CaptureSinglePointCloud.h>

//boost
#include <boost/foreach.hpp>

//std
#include <math.h>

//moveit
#include <moveit/move_group_interface/move_group.h>
#include <moveit/planning_interface/planning_interface.h>
#include <moveit/trajectory_processing/iterative_time_parameterization.h>
#include <moveit_msgs/ExecuteKnownTrajectory.h>
#include <moveit_msgs/DisplayTrajectory.h>

typedef pcl::PointCloud<pcl::PointXYZ> PointCloudXYZ;

//namespace
using namespace cv;
using namespace std;

class linemod_detect
{

    ros::NodeHandle nh;
    image_transport::ImageTransport it;
    ros::Subscriber sub_cam_info;
    ros::Publisher pc_rgb_pub_;
    ros::Publisher extract_pc_pub;
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

    std::string depth_frame_id_;

    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;

    tf::TransformBroadcaster tf_broadcaster;

    //Service client
    ros::ServiceClient ensenso_registImg_client;
    ros::ServiceClient ensenso_singlePc_client;

    //Offset for compensating cropped image
    int bias_x;
    rgbdDetector::IMAGE_WIDTH image_width;

    //Wrapper for key methods
    rgbdDetector rgbd_detector;

    //tf from rgb camera to depth camera
    Eigen::Affine3d pose_rgbTdep;

    //tf from depth camera to rgb camera
    Eigen::Affine3d pose_depTrgb;

    //tf from tool0 to depth camera
    Eigen::Affine3d pose_tool0Tdep;

    //Move group
    boost::shared_ptr<moveit::planning_interface::MoveGroup> group;

    //UR script publisher
    ros::Publisher ur_script_pub;

    //Non maximum suppression neighbor
    double nms_radius;


public:
    linemod_detect(std::string template_file_name,std::string renderer_params_name,std::string mesh_path,float detect_score_threshold,int icp_max_iter,float icp_tr_epsilon,float icp_fitness_threshold, float icp_maxCorresDist, uchar clustering_step,float orientation_clustering_th):
        it(nh),
        depth_frame_id_("camera_link"),
        px_match_min_(0.25f),
        icp_dist_min_(0.06f),
        bias_x(56),
        image_width(rgbdDetector::ENSENSO),
        clustering_step_(clustering_step),
        nms_radius(4)
    {
        //Publisher
        pc_rgb_pub_= nh.advertise<sensor_msgs::PointCloud2>("ensenso_rgb_pc",1);
        extract_pc_pub = nh.advertise<sensor_msgs::PointCloud2>("/render_pc",1);
        ur_script_pub = nh.advertise<std_msgs::String>("/ur_driver/URScript",1);

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

        //Moveit
        group.reset();
        group = boost::make_shared<moveit::planning_interface::MoveGroup>("manipulator");

        group->setPoseReferenceFrame("/base");
        group->setMaxVelocityScalingFactor(0.3);
        group->setMaxAccelerationScalingFactor(0.3);

    }

    virtual ~linemod_detect()
    {
    }

    void detect_cb(const sensor_msgs::Image& msg_rgb,sensor_msgs::PointCloud2& pc,bool is_rgb,vector<ClusterData>& cluster_data)
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
        Mat before_nms=mat_rgb_crop.clone();
        Mat after_nms=mat_rgb_crop.clone();
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
//        for(std::vector<linemod::Match>::iterator it= matches.begin();it != matches.end();it++){
//            std::vector<cv::linemod::Template> templates=detector->getTemplates(it->class_id, it->template_id);
//            drawResponse(templates, 1, display,cv::Point(it->x,it->y), 2);
//        }

        //Clustering based on Row Col Depth
        std::map<std::vector<int>, std::vector<linemod::Match> > map_match;
        rgbd_detector.rcd_voting(Obj_origin_dists,renderer_radius_min,clustering_step_,renderer_radius_step,matches,map_match);

        //Filter based on size of clusters
        uchar thresh=2;
        rgbd_detector.cluster_filter(map_match,thresh);

        //Compute criteria for each cluster
        //Output: Vecotor of ClusterData, each element of which contains index, score, flag of checking.
        t=cv::getTickCount ();
        rgbd_detector.cluster_scoring(map_match,mat_depth,cluster_data);
        t=(cv::getTickCount ()-t)/cv::getTickFrequency ();
        cout<<"Time consumed by scroing: "<<t<<endl;

        for(std::map<std::vector<int>, std::vector<linemod::Match> >::iterator it= map_match.begin();it != map_match.end();it++)
        {
            for(std::vector<linemod::Match>::iterator it2=it->second.begin();it2!=it->second.end();++it2)
            {
                std::vector<cv::linemod::Template> templates=detector->getTemplates(it2->class_id, it2->template_id);
                drawResponse(templates, 1, before_nms,cv::Point(it2->x,it2->y), 2);
            }
        }
        imshow("before nms",before_nms);
        waitKey(1);

        //Non-maxima suppression
        rgbd_detector.nonMaximaSuppression(cluster_data,nms_radius,Rects_,map_match);

        //Display all the bounding box
        for(int ii=0;ii<cluster_data.size();++ii)
        {
            rectangle(after_nms,cluster_data[ii].rect,Scalar(0,0,255),2);
        }
        imshow("after nms",after_nms);
        waitKey(1);

        //Pose refine for one candidates
        vector<ClusterData> tmp;
        for(int k=0;k<cluster_data.size();++k)
        {
            tmp.clear();
            tmp.push_back(cluster_data[k]);

            //Pose average
            t=cv::getTickCount ();
            rgbd_detector.getRoughPoseByClustering(tmp,pc_ptr,Rs_,Ts_,Distances_,Obj_origin_dists,orientation_clustering_th_,renderer_iterator_,renderer_focal_length_x,renderer_focal_length_y,image_width,bias_x);
            t=(cv::getTickCount ()-t)/cv::getTickFrequency ();
            cout<<"Time consumed by pose clustering: "<<t<<endl;

            //Pose refinement
            t=cv::getTickCount ();
            rgbd_detector.icpPoseRefine(tmp,icp,pc_ptr,image_width,bias_x,true);
            t=(cv::getTickCount ()-t)/cv::getTickFrequency ();
            cout<<"Time consumed by pose refinement: "<<t<<endl;


            //Hypothesis verification
            rgbd_detector.hypothesisVerification(tmp,0.004,0.30,true);

            if(tmp.size() == 1)
            {
                break;
            }
        }
        cluster_data.clear();
        cluster_data=tmp;

//        //Pose average
//        t=cv::getTickCount ();
//        rgbd_detector.getRoughPoseByClustering(cluster_data,pc_ptr,Rs_,Ts_,Distances_,Obj_origin_dists,orientation_clustering_th_,renderer_iterator_,renderer_focal_length_x,renderer_focal_length_y,image_width,bias_x);
//        t=(cv::getTickCount ()-t)/cv::getTickFrequency ();
//        cout<<"Time consumed by pose clustering: "<<t<<endl;

//        //Pose refine for all candidates
//        t=cv::getTickCount ();
//        rgbd_detector.icpPoseRefine(cluster_data,icp,pc_ptr,image_width,bias_x,false);
//        t=(cv::getTickCount ()-t)/cv::getTickFrequency ();
//        cout<<"Time consumed by pose refinement: "<<t<<endl;

//        //Hypothesis verification
//        rgbd_detector.hypothesisVerification(cluster_data,0.002,0.10,false);

        //Display all the bounding box
        for(int ii=0;ii<cluster_data.size();++ii)
        {
            rectangle(display,cluster_data[ii].rect,Scalar(0,0,255),2);
        }

        cv::startWindowThread();
        namedWindow("display");
        imshow("display",display);
        cv::waitKey (1);
//        destroyWindow("display");
//        cv::waitKey (1);

        //Viz in point cloud
        //vizResultPclViewer(cluster_data,pc_ptr);

        return;

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

    Eigen::Affine3d getGraspingPose_Tpipe(ClusterData& target)
    {
        //Init
        Eigen::Affine3d grasp_pose;
        double offset = 0.0;
        Eigen::Vector3d grasp_position(target.pose.translation()[0],target.pose.translation()[1],target.pose.translation()[2]);
        Eigen::Vector4f obj_z_axis(target.pose.matrix()(0,2),target.pose.matrix()(1,2),target.pose.matrix()(2,2),0.0);
        Eigen::Vector4f cam_z_axis(0.0,0.0,1.0,0.0);
        Eigen::Vector4f cam_y_axis(0.0,1.0,0.0,0.0);
        Eigen::Vector4f obj_y_axis(target.pose.matrix()(0,1),target.pose.matrix()(1,1),target.pose.matrix()(2,1),0.0);

        double theta1= pcl::getAngle3D(obj_z_axis,cam_z_axis);
        theta1=theta1/M_PI*180.0;

        //Object faces downward
        if(theta1 <= 50.0 && theta1 >=0.0)
        {
           offset = 0.022;
           grasp_position[0] = grasp_position[0] + offset*obj_z_axis[0]*(-1.0);
           grasp_position[1] = grasp_position[1] + offset*obj_z_axis[1]*(-1.0);
           grasp_position[2] = grasp_position[2] + offset*obj_z_axis[2]*(-1.0);

//           grasp_position[0] = grasp_position[0] + 0.008*obj_y_axis[0];
//           grasp_position[1] = grasp_position[1] + 0.008*obj_y_axis[1];
//           grasp_position[2] = grasp_position[2] + 0.008*obj_y_axis[2];

           grasp_pose = target.pose;
           grasp_pose.translation() << grasp_position[0],grasp_position[1],grasp_position[2];
        }//Object faces upward
        else if(theta1 <= 180.0 && theta1 >= 130)
        {
            offset = 0.022;
            //position
            grasp_position[0] = grasp_position[0] + offset*obj_z_axis[0];
            grasp_position[1] = grasp_position[1] + offset*obj_z_axis[1];
            grasp_position[2] = grasp_position[2] + offset*obj_z_axis[2];

//            grasp_position[0] = grasp_position[0] + 0.008*obj_y_axis[0];
//            grasp_position[1] = grasp_position[1] + 0.008*obj_y_axis[1];
//            grasp_position[2] = grasp_position[2] + 0.008*obj_y_axis[2];

            grasp_pose = target.pose;
            grasp_pose.translation() << grasp_position[0],grasp_position[1],grasp_position[2];
            //orientation
            grasp_pose *= Eigen::AngleAxisd(3.14,Eigen::Vector3d(0.0,1.0,0.0));


        }else
            {
            double theta= pcl::getAngle3D(obj_y_axis,cam_z_axis);
            theta=theta/M_PI*180.0;
            if(theta>90)
            {
                offset = 0.008;
                //position
                grasp_position[0] = grasp_position[0] + offset*obj_y_axis[0];
                grasp_position[1] = grasp_position[1] + offset*obj_y_axis[1];
                grasp_position[2] = grasp_position[2] + offset*obj_y_axis[2];
                grasp_pose = target.pose;
                grasp_pose.translation() << grasp_position[0],grasp_position[1],grasp_position[2];
                //orientation
                grasp_pose *= Eigen::AngleAxisd(1.57,Eigen::Vector3d(1.0,0.0,0.0));
            }else
            {
                offset = 0.01;
                //position
                grasp_position[0] = grasp_position[0] + offset*obj_y_axis[0];
                grasp_position[1] = grasp_position[1] + offset*obj_y_axis[1];
                grasp_position[2] = grasp_position[2] + offset*obj_y_axis[2];
                grasp_pose = target.pose;
                grasp_pose.translation() << grasp_position[0],grasp_position[1],grasp_position[2];
                //orientation
                grasp_pose *= Eigen::AngleAxisd(-1.57,Eigen::Vector3d(1.0,0.0,0.0));
            }

        }

        //Viz for test
//        tf::Transform grasp_pose_tf_viz;
//        tf::poseEigenToTF(grasp_pose,grasp_pose_tf_viz);
//        tf::TransformBroadcaster tf_broadcaster;
//        tf_broadcaster.sendTransform (tf::StampedTransform(grasp_pose_tf_viz,ros::Time::now(),"camera_link","grasp_frame"));

        return grasp_pose;
    }

    Eigen::Affine3d getGraspingPose_pipe(ClusterData& target)
    {
        //Init
        Eigen::Affine3d grasp_pose;
        double offset = 0.0;
        Eigen::Vector3d grasp_position(target.pose.translation()[0],target.pose.translation()[1],target.pose.translation()[2]);
        Eigen::Vector4f obj_z_axis(target.pose.matrix()(0,2),target.pose.matrix()(1,2),target.pose.matrix()(2,2),0.0);
        Eigen::Vector4f cam_z_axis(0.0,0.0,1.0,0.0);
        Eigen::Vector4f cam_y_axis(0.0,1.0,0.0,0.0);
        Eigen::Vector4f obj_y_axis(target.pose.matrix()(0,1),target.pose.matrix()(1,1),target.pose.matrix()(2,1),0.0);

        double theta1= pcl::getAngle3D(obj_z_axis,cam_z_axis);
        theta1=theta1/M_PI*180.0;

        grasp_pose = target.pose;

        if(theta1>90.0)
            {
            //Rotate
            grasp_pose *= Eigen::AngleAxisd(M_PI,Eigen::Vector3d(0.0,1.0,0.0));

            offset = 0.022;
            grasp_position[0] = grasp_position[0] + offset*grasp_pose.matrix()(0,2)*(-1.0);
            grasp_position[1] = grasp_position[1] + offset*grasp_pose.matrix()(1,2)*(-1.0);
            grasp_position[2] = grasp_position[2] + offset*grasp_pose.matrix()(2,2)*(-1.0);
        }else
            {
            offset = 0.022;
            grasp_position[0] = grasp_position[0] + offset*grasp_pose.matrix()(0,2)*(-1.0);
            grasp_position[1] = grasp_position[1] + offset*grasp_pose.matrix()(1,2)*(-1.0);
            grasp_position[2] = grasp_position[2] + offset*grasp_pose.matrix()(2,2)*(-1.0);
        }
        grasp_pose.translation()<<grasp_position[0],grasp_position[1],grasp_position[2];



        //Viz for test
//        tf::Transform grasp_pose_tf_viz;
//        tf::poseEigenToTF(grasp_pose,grasp_pose_tf_viz);
//        tf::TransformBroadcaster tf_broadcaster;
//        tf_broadcaster.sendTransform (tf::StampedTransform(grasp_pose_tf_viz,ros::Time::now(),"camera_link","grasp_frame"));

        return grasp_pose;
    }

    Eigen::Affine3d transformPose(Eigen::Affine3d& pose_rgbTgrasp)
    {
        //Get tf from BASE to TOOL0
        tf::TransformListener listener;
        tf::StampedTransform transform_stamped;
        ros::Time now(ros::Time::now());
        listener.waitForTransform("base","tool0",now,ros::Duration(1.5));
        listener.lookupTransform("base","tool0",ros::Time(0),transform_stamped);
        Eigen::Affine3d pose_baseTtool0;
        tf::poseTFToEigen(transform_stamped,pose_baseTtool0);

        //Get tf from BASE to OBJECT
        Eigen::Affine3d pose_baseTgrasp;        
        pose_baseTgrasp = pose_baseTtool0 * pose_tool0Tdep * pose_rgbTgrasp;
        return pose_baseTgrasp;
    }

    Eigen::Affine3d getRGBtoDepthTF(double x, double y,double z, double roll,double pitch,double yaw)
    {
        Eigen::Affine3d pose_rgbTdep_;

        //Translation
        pose_rgbTdep_.translation()<< x/1000.0,y/1000.0,z/1000.0;

        //Rotation
            //Z-Y-X euler ---> yaw-pitch-roll
        Eigen::Matrix3d rotation_matrix;
        rotation_matrix = Eigen::AngleAxisd(yaw/180.0*M_PI,Eigen::Vector3d(0.0,0.0,1.0)) *
                          Eigen::AngleAxisd(pitch/180.0*M_PI,Eigen::Vector3d(0.0,1.0,0.0)) *
                          Eigen::AngleAxisd(roll/180.0*M_PI,Eigen::Vector3d(1.0,0.0,0.0));
        pose_rgbTdep_.linear()=rotation_matrix;

        return pose_rgbTdep_;
    }

    Eigen::Affine3d getTool0toDepthTF(double x, double y,double z, double qw,double qx,double qy,double qz)
    {
        Eigen::Affine3d pose_Tool0Tdep_;

        //Translation
        pose_Tool0Tdep_.translation()<< x,y,z;

        //Rotation
        Eigen::Quaterniond quat(qw,qx,qy,qz);
        pose_Tool0Tdep_.linear() = quat.toRotationMatrix();

        return pose_Tool0Tdep_;
    }

    void broadcastTF(Eigen::Affine3d& pose_eigen, const string& parent_frame, const string& child_frame)
    {
        tf::Transform pose_tf;
        tf::poseEigenToTF(pose_eigen,pose_tf);
        tf::TransformBroadcaster tf_broadcaster;
        tf_broadcaster.sendTransform (tf::StampedTransform(pose_tf,ros::Time::now(),parent_frame,child_frame));

    }

    //Notice: these parameters stands for TF from rgb camera to depth camera
    void setDepthToRGB_broadcastTF(double x, double y,double z, double roll,double pitch,double yaw)
    {
       //Conversion
       pose_rgbTdep = getRGBtoDepthTF(x,y,z,roll,pitch,yaw);
       pose_depTrgb = pose_rgbTdep.inverse();
       //Broadcaster
       broadcastTF(pose_depTrgb,"camera_link","rgb_camera_link");
    }

    void setTool0tDepth_broadcastTF(double x, double y,double z, double qw,double qx,double qy,double qz)
    {
        //Conversion
        pose_tool0Tdep = getTool0toDepthTF(x,y,z,qw,qx,qy,qz);
        //Broadcaster
        broadcastTF(pose_tool0Tdep,"tool0","camera_link");
    }

    //linear interpolation for approaching the taerger. Base frame: /base.
    bool linear_trajectory_planning(Eigen::Affine3d grasp_pose, double line_offset, double num_of_interpolation, moveit_msgs::RobotTrajectory& target_traj )
    {
        tf::Transform grasp_pose_tf;
        tf::poseEigenToTF(grasp_pose,grasp_pose_tf);

        //Z axis
        tf::Matrix3x3 rot_mat=grasp_pose_tf.getBasis();
        tf::Vector3 direc = rot_mat.getColumn(2);;
        double step = line_offset/num_of_interpolation;

        //Interpolate n points
        std::vector<geometry_msgs::Pose> waypoints(num_of_interpolation+1);
        geometry_msgs::Pose grasp_pose_msg;
        tf::poseTFToMsg(grasp_pose_tf,grasp_pose_msg);
        for(int i=0;i<waypoints.size()-1;++i)
        {
            geometry_msgs::Pose tmp_pose;
            tmp_pose.position.x = grasp_pose_msg.position.x - step * direc[0] * (num_of_interpolation-i);
            tmp_pose.position.y = grasp_pose_msg.position.y - step * direc[1] * (num_of_interpolation-i);
            tmp_pose.position.z = grasp_pose_msg.position.z - step * direc[2] * (num_of_interpolation-i);
            tmp_pose.orientation = grasp_pose_msg.orientation;
            waypoints[i]=tmp_pose;
        }
        waypoints[num_of_interpolation]=grasp_pose_msg;

        //ComputeCartesian...
        double score=group->computeCartesianPath(waypoints,0.05,0.0,target_traj);


        if(score > 0.95)
        {
            moveit::core::RobotStatePtr kinematic_state(group->getCurrentState());
            robot_trajectory::RobotTrajectory rt(kinematic_state->getRobotModel(),"manipulator");
            rt.setRobotTrajectoryMsg(*kinematic_state,target_traj);
            trajectory_processing::IterativeParabolicTimeParameterization iptp;

            if(iptp.computeTimeStamps(rt,0.3,0.3))
                {

                rt.getRobotTrajectoryMsg(target_traj);
                return true;
            }

            return true;
        }
        else
        {
            ROS_ERROR("Cartessian Planning fail!");
            return false;
        }


    }

    void moveToPrePickPose(Eigen::Affine3d& grasp_pose,double z_offset)
    {
        geometry_msgs::Pose goalPose;

        //Orientation
        tf::TransformListener listener;
        tf::StampedTransform transform_stamped;
        tf::Transform transform_tmp;
        ros::Time now(ros::Time::now());
        listener.waitForTransform("base","suction_cup",now,ros::Duration(1.5));
        listener.lookupTransform("base","suction_cup",ros::Time(0),transform_stamped);
        transform_tmp= transform_stamped;
        tf::poseTFToMsg(transform_tmp,goalPose);

        //Position
        goalPose.position.x=grasp_pose.translation()[0];
        goalPose.position.y=grasp_pose.translation()[1];
        goalPose.position.z=grasp_pose.translation()[2]+z_offset;

        //Move
        ros::AsyncSpinner spinner(1);
        spinner.start();

        group->setMaxVelocityScalingFactor(0.8);
        group->setMaxAccelerationScalingFactor(0.8);
        group->setPoseTarget(goalPose);
        moveit::planning_interface::MoveGroup::Plan planner;
        bool is_success=group->plan(planner);
        if(is_success)
        {
            group->move();
            sleep(1);
        }else{
            cout<<"Move to pre-pick pose: Planning fail!"<<endl;
        }
        spinner.stop();
    }

    void moveToPrePlacePose(Eigen::Affine3d& grasp_pose,double z_offset)
    {
//        tf::Transform pose_tf;
//        geometry_msgs::Pose pose_msg;
//        tf::poseEigenToTF(grasp_pose,pose_tf);
//        tf::poseTFToMsg(pose_tf,pose_msg);

        geometry_msgs::Pose pose_msg;
        Eigen::Quaterniond quat(grasp_pose.linear());

        pose_msg.position.x= grasp_pose.translation()[0];
        pose_msg.position.y= grasp_pose.translation()[1];
        pose_msg.position.z= grasp_pose.translation()[2]+z_offset;
        pose_msg.orientation.w=quat.w();
        pose_msg.orientation.x=quat.x();
        pose_msg.orientation.y=quat.y();
        pose_msg.orientation.z=quat.z();

        //Move
        ros::AsyncSpinner spinner(1);
        spinner.start();
        group->setMaxVelocityScalingFactor(0.3);
        group->setMaxAccelerationScalingFactor(0.3);
        group->setPoseTarget(pose_msg);
        moveit::planning_interface::MoveGroup::Plan planner;
        bool is_success=group->plan(planner);
        if(is_success)
        {
            group->move();
            sleep(0.5);
        }else{
            cout<<"Move to pre-place pose: Planning fail!"<<endl;
        }
        spinner.stop();


    }

    void moveToPickTargetPose(const moveit_msgs::RobotTrajectory& robot_traj)
    {
        //Moveit
        ros::AsyncSpinner spinner(1);
        spinner.start();

        group->setMaxVelocityScalingFactor(0.6);
        group->setMaxAccelerationScalingFactor(0.6);

        moveit::planning_interface::MoveGroup::Plan planner;
        planner.trajectory_=robot_traj;
        planner.trajectory_.joint_trajectory.points[0].time_from_start=ros::Duration(0);

        //Viz for test
        //        moveit_msgs::DisplayTrajectory display_traj;
        //        display_traj.trajectory_start=planner.start_state_;
        //        display_traj.trajectory.push_back(planner.trajectory_);
        //        traj_publisher.publish(display_traj);

        group->execute(planner);
        //sleep(1.5);
        spinner.stop();
    }

    void moveToLookForTargetsPose(double speed_scale_factor,double time)
    {
        ros::AsyncSpinner spinner(1);
        spinner.start();

        group->setMaxVelocityScalingFactor(speed_scale_factor);
        group->setMaxAccelerationScalingFactor(speed_scale_factor);
        group->setNamedTarget("look_for_targets");
        group->move();
        sleep(time);

        spinner.stop();
    }

    void moveToPLaceTargetsPose(Eigen::Affine3d& grasp_pose, double offset)
    {
        ros::AsyncSpinner spinner(1);
        spinner.start();

        moveit::planning_interface::MoveGroup::Plan planner;
        bool is_success;

//        tf::Transform pose_tf;
//        geometry_msgs::Pose pose_msg;
//        tf::poseEigenToTF(grasp_pose,pose_tf);
//        tf::poseTFToMsg(pose_tf,pose_msg);

//        Eigen::Vector3d z_dir(grasp_pose.matrix()(0,2),grasp_pose.matrix()(1,2),grasp_pose.matrix()(2,2));
//        pose_msg.position.x-=offset*z_dir[0];
//        pose_msg.position.y-=offset*z_dir[1];
//        pose_msg.position.z-=offset*z_dir[2];

//        //Move
//        group->setMaxVelocityScalingFactor(0.4);
//        group->setMaxAccelerationScalingFactor(0.4);
//        group->setPoseTarget(pose_msg);
//        is_success=group->plan(planner);
//        if(is_success)
//        {
//            group->move();
//            sleep(1);
//        }else{
//            cout<<"Move to pre-place pose: Planning fail!"<<endl;
//        }

        //group->setNamedTarget("place_target");
        vector<double> joint_angles(6);
        joint_angles[0]=-16.34/180.0*M_PI;
       joint_angles[1]=-93.10/180.0*M_PI;
        joint_angles[2]=-63.82/180.0*M_PI;
        joint_angles[3]=-115.20/180.0*M_PI;
        joint_angles[4]=85.06/180.0*M_PI;
        joint_angles[5]=-57.22/180.0*M_PI;
        group->setJointValueTarget(joint_angles);        
        is_success=group->plan(planner);
        if(is_success)
        {
            group->move();
            //sleep(0.5);
        }else{
            cout<<"Move to place pose: Planning fail!"<<endl;
        }
        sleep(0.2);
        spinner.stop();
    }

    void pickTarget()
    {
       std_msgs::String cmd;
       cmd.data="set_digital_out(0,True)";
       ur_script_pub.publish(cmd);
       sleep(1);
    }

    void pushTarget(const Eigen::Affine3d& grasp_pose, double push_offset)
    {
        //Params
        Eigen::Vector3d direc(grasp_pose.matrix()(0,2),grasp_pose.matrix()(1,2),grasp_pose.matrix()(2,2));
        Eigen::Affine3d push_pose;
        tf::Transform pose_tf;
        geometry_msgs::Pose pose_msg;
        moveit_msgs::RobotTrajectory traj;
        moveit::planning_interface::MoveGroup::Plan planner;
        //ros::AsyncSpinner spinner(1);

        //Define push pose
        push_pose = grasp_pose;
        push_pose.translation()[0]+= push_offset*direc[0];
        push_pose.translation()[1]+= push_offset*direc[1];
        push_pose.translation()[2]+= push_offset*direc[2];
        linear_trajectory_planning(push_pose,push_offset,5,traj);


        //Move
        moveToPickTargetPose(traj);
    }

    void placeTarget()
    {
       std_msgs::String cmd;
       cmd.data="set_digital_out(0,False)";
       ur_script_pub.publish(cmd);
       sleep(1);
    }

    void getImages(ensenso::RegistImage& srv)
    {
        ensenso_registImg_client.call(srv);
        int p=0;

    }

    void setNonMaximumSuppressionRadisu(double radius)\
    {
        nms_radius=radius;
    }

};

int main(int argc,char** argv)
{
    ros::init (argc,argv,"linemod_detect");

    //Load Linemod parameters
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
    double nms_neighbor_size;

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
    nms_neighbor_size=atof(argv[11]);

    linemod_detect detector(linemod_template_path,renderer_param_path,model_stl_path,
                            detect_score_th,icp_max_iter,icp_tr_epsilon,icp_fitness_th,icp_maxCorresDist,clustering_step,orientation_clustering_step);

    ensenso::RegistImage srv;
    srv.request.is_rgb=true;
   //-----------------------------------images from dataset----------------------------------//
    string img_path=argv[12];
    string pc_path=argv[13];
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

    //Params
    float region_growing_normal_thresh=25.0; //degree
    float region_growing_curvature_thresh=2.0;

    //Publisher for visualize pointcloud in Rviz
    pointcloud_publisher scene_pc_pub(detector.getNodeHandle(),string("/rgbDetect/scene"));
    pointcloud_publisher model_pc_pub(detector.getNodeHandle(),string("/rgbDetect/pick_object"));

    //Bradocast tf from rgb to depth
    detector.setDepthToRGB_broadcastTF(-54.23,43.00,-25.59,0.25,-0.32,1.31);
    detector.setTool0tDepth_broadcastTF(0.0652032, -0.060422, 0.0464063,0.707464, -0.00131294, 0.0106396, 0.706668);

    //Robot initial pose
    detector.moveToLookForTargetsPose(0.3,1.0);

    //Set NMS Param
    detector.setNonMaximumSuppressionRadisu(nms_neighbor_size);


    string cmd;
    while(ros::ok())
    {
        cout<<"Start a new detetcion? Input [y] to begin, or [n] to quit. "<<endl;
//        cin>>cmd;
        cmd = "y";
        if(cmd == "y")
        {
            detector.getImages(srv);

            //Object detection
            vector<ClusterData> targets;
            detector.detect_cb(srv.response.image,srv.response.pointcloud,srv.request.is_rgb,targets);

            if(targets.size() == 0)
            {
                cout<<"No object is found."<<endl;

            }

            //Select one object to pick
            //scene_pc_pub.publish(srv.response.pointcloud);
            for(vector<ClusterData>::iterator it_target=targets.begin();it_target!=targets.end();++it_target)
            {
                //Grasping pose generation
                double t=cv::getTickCount ();
                Eigen::Affine3d grasp_pose_pDepth;
                detector.rgbd_detector.graspingPoseBasedOnRegionGrowing (it_target->dense_scene_pc,it_target->scene_pc,region_growing_normal_thresh,region_growing_curvature_thresh,0.0,grasp_pose_pDepth,true);
                t=(cv::getTickCount()-t)/cv::getTickFrequency ();
                cout<<"Time consumed by grasping pose generation: "<<t<<endl;

                //model_pc_pub.publish(it_target->scene_pc,it_target->pose,cv::Scalar(255,0,0));

                //Transform grasping pose to robot frame
                Eigen::Affine3d grasp_pose_pBase = detector.transformPose(grasp_pose_pDepth);

                //Visuailze the grapsing pose
                //                    tf::Transform grasp_pose_tf_viz;
                //                    tf::poseEigenToTF(grasp_pose_pBase,grasp_pose_tf_viz);
                //                    tf::TransformBroadcaster tf_broadcaster;
                //                    tf_broadcaster.sendTransform (tf::StampedTransform(grasp_pose_tf_viz,ros::Time::now(),"base","grasp_frame"));

                //Move to prepick pose
                detector.moveToPrePickPose(grasp_pose_pBase,0.10);

                //Trajectory from pre-pick to pick
                moveit_msgs::RobotTrajectory traj;
                bool is_succeed=detector.linear_trajectory_planning(grasp_pose_pBase,0.05,10,traj);
                if(is_succeed)
                {
                    detector.moveToPickTargetPose(traj);
                    detector.pickTarget();
                    //detector.pushTarget(grasp_pose_pBase,0.05);
                    //Move back
                    detector.moveToLookForTargetsPose(0.9,0.5);
                    detector.moveToPLaceTargetsPose(grasp_pose_pBase,0.15); // grasp_pose_pBase is used as a pre-place pose
                    detector.placeTarget();
                }else
                {
                    cout<<"Skip to next object"<<endl;
                }
                //Go back to initial state
                detector.moveToLookForTargetsPose(0.7,1.5);

                //                }else if (cmd == "q")
                break;
            }
        }
        else if(cmd == "n")
            break;
    }



    ros::shutdown();
}
