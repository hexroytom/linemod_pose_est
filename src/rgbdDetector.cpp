#include <linemod_pose_estimation/rgbdDetector.h>

//Constructor
rgbdDetector::rgbdDetector()
{
    //Read all linemod-related params
    //linemod_detector=readLinemod (linemod_template_path);
//    readLinemodTemplateParams (linemod_render_path,Rs_,Ts_,Distances_,Obj_origin_dists,
//                               Ks_,Rects_,renderer_n_points,renderer_angle_step,
//                               renderer_radius_min,renderer_radius_max,
//                               renderer_radius_step,renderer_width,renderer_height,
//                               renderer_focal_length_x,renderer_focal_length_y,
//                               renderer_near,renderer_far);

//    //Initiate ICP
//    icp.setMaximumIterations (icp_max_iter);
//    icp.setMaxCorrespondenceDistance (icp_maxCorresDist);
//    icp.setTransformationEpsilon (icp_tr_epsilon);
//    icp.setEuclideanFitnessEpsilon (icp_fitness_threshold);

//    linemod_thresh=detect_score_threshold;
}

//Perform the LINEMOD detection
void rgbdDetector::linemod_detection(Ptr<linemod::Detector> linemod_detector,const vector<Mat>& sources,const float& threshold,std::vector<linemod::Match>& matches)
{
    linemod_detector->match (sources,threshold,matches,std::vector<String>(),noArray());
}

void rgbdDetector::rcd_voting(vector<double>& Obj_origin_dists,const double& renderer_radius_min,const int& vote_row_col_step,const double& renderer_radius_step_,const vector<linemod::Match>& matches,std::map<std::vector<int>, std::vector<linemod::Match> >& map_match)
{
    //----------------3D Voting-----------------------------------------------------//

    int voting_width_step=vote_row_col_step; //Unit: pixel
    int voting_height_step=vote_row_col_step; //Unit: pixel, width step and height step suppose to be the same
    float voting_depth_step=renderer_radius_step_;//Unit: m

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

void rgbdDetector::cluster_filter(std::map<std::vector<int>, std::vector<linemod::Match> >& map_match,int thresh)
{
    assert(map_match.size() != 0);
    std::map<std::vector<int>, std::vector<linemod::Match> >::iterator it= map_match.begin();
    for(;it!=map_match.end();++it)
    {
       if(it->second.size()<thresh)
       {
           map_match.erase(it);
       }
    }


}

void rgbdDetector::cluster_scoring(std::map<std::vector<int>, std::vector<linemod::Match> >& map_match,Mat& depth_img,std::vector<ClusterData>& cluster_data)
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

double rgbdDetector::similarity_score_calc(std::vector<linemod::Match> match_cluster)
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

// compute depth and normal diff for 1 cluster
double rgbdDetector::depth_normal_diff_calc(RendererIterator *renderer_iterator_,Matx33d& K_rgb,vector<Mat>& Rs_,vector<Mat>& Ts_,std::vector<linemod::Match> match_cluster, Mat& depth_img)
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
        sum_normal_diff+=normal_diff(depth_img,depth_template,template_mask,rect,K_rgb);
    }
    sum_depth_diff=sum_depth_diff/match_cluster.size();
    sum_normal_diff=sum_normal_diff/match_cluster.size();
    int p=0;
    return (getClusterScore(sum_depth_diff,sum_normal_diff));
}

double rgbdDetector::depth_diff(Mat& depth_img,Mat& depth_template,cv::Mat& template_mask,cv::Rect& rect)
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

double rgbdDetector::normal_diff(cv::Mat& depth_img,Mat& depth_template,cv::Mat& template_mask,cv::Rect& rect,Matx33d& K_rgb)
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

void rgbdDetector::nonMaximaSuppression(vector<ClusterData>& cluster_data,const double& neighborSize, vector<Rect>& Rects_,std::map<std::vector<int>, std::vector<linemod::Match> >& map_match)
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

double rgbdDetector::getClusterScore(const double& depth_diff_score,const double& normal_diff_score)
{
    //Simply add two scores
    return(depth_diff_score+normal_diff_score);

}

void rgbdDetector::getRoughPoseByClustering(vector<ClusterData>& cluster_data,PointCloudXYZ::Ptr pc,vector<Mat>& Rs_,vector<Mat>& Ts_, vector<double>& Distances_,vector<double>& Obj_origin_dists,float orientation_clustering_th_,RendererIterator *renderer_iterator_,double& renderer_focal_length_x,double& renderer_focal_length_y,IMAGE_WIDTH& image_width,int& bias_x)
{
    //For each cluster
    for(vector<ClusterData>::iterator it = cluster_data.begin();it!=cluster_data.end();++it)
    {
        //Perform clustering
        vector<vector<Eigen::Matrix3d> > orienClusters;
        vector<vector<int> > idClusters;
        vector<vector<pair<int ,int> > > xyClusters;
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
                    xyClusters[i].push_back(pair<int ,int>(it2->x,it2->y));
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

                vector<pair<int,int> > new_xy_cluster;
                new_xy_cluster.push_back(pair<int,int>(it2->x,it2->y));
                xyClusters.push_back(new_xy_cluster);
            }
        }
        //Sort cluster according to the number of poses
        std::sort(orienClusters.begin(),orienClusters.end(),sortOrienCluster);
        std::sort(idClusters.begin(),idClusters.end(),sortIdCluster);
        std::sort(xyClusters.begin(),xyClusters.end(),sortXyCluster);

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
        double Trans_aver=0.0;
        vector<Eigen::Matrix3d>::iterator iter=orienClusters[0].begin();
        bool is_center_hole=false;
        int X=0; int Y=0;
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
            //Get translation
            Trans_aver+=Obj_origin_dists[idClusters[0][i]];
            //Get match position
            X+=xyClusters[0][i].first;
            Y+=xyClusters[0][i].second;

            if(fabs(Distances_[idClusters[0][i]]-Obj_origin_dists[idClusters[0][i]])<0.001)
            {
               is_center_hole=true;
            }
        }

       //Averaging operation
        R_aver/=orienClusters[0].size();
        T_aver/=orienClusters[0].size();
        D_aver/=orienClusters[0].size();
        Trans_aver/=orienClusters[0].size();
        X/=orienClusters[0].size();
        Y/=orienClusters[0].size();
       //normalize the averaged quaternion
        Eigen::Quaterniond quat=Eigen::Quaterniond(R_aver).normalized();
        cv::Mat R_mat;
        Eigen::Matrix3d R_eig;
        R_eig=quat.toRotationMatrix();
        eigen2cv(R_eig,R_mat);

        if(fabs(D_aver-Trans_aver) < 0.001)
            {
            is_center_hole=true;
        }

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

        //Save mask
        it->mask=mask;

        //Save orientation, T_match and K_matrix
        it->orientation=R_match;
        it->T_match=T_match;
        it->dist=D_aver;
        it->K_matrix=(Mat_<double>(3,3)<<renderer_focal_length_x,0.0,rect.width/2,
                                      0.0,renderer_focal_length_y,rect.height/2,
                                      0.0,0.0,1.0);
        it->pose.linear()=R_eig;

        //Save position
//                 int x=it->rect.x+it->rect.width/2;
//                 int y=it->rect.y+it->rect.height/2;
        int x=X+rect.width/2;
        int y=Y+rect.height/2;
        //Add offset to x due to previous cropping operation
        x+=bias_x;
        pcl::PointXYZ bbox_center =pc->at(x,y);

        //Deal with the situation that there is a hole in ROI or in the model pointcloud during rendering
        if(pcl_isnan(bbox_center.x) || pcl_isnan(bbox_center.y) || pcl_isnan(bbox_center.z) || is_center_hole)
        {
            PointCloudXYZ::Ptr pts_tmp(new PointCloudXYZ());
            pcl::PointIndices::Ptr indices_tmp(new pcl::PointIndices());
            vector<int> index_tmp;

            indices_tmp=getPointCloudIndices(it,image_width,bias_x);
            extractPointsByIndices(indices_tmp,pc,pts_tmp,false,false);
            pcl::removeNaNFromPointCloud(*pts_tmp,*pts_tmp,index_tmp);

            //Comptute centroid
            Eigen::Matrix<double,4,1> centroid;
            if(pcl::compute3DCentroid<pcl::PointXYZ,double>(*pts_tmp,centroid)!=0)
            {
                //Replace the Nan center point with centroid
                bbox_center.x=centroid[0];
                bbox_center.y=centroid[1];
                bbox_center.z=centroid[2];
            }
            else
            {
               ROS_ERROR("Pose clustering: unable to compute centroid of the pointcloud!");
            }
        }

        //Notice: No hole in the center
        if(!is_center_hole)
        {
            bbox_center.z+=D_aver;
        }
        it->position=cv::Vec3d(bbox_center.x,bbox_center.y,bbox_center.z);
        it->pose.translation()<< bbox_center.x,bbox_center.y,bbox_center.z;

        //Get render point cloud
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
//                 pcl::PointXYZ pt_scene(bbox_center.x,bbox_center.y,bbox_center.z-D_aver);
//                 pcl::PointXYZ pt_model=it->model_pc->at(it->model_pc->width/2,it->model_pc->height/2);

        pcl::PointXYZ pt_scene(bbox_center.x,bbox_center.y,bbox_center.z);
        pcl::PointXYZ pt_model(0,0,Trans_aver);

        pcl::PointXYZ translation(pt_scene.x -pt_model.x,pt_scene.y -pt_model.y,pt_scene.z -pt_model.z);
        Eigen::Affine3d transform =Eigen::Affine3d::Identity();
        transform.translation()<<translation.x, translation.y, translation.z;

        PointCloudXYZ::Ptr transformed_cloud (new PointCloudXYZ ());
        pcl::transformPointCloud(*it->model_pc,*transformed_cloud,transform);

        //it->model_pc.swap(transformed_cloud);

        pcl::copyPointCloud(*transformed_cloud,*(it->model_pc));
    }
}

bool rgbdDetector::orientationCompare(Eigen::Matrix3d& orien1,Eigen::Matrix3d& orien2,double thresh)
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

void rgbdDetector::icpPoseRefine(vector<ClusterData>& cluster_data,pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ>& icp,PointCloudXYZ::Ptr pc, IMAGE_WIDTH image_width,int bias_x,bool is_viz)
{
    pcl::visualization::PCLVisualizer::Ptr v;
    if(is_viz)
    {
        v=boost::make_shared<pcl::visualization::PCLVisualizer>("view");
    }
    int id=0;
    for(vector<ClusterData>::iterator it = cluster_data.begin();it!=cluster_data.end();++it)
    {
       //Get scene point cloud indices
        pcl::PointIndices::Ptr indices(new pcl::PointIndices);
        indices=getPointCloudIndices(it,image_width,bias_x);
       //Extract scene pc according to indices
        PointCloudXYZ::Ptr scene_pc(new PointCloudXYZ);
        extractPointsByIndices(indices,pc,scene_pc,false,false);

        //Viz for test
        if(is_viz)
        {
            string model_str="model";
            string scene_str="scene";
            stringstream ss;
            ss<<id;
            model_str+=ss.str();
            scene_str+=ss.str();
            pcl::visualization::PointCloudColorHandlerRandom<pcl::PointXYZ> color(it->model_pc);
            v->addPointCloud(scene_pc,scene_str);
            v->addPointCloud(it->model_pc,color,model_str);
            v->spin();
        }

        //Remove Nan points
        vector<int> index;
        it->model_pc->is_dense=false;
        pcl::removeNaNFromPointCloud(*(it->model_pc),*(it->model_pc),index);
        pcl::removeNaNFromPointCloud(*scene_pc,*scene_pc,index);

        //Statistical outlier removal
        statisticalOutlierRemoval(scene_pc,50,1.0);

        //euclidianClustering(scene_pc,0.01);

//                 float leaf_size=0.002;
//                 voxelGridFilter(scene_pc,leaf_size);
//                 voxelGridFilter(it->model_pc,leaf_size);

        //Save scene point cloud for later HV
        it->scene_pc=scene_pc;

        //Coarse alignment
        icp.setRANSACOutlierRejectionThreshold(0.02);
        icp.setInputSource (it->model_pc);
        icp.setInputTarget (scene_pc);
        icp.align (*(it->model_pc));
        if(!icp.hasConverged())
        {
            cout<<"ICP cannot converge"<<endl;
        }
        else{
            //cout<<"ICP fitness score of coarse alignment: "<<icp.getFitnessScore()<<endl;
        }
        //Update pose
        Eigen::Matrix4f tf_mat = icp.getFinalTransformation();
        Eigen::Matrix4d tf_mat_d=tf_mat.cast<double>();
        Eigen::Affine3d tf(tf_mat_d);
        it->pose=tf*it->pose;

        //Viz for test
        if(is_viz)
        {
            string model_str="model";
            string scene_str="scene";
            stringstream ss;
            ss<<id;
            model_str+=ss.str();
            scene_str+=ss.str();
            pcl::visualization::PointCloudColorHandlerRandom<pcl::PointXYZ> color(it->model_pc);
            v->updatePointCloud(scene_pc,scene_str);
            v->updatePointCloud(it->model_pc,color,model_str);
            v->spin();
        }

        //Fine alignment 1
        icp.setEuclideanFitnessEpsilon(1e-4);
        icp.setRANSACOutlierRejectionThreshold(0.01);
        icp.setMaximumIterations(20);
        icp.setMaxCorrespondenceDistance(0.01);
        icp.setInputSource (it->model_pc);
        icp.setInputTarget (scene_pc);
        icp.align (*(it->model_pc));
        if(!icp.hasConverged())
        {
            cout<<"ICP cannot converge"<<endl;
        }
        else{
            //cout<<"ICP fitness score of fine alignment 1: "<<icp.getFitnessScore()<<endl;
        }
        //Update pose
        tf_mat = icp.getFinalTransformation();
        tf_mat_d=tf_mat.cast<double>();
        tf.matrix()=tf_mat_d;
        it->pose=tf*it->pose;

        //Viz for test
        if(is_viz)
        {
            string model_str="model";
            string scene_str="scene";
            stringstream ss;
            ss<<id;
            model_str+=ss.str();
            scene_str+=ss.str();
            pcl::visualization::PointCloudColorHandlerRandom<pcl::PointXYZ> color(it->model_pc);
            v->updatePointCloud(scene_pc,scene_str);
            v->updatePointCloud(it->model_pc,color,model_str);
            v->spin();
        }

//                 //Fine alignment 2
//                 icp.setRANSACOutlierRejectionThreshold(0.005);
//                 icp.setMaximumIterations(10);
//                 icp.setMaxCorrespondenceDistance(0.005);
//                 icp.setInputSource (it->model_pc);
//                 icp.setInputTarget (scene_pc);
//                 icp.align (*(it->model_pc));
//                 if(!icp.hasConverged())
//                 {
//                     cout<<"ICP cannot converge"<<endl;
//                 }
//                 else{
//                     cout<<"ICP fitness score of fine alignment 2: "<<icp.getFitnessScore()<<endl;
//                 }
//                 //Update pose
//                 tf_mat = icp.getFinalTransformation();
//                 tf_mat_d=tf_mat.cast<double>();
//                 tf.matrix()=tf_mat_d;
//                 it->pose=tf*it->pose;

//                 //Viz test
//                 v.updatePointCloud(it->model_pc,color,"model");
//                 v.spin();

        id++;

    }
}

void rgbdDetector::euclidianClustering(PointCloudXYZ::Ptr pts,float dist)
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

void rgbdDetector::statisticalOutlierRemoval(PointCloudXYZ::Ptr pts, int num_neighbor,float stdDevMulThresh)
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

void rgbdDetector::voxelGridFilter(PointCloudXYZ::Ptr pts, float leaf_size)
{
    PointCloudXYZ::Ptr pts_filtered(new PointCloudXYZ);
    pcl::VoxelGrid<pcl::PointXYZ> vg;
    vg.setInputCloud(pts);
    vg.setLeafSize(leaf_size,leaf_size,leaf_size);
    vg.filter(*pts_filtered);
    pcl::copyPointCloud(*pts_filtered,*pts);
}

void rgbdDetector::hypothesisVerification(vector<ClusterData>& cluster_data, float octree_res, float thresh)
{
    vector<ClusterData>::iterator it =cluster_data.begin();
    for(;it!=cluster_data.end();)
    {
        pcl::octree::OctreePointCloud<pcl::PointXYZ> octree(octree_res);
        octree.setInputCloud(it->scene_pc);
        octree.addPointsFromInputCloud();

        int count=0;

        std::vector<pcl::PointXYZ, Eigen::aligned_allocator<pcl::PointXYZ> >::iterator iter_pc=it->model_pc->points.begin();
        for(;iter_pc!=it->model_pc->points.end();++iter_pc)
        {
            if(octree.isVoxelOccupiedAtPoint(*iter_pc))
                count++;
        }
        int model_pts=it->model_pc->points.size();

//        pcl::visualization::PCLVisualizer v("check");
//        pcl::visualization::PointCloudColorHandlerRandom<pcl::PointXYZ> color(it->model_pc);
//        v.addPointCloud(it->scene_pc,"scene");
//        v.addPointCloud(it->model_pc,color,"model");
//        v.spin();
//        v.close();

        double collision_rate = (double)count/(double)model_pts;
        if(collision_rate<thresh)
            {
            it=cluster_data.erase(it);
        }
        else
            {
            it++;
        }

    }

}

void rgbdDetector::icpNonLinearPoseRefine(vector<ClusterData>& cluster_data,PointCloudXYZ::Ptr pc,IMAGE_WIDTH image_width,int bias_x)
{
    for(vector<ClusterData>::iterator it = cluster_data.begin();it!=cluster_data.end();++it)
    {
       //Get scene point cloud indices
        pcl::PointIndices::Ptr indices(new pcl::PointIndices);
        indices=getPointCloudIndices(it,image_width,bias_x);
       //Extract scene pc according to indices
        PointCloudXYZ::Ptr scene_pc(new PointCloudXYZ);
        extractPointsByIndices(indices,pc,scene_pc,false,false);

        //Viz for test
        pcl::visualization::PCLVisualizer v("view_test");
        v.addPointCloud(scene_pc,"scene");
        pcl::visualization::PointCloudColorHandlerRandom<pcl::PointXYZ> color(it->model_pc);
        v.addPointCloud(it->model_pc,color,"model");
        v.spin();

        //Remove Nan points
        vector<int> index;
        it->model_pc->is_dense=false;
        pcl::removeNaNFromPointCloud(*(it->model_pc),*(it->model_pc),index);
        pcl::removeNaNFromPointCloud(*scene_pc,*scene_pc,index);

        //Statistical outlier removal
        statisticalOutlierRemoval(scene_pc,50,1.0);

        //euclidianClustering(scene_pc,0.01);

//                 float leaf_size=0.002;
//                 voxelGridFilter(scene_pc,leaf_size);
//                 voxelGridFilter(it->model_pc,leaf_size);

        //Viz for test
        v.updatePointCloud(scene_pc,"scene");
        v.updatePointCloud(it->model_pc,color,"model");
        v.spin();

        //Instantiate a non-linear ICP object
        pcl::IterativeClosestPointNonLinear<pcl::PointXYZ,pcl::PointXYZ> icp_;
        icp_.setMaxCorrespondenceDistance(0.05);
        icp_.setMaximumIterations(50);
        icp_.setRANSACOutlierRejectionThreshold(0.02);
        icp_.setTransformationEpsilon (1e-8);
        icp_.setEuclideanFitnessEpsilon (0.002);

        //Coarse alignment
        icp_.setInputSource (it->model_pc);
        icp_.setInputTarget (scene_pc);
        icp_.align (*(it->model_pc));
        if(!icp_.hasConverged())
            cout<<"ICP cannot converge"<<endl;
        //Update pose
        Eigen::Matrix4f tf_mat = icp_.getFinalTransformation();
        Eigen::Matrix4d tf_mat_d=tf_mat.cast<double>();
        Eigen::Affine3d tf(tf_mat_d);
        it->pose=tf*it->pose;

        //Fine alignment 1
        icp_.setRANSACOutlierRejectionThreshold(0.01);
        icp_.setMaximumIterations(20);
        icp_.setMaxCorrespondenceDistance(0.02);
        icp_.setInputSource (it->model_pc);
        icp_.setInputTarget (scene_pc);
        icp_.align (*(it->model_pc));
        if(!icp_.hasConverged())
            cout<<"ICP cannot converge"<<endl;
        //Update pose
        tf_mat = icp_.getFinalTransformation();
        tf_mat_d=tf_mat.cast<double>();
        tf.matrix()=tf_mat_d;
        it->pose=tf*it->pose;

        //Fine alignment 2
        icp_.setMaximumIterations(10);
        icp_.setMaxCorrespondenceDistance(0.005);
        icp_.setInputSource (it->model_pc);
        icp_.setInputTarget (scene_pc);
        icp_.align (*(it->model_pc));
        if(!icp_.hasConverged())
            cout<<"ICP cannot converge"<<endl;
        //Update pose
        tf_mat = icp_.getFinalTransformation();
        tf_mat_d=tf_mat.cast<double>();
        tf.matrix()=tf_mat_d;
        it->pose=tf*it->pose;

        //Viz test
        v.updatePointCloud(it->model_pc,color,"model");
        v.spin();

    }
}

//Utility
pcl::PointIndices::Ptr rgbdDetector::getPointCloudIndices(vector<ClusterData>::iterator& it, IMAGE_WIDTH image_width,int bias_x)
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
                int index=y_cropped*image_width+x_cropped+bias_x;
                indices->indices.push_back(index);
            }
        }
    }
    return indices;

}

pcl::PointIndices::Ptr rgbdDetector::getPointCloudIndices(const cv::Rect& rect, IMAGE_WIDTH image_width,int bias_x)
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
            int x_uncropped=x_cropped+bias_x;
            int y_uncropped=y_cropped;
            //Attention: image width of ensenso: 752, image height of ensenso: 480
            int index=y_uncropped*image_width+x_uncropped;
            indices->indices.push_back(index);

        }
    }
    return indices;

}

void rgbdDetector::extractPointsByIndices(pcl::PointIndices::Ptr indices, const PointCloudXYZ::Ptr ref_pts, PointCloudXYZ::Ptr extracted_pts, bool is_negative,bool is_organised)
{
    pcl::ExtractIndices<pcl::PointXYZ> tmp_extractor;
    tmp_extractor.setKeepOrganized(is_organised);
    tmp_extractor.setInputCloud(ref_pts);
    tmp_extractor.setNegative(is_negative);
    tmp_extractor.setIndices(indices);
    tmp_extractor.filter(*extracted_pts);
}


cv::Ptr<cv::linemod::Detector> rgbdDetector::readLinemod(const std::string& filename)
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
void rgbdDetector::readLinemodTemplateParams(const std::string fileName,
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

pointcloud_publisher::pointcloud_publisher(ros::NodeHandle& nh, const string &topic)
{
    publisher = nh.advertise<sensor_msgs::PointCloud2>(topic,1);
    pc_msg.header.frame_id = "/camera_link";
    pc_msg.header.stamp = ros::Time::now();
}

void pointcloud_publisher::publish(PointCloudXYZ::Ptr pc,const Scalar& color)
{
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc_rgb(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::copyPointCloud(*pc,*pc_rgb);

    //pc_msg.header.stamp = ros::Time::now();
    pcl::toROSMsg(*pc_rgb,pc_msg);

    sensor_msgs::PointCloud2Iterator<uint8_t> iter_r(pc_msg, "r");
    sensor_msgs::PointCloud2Iterator<uint8_t> iter_g(pc_msg, "g");
    sensor_msgs::PointCloud2Iterator<uint8_t> iter_b(pc_msg, "b");
    vector<pcl::PointXYZRGB,Eigen::aligned_allocator<pcl::PointXYZRGB> >::iterator it = pc_rgb->begin();

    for(;it!=pc_rgb->end();++it,++iter_r,++iter_g,++iter_b)
    {
        *iter_r = color[0];
        *iter_g = color[1];
        *iter_b = color[2];
    }

    publisher.publish(pc_msg);

}



