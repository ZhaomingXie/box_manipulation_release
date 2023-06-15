#pragma once

#include <stdlib.h>
#include <set>
#include "../../RaisimGymEnv.hpp"

namespace raisim {

class ENVIRONMENT : public RaisimGymEnv {

 public:

  explicit ENVIRONMENT(const std::string& resourceDir, const Yaml::Node& cfg, bool visualizable) :
      RaisimGymEnv(resourceDir, cfg), visualizable_(visualizable) {

    /// create world
    world_ = std::make_unique<raisim::World>();

    /// add objects
    humanoid_ = world_->addArticulatedSystem(resourceDir_+"/virtual_human.urdf", "", {}, raisim::COLLISION(0), -1);
    humanoid_->setName("humanoid");
    humanoid_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);
    world_->addGround(0, "steel", raisim::COLLISION(0));
    world_->setMaterialPairProp("default", "steel", 2.0, 0.0, 0.0001);
    world_->setMaterialPairProp("default", "ball", 2.0, 0.0, 0.2, 1.0, 0.01);
    world_->setMaterialPairProp("ball", "ball", 1.0, 0.0, 0.0001);
    world_->setContactSolverParam(1, 1, 1, 200, 1e-8);
    world_->setContactSolverIterationOrder(true);
    // world_->setERP(1.0, 0.1);

    if (visualizable_) {
      visual_humanoid_ = world_->addArticulatedSystem(resourceDir_+"/virtual_human.urdf");
      visual_humanoid_->setName("visual_humanoid");
      visual_humanoid_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);
    }

    /// get robot data
    gcDim_ = humanoid_->getGeneralizedCoordinateDim();
    gvDim_ = humanoid_->getDOF();
    nJoints_ = gvDim_ - 6;

    /// initialize containers
    gc_.setZero(gcDim_); gc_init_.setZero(gcDim_);
    gv_.setZero(gvDim_); gv_init_.setZero(gvDim_);
    prev_gc_.setZero(gcDim_); prev_gv_.setZero(gvDim_);
    pTarget_.setZero(gcDim_); vTarget_.setZero(gvDim_); pTarget12_.setZero(nJoints_);
    learned_phase_.setZero(9);
    learned_next_phase_.setZero(27);
    reference_.setZero(gcDim_);
    prev_reference_.setZero(gcDim_);
    reference_velocity_.setZero(gvDim_);
    torque_.setZero(gvDim_);
    previous_torque_.setZero(gvDim_-6);

    /// this is nominal configuration of anymal
    gc_init_ << 0., 0., 1., 0.5, 0.5, 0.5, 0.5, 
    1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 
    1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 
    1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 
    1., 0., 0., 0., 0.78192104, -0.09971977, -0.0831469, -0.60970652, 1., 0., 0., 0., 
     0.0,
     1., 0., 0., 0., 0.78192104, -0.09971977, 0.0831469, 0.60970652, 1., 0., 0., 0.,
     0.0;
    gv_init_[0] = 3.0;
    reference_ << 0., 0., 0., 1., 0., 0., 0., 
    1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 
    1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 
    1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 
     1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 
     0.0,
     1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0.,
     0.0;

     setCrouchingPose();

    /// set pd gains
    // Eigen::VectorXd jointPgain_(gvDim_), jointDgain_(gvDim_);
    jointPgain_.setZero(gvDim_); jointPgain_.tail(nJoints_).setConstant(300.0);
    jointDgain_.setZero(gvDim_); jointDgain_.tail(nJoints_).setConstant(30.0);
    
    // // //ankle
    jointPgain_.segment(12, 3).setConstant(100); jointDgain_.segment(12,3).setConstant(10); 
    jointPgain_.segment(21, 3).setConstant(100); jointDgain_.segment(21,3).setConstant(10); 
    //knee
    jointPgain_.segment(9, 3).setConstant(150); jointDgain_.segment(9,3).setConstant(15); 
    jointPgain_.segment(18, 3).setConstant(150); jointDgain_.segment(18,3).setConstant(15); 
    //hip
    jointPgain_.segment(6, 3).setConstant(250); jointDgain_.segment(6,3).setConstant(25); 
    jointPgain_.segment(15, 3).setConstant(250); jointDgain_.segment(15,3).setConstant(25);
    //arms
    jointPgain_.segment(39, 20).setConstant(100); jointDgain_.segment(39, 20).setConstant(5);
    // jointPgain_.segment(39, 20).setConstant(100); jointDgain_.segment(39, 20).setConstant(5);
    jointPgain_[48] = 100, jointDgain_[48] = 10, jointPgain_[58] = 100, jointDgain_[58] = 10;

    
    humanoid_->setPdGains(jointPgain_, jointDgain_);
    humanoid_->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));

    /// MUST BE DONE FOR ALL ENVIRONMENTS
    obDim_ = 132 + 7 + 3 + 68 + 1 + 7 + 1 + 1 + 1 + 2 + 7 + 1 + 1 + 1 + 4; 
    //132 + 4 * 9 + 3 + 68 + box_height + box3 + contact_label + sim_cycle + desired_box_height + desired_box_xy + box4 + box_width + desired_box_height
    actionDim_ = nJoints_; actionMean_.setZero(actionDim_); actionStd_.setZero(actionDim_);
    obDouble_.setZero(obDim_);

    kinematicDouble_.setZero(84);

    /// action scaling
    actionMean_ = gc_init_.tail(nJoints_);
    actionStd_.setConstant(0.3);

    stateDim_ = 20 * 7 + 3 * 7; // 20 bones and 3 boxes

    /// Reward coefficients
    rewards_.initializeFromConfigurationFile (cfg["reward"]);

    /// indices of links that should not make contact with ground
    footIndices_.insert(humanoid_->getBodyIdx("rankle"));
    footIndices_.insert(humanoid_->getBodyIdx("lankle"));
    handIndices_.insert(humanoid_->getBodyIdx("rwrist"));
    handIndices_.insert(humanoid_->getBodyIdx("lwrist"));
    // std::cout << humanoid_->getBodyIdx("rankle") << humanoid_->getBodyIdx("lankle") << humanoid_->getBodyIdx("relbow") << humanoid_->getBodyIdx("lelbow") << std::endl;

    std::string bone_name[20] = {"root", "lhip", "lknee", "lankle", "rhip", "rknee", "rankle", "lowerback",
      "upperback", "chest", "lowerneck", "upperneck", "lclavicle", "lshoulder", "lelbow", "lwrist",
      "rclavicle", "rshoulder", "relbow", "rwrist"};
    for (int i = 0; i < 20; i++)
      boneIndices_.push_back(humanoid_->getFrameIdxByName(bone_name[i]));


    /// visualize if it is the first environment
    if (visualizable_) {
      server_ = std::make_unique<raisim::RaisimServer>(world_.get());
      server_->launchServer(8082);
      server_->focusOn(humanoid_);
    }
  }

  void init() final { }

  void setEnvironmentTask(int i) final {
    enviornment_id_ = 1;//i % 3;//(i % 3)+2;//i % 4;
    box_height_ = 0.15;//(rand() % 11) * 0.015 + 0.15;
    box2_ = world_->addBox(0.5, 0.5, box_height_, 1.0, "ball");
    box_width_ = (enviornment_id_ + 1) * 0.15;

    box3_height_ = 0.15;
    box3_ = world_->addBox(0.4, 0.4, box3_height_, 1.0, "ball");

    box4_height_ = 0.15;
    box4_ = world_->addBox(0.3, 0.3, box4_height_, 1.0, "ball");

    raisim::Mat<3, 3> inertia;
    for (int i = 0; i < 9; i++) {
      if (i == 0 or i == 4 or i == 8)
        inertia[i] = 0.043;
      else
        inertia[i] = 0.0;
    }
    box2_->setInertia(inertia);
    box3_->setInertia(inertia);
    box4_->setInertia(inertia);

    platform_height_ = 0.3;
    platform_ = world_->addBox(0.6, 0.6, 0.5, 10.0, "ball"); // table
    platform_->setPosition(0.5, 0, 0.25);

    platform2_ = world_->addBox(0.6, 0.6, 0.5, 10.0, "ball");  //sofa
    platform2_->setPosition(-1.2, 0.5, 0.25);

    platform3_ = world_->addBox(0.6, 0.6, 0.5, 10.0, "ball");  //TV stand
    platform3_->setPosition(0.5, 1, 0.25);

    platform4_ = world_->addBox(0.3, 1.5, 0.5, 10.0, "ball");  //dining table
    platform4_->setPosition(-10.15, 0, 0.3);
  }

  void reset() final {
    if (testing_ or testing3_) {
      primitive_index_ = 0;
      current_primitive_timer_ = timing_[0];
    }
    // testing_ = true;
    rightHandContactActive_ = false, leftHandContactActive_ = false;
    total_reward_ = 0;
    sim_step_ = 0;
    place_box_ = 0;//(rand() % 10 == 1);
    updatePickUpTaskVariable(2);
    // phase_ = 0;
    // phase_ = rand() % max_phase_;
    speed_ = 0;
    gv_init_[0] = speed_;
    gv_init_[1] = 0;
    humanoid_->setState(reference_, reference_velocity_);
    prev_reference_ << reference_;

    num_boxes_ = 1;//rand() % 3 + 1;
    init_reference_error_x_ = 0; init_reference_error_y_ = 0;


    double box2_offset_x = 0;
    double box3_offset_x = 0;
    double box4_offset_x = 0;
    if (num_boxes_ == 1) {
      box3_offset_x = 10;
      box4_offset_x = -10;
    }
    else if (num_boxes_ == 2) {
      box4_offset_x = 10;
    }

    double x_noise = -0.04;//(rand() % 21-10) * 0.004;
    double y_noise = 0;//(rand() % 21-10) * 0.004;
    double platform_x_offset = -0.03;//(rand() % 21-10) * 0.003;
    double platform_y_offset = 0;//(rand() % 21-10) * 0.003;
    desired_box_x_ = 0.45, desired_box_y_ = -0.055;
    box2_->setPosition(0.55 + x_noise + box2_offset_x + platform_x_offset, -0.0 + y_noise + platform_y_offset, box_height_ / 2 + platform_height_); //0.7  //0.35 for demo pick and place init
    box2_->setOrientation(1.0, 0.0, 0.0, 0.0);
    Vec<3> init_vel; init_vel[0] = 0.0, init_vel[1] = 0.0, init_vel[2] = 0.0;
    box2_->setLinearVelocity(init_vel), box2_->setAngularVelocity(init_vel);

    // box3_->setPosition(0.45, 0.05, 1.05+0.5); // pick and place init
    double box3_init_height = box_height_+platform_height_+box3_height_ / 2;
    if (num_boxes_ <= 1)
      box3_init_height = box3_height_ / 2;
    box3_->setPosition(0.55 + x_noise + box3_offset_x + platform_x_offset, -0.0 + y_noise + platform_y_offset, box3_init_height); // carry init
    box3_->setOrientation(1.0, 0.0, 0.0, 0.0);
    box3_->setLinearVelocity(init_vel), box3_->setAngularVelocity(init_vel);

    double box4_init_height = box_height_+platform_height_+box4_height_/2+box3_height_;
    if (num_boxes_ <= 2)
      box4_init_height = box4_height_ / 2;
    box4_->setPosition(0.55 + x_noise + box4_offset_x + platform_x_offset, -0.0 + y_noise + platform_y_offset, box4_init_height); // carry init
    box4_->setOrientation(1.0, 0.0, 0.0, 0.0);
    box4_->setLinearVelocity(init_vel), box4_->setAngularVelocity(init_vel);


    if (testing3_) {
      box2_->setPosition(0.5, 0, 0.5+0.075);  // on top of TV stand
      raisim::Vec<4> init_orientation;
      init_orientation[0] = 1, init_orientation[1] = 0, init_orientation[2] = 0.0, init_orientation[3] = 0.0;
      box2_->setOrientation(init_orientation);
      
      box3_->setPosition(0.5, 0, 0.5+0.15+0.075); //on top of sofa
      init_orientation[0] = 1, init_orientation[1] = 0, init_orientation[2] = 0.0, init_orientation[3] = -0.0;
      box3_->setOrientation(init_orientation);

      box4_->setPosition(0.5, 0, 0.5+0.3+0.075); //on top of sofa
      init_orientation[0] = 1, init_orientation[1] = 0, init_orientation[2] = 0.0, init_orientation[3] = -0.0;
      box4_->setOrientation(init_orientation);

      box_height_ = 0.15;
      platform_height_ = 0.8;
    }

    humanoid_->updateKinematics();
    
    hand_distance_ = 1.0;
    reach_target_counter = 4;
    distance_to_target_ = std::pow(std::pow(reference_[0]-target_x_, 2) + std::pow(reference_[1] - target_y_, 2), 0.5);
    get_kinematic_quaternion_error();

    humanoid_->getState(gc_, gv_);
    updateUsefulVariable();
    hand_distance_ = std::pow(update_box_to_left_hand_position_[0], 2) + std::pow(update_box_to_left_hand_position_[1], 2) + std::pow(update_box_to_left_hand_position_[2], 2)
      + std::pow(update_box_to_right_hand_position_[0], 2) + std::pow(update_box_to_right_hand_position_[1], 2) + std::pow(update_box_to_right_hand_position_[2], 2);
    previous_hand_distance_ = hand_distance_;
    updateObservation();
  }

  float step(const Eigen::Ref<EigenVec>& action) final {
    updatePickUpTaskVariable(2);
    /// action scaling
    pTarget_.tail(70) = reference_.tail(70);

    std::cout << "reference_" << reference_.segment(0, 3) << std::endl;
    std::cout << "actual" << gc_.segment(0, 3) << std::endl;
    std::cout << "vel" << gv_.segment(0, 3) << std::endl;

    if (primitives_[primitive_index_] >= 2) {
      std::cout << "enter loop" << std::endl;

      jointPgain_.segment(12, 3).setConstant(100); jointDgain_.segment(12,3).setConstant(10); 
      jointPgain_.segment(21, 3).setConstant(100); jointDgain_.segment(21,3).setConstant(10); 
      //knee
      jointPgain_.segment(9, 3).setConstant(150); jointDgain_.segment(9,3).setConstant(15); 
      jointPgain_.segment(18, 3).setConstant(150); jointDgain_.segment(18,3).setConstant(15); 
      //hip
      jointPgain_.segment(6, 3).setConstant(250); jointDgain_.segment(6,3).setConstant(25); 
      jointPgain_.segment(15, 3).setConstant(250); jointDgain_.segment(15,3).setConstant(25);
      //arms
      jointPgain_.segment(39, 20).setConstant(100); jointDgain_.segment(39, 20).setConstant(10);
      jointPgain_.segment(24, 15).setConstant(150); jointDgain_.segment(24, 15).setConstant(15);
      jointPgain_[48] = 100, jointDgain_[48] = 10, jointPgain_[58] = 100, jointDgain_[58] = 10;

      pTarget_.tail(70) = reference_.tail(70);

      Eigen::VectorXd torque = Eigen::VectorXd::Zero(gvDim_);

      torque.segment(6, 53) = (action.segment(0, 53).cast<double>()*0.5 + previous_torque_ * 0.5) * 300.0; //200
      torque.segment(12, 3) /= 3.0;
      torque.segment(21, 3) /= 3.0;
      torque.segment(9, 3) /= 2.0/1.5;
      torque.segment(18, 3) /= 2.0/1.5;
      torque.segment(6, 3) /= 2.0/2.5;
      torque.segment(15, 3) /= 2.0/2.5;
      torque.segment(39, 20) /= 3;
      torque.segment(24, 15) /= 2;

      torque_ << torque;
      previous_torque_ << (action.segment(0, 53).cast<double>()*0.5 + previous_torque_ * 0.5);

      // if (current_primitive_timer_ >= 400 and primitive_index_ == 2)
      //   control_dt_ = 0.016;
      // else
      //   \control_dt_ = 0.032;
      control_dt_ = 0.032;
    }
    else if (primitives_[primitive_index_] <= 1) {
      jointPgain_.segment(12, 3).setConstant(100); jointDgain_.segment(12,3).setConstant(10); 
      jointPgain_.segment(21, 3).setConstant(100); jointDgain_.segment(21,3).setConstant(10); 
      //knee
      jointPgain_.segment(9, 3).setConstant(150); jointDgain_.segment(9,3).setConstant(15); 
      jointPgain_.segment(18, 3).setConstant(150); jointDgain_.segment(18,3).setConstant(15); 
      //hip
      jointPgain_.segment(6, 3).setConstant(250); jointDgain_.segment(6,3).setConstant(25); 
      jointPgain_.segment(15, 3).setConstant(250); jointDgain_.segment(15,3).setConstant(25);
      //arms
      jointPgain_.segment(39, 20).setConstant(100); jointDgain_.segment(39, 20).setConstant(10);
      jointPgain_.segment(24, 15).setConstant(150); jointDgain_.segment(24, 15).setConstant(15);
      jointPgain_[48] = 100, jointDgain_[48] = 10, jointPgain_[58] = 100, jointDgain_[58] = 10;

      pTarget_.tail(70) = reference_.tail(70);

      Eigen::VectorXd torque = Eigen::VectorXd::Zero(gvDim_);

      torque.segment(6, 53) = (action.segment(0, 53).cast<double>()*0.7 + previous_torque_ * 0.3) * 300.0; //200
      torque.segment(12, 3) /= 3.0;
      torque.segment(21, 3) /= 3.0;
      torque.segment(9, 3) /= 2.0/1.5;
      torque.segment(18, 3) /= 2.0/1.5;
      torque.segment(6, 3) /= 2.0/2.5;
      torque.segment(15, 3) /= 2.0/2.5;
      torque.segment(39, 20) /= 3;
      torque.segment(24, 15) /= 2;

      torque_ << torque;
      previous_torque_ << (action.segment(0, 53).cast<double>()*0.7 + previous_torque_ * 0.3);
      control_dt_ = 0.032;

    }

    if (primitives_[primitive_index_] >= 0)
      humanoid_->setPdGains(jointPgain_, (jointPgain_) / 20.0);
    else
      humanoid_->setPdGains(jointPgain_, (jointPgain_) / 10.0);


    if (testing3_) {
      current_primitive_timer_ -= 1;
      if (current_primitive_timer_ == 0)
        humanoid_->setState(gc_, gv_init_);
      if (current_primitive_timer_ == 0 and primitive_index_ == 0) {
        box_height_ = 0.15;
      }
      else if (current_primitive_timer_ == 0 and primitive_index_ == 1) {
       raisim::Vec<3> box_position;
        box3_->getPosition(box_position);
        box3_->setPosition(box_position[0], box_position[1], box_position[2]+0.00); 
      }
      else if (current_primitive_timer_ == 10 and primitive_index_ == 2) {
        raisim::Vec<3> box_position;
        box2_->getPosition(box_position);
        box2_->setPosition(box_position[0], box_position[1], box_position[2]);
      }
      else if (primitive_index_ == 3)
        platform_height_ = 0.6;
      else if (primitive_index_ == 4) {
        platform_height_ = 0.65;
        box_height_ = 0.15;
        num_boxes_ = 1;
        box_width_ = 0.4;
      }
      else if (primitive_index_ == 6) {
        platform_height_ = 0.5;
      }
      else if (primitive_index_ == 8) {
        box_width_ = 0.3;
      }
      else if (primitive_index_ == 9) {
        platform_height_ = 0.65;
      }
      else if (primitive_index_ == 12) {
        platform_height_ = 0.5;
        box_width_ = 0.5;
      }
      else if (primitive_index_ == 16) {
        platform_height_ = 0.65;
        box_width_ = 0.3;
      }
      else if (primitive_index_ == 18) {
        platform_height_ = 0.5;
      } 
      else if (primitive_index_ == 20) {
        box_width_ = 0.4;
      }
      else if (primitive_index_ == 22) {
        platform_height_ = 0.65;
      }
      else if (primitive_index_ == 24) {
        platform_height_ = 0.5;
        box_width_ = 0.3;
      }
      else if (primitive_index_ == 26) {
        platform_height_ = 0.8;
      }
    }

    if (visualizable_) {
      Eigen::VectorXd visual_reference = Eigen::VectorXd::Zero(gcDim_);
      visual_reference << reference_;
      visual_reference[0] += 10;
      visual_humanoid_->setState(visual_reference, gv_init_);
    }

    humanoid_->setPdTarget(pTarget_, vTarget_);

    humanoid_->setGeneralizedForce(torque_);

    for(int i=0; i< int(control_dt_ / simulation_dt_ + 1e-10); i++){
      if(server_) server_->lockVisualizationServerMutex();
      world_->integrate();
      if(server_) server_->unlockVisualizationServerMutex();
    }
    sim_step_ += 1;


    humanoid_->getState(gc_, gv_);
    updateUsefulVariable();
    updateObservation();
    computeReward();
    // total_reward_ += rewards_.multiply();
    double one_step_reward = rewards_["joint"] * rewards_["position"] * rewards_["dynamic_orientation"] * 0.5 + 0.5 * rewards_["box"];
   
    total_reward_ += one_step_reward;

    raisim::Vec<3> box_vel, box_angular_vel;
    box2_->getLinearVelocity(box_vel);
    box2_->getAngularVelocity(box_angular_vel);
    if (std::abs(box_angular_vel[0]) > 10 or std::abs(box_angular_vel[1]) > 10 or std::abs(box_angular_vel[2]) > 10)
      joint_error_flag_ = true;
    if (std::abs(box_vel[0]) > 10 or std::abs(box_vel[1]) > 10 or std::abs(box_vel[2]) > 10)
      joint_error_flag_ = true;
    for (int i = 0; i < 57; i++)
      if (std::abs(gv_[i]) > 20)
        joint_error_flag_ = true;

    if (current_primitive_timer_ == 0 and (testing_ or testing3_)) {
      primitive_index_ += 1;
      current_primitive_timer_ = timing_[primitive_index_];
      if (primitives_[primitive_index_] == 0 or primitives_[primitive_index_] == 1) {
        sim_step_ = start_time_[primitive_index_];
      }
      if (primitive_index_ >= 17) {
        init_reference_error_x_ = reference_[0] - gc_[0];
        init_reference_error_y_ = reference_[1] - gc_[1];
      }
      else {
        init_reference_error_x_ = 0;//reference_[0] - gc_[0];
        init_reference_error_y_ = 0;//reference_[1] - gc_[1];
      }
      updatePickUpTaskVariable(2);
      updateUsefulVariable();
      updateObservation();
    }

    return one_step_reward;
  }

  void computeReward() {
    joint_error_flag_ = false;
    float joint_reward = 0, position_reward = 0, orientation_reward = 0, task_reward = 0, box_reward = 0;

    for (int i = 0; i < 17; i++) {
      float error = std::pow(joint_quat_error_[i][1], 2) + std::pow(joint_quat_error_[i][2], 2) + std::pow(joint_quat_error_[i][3], 2);
      if ((i < 6) and (abs(joint_quat_error_[i][0]) <= 0.9))
        joint_error_flag_ = true;
      if ((i < 11) and (abs(joint_quat_error_[i][0]) <= 0.9)) {
        joint_error_flag_ = true;
      }
      // if (i < 11 and i != 0 and i != 3)
      joint_reward += error;
    }

    position_reward += std::pow(gc_[0] - reference_[0], 2) + std::pow(gc_[1] - reference_[1], 2) + std::pow(reference_[2] - gc_[2], 2);
      // std::cout << gc_.segment(0, 3) << prev_gc_.segment(0, 3) << std::endl;
    // position_reward += 500 * std::pow(gc_[0] - prev_gc_[0] - reference_[0] + prev_reference_[0], 2) + 500 * std::pow(gc_[1] - prev_gc_[1] - reference_[1] + prev_reference_[1], 2) + std::pow(reference_[2] - gc_[2], 2);

    // dynamic orientation reward
    double dynamic_orientation_reward = 0;
    raisim::Vec<4> quat, quat2, quat_error;
    raisim::Mat<3, 3> rot, rot2, rot_error;
    quat[0] = gc_[3]; quat[1] = gc_[4]; 
    quat[2] = gc_[5]; quat[3] = gc_[6];
    quat2[0] = reference_[3]; quat2[1] = reference_[4]; 
    quat2[2] = reference_[5]; quat2[3] = reference_[6];
    raisim::quatToRotMat(quat, rot); raisim::quatToRotMat(quat2, rot2);
    raisim::mattransposematmul(rot, rot2, rot_error);
    raisim::rotMatToQuat(rot_error, quat_error);
    // std::cout << quat_error << std::endl;
    dynamic_orientation_reward += 5 * (std::pow(quat_error[1], 2) + std::pow(quat_error[2], 2) + std::pow(quat_error[3], 2)); 
    dynamic_orientation_reward += 0.1 * (std::pow(gv_[3], 2) + std::pow(gv_[4], 2));// + std::pow(gv_[5], 2));

    if (std::abs(quat_error[3]) > 0.2 or std::abs(quat_error[2]) > 0.2 or std::abs(quat_error[1]) > 0.2) {//0.95{
      joint_error_flag_ = true;
    }

    //compute box reward
    box_reward += std::exp(-10 * (std::pow(update_box_orientation_[1], 2) + std::pow(update_box_orientation_[2], 2) + std::pow(update_box_orientation_[3], 2)));
    hand_distance_ = std::pow(update_box_to_left_hand_position_[0], 2) + std::pow(update_box_to_left_hand_position_[1], 2) + std::pow(update_box_to_left_hand_position_[2], 2)
      + std::pow(update_box_to_right_hand_position_[0], 2) + std::pow(update_box_to_right_hand_position_[1], 2) + std::pow(update_box_to_right_hand_position_[2], 2);

    int sim_step_cycle = sim_step_ % (240/frequency_);
    if (contact_label_ >= 0.9)
      box_reward += std::exp(-4 * hand_distance_);
    else if (contact_label_ <= 0.1)
      box_reward += 1;
    else if (sim_step_cycle <= 80 and not place_box_)
      box_reward += std::exp(-1 * hand_distance_);
    else
      box_reward += 1;

    // double desired_box_height = 0;
    raisim::Vec<3> box_position, platform_position; 
    box2_->getPosition(box_position);
    platform_->getPosition(platform_position);

    if (std::abs(box_position[2]-desired_box_height_) > 0.5){
      joint_error_flag_ = true;
      // std::cout << "something happen" << std::endl;
    }
    if (contact_label_ > 0.9 && hand_distance_ > 0.1) {  //0.05
      joint_error_flag_ = true;
    }

    if (sim_step_cycle <= 40 and place_box_)
      box_reward += std::exp(-2 * (5 * std::pow(box_position[2] - desired_box_height_, 2) + std::pow(box_position[1] - platform_position[1], 2) + std::pow(box_position[0] - platform_position[0], 2)));
    else
      box_reward += std::exp(-10 * (std::pow(box_position[2] - desired_box_height_, 2)));

    if (std::abs(update_box_orientation_[0]) < 0.99) {
        joint_error_flag_ = true;
    }

    //box3 reward
    if (num_boxes_ >= 2) {
      box_reward *= std::exp(-20 * (std::pow(update_box3_orientation_[1], 2) + std::pow(update_box3_orientation_[2], 2) + std::pow(update_box3_orientation_[3], 2)));
      box_reward *= std::exp(-5 * (std::pow(update_box3_position_[0]-update_box_position_[0], 2) + std::pow(update_box3_position_[1]-update_box_position_[1], 2)+
                      std::pow(update_box3_position_[2]-update_box_position_[2]-box_height_/2 - box3_height_/2, 2)));
      if ((std::pow(update_box3_position_[0]-update_box_position_[0], 2) + std::pow(update_box3_position_[1]-update_box_position_[1], 2)+
                      std::pow(update_box3_position_[2]-update_box_position_[2]-box_height_/2-box3_height_/2, 2)) > 0.2)
        joint_error_flag_ = true;
      if (std::abs(update_box_orientation_[0]) < 0.99 or std::abs(update_box3_orientation_[0]) < 0.99)
        joint_error_flag_ = true;
    }

    //box4 reward
    if (num_boxes_ >= 3) {
      box_reward *= std::exp(-20 * (std::pow(update_box4_orientation_[1], 2) + std::pow(update_box4_orientation_[2], 2) + std::pow(update_box4_orientation_[3], 2)));
      box_reward *= std::exp(-5 * (std::pow(update_box4_position_[0]-update_box_position_[0], 2) + std::pow(update_box4_position_[1]-update_box_position_[1], 2)+
                      std::pow(update_box4_position_[2]-update_box_position_[2]-box_height_/2 - box4_height_/2 - box3_height_, 2)));
      if ((std::pow(update_box4_position_[0]-update_box_position_[0], 2) + std::pow(update_box4_position_[1]-update_box_position_[1], 2)+
                      std::pow(update_box4_position_[2]-update_box_position_[2]-box_height_/2-box4_height_/2 - box3_height_, 2)) > 0.2)
        joint_error_flag_ = true;
      if (std::abs(update_box_orientation_[0]) < 0.99 or std::abs(update_box4_orientation_[0]) < 0.99)
        joint_error_flag_ = true;
    }
    // std::cout << sim_step_ << " " << rewards_["dynamic_orientation"] << " " << rewards_["position"] << " " << rewards_["box"] << std::endl;
    // rewards_.record("task", diff);
    // rewards_.record("orientation", angle_diff);
    rewards_.record("joint", std::exp(-2*joint_reward)); //2
    rewards_.record("position", std::exp(-position_reward));
    rewards_.record("dynamic_orientation", std::exp(-dynamic_orientation_reward));
    rewards_.record("box", box_reward);
  }

  void updateObservation() {
    prev_gc_ << gc_;
    prev_gv_ << gv_;
    humanoid_->getState(gc_, gv_);

    //relative orientation observation
    raisim::Vec<4> quat, quat2, quat_error, quat_base;
    raisim::Mat<3,3> rot, rot2, rot_error, rot_base;
    quat[0] = reference_[3]; quat[1] = reference_[4]; 
    quat[2] = reference_[5]; quat[3] = reference_[6];
    quat2[0] = gc_[3]; quat2[1] = gc_[4]; 
    quat2[2] = gc_[5]; quat2[3] = gc_[6];
    raisim::quatToRotMat(quat, rot);
    raisim::quatToRotMat(quat2, rot2);
    raisim::mattransposematmul(rot, rot2, rot_error);
    raisim::rotMatToQuat(rot_error, quat_error);
    if (quat_error[0] < 0) {
      for (int i = 0; i < 4; i++)
        quat_error[i] *= -1;
    }

    quat_base[0] = 0.5; quat_base[1] = 0.5; 
    quat_base[2] = 0.5; quat_base[3] = 0.5;

    // compute euler angle for gc_ root
    raisim::quatToRotMat(quat_base, rot_base);
    raisim::Mat<3,3> z_up_rot;
    raisim::matmattransposemul(rot_base, rot2, z_up_rot);
    double euler_gc[3];
    double quat_gc[4];
    raisim::Vec<4> z_up_quat;
    raisim::rotMatToQuat(z_up_rot, z_up_quat);
    quat_gc[0] = z_up_quat[0], quat_gc[1] = z_up_quat[1], quat_gc[2] = z_up_quat[2], quat_gc[3] = z_up_quat[3]; 
    raisim::quatToEulerVec(quat_gc, euler_gc);    
    for (int i = 0; i < 3; i++) {
      if (euler_gc[i] < -3.14)
        euler_gc[i] += 6.28;
      else if (euler_gc[i] > 3.14)
        euler_gc[i] -= 6.28;
    }

    //compute euler angle for reference root
    raisim::Mat<3,3> z_up_rot_ref;
    raisim::matmattransposemul(rot_base, rot, z_up_rot_ref);
    double euler_ref[3];
    double quat_ref[4];
    raisim::Vec<4> z_up_quat_ref;
    raisim::rotMatToQuat(z_up_rot_ref, z_up_quat_ref);
    quat_ref[0] = z_up_quat_ref[0], quat_ref[1] = z_up_quat_ref[1], quat_ref[2] = z_up_quat_ref[2], quat_ref[3] = z_up_quat_ref[3]; 
    raisim::quatToEulerVec(quat_ref, euler_ref);
    for (int i = 0; i < 3; i++) {
      if (euler_ref[i] < -3.14)
        euler_ref[i] += 6.28;
      else if (euler_ref[i] > 3.14)
        euler_ref[i] -= 6.28;
    }

    //relative velocity and angular velocity
    raisim::Vec<3> current_gv, update_gv, current_angular_gv_, update_angular_gv_, position_error, update_position_error;

    current_gv[0] = gv_[0], current_gv[1] = gv_[1], current_gv[2] = gv_[2];
    current_angular_gv_[0] = gv_[3], current_angular_gv_[1] = gv_[4], current_angular_gv_[2] = gv_[5];

    // if (sim_step_ <= 240)
    // position_error[0] = reference_[0] - gc_[0], position_error[1] = reference_[1] - gc_[1], position_error[2] = 0;
    position_error[0] = reference_[0] - gc_[0] - init_reference_error_x_, position_error[1] = reference_[1] - gc_[1] - init_reference_error_y_, position_error[2] = 0;
    // else
    // position_error[0] = reference_[0] - prev_reference_[0], position_error[1] = reference_[1] - prev_reference_[1], position_error[2] = 0;
    raisim::quatToRotMat(quat, rot);
    raisim::matvecmul(z_up_rot, current_gv, update_gv);
    raisim::matvecmul(z_up_rot, current_angular_gv_, update_angular_gv_);
    raisim::matvecmul(z_up_rot_ref, position_error, update_position_error);

    int sim_step_cycle = (sim_step_)%(240/frequency_);
    if (sim_step_cycle < 0)
      sim_step_cycle = 240 /frequency_+ sim_step_cycle;

    int index_offset = 4;

    obDouble_ << (update_position_error[0]) * 10, (update_position_error[1]) * 10,  gc_[2], /// body height
        quat_error[0], quat_error[1] * 10, quat_error[2] * 10, quat_error[3] * 10, /// body orientation
        z_up_rot[6], z_up_rot[7], z_up_rot[8],
        gc_.tail(68) * 0, /// joint angles error
        gc_.tail(70),
        // gc_.segment(7, 14*4), gc_.segment(14*4+7+1, 12),
        update_gv[0] / 10.0, update_gv[1] / 10.0, update_gv[2] / 10.0, update_angular_gv_[0] / 10.0, update_angular_gv_[1] / 10.0, update_angular_gv_[2] / 10.0, /// body linear&angular velocity
        gv_.tail(53)/10.0,
        // gv_.segment(6, 14*3) / 10.0, gv_.segment(6+14*3+1, 9)/10.0,/// joint velocity
        // box_gc_[0] - gc_[0], box_gc_[1] - gc_[1], box_gc_[2] - gc_[2], box_gc_.segment(3, 4); 
        box_height_, platform_height_, place_box_, update_platform_position_[0], update_platform_position_[1], box_width_, std::sin(sim_step_cycle / (240.0/frequency_) * 3.1415 * 2), std::cos(sim_step_cycle / (240.0/frequency_) * 3.1415 * 2),
        update_box_position_[0], update_box_position_[1], update_box_position_[2], update_box_orientation_[0], (update_box_orientation_[1]), (update_box_orientation_[2]), update_box_orientation_[3], box_width_,
        update_box3_position_[0], update_box3_position_[1], update_box3_position_[2], update_box3_orientation_[0], (update_box3_orientation_[1]), (update_box3_orientation_[2]), update_box3_orientation_[3], 
        update_box4_position_[0], update_box4_position_[1], update_box4_position_[2], update_box4_orientation_[0], (update_box4_orientation_[1]), (update_box4_orientation_[2]), update_box4_orientation_[3];

    if (testing3_) {
      if (primitives_[primitive_index_] == 2 or primitives_[primitive_index_] == 3) {
        for (int i = 0; i < 17; i++) {
          obDouble_[10 + i * 4] = joint_quat_error_[i][0];
          obDouble_[10 + i * 4 + 1] = 10 * joint_quat_error_[i][1], obDouble_[10 + i * 4 + 2] = 10 * joint_quat_error_[i][2], obDouble_[10 + i * 4 + 3] = 10 * joint_quat_error_[i][3];
        }
      }

      //platform obs
      if (primitive_index_ == 0) {
        raisim::Vec<3> new_platform_position, update_new_platform_position;
        raisim::Vec<3> platform_position, box_position;
        platform_->getPosition(platform_position);
        box2_->getPosition(box_position);
        new_platform_position[0] = platform_position[0], new_platform_position[1] = platform_position[1], new_platform_position[2] = 0.4;
        for (int i = 0; i < 3; i++)
          new_platform_position[i] = new_platform_position[i] - gc_[i];
        raisim::matvecmul(local_root_matrix_, new_platform_position, update_new_platform_position);
        obDouble_[206+index_offset] = update_new_platform_position[0];
        obDouble_[207+index_offset] = update_new_platform_position[1];
        std::cout << "pos" << update_new_platform_position << update_platform_position_ << update_box_position_ << std::endl;
      }
      else if (primitive_index_ == 2) {
        raisim::Vec<3> new_platform_position, update_new_platform_position;
        raisim::Vec<3> platform_position, box_position;
        platform3_->getPosition(platform_position);
        new_platform_position[0] = platform_position[0], new_platform_position[1] = platform_position[1], new_platform_position[2] = 0.25;
        for (int i = 0; i < 3; i++)
          new_platform_position[i] = new_platform_position[i] - gc_[i];
        raisim::matvecmul(local_root_matrix_, new_platform_position, update_new_platform_position);
        obDouble_[206+index_offset] = update_new_platform_position[0];
        obDouble_[207+index_offset] = update_new_platform_position[1];
        std::cout << "pos" << update_new_platform_position << update_platform_position_ << update_box_position_ << std::endl;
      }
      else if (primitive_index_ == 4) {
        raisim::Vec<3> new_platform_position, update_new_platform_position;
        raisim::Vec<3> platform_position, box_position;
        platform_->getPosition(new_platform_position);
        new_platform_position[2] = 0.325;
        for (int i = 0; i < 3; i++)
          new_platform_position[i] = new_platform_position[i] - gc_[i];
        raisim::matvecmul(local_root_matrix_, new_platform_position, update_new_platform_position);
        obDouble_[206+index_offset] = update_new_platform_position[0];// -0.2;
        obDouble_[207+index_offset] = update_new_platform_position[1];
        std::cout << "pos" << update_new_platform_position << update_platform_position_ << update_box_position_ << std::endl;
      }
      else if (primitive_index_ ==  6) {
        raisim::Vec<3> new_platform_position, update_new_platform_position;
        raisim::Vec<3> platform_position, box_position;
        platform2_->getPosition(platform_position);
        box2_->getPosition(box_position);
        new_platform_position[0] = platform_position[0], new_platform_position[1] = platform_position[1], new_platform_position[2] = 0.25;
        for (int i = 0; i < 3; i++)
          new_platform_position[i] = new_platform_position[i] - gc_[i];
        raisim::matvecmul(local_root_matrix_, new_platform_position, update_new_platform_position);
        obDouble_[206+index_offset] = update_new_platform_position[0];
        obDouble_[207+index_offset] = update_new_platform_position[1];
        std::cout << "pos" << update_new_platform_position << update_platform_position_ << update_box_position_ << std::endl;
      }
      else if (primitive_index_ == 8) {
        raisim::Vec<3> new_platform_position, update_new_platform_position;
        raisim::Vec<3> platform_position, box_position;
        platform3_->getPosition(platform_position);
        box2_->getPosition(box_position);
        new_platform_position[0] = platform_position[0], new_platform_position[1] = platform_position[1], new_platform_position[2] = 0.25;
        for (int i = 0; i < 3; i++)
          new_platform_position[i] = new_platform_position[i] - gc_[i];
        raisim::matvecmul(local_root_matrix_, new_platform_position, update_new_platform_position);
        obDouble_[206+index_offset] = update_new_platform_position[0];
        obDouble_[207+index_offset] = update_new_platform_position[1];
        std::cout << "pos" << update_new_platform_position << update_platform_position_ << update_box_position_ << std::endl;
      }
      else if (primitive_index_ == 10) {
        raisim::Vec<3> new_platform_position, update_new_platform_position;
        raisim::Vec<3> platform_position, box_position;
        platform2_->getPosition(platform_position);
        box2_->getPosition(box_position);
        new_platform_position[0] = platform_position[0], new_platform_position[1] = platform_position[1], new_platform_position[2] = 0.325;
        for (int i = 0; i < 3; i++)
          new_platform_position[i] = new_platform_position[i] - gc_[i];
        raisim::matvecmul(local_root_matrix_, new_platform_position, update_new_platform_position);
        obDouble_[206+index_offset] = update_new_platform_position[0];
        obDouble_[207+index_offset] = update_new_platform_position[1];
        std::cout << "pos" << update_new_platform_position << update_platform_position_ << update_box_position_ << std::endl;
      }
      else if (primitive_index_ == 12){
        raisim::Vec<3> new_platform_position, update_new_platform_position;
        raisim::Vec<3> platform_position, box_position;
        platform_->getPosition(platform_position);
        box2_->getPosition(box_position);
        new_platform_position[0] = platform_position[0], new_platform_position[1] = platform_position[1], new_platform_position[2] = 0.25;
        for (int i = 0; i < 3; i++)
          new_platform_position[i] = new_platform_position[i] - gc_[i];
        raisim::matvecmul(local_root_matrix_, new_platform_position, update_new_platform_position);
        obDouble_[206+index_offset] = update_new_platform_position[0];
        obDouble_[207+index_offset] = update_new_platform_position[1];
        std::cout << "pos" << update_new_platform_position << update_platform_position_ << update_box_position_ << std::endl;
      }
      else if (primitive_index_ == 14){
        raisim::Vec<3> new_platform_position, update_new_platform_position;
        raisim::Vec<3> platform_position, box_position;
        platform3_->getPosition(platform_position);
        box2_->getPosition(box_position);
        new_platform_position[0] = platform_position[0], new_platform_position[1] = platform_position[1], new_platform_position[2] = 0.25;
        for (int i = 0; i < 3; i++)
          new_platform_position[i] = new_platform_position[i] - gc_[i];
        raisim::matvecmul(local_root_matrix_, new_platform_position, update_new_platform_position);
        obDouble_[206+index_offset] = update_new_platform_position[0];
        obDouble_[207+index_offset] = update_new_platform_position[1];
        std::cout << "pos" << update_new_platform_position << update_platform_position_ << update_box_position_ << std::endl;
      }
      else if (primitive_index_ == 16){
        raisim::Vec<3> new_platform_position, update_new_platform_position;
        raisim::Vec<3> platform_position, box_position;
        platform2_->getPosition(platform_position);
        box2_->getPosition(box_position);
        new_platform_position[0] = platform_position[0], new_platform_position[1] = platform_position[1], new_platform_position[2] = 0.325;
        for (int i = 0; i < 3; i++)
          new_platform_position[i] = new_platform_position[i] - gc_[i];
        raisim::matvecmul(local_root_matrix_, new_platform_position, update_new_platform_position);
        obDouble_[206+index_offset] = update_new_platform_position[0];
        obDouble_[207+index_offset] = update_new_platform_position[1];
        std::cout << "pos" << update_new_platform_position << update_platform_position_ << update_box_position_ << std::endl;
      }
      else if (primitive_index_ == 18){
        raisim::Vec<3> new_platform_position, update_new_platform_position;
        raisim::Vec<3> platform_position, box_position;
        platform_->getPosition(platform_position);
        box2_->getPosition(box_position);
        new_platform_position[0] = platform_position[0], new_platform_position[1] = platform_position[1], new_platform_position[2] = 0.25;
        for (int i = 0; i < 3; i++)
          new_platform_position[i] = new_platform_position[i] - gc_[i];
        raisim::matvecmul(local_root_matrix_, new_platform_position, update_new_platform_position);
        obDouble_[206+index_offset] = update_new_platform_position[0];
        obDouble_[207+index_offset] = update_new_platform_position[1];
        std::cout << "pos" << update_new_platform_position << update_platform_position_ << update_box_position_ << std::endl;
      }
      else if (primitive_index_ == 20){
        raisim::Vec<3> new_platform_position, update_new_platform_position;
        raisim::Vec<3> platform_position, box_position;
        platform2_->getPosition(platform_position);
        box2_->getPosition(box_position);
        new_platform_position[0] = platform_position[0], new_platform_position[1] = platform_position[1], new_platform_position[2] = 0.25;
        for (int i = 0; i < 3; i++)
          new_platform_position[i] = new_platform_position[i] - gc_[i];
        raisim::matvecmul(local_root_matrix_, new_platform_position, update_new_platform_position);
        obDouble_[206+index_offset] = update_new_platform_position[0];
        obDouble_[207+index_offset] = update_new_platform_position[1];
        std::cout << "pos" << update_new_platform_position << update_platform_position_ << update_box_position_ << std::endl;
      }
      else if (primitive_index_ == 22){
        raisim::Vec<3> new_platform_position, update_new_platform_position;
        raisim::Vec<3> platform_position, box_position;
        platform3_->getPosition(platform_position);
        box2_->getPosition(box_position);
        new_platform_position[0] = platform_position[0], new_platform_position[1] = platform_position[1], new_platform_position[2] = 0.325;
        for (int i = 0; i < 3; i++)
          new_platform_position[i] = new_platform_position[i] - gc_[i];
        raisim::matvecmul(local_root_matrix_, new_platform_position, update_new_platform_position);
        obDouble_[206+index_offset] = update_new_platform_position[0];
        obDouble_[207+index_offset] = update_new_platform_position[1];
        std::cout << "pos" << update_new_platform_position << update_platform_position_ << update_box_position_ << std::endl;
      }
      else if (primitive_index_ == 24){
        raisim::Vec<3> new_platform_position, update_new_platform_position;
        raisim::Vec<3> platform_position, box_position;
        platform_->getPosition(platform_position);
        box2_->getPosition(box_position);
        new_platform_position[0] = platform_position[0], new_platform_position[1] = platform_position[1], new_platform_position[2] = 0.25;
        for (int i = 0; i < 3; i++)
          new_platform_position[i] = new_platform_position[i] - gc_[i];
        raisim::matvecmul(local_root_matrix_, new_platform_position, update_new_platform_position);
        obDouble_[206+index_offset] = update_new_platform_position[0];
        obDouble_[207+index_offset] = update_new_platform_position[1];
        std::cout << "pos" << update_new_platform_position << update_platform_position_ << update_box_position_ << std::endl;
      }
      else if (primitive_index_ == 26){
        raisim::Vec<3> new_platform_position, update_new_platform_position;
        raisim::Vec<3> platform_position, box_position;
        platform3_->getPosition(platform_position);
        box2_->getPosition(box_position);
        new_platform_position[0] = platform_position[0], new_platform_position[1] = platform_position[1], new_platform_position[2] = 0.4;
        for (int i = 0; i < 3; i++)
          new_platform_position[i] = new_platform_position[i] - gc_[i];
        raisim::matvecmul(local_root_matrix_, new_platform_position, update_new_platform_position);
        obDouble_[206+index_offset] = update_new_platform_position[0];
        obDouble_[207+index_offset] = update_new_platform_position[1];
        std::cout << "pos" << update_new_platform_position << update_platform_position_ << update_box_position_ << std::endl;
      }

      // use box 3 first
      if (primitive_index_ <= 2 or primitive_index_ == 8 or primitive_index_ == 9 or primitive_index_ == 10 or primitive_index_ == 16 or primitive_index_ == 17 or primitive_index_ == 18 or primitive_index_ == 24 or primitive_index_ == 25 or primitive_index_ == 26) {
        obDouble_.segment(211+index_offset, 7) << update_box4_position_[0], update_box4_position_[1], update_box4_position_[2], update_box4_orientation_[0], (update_box4_orientation_[1]), (update_box4_orientation_[2]), update_box4_orientation_[3];  
        obDouble_.segment(219+index_offset, 7) << 0, 0, 0, 0, 0, 0, 0;
        // obDouble_[204] = 0.3 + 0.3;
        obDouble_[203+index_offset] = 0.15;
      }
      else if (primitive_index_ <= 6 or primitive_index_ == 20 or primitive_index_ == 21 or primitive_index_ == 22) {
        obDouble_.segment(211+index_offset, 7) << update_box3_position_[0], update_box3_position_[1], update_box3_position_[2], update_box3_orientation_[0], (update_box3_orientation_[1]), (update_box3_orientation_[2]), update_box3_orientation_[3];  
        obDouble_.segment(219+index_offset, 7) << 0, 0, 0, 0, 0, 0, 0;
        // obDouble_[204] = 0.3 + 0.3;
        obDouble_[203+index_offset] = 0.15;
      }
      else if (primitive_index_ == 12 or primitive_index_ == 13 or primitive_index_ == 14) {
        obDouble_.segment(211+index_offset, 7) << update_box_position_[0], update_box_position_[1], update_box_position_[2], update_box_orientation_[0], (update_box_orientation_[1]), (update_box_orientation_[2]), update_box_orientation_[3];  
        obDouble_.segment(219+index_offset, 7) << 0, 0, 0, 0, 0, 0, 0;
        // obDouble_[204] = 0.3 + 0.3;
        obDouble_[203+index_offset] = 0.15;
      }

      if (primitives_[primitive_index_] == 1 and sim_step_cycle >= 80)
        obDouble_.tail(30).setConstant(0.0);
    }

    if (testing2_ and sim_step_ <= 480) {
     obDouble_.segment(211, 7) << update_box3_position_[0], update_box3_position_[1], update_box3_position_[2], update_box3_orientation_[0], (update_box3_orientation_[1]), (update_box3_orientation_[2]), update_box3_orientation_[3];  
     obDouble_.segment(219, 7) << 0, 0, 0, 0, 0, 0, 0;
     obDouble_[204] = 0.3 + 0.3;
     obDouble_[203] = 0.15;
    }

    if (place_box_ and sim_step_cycle >= 80 and not (testing_ or testing3_))
      obDouble_.tail(22).setConstant(0.0);
    if (place_box_ and sim_step_cycle >= 80 and not (testing_ or testing3_))
      obDouble_.tail(30).setConstant(0.0);
    // std::cout << "sim cycle" << sim_step_cycle << std::endl;

    obDouble_[209+index_offset] = std::sin(sim_step_cycle / (240.0/frequency_) * 3.1415 * 2);
    obDouble_[210+index_offset] = std::cos(sim_step_cycle / (240.0/frequency_) * 3.1415 * 2);
    if (num_boxes_ == 2) {
      obDouble_.tail(7).setConstant(0.0);
    }
    else if (num_boxes_ == 1) {
      obDouble_.tail(14).setConstant(0.0);
    }

  }

  void updateUsefulVariable() {
    //character local rotation
    raisim::Vec<3> left_hip_location, right_hip_location, left_to_right_hip, world_up, local_x, normal_left_to_right_hip;
    humanoid_->getFramePosition(humanoid_->getFrameIdxByName("lhip"), left_hip_location);
    humanoid_->getFramePosition(humanoid_->getFrameIdxByName("rhip"), right_hip_location);
    raisim::vecsub(left_hip_location, right_hip_location, left_to_right_hip);
    double norm = std::sqrt(left_to_right_hip[0]*left_to_right_hip[0]+left_to_right_hip[1]*left_to_right_hip[1]+left_to_right_hip[2]*left_to_right_hip[2]);
    raisim::vecScalarMul(1.0 / norm, left_to_right_hip, normal_left_to_right_hip);
    world_up[0] = 0.0, world_up[1] = 0.0, world_up[2] = 1.0;
    raisim::cross(normal_left_to_right_hip, world_up, local_x);
    for (int i = 0; i < 3; i++) {
        local_root_matrix_[3 * i + 0] = local_x[i];
        local_root_matrix_[3 * i + 1] = normal_left_to_right_hip[i];
        local_root_matrix_[3 * i + 2] = world_up[i]; 
    }

    // box state and hand state
    raisim::Vec<3> box_position, box_to_left_hand_position, box_to_right_hand_position; 
    raisim::Vec<4> box_orientation;
    box2_->getPosition(box_position); box2_->getQuaternion(box_orientation);
    humanoid_->getFramePosition(humanoid_->getFrameIdxByName("lwrist"), leftHandPosition_);
    humanoid_->getFramePosition(humanoid_->getFrameIdxByName("rwrist"), rightHandPosition_);
    previous_hand_distance_ = hand_distance_;
    raisim::vecsub(box_position, leftHandPosition_, box_to_left_hand_position);
    raisim::vecsub(box_position, rightHandPosition_, box_to_right_hand_position);
    raisim::matvecmul(local_root_matrix_, box_to_left_hand_position, update_box_to_left_hand_position_);
    raisim::matvecmul(local_root_matrix_, box_to_right_hand_position, update_box_to_right_hand_position_);

    // std::cout << "left" << update_box_to_left_hand_position_ << std::endl;
    // std::cout << "right" << update_box_to_right_hand_position_ << std::endl;
    update_box_to_left_hand_position_[1] += box_width_ / 2;
    update_box_to_right_hand_position_[1] -= box_width_ / 2;



    for (int i = 0; i < 3; i++)
      box_position[i] = box_position[i] - gc_[i];
    raisim::Mat<3,3> box_rotation_matrix, update_box_rotation_matrix;
    raisim::matvecmul(local_root_matrix_, box_position, update_box_position_);
    raisim::quatToRotMat(box_orientation, box_rotation_matrix);
    raisim::matmul(local_root_matrix_, box_rotation_matrix, update_box_rotation_matrix);
    raisim::rotMatToQuat(update_box_rotation_matrix, update_box_orientation_);

    //box 3 state
    raisim::Vec<3> box3_position; 
    raisim::Vec<4> box3_orientation;
    box3_->getPosition(box3_position); box3_->getQuaternion(box3_orientation);
    for (int i = 0; i < 3; i++)
      box3_position[i] = box3_position[i] - gc_[i];
    raisim::Mat<3,3> box3_rotation_matrix, update_box3_rotation_matrix;
    raisim::matvecmul(local_root_matrix_, box3_position, update_box3_position_);
    raisim::quatToRotMat(box3_orientation, box3_rotation_matrix);
    raisim::matmul(local_root_matrix_, box3_rotation_matrix, update_box3_rotation_matrix);
    raisim::rotMatToQuat(update_box3_rotation_matrix, update_box3_orientation_);

    //box 4 state
    raisim::Vec<3> box4_position; 
    raisim::Vec<4> box4_orientation;
    box4_->getPosition(box4_position); box4_->getQuaternion(box4_orientation);
    for (int i = 0; i < 3; i++)
      box4_position[i] = box4_position[i] - gc_[i];
    raisim::Mat<3,3> box4_rotation_matrix, update_box4_rotation_matrix;
    raisim::matvecmul(local_root_matrix_, box4_position, update_box4_position_);
    raisim::quatToRotMat(box4_orientation, box4_rotation_matrix);
    raisim::matmul(local_root_matrix_, box4_rotation_matrix, update_box4_rotation_matrix);
    raisim::rotMatToQuat(update_box4_rotation_matrix, update_box4_orientation_);

    //platform state
    raisim::Vec<3> platform_position; 
    raisim::Vec<4> platform_orientation;
    platform_->getPosition(platform_position); platform_->getQuaternion(platform_orientation);
    for (int i = 0; i < 3; i++)
      platform_position[i] = platform_position[i] - gc_[i];
    raisim::Mat<3,3> platform_rotation_matrix, update_platform_rotation_matrix;
    raisim::matvecmul(local_root_matrix_, platform_position, update_platform_position_);
    raisim::quatToRotMat(platform_orientation, platform_rotation_matrix);
    raisim::matmul(local_root_matrix_, platform_rotation_matrix, update_platform_rotation_matrix);
    raisim::rotMatToQuat(update_platform_rotation_matrix, update_platform_orientation_);

    //desired box state
    raisim::Vec<3> update_desired_box_position;
    raisim::Vec<3> desired_box_position;
    desired_box_position[0] = desired_box_x_; desired_box_position[1] = desired_box_y_; desired_box_position[2] = 0;
    raisim::matvecmul(local_root_matrix_, desired_box_position, update_desired_box_position);
    update_desired_box_x_ = update_desired_box_position[0], update_desired_box_y_ = update_desired_box_position[1];


    // joint error
    raisim::Vec<4> joint_quat[17];
    raisim::Vec<4> desired_joint_quat[17];
    raisim::Mat<3,3> joint_rot[17], desired_joint_rot[17], joint_rot_error[17];
    for (int i = 0; i < 17; i++) {
      for (int j = 0; j < 4; j++) {
        int index_offset = 0;
        if (i >= 14)
          index_offset = 1;
        joint_quat[i][j] = gc_[7 + 4 * i + j + index_offset];
        desired_joint_quat[i][j] = reference_[7 + 4 * i + j + index_offset];
      }
      
      raisim::quatToRotMat(joint_quat[i], joint_rot[i]);
      raisim::quatToRotMat(desired_joint_quat[i], desired_joint_rot[i]);
      raisim::mattransposematmul(joint_rot[i], desired_joint_rot[i], joint_rot_error[i]);
      raisim::rotMatToQuat(joint_rot_error[i], joint_quat_error_[i]);
      if (joint_quat_error_[i][0] < 0) {
        joint_quat_error_[i][0] *= -1, joint_quat_error_[i][1] *= -1, joint_quat_error_[i][2] *= -1, joint_quat_error_[i][3] *= -1;
      }
    }
  }

  void updatePickUpTaskVariable(int place_box) {
    int sim_step_cycle = sim_step_%(240/frequency_);

    if (sim_step_ < 240/frequency_)
      place_box_ = place_box_;
    else if (sim_step_cycle == 0)
      place_box_ = not place_box_;

    if (testing_ or testing3_) {
      place_box_ = primitives_[primitive_index_];
    }

    if (sim_step_ < 240/frequency_ and place_box_)
      contact_label_ = 0.0;
    else if (sim_step_cycle <= 40/frequency_ and place_box_)
      contact_label_ = 1.0;
    else if (sim_step_cycle <= 40/frequency_)
      contact_label_ = 0.5;
    else if (sim_step_cycle > 80/frequency_) {
      if (place_box_)
        contact_label_ = 0.0;
      else
        contact_label_ = 1.0;
    }
    else
      contact_label_ = 0.5;

    double slerp_variable = 0;
    if (sim_step_cycle <= 40/frequency_ and not place_box_)
      slerp_variable = (40/frequency_ - sim_step_cycle) / (40.0/frequency_);
    else if (sim_step_cycle >= 90/frequency_ and sim_step_cycle <= 140/frequency_ and place_box_)
      slerp_variable = (sim_step_cycle - 90/frequency_) / (50.0/frequency_);
    else if (sim_step_cycle >= 140/frequency_ and sim_step_cycle <= 240/frequency_ and place_box_)
      slerp_variable = 1.0;
    
    if (primitives_[primitive_index_] == 2)
      slerp_variable = 0;
    else if (primitives_[primitive_index_] == 3)
      slerp_variable = 0;
    slerp_for_arm(slerp_variable);

    if (sim_step_ < 240/frequency_ and place_box_)
      desired_box_height_ = box_height_ / 2.0 + platform_height_;
    else if (sim_step_cycle < 40/frequency_ and not place_box_)
      desired_box_height_ = box_height_ / 2.0 + platform_height_;
    else if (sim_step_cycle < 40/frequency_ and place_box_)
      desired_box_height_ = (box_height_ / 2.0 + platform_height_) * (1-(40/frequency_-sim_step_cycle) / (40.0/frequency_))+ 1.2 * ((40/frequency_ - sim_step_cycle) / (40.0/frequency_));
    else if (sim_step_cycle >= 90/frequency_ and sim_step_cycle <= 140/frequency_ and contact_label_ > 0.9)
      desired_box_height_ = 1.2 * (sim_step_cycle - 90/frequency_) / (50.0/frequency_) + (box_height_ / 2.0 + platform_height_) * (1 - (sim_step_cycle - 90/frequency_) / (50.0/frequency_));
    else if (sim_step_cycle >= 90/frequency_ and sim_step_cycle <= 140/frequency_ and contact_label_ < 0.1)
      desired_box_height_ = box_height_ / 2.0 + platform_height_;
    else if (sim_step_cycle >= 140/frequency_ && sim_step_cycle <= 240/frequency_ and contact_label_ > 0.9)
      desired_box_height_ = 1.2;
    else if (sim_step_cycle >= 140/frequency_ && sim_step_cycle <= 240/frequency_ and contact_label_ < 0.1)
      desired_box_height_ = box_height_ / 2.0 + platform_height_;
    else if (sim_step_cycle <= 90/frequency_ and sim_step_cycle >= 40/frequency_)
      desired_box_height_ = box_height_ / 2.0 + platform_height_;
  }

  void observe(Eigen::Ref<EigenVec> ob) final {
    /// convert it to float
    ob = obDouble_.cast<float>();
  }

  bool isTerminalState(float& terminalReward) final {
    
    terminalReward = float(terminalRewardCoeff_) * 0.0f;
    bool termination = false;

    //use contact_label to determine whether hand should be in contact
    bool contact_with_box = false;
    for(auto& contact: humanoid_->getContacts()) {
      if ((world_->getObject(contact.getPairObjectIndex())->getIndexInWorld() == platform_->getIndexInWorld()))
        termination = true;
      if(footIndices_.find(contact.getlocalBodyIndex()) == footIndices_.end()) {   //if the contact body is not the foot
        if (handIndices_.find(contact.getlocalBodyIndex()) == handIndices_.end())  //if the contact body is not the hand either, terminate
          //termination =  true;
          {
            if (sim_step_ % 240 <= 90)
              termination = true;
            // else
            //   continue;
          }
        else {
          // if (contact.getlocalBodyIndex() == humanoid_->getBodyIdx("rwrist"))
          //   std::cout << "right contact force" << contact.getImpulse().e() << std::endl;
          // else if (contact.getlocalBodyIndex() == humanoid_->getBodyIdx("lwrist"))
            // std::cout << "left contact force" << contact.getImpulse().e() << std::endl;
          if (world_->getObject(contact.getPairObjectIndex())->getIndexInWorld() != box2_->getIndexInWorld()) {//if hand in contact other than the box, terminate
            // termination =  true;
          }
          else if (contact_label_ < 0.1) { // if contact label is not true, terminate
            termination = true;
          }
          else {
            contact_with_box = true;
          }
        }
      }
      // else {
      //   if (contact.getlocalBodyIndex() == humanoid_->getBodyIdx("rankle"))
      //       std::cout << "right foot contact force" << contact.getImpulse().e() << std::endl;
      //     else if (contact.getlocalBodyIndex() == humanoid_->getBodyIdx("lankle"))
      //       std::cout << "left foot contact force" << contact.getImpulse().e() << std::endl;
      // }
    }
    if (!contact_with_box && contact_label_ > 0.9){
      termination = true;
    }

    raisim::Vec<3> footPosition;
    humanoid_->getFramePosition(humanoid_->getFrameIdxByName("lankle"), footPosition);
    if (footPosition[0] > 0.1) {
      // std::cout << "something happen1" << std::endl;
      termination = true;
    }
    humanoid_->getFramePosition(humanoid_->getFrameIdxByName("rankle"), footPosition);
    if (footPosition[0] > 0.1) {
      // std::cout << "something happen2" << footPosition[0] << std::endl;
      termination = true;
    }

    if (use_kinematics_) {
      if (rewards_["orientation"] < -10 || (rewards_["task"] < -10000 && sim_step_ > 10))
        termination = true;
    }
    if (!use_kinematics_) {
      if (joint_error_flag_ && sim_step_ > 10) {
        termination = true;
      }
    }

    if (testing_ or testing2_ or testing3_)
      return false;
    
    return termination;
  }

  void setReference(const Eigen::Ref<EigenVec>& ref) final{
    learned_phase_ << ref.cast<double>().segment(0, 9);
    learned_next_phase_ << ref.cast<double>().segment(9, 27);
    prev_reference_ << reference_;
    // reference_ = ref.cast<double>().tail(75);
    reference_.segment(0, 11*4+7) << ref.cast<double>().segment(0+36, 11*4+7);
    reference_.segment(11*4+7, 12) << ref.cast<double>().segment(11*4+7+36, 12);
    reference_[11*4+7+12] = 0.0;
    reference_.segment(11*4+7+12+1, 12) << ref.cast<double>().segment(11*4+7+12+36, 12);
    reference_[11*4+7+12+1+12] = 0.0;
    reference_[2] += 0.05;

    std::cout << "primitive_index_" << primitive_index_ << "primitives_" << primitives_[primitive_index_] << std::endl;

    if (primitives_[primitive_index_] <= 1) {
      reference_.segment(7, 70) << gc_init_.segment(7, 70);
      double interpolation = 0;
      int phase = sim_step_ % (240/frequency_);
      if (phase <= 40/frequency_) {
        interpolation = (40/frequency_ - phase) / (40.0/frequency_);
      }
      else if (phase >= 90/frequency_ and phase <= 140/frequency_) {
        interpolation = (phase - 90/frequency_) / (50.0/frequency_);
      }
      else if (phase >= 140/frequency_)
        interpolation = 1.0;

      // interpolation = 0.0;
      // raisim::Vec<3> footPosition, root_position;
      // humanoid_->getFramePosition(humanoid_->getFrameIdxByName("lankle"), footPosition);
      // humanoid_->getFramePosition(0, root_position);
      // std::cout << "pos diff" << root_position[0] - footPosition[0] << " " << root_position[2] - footPosition[2] << std::endl;

      slerp_for_lower_body(interpolation);
      reference_.segment(64, 12) << 1.0, 0.0, 0.0, 0.0, 0.88, -0.13, 0.13, 0.438, 0.85, -0.2, 0.35, 0.34;
      reference_.segment(51, 12) << 1.0, 0.0, 0.0, 0.0, 0.88, -0.13, -0.13, -0.438, 0.85, -0.2, -0.35, -0.34;
        }
  }

  void slerp_for_lower_body(double t) {
    double degree = (1-t) * 3.1415 / 180.0 *67.5;
    reference_[0] = -0.28 * (1-t) + reference_[0];
    reference_[2] = 0.58 + 0.4 * std::cos(degree);//0.98 - 0.283 * (1-t);

    raisim::Vec<4> current_root_quat; raisim::Mat<3, 3> current_root_rot;
    current_root_quat[0] = reference_[3]; current_root_quat[1] = reference_[4]; current_root_quat[2] = reference_[6]; current_root_quat[3] = reference_[5];
    raisim::quatToRotMat(current_root_quat, current_root_rot);

    for (int i = 0; i < 7; i++) {
      int offset = 0;
      raisim::Vec<4> quatA, quatB;
      quatA[0] = gc_crouching_[3 + i * 4 + offset], quatA[1] = gc_crouching_[4 + i * 4 + offset], quatA[2] = gc_crouching_[5 + i * 4 + offset], quatA[3] = gc_crouching_[6 + i * 4 + offset];
      quatB[0] = 1.0, quatB[1] = 0.0, quatB[2] = 0.0, quatB[3] = 0.0;
      if (i == 0) {
        quatB[0] = 0.5, quatB[1] = 0.5, quatB[2] = 0.5, quatB[3] = 0.5;
      }
      double cosHalfTheta = quatA[0] * quatB[0] + quatA[1] * quatB[1] + quatA[2] * quatB[2] + quatA[3] * quatB[3];
      // if qa=qb or qa=-qb then theta = 0 and we can return qa
      if (std::abs(cosHalfTheta) >= 1.0){
          reference_[7 + i * 4 + offset] = quatA[0];
          reference_[8 + i * 4 + offset] = quatA[1];
          reference_[9 + i * 4 + offset] = quatA[2];
          reference_[10 + i * 4 + offset] = quatA[3];
          continue;
      }
      // Calculate temporary values.
      double halfTheta = std::acos(cosHalfTheta);
      double sinHalfTheta = std::sqrt(1.0 - cosHalfTheta*cosHalfTheta);
      // if theta = 180 degrees then result is not fully defined
      // we could rotate around any axis normal to qa or qb
      if (fabs(sinHalfTheta) < 0.001){ // fabs is floating point absolute
          reference_[3 + i * 4 + offset] = (quatA[0] * 0.5 + quatB[0] * 0.5);
          reference_[4 + i * 4 + offset] = (quatA[1] * 0.5 + quatB[1] * 0.5);
          reference_[5 + i * 4 + offset] = (quatA[2] * 0.5 + quatB[2] * 0.5);
          reference_[6 + i * 4 + offset] = (quatA[3] * 0.5 + quatB[3] * 0.5);
          continue;
      }
      double ratioA = std::sin((1 - t) * halfTheta) / sinHalfTheta;
      double ratioB = std::sin(t * halfTheta) / sinHalfTheta; 
      //calculate Quaternion.
      reference_[3 + i * 4 + offset] = (quatA[0] * ratioA + quatB[0] * ratioB);
      reference_[4 + i * 4 + offset] = (quatA[1] * ratioA + quatB[1] * ratioB);
      reference_[5 + i * 4 + offset] = (quatA[2] * ratioA + quatB[2] * ratioB);
      reference_[6 + i * 4 + offset] = (quatA[3] * ratioA + quatB[3] * ratioB);
    }

    raisim::Mat<3,3> reference_root_rot; raisim::Vec<4> reference_root_quat;
    raisim::Mat<3,3> result_root_rot; raisim::Vec<4> result_root_quat;
    for (int i = 0; i < 4; i++)
      reference_root_quat[i] = reference_[3+i];
    raisim::quatToRotMat(reference_root_quat, reference_root_rot);
    raisim::matmul(current_root_rot, reference_root_rot, result_root_rot);
    raisim::rotMatToQuat(result_root_rot, result_root_quat);
    for (int i = 0; i < 4; i++)
      reference_[3+i] = result_root_quat[i];
    std::cout << "result_root_rot" << result_root_rot;
    std::cout << "reference_root_rot" << reference_root_rot;
    std::cout << "current_root_rot" << current_root_rot;

  }

  void setReferenceVelocity(const Eigen::Ref<EigenVec>& reference_velocity) final {
    // reference_velocity_ << reference_velocity.cast<double>();
    
      reference_velocity_.segment(0, 11*3+6) << reference_velocity.cast<double>().segment(0, 11*3+6);
      reference_velocity_.segment(11*3+6, 9) << reference_velocity.cast<double>().segment(11*3+6, 9);
      reference_velocity_[11*3+6+9] = 0.0;
      reference_velocity_.segment(11*3+6+9+1, 9) << reference_velocity.cast<double>().segment(11*3+6+9, 9);
      reference_velocity_[11*3+6+9+1+9] = 0.0;

      reference_velocity_[1] = -reference_velocity_[2];
      reference_velocity_[2] = 0;
    }

  double get_kinematic_quaternion_error() {
    raisim::Vec<4> quat, quat2;
    raisim::Mat<3,3> rot, rot2, rot_error;

    quat[0] = reference_[3]; quat[1] = reference_[4]; 
    quat[2] = reference_[5]; quat[3] = reference_[6];
    raisim::quatToRotMat(quat, rot);
    quat2[0] = 0.5; quat2[1] = 0.5; 
    quat2[2] = 0.5; quat2[3] = 0.5;
    raisim::quatToRotMat(quat2, rot2);

    raisim::Mat<3, 3> yaw_mat;
    raisim::Vec<3> yaw_euler;
    raisim::Vec<4> yaw_quat;
    yaw_euler[0] = 0; yaw_euler[1] = 0; yaw_euler[2] = desired_yaw_;
    raisim::eulerVecToQuat(yaw_euler, yaw_quat);
    raisim::quatToRotMat(yaw_quat, yaw_mat);

    raisim::Vec<4> desired_quat; raisim::Mat<3, 3> desired_mat;
    raisim::mattransposematmul(yaw_mat, rot2, desired_mat);
    raisim::matmattransposemul(desired_mat, rot, rot_error);

    raisim::rotMatToQuat(rot_error, quat_error_);

    // transform rot so transforming to euler become meaningful
    raisim::Mat<3, 3> rotated_rot;
    raisim::Vec<4> rotated_quat;
    raisim::matmattransposemul(rot, rot2, rotated_rot);
    raisim::rotMatToQuat(rotated_rot, rotated_quat);
    double euler_ref[3];
    double quat_ref[4];
    quat_ref[0] = rotated_quat[0], quat_ref[1] = rotated_quat[1], quat_ref[2] = rotated_quat[2], quat_ref[3] = rotated_quat[3];
    raisim::quatToEulerVec(quat_ref, euler_ref);
    double current_angle_to_target = desired_yaw_ - euler_ref[2];
    double angle_diff = std::cos(current_angle_to_target - angle_to_target_);
    angle_to_target_ = current_angle_to_target;

    return angle_diff;
  }

  void getReference(Eigen::Ref<EigenVec> ref) final {
    get_kinematic_quaternion_error();
    kinematicDouble_ << target_x_ - reference_[0], target_y_ - reference_[1], reference_[2], /// body height
        // quat_error[0], quat_error[1] * 10, quat_error[2] * 10, quat_error[3] * 10, /// body orientation
        quat_error_[0], quat_error_[1] * 10, quat_error_[2] * 10, quat_error_[3] * 10,
        // rot2.e().row(2).transpose(), 
        reference_.tail(68), /// joint angles
        learned_phase_.segment(0, 6), learned_phase_.tail(3) / 10;
    ref = kinematicDouble_.cast<float>();
  }

  void getState(Eigen::Ref<EigenVec> state) final {
    for (int i = 0; i < 20; i++){
      raisim::Vec<3> frame_position;
      raisim::Mat<3,3> frame_orientation;
      raisim::Vec<4> frame_quaternion;
      int bodyIndex = boneIndices_[i];
      if (i == 0)
        bodyIndex = 0;
      humanoid_->getFramePosition(bodyIndex, frame_position);
      humanoid_->getFrameOrientation(bodyIndex, frame_orientation);
      raisim::rotMatToQuat(frame_orientation, frame_quaternion);
      state.segment(7*i, 7) << frame_position[0],frame_position[1],frame_position[2],
        frame_quaternion[0], frame_quaternion[1], frame_quaternion[2], frame_quaternion[3];
    }
    raisim::Vec<3> box_position; raisim::Vec<4> box_orientation;
    box2_->getPosition(box_position), box2_->getQuaternion(box_orientation);
    state.segment(7*20, 7) << box_position[0], box_position[1], box_position[2], box_orientation[0], box_orientation[1],
      box_orientation[2], box_orientation[3];
    box3_->getPosition(box_position), box3_->getQuaternion(box_orientation);
    state.segment(7*21, 7) << box_position[0], box_position[1], box_position[2], box_orientation[0], box_orientation[1],
      box_orientation[2], box_orientation[3];
    platform_->getPosition(box_position), platform_->getQuaternion(box_orientation);
    state.segment(7*22, 7) << box_position[0], box_position[1], box_position[2], box_orientation[0], box_orientation[1],
      box_orientation[2], box_orientation[3];
  }


  void curriculumUpdate() {
  };

  void slerp_for_arm(double t) {
    for (int i = 11; i < 17; i++) {
      raisim::Vec<4> quatA, quatB;
      int index_offset = 0;
      if (i >= 14)
        index_offset = 1;
      quatA[0] = reference_[7 + i * 4 + index_offset], quatA[1] = reference_[8 + i * 4 + index_offset], quatA[2] = reference_[9 + i * 4 + index_offset], quatA[3] = reference_[10 + i * 4 + index_offset];
      if (i == 11 or i == 14 or i == 13 or i == 16)
        quatB[0] = 1.0, quatB[1] = 0.0, quatB[2] = 0.0, quatB[3] = 0.0;
      else if (i == 12) {
        quatB[0] = 0.78192104, quatB[1] = -0.09971977, quatB[2] = -0.0831469, quatB[3] = -0.60970652;
      }
      else if (i == 15)
        quatB[0] = 0.78192104, quatB[1] = -0.09971977, quatB[2] = 0.0831469, quatB[3] = 0.60970652;

      if ((testing_ or testing3_) and primitives_[primitive_index_] == 2) {
        if (i == 12) {
          quatB[0] = 0.88, quatB[1] = -0.13, quatB[2] = -0.13, quatB[3] = -0.438;
        }
        else if (i == 13) {
          quatB[0] = 0.85, quatB[1] = -0.2, quatB[2] = -0.35, quatB[3] = -0.34;
        }
        else if (i == 15) {
          quatB[0] = 0.88, quatB[1] = -0.13, quatB[2] = 0.13, quatB[3] = 0.438;
        }
        else if (i == 16) {
          quatB[0] = 0.85, quatB[1] = -0.2, quatB[2] = 0.35, quatB[3] = 0.34;
        }
      }


      // Calculate angle between them.
      double cosHalfTheta = quatA[0] * quatB[0] + quatA[1] * quatB[1] + quatA[2] * quatB[2] + quatA[3] * quatB[3];
      // if qa=qb or qa=-qb then theta = 0 and we can return qa
      if (std::abs(cosHalfTheta) >= 1.0){
          reference_[7 + i * 4 + index_offset] = quatA[0];
          reference_[8 + i * 4 + index_offset] = quatA[1];
          reference_[9 + i * 4 + index_offset] = quatA[2];
          reference_[10 + i * 4 + index_offset] = quatA[3];
          continue;
      }
      // Calculate temporary values.
      double halfTheta = std::acos(cosHalfTheta);
      double sinHalfTheta = std::sqrt(1.0 - cosHalfTheta*cosHalfTheta);
      // if theta = 180 degrees then result is not fully defined
      // we could rotate around any axis normal to qa or qb
      if (fabs(sinHalfTheta) < 0.001){ // fabs is floating point absolute
          reference_[7 + i * 4 + index_offset] = (quatA[0] * 0.5 + quatB[0] * 0.5);
          reference_[8 + i * 4 + index_offset] = (quatA[1] * 0.5 + quatB[1] * 0.5);
          reference_[9 + i * 4 + index_offset] = (quatA[2] * 0.5 + quatB[2] * 0.5);
          reference_[10 + i * 4 + index_offset] = (quatA[3] * 0.5 + quatB[3] * 0.5);
          continue;
      }
      double ratioA = std::sin((1 - t) * halfTheta) / sinHalfTheta;
      double ratioB = std::sin(t * halfTheta) / sinHalfTheta; 
      //calculate Quaternion.
      reference_[7 + i * 4 + index_offset] = (quatA[0] * ratioA + quatB[0] * ratioB);
      reference_[8 + i * 4 + index_offset] = (quatA[1] * ratioA + quatB[1] * ratioB);
      reference_[9 + i * 4 + index_offset] = (quatA[2] * ratioA + quatB[2] * ratioB);
      reference_[10 + i * 4 + index_offset] = (quatA[3] * ratioA + quatB[3] * ratioB);
    }
  }

  float get_total_reward() {
    return float(total_reward_);
  }

  bool time_limit_reached() {
    return sim_step_ >= max_sim_step_;
  }

  void setCrouchingPose() {
    gc_crouching_ = Eigen::VectorXd::Zero(77);
    for (int i = 0; i < 17; i++) {
      int index_offset = 0;
      if (i >= 14)
        index_offset = 1;
      gc_crouching_[7+4*i+index_offset] = 1.0, gc_crouching_[8+4*i+index_offset] = .0, gc_crouching_[9+4*i+index_offset] = .0, gc_crouching_[10+4*i+index_offset] = .0;
    }
    gc_crouching_[0] = 0, gc_crouching_[1] = 1, gc_crouching_[2] = 2;
    gc_crouching_[3] = 0.5; gc_crouching_[4] = 0.5, gc_crouching_[5] = 0.5, gc_crouching_[6] = 0.5;

    raisim::Mat<3,3> root_rotation; raisim::Vec<4> root_rotation_quat;
    raisim::Mat<3,3> root_default; raisim::Vec<4> root_default_quat;
    raisim::Mat<3,3> root_result; raisim::Vec<4> root_result_quat;
    root_rotation_quat[0] = 0.9238795, root_rotation_quat[1] = 0, root_rotation_quat[2] = 0.3826834, root_rotation_quat[3] = 0;
    root_default_quat[0] = 0.5, root_default_quat[1] = 0.5, root_default_quat[2] = 0.5, root_default_quat[3] = 0.5;
    raisim::quatToRotMat(root_rotation_quat, root_rotation);
    raisim::quatToRotMat(root_default_quat, root_default);
    raisim::matmul(root_rotation, root_default, root_result);
    raisim::rotMatToQuat(root_result, root_result_quat);
    gc_crouching_[3] = root_result_quat[0]; gc_crouching_[4] = root_result_quat[1], gc_crouching_[5] = root_result_quat[2], gc_crouching_[6] = root_result_quat[3];

    
    gc_crouching_[7] = 0.5555702, gc_crouching_[8] = -0.8314696, gc_crouching_[9] = 0.0, gc_crouching_[10] = 0.0;
    gc_crouching_[19] = 0.5555702, gc_crouching_[20] = -0.8314696, gc_crouching_[21] = 0.0, gc_crouching_[22] = 0.0;

    gc_crouching_[11] = 0.8314696 , gc_crouching_[12] = 0.5555702, gc_crouching_[13] = 0, gc_crouching_[14] = 0.0;
    gc_crouching_[23] = 0.8314696 , gc_crouching_[24] = 0.5555702, gc_crouching_[25] = 0, gc_crouching_[26] = 0.0;
    gc_crouching_[2] = 1;
  }


 private:
  int gcDim_, gvDim_, nJoints_;
  int phase_ = 0, sim_step_ = 0, max_sim_step_ = 1000;
  int max_phase_ = 40;
  bool visualizable_ = false;
  bool use_kinematics_ = false;
  raisim::ArticulatedSystem* humanoid_;
  raisim::ArticulatedSystem* visual_humanoid_;
  raisim::Box* box2_;
  raisim::Box* box3_;
  raisim::Box* box4_;
  raisim::Box* platform_;
  raisim::Box* platform2_;
  raisim::Box* platform3_;
  raisim::Box* platform4_;
  double platform_height_;
  std::vector<raisim::Box*> boxes_;
  Eigen::VectorXd gc_init_, gv_init_, gc_, gv_, pTarget_, pTarget12_, vTarget_, reference_, prev_reference_, jointPgain_, jointDgain_, prev_gc_, prev_gv_, torque_, previous_torque_;
  Eigen::VectorXd box_gc_init_, box_gv_init_, box_gc_, box_gv_;
  Eigen::VectorXd learned_phase_, learned_next_phase_, reference_velocity_;
  double terminalRewardCoeff_ = -10.;
  double speed_;
  double total_reward_ = 0;
  Eigen::VectorXd actionMean_, actionStd_, obDouble_, kinematicDouble_;
  Eigen::Vector3d bodyLinearVel_, bodyAngularVel_;
  std::set<size_t> footIndices_, handIndices_;
  std::vector<size_t> boneIndices_;
  bool joint_error_flag_;
  double target_x_, target_y_;
  double distance_to_target_ = 0, angle_to_target_ = 0;
  raisim::Visuals *visual_ball;
  int reach_target_counter = 10;
  raisim::Vec<4> quat_error_;
  double desired_yaw_ = 0;
  raisim::Vec<3> leftHandPosition_, rightHandPosition_;
  double hand_distance_ = 0, previous_hand_distance_;
  bool rightHandContactActive_, leftHandContactActive_;
  double box_width_ = 0.0;
  int enviornment_id_ = 0;
  raisim::Vec<4> update_box_orientation_, update_box3_orientation_, update_box4_orientation_; raisim::Vec<3> update_box_position_, update_box3_position_, update_box4_position_;
  raisim::Vec<4> update_platform_orientation_; raisim::Vec<3> update_platform_position_;
  raisim::Mat<3, 3> local_root_matrix_;
  raisim::Vec<3> update_box_to_left_hand_position_, update_box_to_right_hand_position_;
  raisim::Vec<4> joint_quat_error_[17];
  double contact_label_ = 0;
  double box_height_ = 0;
  double box3_height_ = 0;
  double box4_height_ = 0;
  bool place_box_ = false;
  double desired_box_height_ = 0;
  double desired_box_x_, desired_box_y_, update_desired_box_x_, update_desired_box_y_;
  int num_boxes_ = 0;
  Eigen::VectorXd gc_crouching_;

  bool testing_ = false;
  bool testing2_ = false;
  bool testing3_ = true;

  //set up for testing3
  // 0 pick up, 1 put down, 2 walk carry, 3 walk
  #define NUM_PRIMITIVES 27
  int primitives_[NUM_PRIMITIVES] = {0,   2,   1,      3,         0,      2,  1,  3, 0, 2, 1, 3, 0, 2, 1, 3, 0, 2, 1, 3, 0, 2, 1, 3, 0, 2, 1};
  int timing_[NUM_PRIMITIVES] = {75, 120,   75,     120,     75,    160,   75,  160, 75, 160, 75, 160, 75, 120, 75, 160, 75, 160, 75, 160, 75, 160, 75, 120, 75, 120, 75};
  int start_time_[NUM_PRIMITIVES] = {0,   0,   0,    0,       0,       0,      0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  int primitive_index_ = 0;
  int current_primitive_timer_ = 0;
  int frequency_ = 2;
  double init_reference_error_x_, init_reference_error_y_;
};
}