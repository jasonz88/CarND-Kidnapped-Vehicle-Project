/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
    // TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
    //   x, y, theta and their uncertainties from GPS) and all weights to 1. 
    // Add random Gaussian noise to each particle.
    // NOTE: Consult particle_filter.h for more information about this method (and others in this file).
    
    num_particles = 100;
    // Resize weights and particles vector based on num_particles
    particles.resize(num_particles);

    default_random_engine gen;

    // Create normal distributions for x, y and theta
    normal_distribution<double> dist_x(x, std[0]);

    normal_distribution<double> dist_y(y, std[1]);
    normal_distribution<double> dist_theta(theta, std[2]);

    // Initializes particles - from the normal distributions set above
    for (int i = 0; i < num_particles; ++i) {

        // Add generated particle data to particles class
        particles[i].id = i;
        particles[i].x = dist_x(gen);
        particles[i].y = dist_y(gen);
        particles[i].theta = dist_theta(gen);
        particles[i].weight = 1.0;
    }

    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
    // TODO: Add measurements to each particle and add random Gaussian noise.
    // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
    //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
    //  http://www.cplusplus.com/reference/random/default_random_engine/
    // Engine for later generation of particles
    default_random_engine gen;

    // Make distributions for adding noise
    normal_distribution<double> dist_x(0, std_pos[0]);
    normal_distribution<double> dist_y(0, std_pos[1]);
    normal_distribution<double> dist_theta(0, std_pos[2]);

    // Different equations based on if yaw_rate is zero or not
    for (int i = 0; i < num_particles; ++i) {
        if (fabs(yaw_rate) > 1e-4) {
            // Add measurements to particles
            particles[i].x += (velocity/yaw_rate) * (sin(particles[i].theta + (yaw_rate * delta_t)) - sin(particles[i].theta));
            particles[i].y += (velocity/yaw_rate) * (cos(particles[i].theta) - cos(particles[i].theta + (yaw_rate * delta_t)));
            particles[i].theta += yaw_rate * delta_t;
        } else {
            // Add measurements to particles
            particles[i].x += velocity * delta_t * cos(particles[i].theta);
            particles[i].y += velocity * delta_t * sin(particles[i].theta);
            // Theta will stay the same due to no yaw_rate
        }

        // Add noise to the particles
        particles[i].x += dist_x(gen);
        particles[i].y += dist_y(gen);
        particles[i].theta += dist_theta(gen);
    }

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
    // TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
    //   observed measurement to this particular landmark.
    // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
    //   implement this method and use it as a helper during the updateWeights phase.

    for (auto &ob: observations) {
        double min_error = numeric_limits<double>::max();

        for (const auto &pred: predicted){
            const double dx = pred.x - ob.x;
            const double dy = pred.y - ob.y;
            const double error = dx * dx + dy * dy;

            if (error < min_error) {
                min_error = error;
                ob.id = pred.id;
            }
        }
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
    std::vector<LandmarkObs> observations, Map map_landmarks) {
    // TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
    //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
    // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
    //   according to the MAP'S coordinate system. You will need to transform between the two systems.
    //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
    //   The following is a good resource for the theory:
    //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
    //   and the following is a good resource for the actual equation to implement (look at equation 
    //   3.33
    //   http://planning.cs.uiuc.edu/node99.html
    
    // constants used later for calculating the new weights

    for (auto &p: particles) {

        // transform each observations to map coordinates
        // assume observations are made in the particle's perspective
        vector<LandmarkObs> map_observations;
        double cos_theta = cos(p.theta);
        double sin_theta = sin(p.theta);

        for (const auto &ob: observations) {
            LandmarkObs lm;
            lm.x = ob.x * cos_theta - ob.y * sin_theta + p.x;
            lm.y = ob.x * sin_theta + ob.y * cos_theta + p.y;
            lm.id = ob.id;
            map_observations.push_back(lm);
        }

        // Find map landmarks within the sensor range
        vector<LandmarkObs> landmarks_in_range;
        for (const auto &lm: map_landmarks.landmark_list) {
            double dx = p.x - lm.x_f, dy = p.y - lm.y_f;
            double dist = sqrt(dx * dx + dy * dy);
            if ( dist < sensor_range) {
                landmarks_in_range.push_back(LandmarkObs{lm.id_i, lm.x_f, lm.y_f});
            }
        }
        // return if no landmarks within the sensor range
        if (landmarks_in_range.empty()) {
            return;
        }

        // Associate landmark in range (id) to landmark observations
        dataAssociation(landmarks_in_range, map_observations);

        // Compare each observation by the vehicle (map_observations)
        // to corresponding observation by the particle (landmark_in_range)
        // update the particle weight based on this

        double nx = (2 * pow(std_landmark[0], 2));
        double ny = (2 * pow(std_landmark[1], 2));
        double nd = (2 * M_PI * std_landmark[0] * std_landmark[1]);
        p.weight = 1.0;
        for (const auto &ob: map_observations) {
            Map::single_landmark_s lm = map_landmarks.landmark_list.at(ob.id-1);
            double x_term = pow(ob.x - lm.x_f, 2) / nx;
            double y_term = pow(ob.y - lm.y_f, 2) / ny;
            double w = exp(-(x_term + y_term)) / nd;
            p.weight *=  w;
        }
        weights.push_back(p.weight);
    }
}

void ParticleFilter::resample() {
    // TODO: Resample particles with replacement with probability proportional to their weight. 
    // NOTE: You may find std::discrete_distribution helpful here.
    //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    // generate distribution according to weights

    random_device rd;
    mt19937 gen(rd());
    discrete_distribution<> dist(weights.begin(), weights.end());

    // create resampled particles
    vector<Particle> resampled_particles;
    resampled_particles.resize(num_particles);

    // resample the particles according to weights
    for(int i=0; i < num_particles; i++){
        int idx = dist(gen);
        resampled_particles[i] = particles[idx];
    }

    // assign the resampled_particles to the previous particles, clear weights for the next round
    particles = resampled_particles;
    weights.clear();
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    //Clear the previous associations
    particle.associations.clear();
    particle.sense_x.clear();
    particle.sense_y.clear();

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;

    return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
    vector<int> v = best.associations;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
    vector<double> v = best.sense_x;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
    vector<double> v = best.sense_y;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
