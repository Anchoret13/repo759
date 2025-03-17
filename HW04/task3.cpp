#include <cstddef>
#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <random>
#include <omp.h>
#include <string>

using std::chrono::high_resolution_clock;
using std::chrono::duration;

// Constants
const double G = 1.0;          // Gravitational constant
const double softening = 0.1;  // Softening length
const double dt = 0.01;        // Time step
const double board_size = 4.0; // Size of the board

// Function to calculate acceleration due to gravity with different scheduling options
void getAcc(const double pos[][3], const double mass[], double acc[][3], int N, int num_threads, const std::string& schedule) {
    // Initialize acceleration array to zeros
    if (schedule == "static") {
        #pragma omp parallel for num_threads(num_threads) schedule(static)
        for (int i = 0; i < N; i++) {
            acc[i][0] = 0.0;
            acc[i][1] = 0.0;
            acc[i][2] = 0.0;
        }
    } else if (schedule == "dynamic") {
        #pragma omp parallel for num_threads(num_threads) schedule(dynamic)
        for (int i = 0; i < N; i++) {
            acc[i][0] = 0.0;
            acc[i][1] = 0.0;
            acc[i][2] = 0.0;
        }
    } else if (schedule == "guided") {
        #pragma omp parallel for num_threads(num_threads) schedule(guided)
        for (int i = 0; i < N; i++) {
            acc[i][0] = 0.0;
            acc[i][1] = 0.0;
            acc[i][2] = 0.0;
        }
    }

    // Calculate pairwise gravitational forces
    if (schedule == "static") {
        #pragma omp parallel for num_threads(num_threads) schedule(static)
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                if (i != j) {
                    double dx = pos[j][0] - pos[i][0];
                    double dy = pos[j][1] - pos[i][1];
                    double dz = pos[j][2] - pos[i][2];
                    
                    double inv_r3 = pow(dx*dx + dy*dy + dz*dz + softening*softening, -1.5);
                    
                    acc[i][0] += G * (dx * inv_r3) * mass[j];
                    acc[i][1] += G * (dy * inv_r3) * mass[j];
                    acc[i][2] += G * (dz * inv_r3) * mass[j];
                }
            }
        }
    } else if (schedule == "dynamic") {
        #pragma omp parallel for num_threads(num_threads) schedule(dynamic)
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                if (i != j) {
                    double dx = pos[j][0] - pos[i][0];
                    double dy = pos[j][1] - pos[i][1];
                    double dz = pos[j][2] - pos[i][2];
                    
                    double inv_r3 = pow(dx*dx + dy*dy + dz*dz + softening*softening, -1.5);
                    
                    acc[i][0] += G * (dx * inv_r3) * mass[j];
                    acc[i][1] += G * (dy * inv_r3) * mass[j];
                    acc[i][2] += G * (dz * inv_r3) * mass[j];
                }
            }
        }
    } else if (schedule == "guided") {
        #pragma omp parallel for num_threads(num_threads) schedule(guided)
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                if (i != j) {
                    double dx = pos[j][0] - pos[i][0];
                    double dy = pos[j][1] - pos[i][1];
                    double dz = pos[j][2] - pos[i][2];
                    
                    double inv_r3 = pow(dx*dx + dy*dy + dz*dz + softening*softening, -1.5);
                    
                    acc[i][0] += G * (dx * inv_r3) * mass[j];
                    acc[i][1] += G * (dy * inv_r3) * mass[j];
                    acc[i][2] += G * (dz * inv_r3) * mass[j];
                }
            }
        }
    }
}

int main(int argc, char *argv[]) {
    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, std::milli> duration_sec;
    start = high_resolution_clock::now();

    // Check if correct number of arguments are provided
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " <number_of_particles> <simulation_end_time> <num_threads> <schedule>" << std::endl;
        std::cerr << "Schedule options: static, dynamic, guided" << std::endl;
        return 1;
    }

    // Read N, tEnd, and num_threads from command line
    int N = std::stoi(argv[1]);             // Number of particles
    double tEnd = std::stod(argv[2]);       // Time at which simulation ends
    int num_threads = std::stoi(argv[3]);   // Number of OpenMP threads to use
    std::string schedule = argv[4];         // Scheduling policy

    // Validate schedule option
    if (schedule != "static" && schedule != "dynamic" && schedule != "guided") {
        std::cerr << "Invalid scheduling option. Use: static, dynamic, or guided" << std::endl;
        return 1;
    }

    // Set the number of threads for OpenMP
    omp_set_num_threads(num_threads);

    // Print information about parallelization
    std::cout << "Using " << num_threads << " OpenMP threads with " << schedule << " scheduling" << std::endl;

    // Allocate dynamic arrays based on N
    double* mass = new double[N];
    double(*pos)[3] = new double[N][3];
    double(*vel)[3] = new double[N][3];
    double(*acc)[3] = new double[N][3];

    // Create a random number engine with fixed seed for reproducibility
    std::mt19937 generator(42);  // Fixed seed for consistent results across experiments

    // Create random distributions
    std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);
    std::normal_distribution<double> normal_dist(0.0, 1.0);

    // Simulation parameters
    double t = 0.0;

    // Set initial masses and random positions/velocities
    for (int i = 0; i < N; i++) {
        mass[i] = uniform_dist(generator);

        pos[i][0] = normal_dist(generator);
        pos[i][1] = normal_dist(generator);
        pos[i][2] = normal_dist(generator);

        vel[i][0] = normal_dist(generator);
        vel[i][1] = normal_dist(generator);
        vel[i][2] = normal_dist(generator);
    }

    // Convert to Center-of-Mass frame
    double velCM[3] = {0.0, 0.0, 0.0};
    double totalMass = 0.0;
    
    if (schedule == "static") {
        #pragma omp parallel num_threads(num_threads)
        {
            double local_velCM[3] = {0.0, 0.0, 0.0};
            double local_totalMass = 0.0;
            
            #pragma omp for schedule(static)
            for (int i = 0; i < N; i++) {
                local_velCM[0] += vel[i][0] * mass[i];
                local_velCM[1] += vel[i][1] * mass[i];
                local_velCM[2] += vel[i][2] * mass[i];
                local_totalMass += mass[i];
            }
            
            #pragma omp critical
            {
                velCM[0] += local_velCM[0];
                velCM[1] += local_velCM[1];
                velCM[2] += local_velCM[2];
                totalMass += local_totalMass;
            }
        }
    } else if (schedule == "dynamic") {
        #pragma omp parallel num_threads(num_threads)
        {
            double local_velCM[3] = {0.0, 0.0, 0.0};
            double local_totalMass = 0.0;
            
            #pragma omp for schedule(dynamic)
            for (int i = 0; i < N; i++) {
                local_velCM[0] += vel[i][0] * mass[i];
                local_velCM[1] += vel[i][1] * mass[i];
                local_velCM[2] += vel[i][2] * mass[i];
                local_totalMass += mass[i];
            }
            
            #pragma omp critical
            {
                velCM[0] += local_velCM[0];
                velCM[1] += local_velCM[1];
                velCM[2] += local_velCM[2];
                totalMass += local_totalMass;
            }
        }
    } else if (schedule == "guided") {
        #pragma omp parallel num_threads(num_threads)
        {
            double local_velCM[3] = {0.0, 0.0, 0.0};
            double local_totalMass = 0.0;
            
            #pragma omp for schedule(guided)
            for (int i = 0; i < N; i++) {
                local_velCM[0] += vel[i][0] * mass[i];
                local_velCM[1] += vel[i][1] * mass[i];
                local_velCM[2] += vel[i][2] * mass[i];
                local_totalMass += mass[i];
            }
            
            #pragma omp critical
            {
                velCM[0] += local_velCM[0];
                velCM[1] += local_velCM[1];
                velCM[2] += local_velCM[2];
                totalMass += local_totalMass;
            }
        }
    }

    velCM[0] /= totalMass;
    velCM[1] /= totalMass;
    velCM[2] /= totalMass;

    // Apply velocity correction with appropriate scheduling
    if (schedule == "static") {
        #pragma omp parallel for num_threads(num_threads) schedule(static)
        for (int i = 0; i < N; i++) {
            vel[i][0] -= velCM[0];
            vel[i][1] -= velCM[1];
            vel[i][2] -= velCM[2];
        }
    } else if (schedule == "dynamic") {
        #pragma omp parallel for num_threads(num_threads) schedule(dynamic)
        for (int i = 0; i < N; i++) {
            vel[i][0] -= velCM[0];
            vel[i][1] -= velCM[1];
            vel[i][2] -= velCM[2];
        }
    } else if (schedule == "guided") {
        #pragma omp parallel for num_threads(num_threads) schedule(guided)
        for (int i = 0; i < N; i++) {
            vel[i][0] -= velCM[0];
            vel[i][1] -= velCM[1];
            vel[i][2] -= velCM[2];
        }
    }

    // Initial accelerations
    getAcc(pos, mass, acc, N, num_threads, schedule);

    // Number of timesteps
    int Nt = int(tEnd / dt);

    // Main simulation loop
    for (int step = 0; step < Nt; step++) {
        
        // (1/2) kick with appropriate scheduling
        if (schedule == "static") {
            #pragma omp parallel for num_threads(num_threads) schedule(static)
            for (int i = 0; i < N; i++) {
                vel[i][0] += acc[i][0] * dt / 2.0;
                vel[i][1] += acc[i][1] * dt / 2.0;
                vel[i][2] += acc[i][2] * dt / 2.0;
            }
        } else if (schedule == "dynamic") {
            #pragma omp parallel for num_threads(num_threads) schedule(dynamic)
            for (int i = 0; i < N; i++) {
                vel[i][0] += acc[i][0] * dt / 2.0;
                vel[i][1] += acc[i][1] * dt / 2.0;
                vel[i][2] += acc[i][2] * dt / 2.0;
            }
        } else if (schedule == "guided") {
            #pragma omp parallel for num_threads(num_threads) schedule(guided)
            for (int i = 0; i < N; i++) {
                vel[i][0] += acc[i][0] * dt / 2.0;
                vel[i][1] += acc[i][1] * dt / 2.0;
                vel[i][2] += acc[i][2] * dt / 2.0;
            }
        }

        // Drift with appropriate scheduling
        if (schedule == "static") {
            #pragma omp parallel for num_threads(num_threads) schedule(static)
            for (int i = 0; i < N; i++) {
                pos[i][0] += vel[i][0] * dt;
                pos[i][1] += vel[i][1] * dt;
                pos[i][2] += vel[i][2] * dt;
            }
        } else if (schedule == "dynamic") {
            #pragma omp parallel for num_threads(num_threads) schedule(dynamic)
            for (int i = 0; i < N; i++) {
                pos[i][0] += vel[i][0] * dt;
                pos[i][1] += vel[i][1] * dt;
                pos[i][2] += vel[i][2] * dt;
            }
        } else if (schedule == "guided") {
            #pragma omp parallel for num_threads(num_threads) schedule(guided)
            for (int i = 0; i < N; i++) {
                pos[i][0] += vel[i][0] * dt;
                pos[i][1] += vel[i][1] * dt;
                pos[i][2] += vel[i][2] * dt;
            }
        }

        // Ensure particles stay within the board limits with appropriate scheduling
        if (schedule == "static") {
            #pragma omp parallel for num_threads(num_threads) schedule(static)
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < 3; j++) {
                    if (pos[i][j] > board_size) pos[i][j] = board_size;
                    if (pos[i][j] < -board_size) pos[i][j] = -board_size;
                }
            }
        } else if (schedule == "dynamic") {
            #pragma omp parallel for num_threads(num_threads) schedule(dynamic)
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < 3; j++) {
                    if (pos[i][j] > board_size) pos[i][j] = board_size;
                    if (pos[i][j] < -board_size) pos[i][j] = -board_size;
                }
            }
        } else if (schedule == "guided") {
            #pragma omp parallel for num_threads(num_threads) schedule(guided)
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < 3; j++) {
                    if (pos[i][j] > board_size) pos[i][j] = board_size;
                    if (pos[i][j] < -board_size) pos[i][j] = -board_size;
                }
            }
        }

        // Update accelerations
        getAcc(pos, mass, acc, N, num_threads, schedule);

        // (1/2) kick with appropriate scheduling
        if (schedule == "static") {
            #pragma omp parallel for num_threads(num_threads) schedule(static)
            for (int i = 0; i < N; i++) {
                vel[i][0] += acc[i][0] * dt / 2.0;
                vel[i][1] += acc[i][1] * dt / 2.0;
                vel[i][2] += acc[i][2] * dt / 2.0;
            }
        } else if (schedule == "dynamic") {
            #pragma omp parallel for num_threads(num_threads) schedule(dynamic)
            for (int i = 0; i < N; i++) {
                vel[i][0] += acc[i][0] * dt / 2.0;
                vel[i][1] += acc[i][1] * dt / 2.0;
                vel[i][2] += acc[i][2] * dt / 2.0;
            }
        } else if (schedule == "guided") {
            #pragma omp parallel for num_threads(num_threads) schedule(guided)
            for (int i = 0; i < N; i++) {
                vel[i][0] += acc[i][0] * dt / 2.0;
                vel[i][1] += acc[i][1] * dt / 2.0;
                vel[i][2] += acc[i][2] * dt / 2.0;
            }
        }

        // Update time
        t += dt;
    }

    // Clean up dynamically allocated memory
    delete[] mass;
    delete[] pos;
    delete[] vel;
    delete[] acc;

    end = high_resolution_clock::now();
    duration_sec = std::chrono::duration_cast<duration<double, std::milli> >(end - start);
    
    // Print the result in a format suitable for data collection
    std::cout << "Schedule: " << schedule << ", Threads: " << num_threads 
              << ", Particles: " << N << ", Time: " << duration_sec.count() << " ms" << std::endl;

    return 0;
}