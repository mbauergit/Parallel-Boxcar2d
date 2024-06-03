#include "box2d/box2d.h"
#include "car.hpp"
#include "configurations.hpp"
#include "ground.hpp"
#include "json.hpp"
#include "population.hpp"
#include "randomgen.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdio.h>
#include <time.h>
#include <vector>
#include <mpi.h>
#include <chrono>
#include <cstring>

using json = nlohmann::json;

char *getCmdOption(char **begin, char **end, const std::string &option)
{
    char **it = std::find(begin, end, option);
    if (it != end && ++it != end)
    {
        return *it;
    }
    return 0;
}

bool cmdOptionExists(char **begin, char **end, const std::string &option)
{
    return std::find(begin, end, option) != end;
}

int find_arg_idx(int argc, char **argv, const char *option)
{
    for (int i = 1; i < argc; ++i)
    {
        if (strcmp(argv[i], option) == 0)
        {
            return i;
        }
    }
    return -1;
}

int find_int_arg(int argc, char **argv, const char *option, int default_value)
{
    int iplace = find_arg_idx(argc, argv, option);

    if (iplace >= 0 && iplace < argc - 1)
    {
        return std::stoi(argv[iplace + 1]);
    }

    return default_value;
}

char *find_string_option(int argc, char **argv, const char *option, char *default_value)
{
    int iplace = find_arg_idx(argc, argv, option);

    if (iplace >= 0 && iplace < argc - 1)
    {
        return argv[iplace + 1];
    }

    return default_value;
}

MPI_Datatype GENOME;

int main(int argc, char *argv[])
{
    if (cmdOptionExists(argv, argv + argc, "-h") ||
        cmdOptionExists(argv, argv + argc, "--help"))
    {
        printf("Usage: %s [options]\n", argv[0]);
        printf("Options:\n");
        printf("  -h, --help\t\tShow this help message\n");
        return 0;
    }

    int num_procs, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Create MPI Genome Type

    int count = 4; // We consider the main arrays and vectors
    MPI_Datatype types[] = {
        MPI_FLOAT, // legAngleWeight (8 elements)
        MPI_FLOAT, // legLength (8 elements)
        MPI_INT,   // wheelLeg (2 elements)
        MPI_FLOAT  // wheelLength (2 elements)
    };
    int block_lengths[] = {8, 8, 2, 2};
    MPI_Aint offsets[] = {
        offsetof(Genome, legAngleWeight),
        offsetof(Genome, legLength),
        offsetof(Genome, wheelLeg),
        offsetof(Genome, wheelLength)};

    MPI_Type_create_struct(count, block_lengths, offsets, types, &GENOME);
    MPI_Type_commit(&GENOME);

    int generationNumber = 0;
    int carNumber = 0; // testing which car in population

    // Time tracking
    u_int16_t frames = 0;             // number of frames passed
    u_int32_t simTimeElapsed = 0;     // simulation elapsed time in microseconds
    u_int32_t generationDuration = 0; // time spent on current generation

    float timeGoal = 0; // if the currMaxDist is not greater than timeGoal in
                        // conf::timeLimit, than end the car trial
    float currMaxDist = 0;
    float recordDist = 0;
    int extensions = 0; // how many times the ground has been extended
    float extensionUnit =
        conf::drawScale * ((float)conf::extendNumber * conf::extendUnitLength);

    std::array<float, 8> bestLegAngleWeight;
    std::array<float, 8> bestLegLength;
    std::array<int, 2> bestWheelLeg;
    std::array<float, 2> bestWheelLength;
    int terrainSeed = conf::terrainSeed;
    int genomeSeed = conf::genomeSeed;
    int numCars = conf::populationSize;

    // Set seeds for random number generators
    std::mt19937 terrainRNG(terrainSeed);
    std::mt19937 genomeRNG(genomeSeed);

    Population population(numCars, genomeRNG); // create population (kept track by rank 0)
    Genome my_car(genomeRNG);                  // for all other ranks

    // Initialize Population of Cars and send to rest of processors only if you are "master" rank
    if (rank == 0)
    {
        // send genomes to respective processors
        for (int i = 0; i < numCars; ++i)
        {
            MPI_Send(&population.car[i], 1, GENOME, i + 1, 0, MPI_COMM_WORLD);
            std::cout << "rank " << rank << " sent to rank " << i + 1 << std::endl;
        }
    }
    // Recieve specific car genome if you are "slave" rank
    else
    {
        if (rank < numCars + 1)
        {
            MPI_Recv(&my_car, 1, GENOME, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            std::cout << "rank " << rank << " recieved from rank " << 0 << std::endl;
        }
    }

    // GA algorithm starts
    // every iteration of the alg until target score is met:
    // barrier wait for all processors to finish simulating
    //  if master: calculate
    //  if slave:
    auto start_time = std::chrono::steady_clock::now();
    bool global_finished = false;
    float target_score = 15000; // target distance to reach
    Genome bestCarGenome = population.car[0];

    while (!global_finished)
    {
        // wait for simulations to finish
        if (rank > 0 and !(rank >= numCars + 1))
        {

            std::cout << "rank " << rank << " starting sim" << std::endl;
            // srand(ground_seed);
            // b2World world(conf::gravity);
            // b2Vec2 center;
            // Ground ground(&world);
            // srand(time(NULL));
            // Car car(my_car, &world);

            b2World world(conf::gravity);
            b2Vec2 center;
            Car car(my_car, &world);
            Ground ground(&world, terrainRNG);

            // checking cargenome was recieved properly
            //  for (int i = 0; i < 8; ++i) {
            //      std::cout << "rank " << rank << "my car genome"<< my_car.legAngleWeight[i]<< std::endl;
            //  }
            //  for (int i = 0; i < 8; ++i) {
            //      std::cout << "rank " << rank << "my car genome"<< my_car.legLength[i]<< std::endl;
            //  }
            //  for (int i = 0; i < 2; ++i) {
            //      std::cout << "rank " << rank << "my car genome"<< my_car.wheelLeg[i]<< std::endl;
            //  }
            //  for (int i = 0; i < 2; ++i) {
            //      std::cout << "rank " << rank << "my car genome"<< my_car.wheelLength[i]<< std::endl;
            //  }

            // Time tracking
            u_int16_t frames = 0;         // number of frames passed
            u_int32_t simTimeElapsed = 0; // elapsed time in simulation

            float timeGoal = 0.0f;
            float currMaxDist = 0.0f;
            float terrainExtensionFactor = 1.0f;
            int extensions = 0; // how many times the ground has been extended
            float extensionUnit =
                conf::drawScale * ((float)conf::extendNumber * conf::extendUnitLength);

            bool finished = false;

            while (currMaxDist < target_score && !finished)
            {
                center = car.getCenter();
                if (center.x > currMaxDist)
                // update currMaxDist
                {
                    currMaxDist = center.x;
                    if (currMaxDist >= extensions * extensionUnit + extensionUnit / 2)
                    // update and extend ground
                    {
                        extensions++;
                        ground.extend(terrainExtensionFactor);
                        if (terrainExtensionFactor <= 2.0)
                        { // Cap at 2.0
                            terrainExtensionFactor += conf::terrainExtensionIncrement;
                        }
                    }
                }
                frames++;
                simTimeElapsed += (1000.0f / conf::fps) * conf::timeMultiplier;

                if (frames == (int)conf::fps / conf::timeMultiplier)
                {
                    frames = 0;
                    if (currMaxDist <= timeGoal + 10)
                    {
                        // next Car checks and steps
                        if (currMaxDist <= conf::minScore)
                        {
                            currMaxDist = conf::minScore;
                        }
                        // Send score back to master process
                        MPI_Send(&currMaxDist, 1, MPI_FLOAT, 0, 1, MPI_COMM_WORLD); // tag is 1
                        // car.destroy();
                        finished = true;
                        terrainExtensionFactor = 1.0f;
                    }
                    else
                    {
                        timeGoal = currMaxDist;
                    }
                }

                if (!finished)
                {
                    world.Step((1.0f / (float)conf::fps) * conf::timeMultiplier,
                               conf::velocityIterations, conf::positionIterations);
                }
            }

            if (currMaxDist >= target_score)
            {                                                               // it didn't get a chance to send so we send info if it exceeds target
                MPI_Send(&currMaxDist, 1, MPI_FLOAT, 0, 1, MPI_COMM_WORLD); // tag is 1
            }

            std::cout << "Processor " << rank << " done processing, my car got:  " << currMaxDist << "m" << std::endl;
        }
        MPI_Barrier(MPI_COMM_WORLD); // wait for all sims to be done
        // Master rank recieves all the genomes from all slave processes
        std::cout << "Processor " << rank << " over here! " << std::endl;
        if (rank == 0)
        {
            std::cout << "Processor " << rank << " Calculating! " << std::endl;
            for (int i = 0; i < numCars; ++i)
            {
                MPI_Recv(&population.scores[i], 1, MPI_FLOAT, i + 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE); // tag is 1
            }
            // update recordDist
            for (int i = 0; i < numCars; ++i)
            {
                if (population.scores[i] > recordDist)
                {
                    recordDist = population.scores[i];
                    bestCarGenome = population.car[i];
                }
            }

            std::cout << "Record Distance reached by population:" << recordDist << "m" << std::endl;

            // if target_dist reached finish entire script
            if (recordDist >= target_score)
            {
                global_finished = true;
                std::cout << "FINISHED script must END!" << std::endl;
            }
        }

        // master process does neccesary GA work and sends also if things need to stop
        if (rank == 0)
        {
            generationNumber++;
            population.passOn();
            for (int i = 0; i < numCars; ++i)
            {
                MPI_Send(&population.car[i], 1, GENOME, i + 1, 0, MPI_COMM_WORLD);
            }
        }
        else
        {
            if (rank < numCars + 1)
            {
                MPI_Recv(&my_car, 1, GENOME, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }

        MPI_Bcast(&global_finished, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD); // all processes know when to quit globally and also acts as a barrier to ensure all processes get their new set of cars
        // std::cout << "Here now!: rank: " << rank << " global finished: " << global_finished << std::endl;
        // MPI_Barrier(MPI_COMM_WORLD);
    }

    auto end_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = end_time - start_time;
    double seconds = diff.count();

    // Finalize
    if (rank == 0)
    {
        printf("Best car distance: %f\n", recordDist);
        std::cout << "Simulation Time = " << seconds << " seconds for " << numCars
                  << std::endl;

        // Always save best car genome
        json j = {{"bestCarGenome",
                   {
                       {"legAngleWeight", bestCarGenome.legAngleWeight},
                       {"legLength", bestCarGenome.legLength},
                       {"wheelLeg", bestCarGenome.wheelLeg},
                       {"wheelLength", bestCarGenome.wheelLength},
                       {"color", bestCarGenome.color},
                   }},
                  {"conf",
                   {
                       {"windowLength", conf::windowLength},
                       {"windowHeight", conf::windowHeight},
                       {"fps", conf::fps},
                       {"timeMultiplier", conf::timeMultiplier},
                       {"velocityIterations", conf::velocityIterations},
                       {"positionIterations", conf::positionIterations},
                       {"gravity", {conf::gravity.x, conf::gravity.y}},
                       {"minAngleWeight", conf::minAngleWeight},
                       {"maxLength", conf::maxLength},
                       {"maxWheelSize", conf::maxWheelSize},
                       {"carDensity", conf::carDensity},
                       {"carFriction", conf::carFriction},
                       {"carRestitution", conf::carRestitution},
                       {"wheelLocationRatio", conf::wheelLocationRatio},
                       {"wheelDensity", conf::wheelDensity},
                       {"wheelFriction", conf::wheelFriction},
                       {"wheelRestitution", conf::wheelRestitution},
                       {"axisSpeed", conf::axisSpeed},
                       {"maxTorque", conf::maxTorque},
                       {"drawScale", conf::drawScale},
                       {"drawBorder", conf::drawBorder},
                       {"groundThickness", conf::groundThickness},
                       {"extendNumber", conf::extendNumber},
                       {"extendUnitLength", conf::extendUnitLength},
                       {"terrainSeed", conf::terrainSeed},
                       {"genomeSeed", conf::genomeSeed},
                       {"baseHillyChange", conf::baseHillyChange},
                       {"terrainExtensionIncrement", conf::terrainExtensionIncrement},
                       {"mutationRate", conf::mutationRate},
                       {"populationSize", conf::populationSize},
                       {"minScore", conf::minScore},
                   }},
                  {"generationNumber", generationNumber},
                  {"simTimeElapsed", seconds},
                  {"distance", recordDist}};

        std::ofstream o(conf::saveConfigName);
        if (o)
        {
            o << j << std::endl;
            o.close();
            printf("Saved best car genome to %s\n", conf::saveConfigName);
        }
        else
        {
            std::cerr << "Error opening file for writing" << std::endl;
        }
    }

    // delete &population; #TODO: should I delete stuff?
    // delete &my_car;
    MPI_Finalize();
    return 0;
}
