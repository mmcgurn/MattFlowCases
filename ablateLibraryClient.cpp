#include <environment/runEnvironment.hpp>
#include <mathFunctions/functionFactory.hpp>
#include <memory>
#include <mesh/boxMesh.hpp>
#include <monitors/flow/hdf5OutputFlow.hpp>
#include <parameters/mapParameters.hpp>
#include "builder.hpp"
#include "flow/flow.hpp"
#include "flow/incompressibleFlow.hpp"
#include "solve/timeStepper.hpp"
#include "utilities/petscOptions.hpp"

int main(int argc, char **argv) {
    // initialize petsc and mpi
    PetscErrorCode ierr = PetscInitialize(&argc, &argv, NULL, NULL);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);

    {
        // setup the run environment
        ablate::parameters::MapParameters runEnvironmentParameters(std::map<std::string, std::string>{{"title", "clientExample"}});
        ablate::environment::RunEnvironment::Setup(runEnvironmentParameters);

        // setup any global arguments
        ablate::utilities::PetscOptions::Set({{"dm_plex_separate_marker", ""}});

        // create a time stepper
        auto timeStepper = ablate::solve::TimeStepper("timeStepper",
                                                      {{"ts_dt", ".01"},
                                                       {"ts_max_steps", "15"},
                                                       {"ksp_type", "fgmres"},
                                                       {"ksp_gmres_restart", "10"},
                                                       {"ksp_rtol", "1.0e-9"},
                                                       {"ksp_atol", "1.0e-14"},
                                                       {"ksp_error_if_not_converged", ""},
                                                       {"pc_type", "fieldsplit"},
                                                       {"pc_fieldsplit_0_fields", "0,2"},
                                                       {"pc_fieldsplit_1_fields", "1"},
                                                       {"pc_fieldsplit_type", "schur"},
                                                       {"pc_fieldsplit_schur_factorization_type", "full"},
                                                       {"fieldsplit_0_pc_type", "lu"},
                                                       {"fieldsplit_pressure_ksp_rtol", "1E-10"},
                                                       {"fieldsplit_pressure_ksp_atol", "1E-12"},
                                                       {"fieldsplit_pressure_pc_type", "jacobi"}});

        auto mesh = std::make_shared<ablate::mesh::BoxMesh>("simpleMesh",
                                                            std::map<std::string, std::string>{{"dm_refine", "0"},
                                                                                               {"vel_petscspace_degree", "3"},
                                                                                               {"pres_petscspace_degree", "2"},
                                                                                               {"temp_petscspace_degree", "2"}},
                                                            std::vector<int>{2, 3},
                                                            std::vector<double>{.1, .1},
                                                            std::vector<double>{.2, .2});

        // setup a flow parameters
        auto parameters = std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{{"strouhal", "1.0"},
                                                                                                                 {"reynolds", "33149.2"},
                                                                                                                 {"peclet", "23469.3"},
                                                                                                                 {"froude", "0.451754"},
                                                                                                                 {"heatRelease", "0.0"},
                                                                                                                 {"gamma", "0.0"},
                                                                                                                 {"pth", "1.0"},
                                                                                                                 {"beta", "0.0"},
                                                                                                                 {"mu", "1.0"},
                                                                                                                 {"k", "1.0"},
                                                                                                                 {"cp", "1.0"},
                                                                                                                 {"gravityDirection", "1"}});

        // create a simple flow
        auto flow = std::make_shared<ablate::flow::IncompressibleFlow>(
            "FlowField",
            mesh,
            std::map<std::string, std::string>{},
            parameters,
            std::vector<std::shared_ptr<ablate::flow::FlowFieldSolution>>{
                std::make_shared<ablate::flow::FlowFieldSolution>(
                    "velocity", ablate::mathFunctions::Create("t + x^2 + y^2, t + 2*x^2 - x*y"), ablate::mathFunctions::Create("1.0, 1.0")),
                std::make_shared<ablate::flow::FlowFieldSolution>(
                    "pressure", ablate::mathFunctions::Create("x + y - 1"), ablate::mathFunctions::Create("1.0")),
                std::make_shared<ablate::flow::FlowFieldSolution>(
                    "temperature", ablate::mathFunctions::Create("t + x + y"), ablate::mathFunctions::Create("1.0")),
            },
            std::vector<std::shared_ptr<ablate::flow::BoundaryCondition>>{
                std::make_shared<ablate::flow::BoundaryCondition>(
                    "velocity",
                    "wall velocity",
                    "marker",
                    1,
                    ablate::mathFunctions::Create([](PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar* u, auto ctx ){
                        u[0] = time + x[0]*x[0] + x[1]*x[1];
                        u[1] = time + 2*x[0]*x[0] - x[0]*x[1];
                        return 0;
                    }),/**example showing lambda input**/
                    ablate::mathFunctions::Create("1.0, 1.0")),
                std::make_shared<ablate::flow::BoundaryCondition>(
                    "temperature", "wall temp", "marker", 1, ablate::mathFunctions::Create("t + x + y + z"), ablate::mathFunctions::Create("1.0"))},
            std::vector<std::shared_ptr<ablate::flow::FlowFieldSolution>>{});

        // assume one flow field right now
        flow->SetupSolve(timeStepper.GetTS());

        // setup a monitor
        auto monitor = std::make_shared<ablate::monitors::flow::Hdf5OutputFlow>();
        monitor->Register(flow);
        timeStepper.AddMonitor(monitor);

        // run
        timeStepper.Solve(flow);
    }
    return PetscFinalize();
}