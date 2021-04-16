#include <petsc.h>
#include "compressibleFlow.h"
#include "mesh.h"

#define DIM 2                   /* Geometric dimension */

typedef struct {
    PetscReal gamma;
    PetscReal rhoIn;
    PetscReal rhoOut;
    PetscReal uIn;
    PetscReal uOut;
    PetscReal pIn;
    PetscReal pOut;
    PetscReal r;
} InitialConditions;

typedef struct {
    InitialConditions initialConditions;
} ProblemSetup;

static PetscReal Radius(PetscInt dim, const PetscReal x[]){
    PetscReal r = 0.0;
    for(PetscInt i =0; i < dim; i++){
        r += PetscSqr(x[i]);
    }
    return PetscSqrtReal(r);
}

static void Norm(PetscInt dim, const PetscReal in[], PetscReal out[]){
    PetscReal mag = 0.0;
    for(PetscInt i =0; i < dim; i++){
        mag += PetscSqr(in[i]);
    }
    mag = PetscSqrtReal(mag);

    for(PetscInt i =0; i < dim; i++){
        out[i] = in[i]/mag;
    }
}

static PetscErrorCode SetInitialRho(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx) {
    InitialConditions *initialConditions = (InitialConditions *)ctx;

    PetscReal r = Radius(dim, x);

    if (r < initialConditions->r) {
        u[0] = initialConditions->rhoIn;
    } else {
        u[0] = initialConditions->rhoOut;
    }

    return 0;
}

static PetscErrorCode SetInitialRhoU(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx) {
    InitialConditions *initialConditions = (InitialConditions *)ctx;

    PetscReal r = Radius(dim, x);
    PetscReal outwardDir[3];
    Norm(dim, x, outwardDir);

    if (r < initialConditions->r) {
        u[0] = initialConditions->rhoIn * initialConditions->uIn*outwardDir[0];
        u[1] = initialConditions->rhoIn * initialConditions->uIn*outwardDir[1];
    } else {
        u[0] = initialConditions->rhoOut * initialConditions->uOut*outwardDir[0];
        u[1] = initialConditions->rhoOut * initialConditions->uOut*outwardDir[1];
    }
    return 0;
}

static PetscErrorCode SetInitialRhoE(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx) {
    InitialConditions *initialConditions = (InitialConditions *)ctx;

    PetscReal r = Radius(dim, x);
    if (r < initialConditions->r) {
        PetscReal e = initialConditions->pIn / ((initialConditions->gamma - 1.0) * initialConditions->rhoIn);
        PetscReal et = e + 0.5 * PetscSqr(initialConditions->uIn);
        u[0] = et * initialConditions->rhoIn;
    } else {
        PetscReal e = initialConditions->pOut / ((initialConditions->gamma - 1.0) * initialConditions->rhoOut);
        PetscReal et = e + 0.5 * PetscSqr(initialConditions->uOut);
        u[0] = et * initialConditions->rhoOut;
    }
    return 0;
}

static PetscErrorCode MonitorError(TS ts, PetscInt step, PetscReal time, Vec u, void *ctx) {
    PetscFunctionBeginUser;
    PetscErrorCode     ierr;

    // Get the DM
    DM dm;
    ierr = TSGetDM(ts, &dm);CHKERRQ(ierr);

    ierr = VecViewFromOptions(u, NULL, "-vec_view");CHKERRQ(ierr);
    ierr = PetscPrintf(PetscObjectComm((PetscObject)dm), "TS at %f\n", time);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

static PetscErrorCode PhysicsBoundary_Euler(PetscReal time, const PetscReal *c, const PetscReal *n, const PetscScalar *a_xI, PetscScalar *a_xG, void *ctx) {
    InitialConditions *initialConditions = (InitialConditions *)ctx;

    a_xG[0] = initialConditions->rhoOut;

    PetscReal outwardDir[3];
    Norm(2, c, outwardDir);

    a_xG[1] = initialConditions->rhoOut * initialConditions->uOut*outwardDir[0];
    a_xG[2] = initialConditions->rhoOut * initialConditions->uOut*outwardDir[1];


    PetscReal e = initialConditions->pOut / ((initialConditions->gamma - 1.0) * initialConditions->rhoOut);
    PetscReal et = e + 0.5 * PetscSqr(initialConditions->uOut);
    a_xG[3] = et * initialConditions->rhoOut;

    PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
    PetscErrorCode ierr;
    // create the mesh
    // setup the ts
    DM dm;                 /* problem definition */
    TS ts;                 /* timestepper */

    // initialize petsc and mpi
    PetscInitialize(&argc, &argv, NULL, "HELP");

    // Create a ts
    ierr = TSCreate(PETSC_COMM_WORLD, &ts);CHKERRQ(ierr);
    ierr = TSSetProblemType(ts, TS_NONLINEAR);CHKERRQ(ierr);
    ierr = TSSetType(ts, TSEULER);CHKERRQ(ierr);
    ierr = TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);

    // Create a mesh
    // hard code the problem setup
    PetscReal start[] = {-1, -1};
    PetscReal end[] = {1.0, 1};
    PetscInt nx[] = {50, 50};
    DMBoundaryType bcType[] = {DM_BOUNDARY_NONE, DM_BOUNDARY_NONE};
    ierr = DMPlexCreateBoxMesh(PETSC_COMM_WORLD, DIM, PETSC_FALSE, nx, start, end, bcType, PETSC_TRUE, &dm);CHKERRQ(ierr);

// Setup the problem
    ProblemSetup problem;

    // case 2
    problem.initialConditions.rhoIn=1.0;
    problem.initialConditions.uIn=0.0;
    problem.initialConditions.pIn=3;
    problem.initialConditions.rhoOut=1.0;
    problem.initialConditions.uOut=0.0;
    problem.initialConditions.pOut=1.0;
    problem.initialConditions.r = .25;
    problem.initialConditions.gamma = 1.4;

    // Setup the flow data
    FlowData flowData;     /* store some of the flow data*/
    ierr = FlowCreate(&flowData);CHKERRQ(ierr);

    //Setup
    CompressibleFlow_SetupDiscretization(flowData, dm);

    // Add in the flow parameters
    PetscScalar params[TOTAL_COMPRESSIBLE_FLOW_PARAMETERS];
    params[CFL] = 0.5;
    params[GAMMA] = problem.initialConditions.gamma;

    // set up the finite volume fluxes
    CompressibleFlow_StartProblemSetup(flowData, TOTAL_COMPRESSIBLE_FLOW_PARAMETERS, params);
    DMView(flowData->dm, PETSC_VIEWER_STDERR_SELF);
    // Add in any boundary conditions
    PetscDS prob;
    ierr = DMGetDS(flowData->dm, &prob);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);
    const PetscInt idsLeft[]= {1, 2, 3, 4};
    ierr = PetscDSAddBoundary(prob, DM_BC_NATURAL_RIEMANN, "wall left", "Face Sets", 0, 0, NULL, (void (*)(void))PhysicsBoundary_Euler, NULL, 4, idsLeft, &problem);CHKERRQ(ierr);

    // Complete the problem setup
    ierr = CompressibleFlow_CompleteProblemSetup(flowData, ts);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);

    // Name the flow field
    ierr = PetscObjectSetName(((PetscObject)flowData->flowField), "Numerical Solution");
    CHKERRABORT(PETSC_COMM_WORLD, ierr);

    // Setup the TS
    ierr = TSSetFromOptions(ts);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);
    ierr = TSMonitorSet(ts, MonitorError, &problem, NULL);CHKERRQ(ierr);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);
    ierr = TSSetMaxTime(ts, 1.0);CHKERRQ(ierr);

    // set the initial conditions
    PetscErrorCode     (*func[3]) (PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx) = {SetInitialRho, SetInitialRhoU, SetInitialRhoE};
    void* ctxs[3] ={&problem, &problem, &problem};
    ierr    = DMProjectFunction(flowData->dm,0.0,func,ctxs,INSERT_ALL_VALUES,flowData->flowField);CHKERRQ(ierr);

    // Output the mesh
    ierr = DMViewFromOptions(flowData->dm, NULL, "-dm_view");CHKERRQ(ierr);

    ierr = TSSolve(ts,flowData->flowField);CHKERRQ(ierr);

    return PetscFinalize();

}